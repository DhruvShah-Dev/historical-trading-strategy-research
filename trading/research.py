"""Leakage-aware five-session direction model and comparable strategy backtests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import SYMBOLS, DataError

HORIZON = 5
EMBARGO = HORIZON + 1  # The label exits at open t+6.
COST_PER_SIDE = 0.001
FEATURES = ("return_1", "return_5", "return_20", "volatility_20", "close_to_sma20", "sma20_to_sma50", "rsi_14", "volume_ratio")


@dataclass
class ResearchResult:
    panel: pd.DataFrame
    scored: pd.DataFrame
    performance: pd.DataFrame
    evaluation: pd.DataFrame
    test_dates: tuple
    rule_history: pd.DataFrame
    rule_performance: pd.DataFrame


def add_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    close = group["close"]
    returns = close.pct_change(fill_method=None)
    group["return_1"] = returns
    group["return_5"] = close.pct_change(5, fill_method=None)
    group["return_20"] = close.pct_change(20, fill_method=None)
    group["volatility_20"] = returns.rolling(20).std()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    group["close_to_sma20"] = close / sma20 - 1
    group["sma20_to_sma50"] = sma20 / sma50 - 1
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    group["rsi_14"] = 100 * gain / (gain + loss)
    group["volume_ratio"] = group["volume"] / group["volume"].rolling(20).mean() - 1
    # Signal is formed after close t. Entry is the next open; exit is five opens later.
    entry = group["open"].shift(-1)
    exit_ = group["open"].shift(-(HORIZON + 1))
    forward_dividends = sum(group["dividend"].shift(-i) for i in range(2, HORIZON + 2))
    group["forward_return_5"] = (exit_ + forward_dividends) / entry - 1
    group["target_up_5"] = (group["forward_return_5"] > 0).astype(float)
    group.loc[group["forward_return_5"].isna(), "target_up_5"] = np.nan
    # Holding at open t receives the dividend whose ex-date is the next open.
    group["next_open_return"] = (group["open"].shift(-1) + group["dividend"].shift(-1)) / group["open"] - 1
    group["sma_signal"] = (sma20 > sma50).astype(int)
    rsi_position = 0
    rsi_signals = []
    for rsi in group["rsi_14"]:
        if pd.notna(rsi) and rsi <= 30:
            rsi_position = 1
        elif pd.notna(rsi) and rsi >= 70:
            rsi_position = 0
        rsi_signals.append(rsi_position)
    group["rsi_signal"] = rsi_signals
    return group


def prepare_panel(prices: pd.DataFrame) -> pd.DataFrame:
    if set(prices["symbol"]) != set(SYMBOLS):
        raise DataError("The full watchlist is required for pooled modeling.")
    panel = pd.concat([add_features(group) for _, group in prices.groupby("symbol", sort=False)], ignore_index=True)
    panel[list(FEATURES)] = panel[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def split_dates(panel: pd.DataFrame) -> dict[str, list]:
    # Use shared dates with complete features and matured five-session labels.
    eligible = panel.dropna(subset=[*FEATURES, "target_up_5"])
    counts = eligible.groupby("date")["symbol"].nunique()
    dates = sorted(counts[counts == len(SYMBOLS)].index)
    if len(dates) < 350:
        raise DataError(f"Only {len(dates)} complete labeled dates; need at least 350.")
    test = dates[-126:]
    validation = dates[-(126 + EMBARGO + 63):-(126 + EMBARGO)]
    train = dates[:-(126 + EMBARGO + 63 + EMBARGO)]
    if len(train) < 150 or len(validation) != 63 or len(test) != 126:
        raise DataError("Insufficient history for time-ordered train, validation, and test windows.")
    return {"train": train, "validation": validation, "test": test}


def _model() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, list(FEATURES)),
        ("symbol", OneHotEncoder(handle_unknown="ignore"), ["symbol"]),
    ])
    return Pipeline([("prepare", preprocessor), ("classify", LogisticRegression(max_iter=1000, C=0.5))])


def _evaluation(frame: pd.DataFrame, segment: str) -> dict:
    actual = frame["target_up_5"].astype(int)
    probability = frame["probability_up_5"]
    return {
        "segment": segment,
        "observations": len(frame),
        "accuracy": accuracy_score(actual, probability >= 0.5),
        "brier_score": brier_score_loss(actual, probability),
        "roc_auc": roc_auc_score(actual, probability) if actual.nunique() == 2 else np.nan,
        "positive_rate": actual.mean(),
    }


def _backtest_symbol(group: pd.DataFrame, test_dates: set) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    signals = {
        "Buy and hold": pd.Series(1, index=group.index),
        "Moving average 20/50": group["sma_signal"],
        "RSI 14 (30/70)": group["rsi_signal"],
        "Model (p ≥ 0.50)": (group["probability_up_5"] >= 0.5).astype(int),
    }
    rows = []
    for name, signal in signals.items():
        # Yesterday's close determines today's opening position.
        position = (signal if name == "Buy and hold" else signal.shift(1).fillna(0)).astype(int)
        selected = group.loc[group["date"].isin(test_dates) & group["next_open_return"].notna()].copy()
        selected["strategy"] = name
        selected["position"] = position.loc[selected.index]
        selected["trade_sides"] = selected["position"].diff().abs().fillna(selected["position"]).astype(int)
        selected["strategy_return"] = selected["position"] * selected["next_open_return"] - COST_PER_SIDE * selected["trade_sides"]
        selected["equity"] = (1 + selected["strategy_return"]).cumprod()
        rows.append(selected[["date", "symbol", "strategy", "position", "trade_sides", "strategy_return", "equity"]])
    return pd.concat(rows, ignore_index=True)


def _performance(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (symbol, strategy), group in scored.groupby(["symbol", "strategy"], sort=False):
        group = group.sort_values("date")
        returns = group["strategy_return"]
        equity = group["equity"]
        peak = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)]).cummax().iloc[1:].to_numpy()
        annual_vol = returns.std(ddof=1) * np.sqrt(252)
        rows.append({
            "symbol": symbol,
            "strategy": strategy,
            "sessions": len(group),
            "total_return": equity.iloc[-1] - 1,
            "annualized_return": equity.iloc[-1] ** (252 / len(group)) - 1,
            "annualized_volatility": annual_vol,
            "sharpe_zero_cash_rate": returns.mean() / returns.std(ddof=1) * np.sqrt(252) if returns.std(ddof=1) > 0 else np.nan,
            "max_drawdown": np.min(equity.to_numpy() / peak - 1),
            "trade_sides": int(group["trade_sides"].sum()),
            "exposure": group["position"].mean(),
        })
    return pd.DataFrame(rows)


def run_research(prices: pd.DataFrame) -> ResearchResult:
    panel = prepare_panel(prices)
    windows = split_dates(panel)
    train = panel[panel["date"].isin(windows["train"])].dropna(subset=[*FEATURES, "target_up_5"])
    validation = panel[panel["date"].isin(windows["validation"])].dropna(subset=[*FEATURES, "target_up_5"])
    test = panel[panel["date"].isin(windows["test"])].dropna(subset=[*FEATURES, "target_up_5"])
    if train["target_up_5"].nunique() < 2:
        raise DataError("Training labels have only one class.")
    model = _model()
    model.fit(train[[*FEATURES, "symbol"]], train["target_up_5"].astype(int))
    eligible_features = panel.dropna(subset=list(FEATURES))
    panel["probability_up_5"] = np.nan
    panel.loc[eligible_features.index, "probability_up_5"] = model.predict_proba(eligible_features[[*FEATURES, "symbol"]])[:, 1]
    validation = panel.loc[validation.index]
    test = panel.loc[test.index]
    test_dates = set(windows["test"])
    scored = pd.concat([_backtest_symbol(group, test_dates) for _, group in panel.groupby("symbol", sort=False)], ignore_index=True)
    if scored["strategy_return"].isna().any():
        raise DataError("Backtest contains a missing interval return.")
    rule_history = pd.concat(
        [_backtest_symbol(group, set(group["date"])) for _, group in panel.groupby("symbol", sort=False)],
        ignore_index=True,
    )
    rule_history = rule_history[rule_history["strategy"] != "Model (p ≥ 0.50)"].reset_index(drop=True)
    evaluation = pd.DataFrame([_evaluation(validation, "Validation"), _evaluation(test, "Holdout")])
    return ResearchResult(panel, scored, _performance(scored), evaluation, tuple(windows["test"]), rule_history, _performance(rule_history))
