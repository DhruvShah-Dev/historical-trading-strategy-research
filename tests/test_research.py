from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading import data
from trading.data import SYMBOLS, DataError, normalize_symbol, validate_panel
from trading.research import COST_PER_SIDE, EMBARGO, HORIZON, _backtest_symbol, add_features, run_research, split_dates


def sample_prices(length=510):
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=length).date
    frames = []
    for number, symbol in enumerate(SYMBOLS):
        x = np.arange(length)
        close = 100 + 4 * number + 0.04 * x + 4 * np.sin(x / (12 + number))
        frames.append(pd.DataFrame({
            "date": days, "symbol": symbol, "open": close - 0.2,
            "high": close + 1, "low": close - 1, "close": close,
            "volume": 1_000_000 + 1000 * x, "dividend": np.where(x % 60 == 0, 0.25, 0.0),
        }))
    return pd.concat(frames, ignore_index=True)


def test_feature_and_label_timing():
    prices = sample_prices(90).query("symbol == 'SPY'").reset_index(drop=True)
    features = add_features(prices)
    t = 60
    expected = (prices.loc[t + HORIZON + 1, "open"] + prices.loc[t + 2:t + HORIZON + 1, "dividend"].sum()) / prices.loc[t + 1, "open"] - 1
    assert features.loc[t, "forward_return_5"] == pytest.approx(expected)
    assert features.loc[t, "next_open_return"] == pytest.approx((prices.loc[t + 1, "open"] + prices.loc[t + 1, "dividend"]) / prices.loc[t, "open"] - 1)
    altered = prices.copy()
    altered.loc[t + 1:, "close"] *= 1.5
    assert add_features(altered).loc[t, "return_20"] == pytest.approx(features.loc[t, "return_20"])


def test_backtest_next_open_cost_and_dividend():
    frame = pd.DataFrame({
        "date": [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)],
        "symbol": "SPY", "next_open_return": [0.04, -0.02, 0.01],
        "sma_signal": [1, 0, 0], "rsi_signal": [0, 0, 0],
        "probability_up_5": [0.6, 0.4, 0.4],
    })
    result = _backtest_symbol(frame, set(frame["date"]))
    model = result[result["strategy"] == "Model (p ≥ 0.50)"].reset_index(drop=True)
    assert model["position"].tolist() == [0, 1, 0]
    assert model["strategy_return"].tolist() == pytest.approx([0, -0.02 - COST_PER_SIDE, -COST_PER_SIDE])
    hold = result[result["strategy"] == "Buy and hold"].reset_index(drop=True)
    assert hold["strategy_return"].tolist() == pytest.approx([0.04 - COST_PER_SIDE, -0.02, 0.01])


def test_full_research_has_disjoint_windows_and_shared_holdout():
    panel = sample_prices()
    prepared = pd.concat([add_features(group) for _, group in panel.groupby("symbol")])
    windows = split_dates(prepared)
    assert set(windows["train"]).isdisjoint(windows["validation"])
    assert set(windows["validation"]).isdisjoint(windows["test"])
    all_dates = sorted(prepared["date"].unique())
    assert all_dates.index(windows["validation"][0]) - all_dates.index(windows["train"][-1]) > EMBARGO
    result = run_research(panel)
    assert len(result.test_dates) == 126
    assert set(result.scored.groupby(["symbol", "strategy"]).size()) == {126}
    assert not result.scored["strategy_return"].isna().any()
    assert set(result.evaluation["segment"]) == {"Validation", "Holdout"}


def test_normalization_rejects_bad_prices_and_dividends():
    bar_time = int(pd.Timestamp("2026-01-06", tz="America/New_York").timestamp() * 1000)
    bars = {"results": [{"t": bar_time, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}]}
    valid_dividend = [{"id": "one", "ticker": "SPY", "currency": "USD", "ex_dividend_date": "2026-01-06", "split_adjusted_cash_amount": 0.5}]
    assert normalize_symbol("SPY", bars, valid_dividend)["dividend"].iloc[0] == 0.5
    bad = {"results": [{**bars["results"][0], "h": 90}]}
    with pytest.raises(DataError):
        normalize_symbol("SPY", bad, [])
    with pytest.raises(DataError):
        normalize_symbol("SPY", bars, valid_dividend * 2)


def test_panel_validation_rejects_missing_symbol():
    with pytest.raises(DataError, match="watchlist"):
        validate_panel(sample_prices().query("symbol != 'NVDA'"))


def test_panel_validation_rejects_large_shared_gap():
    prices = sample_prices()
    skipped = sorted(prices["date"].unique())[200:210]
    with pytest.raises(DataError, match="gap"):
        validate_panel(prices[~prices["date"].isin(skipped)])


def test_refresh_writes_real_response_shape_and_can_repeat(tmp_path, monkeypatch):
    prices = sample_prices()
    monkeypatch.setattr(data, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path / "processed")

    class FakeClient:
        def __init__(self):
            self.requests = []

        def get(self, path, params):
            self.requests.append((path, params.copy()))
            if "dividends" in path:
                return {"status": "OK", "results": []}
            symbol = path.split("/")[4]
            rows = prices[prices["symbol"] == symbol]
            return {
                "status": "OK", "ticker": symbol, "adjusted": True,
                "results": [{"t": int(pd.Timestamp(row.date, tz="America/New_York").timestamp() * 1000),
                             "o": row.open, "h": row.high, "l": row.low, "c": row.close, "v": row.volume}
                            for row in rows.itertuples(index=False)],
            }

    client = FakeClient()
    for _ in range(2):
        saved = data.refresh(client=client)
        assert len(saved) == 510 * len(SYMBOLS)
        assert len(data.load_panel()) == len(saved)
    assert len(client.requests) == 24
    assert all(params.get("adjusted") == "true" for path, params in client.requests if "/aggs/" in path)
    assert (tmp_path / "raw" / "massive_responses.json").exists()


def test_refresh_rejects_incomplete_source_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path / "processed")
    data.PROCESSED_DIR.mkdir(parents=True)
    saved_file = data.PROCESSED_DIR / "daily_prices.csv"
    saved_file.write_text("previous good data", encoding="utf-8")

    class EmptyClient:
        def get(self, path, params):
            if "dividends" in path:
                return {"status": "OK", "results": []}
            return {"ticker": "SPY", "adjusted": True, "results": []}

    with pytest.raises(DataError, match="No bars"):
        data.refresh(client=EmptyClient())
    assert saved_file.read_text(encoding="utf-8") == "previous good data"
