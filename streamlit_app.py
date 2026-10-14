from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trading.data import SYMBOLS, DataError, load_panel
from trading.research import run_research

st.set_page_config(page_title="Trading Strategy Research", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Training model and running backtests…")
def load_research(data_mtime: float):
    del data_mtime
    return run_research(load_panel())


st.title("Historical trading strategy research")
st.caption("US stocks and ETFs · end-of-day data · five-session direction · personal research")

try:
    data_file = Path(__file__).resolve().parent / "data" / "processed" / "daily_prices.csv"
    result = load_research(data_file.stat().st_mtime)
except (DataError, OSError, ValueError) as exc:
    st.info(str(exc))
    st.code("$env:MASSIVE_API_KEY='your-key'\npython scripts/refresh_data.py", language="powershell")
    st.stop()

symbol = st.sidebar.selectbox("Symbol", SYMBOLS)
st.sidebar.caption("Refresh data from the terminal: `python scripts/refresh_data.py`")
price = result.panel[result.panel["symbol"] == symbol].sort_values("date")
scored = result.scored[result.scored["symbol"] == symbol]
performance = result.performance[result.performance["symbol"] == symbol].copy()
holdout = result.evaluation[result.evaluation["segment"] == "Holdout"].iloc[0]
latest = price.iloc[-1]
first_test, last_test = result.test_dates[0], result.test_dates[-1]

st.caption(f"Source: Massive · available data: {price['date'].min()} to {latest['date']} ({len(price):,} daily bars) · model holdout: {first_test} to {last_test} ({len(result.test_dates)} sessions)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest split-adjusted close", f"${latest['close']:,.2f}")
col2.metric("Five-session up probability", f"{latest['probability_up_5']:.1%}")
col3.metric("Holdout direction accuracy", f"{holdout['accuracy']:.1%}")
col4.metric("Holdout observations", f"{int(holdout['observations']):,} across 6 symbols")
baseline_direction = "up" if holdout["positive_rate"] >= 0.5 else "down"
baseline_accuracy = max(holdout["positive_rate"], 1 - holdout["positive_rate"])
if holdout["accuracy"] <= baseline_accuracy or holdout["roc_auc"] <= 0.5:
    st.warning(
        f"No demonstrated predictive edge on this holdout: model accuracy {holdout['accuracy']:.1%}, "
        f"always-{baseline_direction} accuracy {baseline_accuracy:.1%}, ROC AUC {holdout['roc_auc']:.3f}. "
        "Treat the latest probability as an experimental estimate."
    )

st.subheader("Price and model signal")
chart = px.line(price, x="date", y="close", title=f"{symbol} split-adjusted close")
chart.update_layout(yaxis_title="USD", xaxis_title="", height=340)
st.plotly_chart(chart, width="stretch")
prob = px.line(price.dropna(subset=["probability_up_5"]), x="date", y="probability_up_5", title="Estimated probability of a positive five-session return")
prob.add_hline(y=0.5, line_dash="dash", line_color="gray")
prob.update_layout(yaxis_title="Probability", yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="", height=280)
st.plotly_chart(prob, width="stretch")

st.subheader("Rule-based strategies across the available data")
long_rules = result.rule_history[result.rule_history["symbol"] == symbol]
long_performance = result.rule_performance[result.rule_performance["symbol"] == symbol]
long_chart = px.line(long_rules, x="date", y="equity", color="strategy", title="Growth of $1 from the first available date")
long_chart.update_layout(yaxis_title="Portfolio value ($)", xaxis_title="", height=380, legend_title_text="Strategy")
st.plotly_chart(long_chart, width="stretch")
long_display = long_performance[["strategy", "total_return", "max_drawdown", "trade_sides"]].copy()
long_display.columns = ["Strategy", "Total return", "Max drawdown", "Trade sides"]
st.dataframe(long_display.style.format({"Total return": "{:.1%}", "Max drawdown": "{:.1%}"}), hide_index=True, width="stretch")
st.caption("These fixed rules use the complete downloaded window. The model appears in the out-of-sample comparison below.")

st.subheader("Strategy comparison on the same holdout dates")
equity_chart = px.line(scored, x="date", y="equity", color="strategy", title="Growth of $1, after transaction costs and with dividends")
equity_chart.update_layout(yaxis_title="Portfolio value ($)", xaxis_title="", height=380, legend_title_text="Strategy")
st.plotly_chart(equity_chart, width="stretch")
display = performance[["strategy", "total_return", "annualized_return", "max_drawdown", "annualized_volatility", "sharpe_zero_cash_rate", "trade_sides", "exposure"]].copy()
display.columns = ["Strategy", "Total return", "Annualized return", "Max drawdown", "Annualized volatility", "Sharpe (0% cash)", "Trade sides", "Exposure"]
st.dataframe(display.style.format({"Total return": "{:.1%}", "Annualized return": "{:.1%}", "Max drawdown": "{:.1%}", "Annualized volatility": "{:.1%}", "Sharpe (0% cash)": "{:.2f}", "Exposure": "{:.1%}"}), hide_index=True, width="stretch")

st.subheader("Model evaluation")
eval_display = result.evaluation.copy()
eval_display.columns = ["Segment", "Observations", "Accuracy", "Brier score", "ROC AUC", "Positive rate"]
st.dataframe(eval_display.style.format({"Accuracy": "{:.1%}", "Brier score": "{:.3f}", "ROC AUC": "{:.3f}", "Positive rate": "{:.1%}"}), hide_index=True, width="stretch")
st.caption("Validation and holdout results pool all six symbols. Lower Brier score is better; 0.5 ROC AUC is chance level. The model trains on older dates only.")

st.subheader("Download results")
st.download_button("Download selected symbol's backtest CSV", scored.to_csv(index=False).encode("utf-8"), f"{symbol.lower()}_strategy_backtest.csv", "text/csv")

with st.expander("Method and limits"):
    st.markdown("""
- Features use prices and volume available through each day's close. Trades execute at the next session's open. The model predicts whether the **next five open-to-open sessions**, including dividends, have a positive return.
- All strategies are long or cash. The moving-average rule holds when SMA 20 exceeds SMA 50. RSI enters at 30 or below and exits at 70 or above. The model holds when its probability is at least 50%.
- Massive prices are split-adjusted; cash dividends are credited on the ex-date to positions held across that opening. Costs are 0.10% each time a position enters or exits. Cash earns 0%; taxes, financing, and market impact are excluded.
- The latest probability is a model estimate, not a validated forecast. The fixed watchlist is selected in hindsight, and six months of holdout performance cannot establish future profitability.
""")
