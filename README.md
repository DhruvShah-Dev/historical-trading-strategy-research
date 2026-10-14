# Historical Trading Strategy Research

A reproducible Python and Streamlit project for exploring US stock and ETF history, testing trading rules, and evaluating a five-session direction model. It downloads end-of-day prices and dividends from Massive for **SPY, QQQ, IWM, AAPL, MSFT, and NVDA**. The dashboard compares buy-and-hold, moving-average, RSI, and model-based long-or-cash strategies. This is a personal research tool; it does not place trades.

## What the dashboard shows

- Price history, the latest model probability, and the data's last available date.
- Buy-and-hold, 20/50-day moving-average, and RSI 14 rules across the downloaded window.
- A four-strategy comparison on the same 126-session out-of-sample holdout, including 0.10% per transaction side and cash dividends.
- Direction accuracy, Brier score, and ROC AUC, with a simple direction baseline when the model has no demonstrated edge.
- A downloadable CSV of the selected symbol's holdout backtest.

The free Massive Stocks Basic tier currently provides a rolling **two-year** history and five API requests per minute. The downloader spaces its calls by at least 12.5 seconds. See [Massive pricing](https://massive.com/pricing?product=stocks) for current limits.

## Quick start

Run these commands from this repository in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MASSIVE_API_KEY = 'your-massive-api-key'
.\.venv\Scripts\python.exe scripts\refresh_data.py
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Get a Stocks Basic API key from [Massive](https://massive.com/). Keep the key in your local environment; never commit it. To update prices later, set the environment variable in that PowerShell session and rerun `scripts\refresh_data.py`.

The refresh saves source responses in `data/raw/massive_responses.json` and validated records in `data/processed/daily_prices.csv`. Both directories are ignored by Git. The dashboard needs a successful download and never substitutes generated market data. A failed download or validation leaves the previous saved dataset intact.

## Research design

| Component | Method |
| --- | --- |
| Data | Daily split-adjusted OHLCV bars and split-adjusted cash dividends from Massive. |
| Prediction target | Whether total return from the next session's open through the open five sessions later is positive. |
| Features | Past 1-, 5-, and 20-day returns, 20-day volatility, moving-average ratios, RSI 14, volume ratio, and symbol. |
| Model | One pooled logistic regression for the six symbols; a fixed 50% threshold forms a long-or-cash rule. |
| Evaluation | Older training dates, six-session gap, 63 validation sessions, six-session gap, then the latest 126 labeled sessions as holdout. |
| Execution | A signal formed after a close takes effect at the next open; positions earn the following open-to-open return. |
| Costs | 0.10% per entry or exit side; cash earns 0%. |

Massive's bars are [adjusted for splits, not dividends](https://massive.com/knowledge-base/article/is-massives-stock-data-adjusted-for-splits-or-dividends). The project uses its [dividend records](https://www.massive.com/docs/rest/stocks/corporate-actions/dividends) to credit cash payouts to positions held across the ex-date open. The six-session gaps keep training outcomes from using prices in the next evaluation segment. All four strategies use identical holdout dates.

The strategy table reports cumulative and annualized return, volatility, maximum drawdown, trade sides, exposure, and a Sharpe ratio with a zero cash rate. Annualized results from a six-month holdout are unstable. A model probability is an estimate, not a validated forecast.

## Project layout

```text
trading/data.py         Massive requests, normalization, and data checks
trading/research.py     Features, model, evaluation, and backtests
scripts/refresh_data.py  Manual download command
streamlit_app.py        Interactive dashboard
tests/test_research.py  Timing, data-quality, and calculation checks
```

Run the checks with `python -m pytest -q` (or `.\.venv\Scripts\python.exe -m pytest -q` on Windows).

## Limits

The fixed watchlist was selected in hindsight and excludes delisted symbols. Two years of free-tier history and six months of holdout results cannot establish future profitability. The simulation does not account for taxes, cash interest, financing, market impact, or partial fills. Results depend on the download date because the free history window rolls forward.
