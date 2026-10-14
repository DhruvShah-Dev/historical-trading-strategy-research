"""Massive end-of-day acquisition and strict source validation."""
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

SYMBOLS = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA")
BASE_URL = "https://api.massive.com"
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MIN_SECONDS_BETWEEN_CALLS = 12.5  # Basic: five requests/minute.


class DataError(ValueError):
    pass


class MassiveClient:
    def __init__(self, api_key: str, session: requests.Session | None = None, throttle: bool = True):
        if not api_key:
            raise DataError("Set MASSIVE_API_KEY before refreshing data.")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.throttle = throttle
        self.last_call = 0.0

    def get(self, path: str, params: dict | None = None) -> dict:
        if self.throttle and self.last_call:
            time.sleep(max(0.0, MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self.last_call)))
        url = path if path.startswith("https://") else BASE_URL + path
        if urlparse(url).netloc != "api.massive.com":
            raise DataError("Unexpected pagination host from data provider.")
        params = dict(params or {})
        params["apiKey"] = self.api_key
        for attempt in range(3):
            self.last_call = time.monotonic()
            try:
                response = self.session.get(url, params=params, timeout=45)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(max(15, int(response.headers.get("Retry-After", "15"))))
                    continue
                if response.status_code >= 400:
                    raise DataError(f"Massive returned HTTP {response.status_code}; check API access and the requested data window.")
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise DataError(f"Massive request failed ({type(exc).__name__}); check connectivity and try again.") from exc
            if payload.get("status") == "ERROR" or "error" in payload:
                raise DataError(f"Massive returned an error: {payload.get('error', payload.get('message', 'unknown'))}")
            return payload
        raise DataError("Massive request failed after retries.")


def fetch_symbol(client: MassiveClient, symbol: str, start: date, end: date) -> tuple[dict, list[dict]]:
    if symbol not in SYMBOLS:
        raise DataError(f"Unsupported symbol: {symbol}")
    bars = client.get(
        f"/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}",
        {"adjusted": "true", "sort": "asc", "limit": 5000},
    )
    if bars.get("next_url"):
        raise DataError(f"Unexpected paginated daily bars for {symbol}; refusing incomplete data.")
    if bars.get("ticker") != symbol or bars.get("adjusted") is not True:
        raise DataError(f"Bars for {symbol} were not confirmed split-adjusted.")
    dividends: list[dict] = []
    next_url = "/stocks/v1/dividends"
    params = {"ticker": symbol, "ex_dividend_date.gte": start.isoformat(), "ex_dividend_date.lte": end.isoformat(), "limit": 5000}
    while next_url:
        page = client.get(next_url, params)
        if page.get("status") != "OK":
            raise DataError(f"Dividend response for {symbol} was not OK.")
        dividends.extend(page.get("results", []))
        next_url = page.get("next_url")
        params = {}  # next_url carries provider pagination filters
    return bars, dividends


def normalize_symbol(symbol: str, bars: dict, dividends: list[dict]) -> pd.DataFrame:
    rows = bars.get("results", [])
    if not rows:
        raise DataError(f"No bars for {symbol}.")
    frame = pd.DataFrame(rows)
    required = {"t", "o", "h", "l", "c", "v"}
    if not required.issubset(frame.columns):
        raise DataError(f"Missing daily price fields for {symbol}.")
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    frame["date"] = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.date
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise DataError(f"Duplicate or unordered dates for {symbol}.")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    prices = frame[["open", "high", "low", "close"]]
    if not np.isfinite(frame[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise DataError(f"Nonfinite prices or volume for {symbol}.")
    if (prices <= 0).any().any() or (frame["volume"] < 0).any():
        raise DataError(f"Invalid nonpositive price or negative volume for {symbol}.")
    if ((frame["high"] < prices[["open", "close", "low"]].max(axis=1)) | (frame["low"] > prices[["open", "close", "high"]].min(axis=1))).any():
        raise DataError(f"Inconsistent OHLC values for {symbol}.")
    amounts: dict[date, float] = {}
    dividend_ids: set[str] = set()
    for dividend in dividends:
        if dividend.get("ticker") != symbol or dividend.get("currency") != "USD":
            raise DataError(f"Unexpected dividend ticker or currency for {symbol}.")
        event_id = dividend.get("id")
        if event_id and event_id in dividend_ids:
            raise DataError(f"Duplicate dividend event for {symbol}.")
        if event_id:
            dividend_ids.add(event_id)
        try:
            ex_date = date.fromisoformat(dividend["ex_dividend_date"])
            amount = float(dividend["split_adjusted_cash_amount"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"Invalid split-adjusted dividend for {symbol}.") from exc
        if not np.isfinite(amount) or amount < 0:
            raise DataError(f"Invalid dividend amount for {symbol}.")
        amounts[ex_date] = amounts.get(ex_date, 0.0) + amount
    unmatched = set(amounts).difference(frame["date"])
    if unmatched:
        raise DataError(f"Dividend ex-date has no price bar for {symbol}: {sorted(unmatched)[0]}")
    frame["dividend"] = frame["date"].map(amounts).fillna(0.0)
    frame["symbol"] = symbol
    return frame[["date", "symbol", "open", "high", "low", "close", "volume", "dividend"]].copy()


def validate_panel(panel: pd.DataFrame) -> None:
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "dividend"}
    if not required.issubset(panel.columns):
        raise DataError("The saved price file is missing required columns.")
    if set(panel["symbol"]) != set(SYMBOLS):
        raise DataError("The watchlist is incomplete.")
    if panel.duplicated(["date", "symbol"]).any():
        raise DataError("Duplicate symbol/date rows.")
    numeric = panel[["open", "high", "low", "close", "volume", "dividend"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise DataError("The price file contains nonfinite values.")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any() or (numeric[["volume", "dividend"]] < 0).any().any():
        raise DataError("The price file contains an invalid price, volume, or dividend.")
    if ((numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)) | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))).any():
        raise DataError("The price file contains inconsistent OHLC values.")
    counts = panel.groupby("symbol").size()
    if (counts < 400).any():
        raise DataError(f"Too few daily bars for research: {counts.to_dict()}")
    calendar = set(panel.loc[panel["symbol"] == "SPY", "date"])
    for symbol in SYMBOLS:
        dates = set(panel.loc[panel["symbol"] == symbol, "date"])
        if len(calendar.symmetric_difference(dates)) > max(3, int(0.01 * len(calendar))):
            raise DataError(f"Trading dates differ materially from SPY for {symbol}.")
        if any((right - left).days > 10 for left, right in zip(sorted(dates), sorted(dates)[1:])):
            raise DataError(f"Unexpected gap over ten calendar days for {symbol}.")
    if panel["date"].max() < date.today() - timedelta(days=7):
        raise DataError("Market data is more than seven calendar days old.")


def refresh(api_key: str | None = None, end: date | None = None, client: MassiveClient | None = None) -> pd.DataFrame:
    end = end or date.today()
    start = end - timedelta(days=729)  # inside the rolling two-year Basic window
    client = client or MassiveClient(api_key or os.environ.get("MASSIVE_API_KEY", ""))
    raw: dict[str, dict] = {}
    frames = []
    for symbol in SYMBOLS:
        bars, dividends = fetch_symbol(client, symbol, start, end)
        raw[symbol] = {"bars": bars, "dividends": dividends}
        frames.append(normalize_symbol(symbol, bars, dividends))
    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    validate_panel(panel)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "massive_responses.json"
    panel_path = PROCESSED_DIR / "daily_prices.csv"
    raw_temp = raw_path.with_suffix(".tmp")
    panel_temp = panel_path.with_suffix(".tmp")
    raw_temp.write_text(json.dumps({"source": "Massive", "start": start.isoformat(), "end": end.isoformat(), "symbols": raw}), encoding="utf-8")
    panel.to_csv(panel_temp, index=False)
    raw_temp.replace(raw_path)
    panel_temp.replace(panel_path)
    return panel


def load_panel() -> pd.DataFrame:
    path = PROCESSED_DIR / "daily_prices.csv"
    if not path.exists():
        raise DataError("No downloaded data. Set MASSIVE_API_KEY and run python scripts/refresh_data.py first.")
    panel = pd.read_csv(path, parse_dates=["date"])
    panel["date"] = panel["date"].dt.date
    validate_panel(panel)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)
