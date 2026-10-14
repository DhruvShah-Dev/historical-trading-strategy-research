from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.data import DataError, refresh


if __name__ == "__main__":
    try:
        panel = refresh()
    except DataError as exc:
        raise SystemExit(f"Data refresh failed: {exc}") from exc
    print(f"Saved {len(panel):,} rows for {panel['symbol'].nunique()} symbols, through {panel['date'].max()}.")
