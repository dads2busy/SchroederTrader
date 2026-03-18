"""Download and cache SPY historical data from yfinance."""
import sys
from pathlib import Path

import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"


def download_spy(start: str = "1993-01-29", end: str | None = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "spy_daily.csv"

    print(f"Downloading SPY data from {start}...")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True)
    spy.to_csv(output)
    print(f"Saved {len(spy)} rows to {output}")
    return output


if __name__ == "__main__":
    download_spy()
