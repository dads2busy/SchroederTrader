"""Download and cache external feature data from yfinance and FRED.

Produces a CSV with raw columns plus derived features needed by the
composite model: credit_spread and dollar_momentum.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"

# yfinance tickers
YF_TICKERS = {
    "vix_close": "^VIX",
    "vix3m_close": "^VIX3M",
    "hyg_close": "HYG",
    "lqd_close": "LQD",
    "gld_close": "GLD",
    "uup_close": "UUP",
    "tlt_close": "TLT",
    "eem_close": "EEM",
}

# FRED series (public CSV endpoint, no API key needed)
FRED_SERIES = {
    "dgs10": "DGS10",
    "dgs2": "DGS2",
}

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"


def download_yfinance(start: str, end: str) -> pd.DataFrame:
    """Download close prices for all yfinance tickers."""
    frames = {}
    for col_name, ticker in YF_TICKERS.items():
        print(f"  Downloading {ticker}...")
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if len(data) == 0:
            print(f"  WARNING: No data for {ticker}")
            continue
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        frames[col_name] = data["Close"].rename(col_name)
        time.sleep(0.5)  # rate limit

    return pd.concat(frames.values(), axis=1).sort_index()


def download_fred(start: str, end: str) -> pd.DataFrame:
    """Download yield data from FRED public CSV endpoint."""
    frames = {}
    for col_name, series in FRED_SERIES.items():
        url = FRED_URL.format(series=series, start=start, end=end)
        print(f"  Downloading FRED {series}...")
        df = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date", na_values=".")
        df.columns = [col_name]
        # Forward-fill gaps (max 5 business days)
        df = df.ffill(limit=5)
        frames[col_name] = df[col_name]

    return pd.concat(frames.values(), axis=1).sort_index()


def compute_derived_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features needed by the composite model."""
    result = combined.copy()

    # Credit spread: 20-day change in log(HYG/LQD)
    if "hyg_close" in result.columns and "lqd_close" in result.columns:
        log_ratio = np.log(result["hyg_close"] / result["lqd_close"])
        result["credit_spread"] = log_ratio - log_ratio.shift(20)

    # Dollar momentum: 20-day log return of UUP
    if "uup_close" in result.columns:
        result["dollar_momentum"] = np.log(result["uup_close"] / result["uup_close"].shift(20))

    return result


def download_all(start: str = "1993-01-01", end: str | None = None, force: bool = False) -> Path:
    """Download all external feature data and save to CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "features_daily.csv"

    # Idempotency: skip if recent file exists
    if not force and output.exists():
        age_hours = (time.time() - output.stat().st_mtime) / 3600
        if age_hours < 24:
            print(f"Using cached data ({age_hours:.1f}h old). Use --force to re-download.")
            return output

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print("Downloading yfinance data...")
    yf_data = download_yfinance(start, end)

    print("Downloading FRED data...")
    fred_data = download_fred(start, end)

    # Merge on date (inner join — only keep days where all sources have data)
    combined = yf_data.join(fred_data, how="inner")

    # Compute derived features for the composite model
    combined = compute_derived_features(combined)

    combined.index.name = "date"
    combined.to_csv(output)
    print(f"Saved {len(combined)} rows to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download external feature data")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cache exists")
    args = parser.parse_args()
    download_all(force=args.force)
