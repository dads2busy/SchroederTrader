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
    "dgs10": "^TNX",
    "dgs2": "2YY=F",
}


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


def download_dtb3(force: bool = False) -> Path:
    """Download the FRED DTB3 series (3-month T-bill rate, percent) to its own
    cached CSV. Kept separate from features_daily.csv so fetching it never
    perturbs the production feature file. Used by the pre-registered
    levered-brake evaluation (financing cost + cash yield)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "dtb3.csv"
    if not force and output.exists():
        age_hours = (time.time() - output.stat().st_mtime) / 3600
        if age_hours < 24 * 7:
            print(f"Using cached DTB3 ({age_hours/24:.1f}d old). Use --force to re-download.")
            return output
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
        df = pd.read_csv(url, parse_dates=["observation_date"], na_values=".")
        df = df.rename(columns={"observation_date": "date", "DTB3": "dtb3"}).set_index("date")
        print("DTB3 source: FRED")
    except Exception as exc:
        # PREREGISTRATION_levered_brake_v1 AMENDMENT 1: FRED unreachable from this
        # environment (HTTP 504); ^IRX is the same instrument (13-week T-bill
        # yield, percent) under a discount-rate quote convention (differs by bps).
        print(f"FRED failed ({exc}); falling back to yfinance ^IRX")
        data = yf.download("^IRX", start="1993-01-01", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = data[["Close"]].rename(columns={"Close": "dtb3"})
        df.index.name = "date"
        print("DTB3 source: yfinance ^IRX (fallback)")
    df.to_csv(output)
    print(f"Saved {len(df)} DTB3 rows to {output}")
    return output


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
    combined = download_yfinance(start, end)

    # Compute derived features for the composite model
    combined = compute_derived_features(combined)

    combined.index.name = "date"
    combined.to_csv(output)
    print(f"Saved {len(combined)} rows to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download external feature data")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cache exists")
    parser.add_argument("--dtb3-only", action="store_true", help="Fetch only the FRED DTB3 series")
    args = parser.parse_args()
    if args.dtb3_only:
        download_dtb3(force=args.force)
    else:
        download_all(force=args.force)
