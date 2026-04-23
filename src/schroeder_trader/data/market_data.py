import logging
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import GetCalendarRequest

from schroeder_trader.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from schroeder_trader.execution.broker import _get_trading_client, _retry_on_connection_error

logger = logging.getLogger(__name__)

_data_client = None


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _data_client


@_retry_on_connection_error()
def fetch_daily_bars(ticker: str, days: int = 365) -> pd.DataFrame:
    client = _get_data_client()
    end = datetime.now()
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    # If multi-index (symbol, timestamp), drop the symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")

    logger.info("Fetched %d bars for %s", len(df), ticker)
    return df[["open", "high", "low", "close", "volume"]]


@_retry_on_connection_error()
def is_market_open_today(date_str: str | None = None) -> bool:
    client = _get_trading_client()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    calendar = client.get_calendar(GetCalendarRequest(start=date_str, end=date_str))
    if not calendar:
        return False

    return str(calendar[0].date) == date_str
