from schroeder_trader.config import TICKER, SMA_SHORT_WINDOW, SMA_LONG_WINDOW, DB_PATH


def test_ticker_is_spy():
    assert TICKER == "SPY"


def test_sma_windows():
    assert SMA_SHORT_WINDOW == 50
    assert SMA_LONG_WINDOW == 200


def test_db_path_is_absolute():
    assert DB_PATH.is_absolute()
