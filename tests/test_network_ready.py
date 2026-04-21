import socket
from unittest.mock import patch

import pytest

from schroeder_trader.main import _wait_for_network


@patch("schroeder_trader.main.socket.gethostbyname")
def test_wait_for_network_succeeds_first_try(mock_resolve):
    mock_resolve.return_value = "1.2.3.4"
    _wait_for_network("example.com")
    assert mock_resolve.call_count == 1


@patch("schroeder_trader.main.time.sleep")
@patch("schroeder_trader.main.socket.gethostbyname")
def test_wait_for_network_retries_then_succeeds(mock_resolve, mock_sleep):
    mock_resolve.side_effect = [socket.gaierror(8, "not known"), "1.2.3.4"]
    _wait_for_network("example.com")
    assert mock_resolve.call_count == 2
    mock_sleep.assert_called_once_with(15)  # second attempt waits 15s


@patch("schroeder_trader.main.time.sleep")
@patch("schroeder_trader.main.socket.gethostbyname")
def test_wait_for_network_exhausts_and_raises(mock_resolve, mock_sleep):
    mock_resolve.side_effect = socket.gaierror(8, "not known")
    with pytest.raises(RuntimeError, match="DNS resolution failed"):
        _wait_for_network("example.com")
    assert mock_resolve.call_count == 4  # matches _NETWORK_READY_BACKOFFS length
