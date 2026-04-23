from unittest.mock import MagicMock, patch

import pytest

from schroeder_trader.main import _wait_for_network


def _fake_ok():
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


@patch("schroeder_trader.main.socket.create_connection")
def test_wait_for_network_succeeds_first_try(mock_connect):
    mock_connect.return_value = _fake_ok()
    _wait_for_network([("alpaca.test", 443), ("smtp.test", 465)])
    assert mock_connect.call_count == 2  # one per host


@patch("schroeder_trader.main.time.sleep")
@patch("schroeder_trader.main.socket.create_connection")
def test_wait_for_network_retries_then_succeeds(mock_connect, mock_sleep):
    mock_connect.side_effect = [OSError("first fail"), _fake_ok()]
    _wait_for_network([("alpaca.test", 443)])
    assert mock_connect.call_count == 2
    mock_sleep.assert_called_once_with(15)  # second backoff


@patch("schroeder_trader.main.time.sleep")
@patch("schroeder_trader.main.socket.create_connection")
def test_wait_for_network_catches_gaierror_as_oserror(mock_connect, mock_sleep):
    import socket as _s
    mock_connect.side_effect = [_s.gaierror(8, "not known"), _fake_ok()]
    _wait_for_network([("alpaca.test", 443)])
    assert mock_connect.call_count == 2


@patch("schroeder_trader.main.time.sleep")
@patch("schroeder_trader.main.socket.create_connection")
def test_wait_for_network_exhausts_and_raises(mock_connect, mock_sleep):
    mock_connect.side_effect = OSError("persistent")
    with pytest.raises(RuntimeError, match="Network unreachable"):
        _wait_for_network([("alpaca.test", 443)])
    assert mock_connect.call_count == 4  # 4 attempts total


@patch("schroeder_trader.main.socket.create_connection")
def test_wait_for_network_probes_multiple_hosts(mock_connect):
    mock_connect.return_value = _fake_ok()
    _wait_for_network([("a.test", 443), ("b.test", 465), ("c.test", 80)])
    assert mock_connect.call_count == 3
    called_addrs = [call.args[0] for call in mock_connect.call_args_list]
    assert ("a.test", 443) in called_addrs
    assert ("b.test", 465) in called_addrs
    assert ("c.test", 80) in called_addrs
