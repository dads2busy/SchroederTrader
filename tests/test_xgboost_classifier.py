import numpy as np
import pandas as pd

from schroeder_trader.strategy.xgboost_classifier import (
    train_model,
    predict_signal,
    save_model,
    load_model,
)
from schroeder_trader.strategy.sma_crossover import Signal


def _make_training_data(n: int = 500):
    """Create synthetic training data with 3 classes."""
    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    # Labels correlated with f1: positive f1 → UP, negative → DOWN
    y = pd.Series(np.where(X["f1"] > 0.5, 2, np.where(X["f1"] < -0.5, 0, 1)))
    return X, y


def test_train_model_returns_classifier():
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_predict_signal_returns_signal_and_proba():
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    signal, proba = predict_signal(model, X.iloc[[0]])
    assert isinstance(signal, Signal)
    assert "DOWN" in proba
    assert "FLAT" in proba
    assert "UP" in proba
    assert abs(sum(proba.values()) - 1.0) < 0.01


def test_predict_signal_buy_on_high_up_proba():
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    # Find a row with strong UP signal (f1 >> 0.5)
    strong_up = pd.DataFrame({"f1": [3.0], "f2": [0.0], "f3": [0.0]})
    signal, proba = predict_signal(model, strong_up)
    # With clear signal, should predict BUY
    assert signal == Signal.BUY or proba["UP"] > 0.3  # model may not be perfect on synthetic


def test_predict_signal_sell_on_high_down_proba():
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    strong_down = pd.DataFrame({"f1": [-3.0], "f2": [0.0], "f3": [0.0]})
    signal, proba = predict_signal(model, strong_down)
    assert signal == Signal.SELL or proba["DOWN"] > 0.3


def test_save_and_load_model(tmp_path):
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    model_path = tmp_path / "test_model.json"
    save_model(model, model_path)
    assert model_path.exists()

    loaded = load_model(model_path)
    # Verify loaded model produces same predictions
    signal_orig, proba_orig = predict_signal(model, X.iloc[[0]])
    signal_loaded, proba_loaded = predict_signal(loaded, X.iloc[[0]])
    assert signal_orig == signal_loaded
    assert abs(proba_orig["UP"] - proba_loaded["UP"]) < 0.001


def test_load_model_nonexistent_returns_none(tmp_path):
    result = load_model(tmp_path / "nonexistent.json")
    assert result is None
