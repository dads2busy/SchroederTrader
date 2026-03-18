import logging
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from schroeder_trader.strategy.feature_engineer import CLASS_DOWN, CLASS_FLAT, CLASS_UP, CLASS_NAMES
from schroeder_trader.strategy.sma_crossover import Signal

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 20,
) -> XGBClassifier:
    """Train an XGBoost classifier with early stopping."""
    model = XGBClassifier(
        **DEFAULT_PARAMS,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("Model trained: %d trees, best iteration %d", model.n_estimators, model.best_iteration)
    return model


def predict_signal(model: XGBClassifier, features_row: pd.DataFrame) -> tuple[Signal, dict]:
    """Generate a trading signal from model prediction."""
    proba = model.predict_proba(features_row)[0]
    proba_dict = {
        "DOWN": float(proba[CLASS_DOWN]),
        "FLAT": float(proba[CLASS_FLAT]),
        "UP": float(proba[CLASS_UP]),
    }

    predicted_class = int(np.argmax(proba))

    if predicted_class == CLASS_UP and proba[CLASS_UP] > 0.5:
        return Signal.BUY, proba_dict
    elif predicted_class == CLASS_DOWN and proba[CLASS_DOWN] > 0.5:
        return Signal.SELL, proba_dict
    else:
        return Signal.HOLD, proba_dict


def save_model(model: XGBClassifier, path: Path) -> None:
    """Save model to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> XGBClassifier | None:
    """Load model from JSON file. Returns None if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        logger.info("No model file at %s", path)
        return None
    model = XGBClassifier()
    model.load_model(str(path))
    logger.info("Model loaded from %s", path)
    return model
