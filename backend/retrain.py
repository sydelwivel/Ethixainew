# backend/retrain.py
"""
Retrain backend with support for classification and regression.
Provides:
  - train_classification(X_train, y_train, X_test, y_test)
  - train_regression(X_train, y_train, X_test, y_test)
  - train_and_evaluate(...) unified wrapper
Notes:
  - Expects X_train/X_test to be pandas DataFrames (features numeric after get_dummies)
  - Returns (model, metrics_dict, preds)
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# sklearn models & metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_classification(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, Dict[str,float], np.ndarray]:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0))
    }
    return model, metrics, preds

def train_regression(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, Dict[str,float], np.ndarray]:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds))
    }
    return model, metrics, preds

def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str = "Classification") -> Tuple[Any, Dict[str,float], np.ndarray]:
    """
    Unified function used by frontend.
    problem_type: "Classification" or "Regression"
    Returns: (model, metrics_dict, preds)
    """
    if problem_type.lower().startswith("class"):
        return train_classification(X_train, y_train, X_test, y_test)
    else:
        return train_regression(X_train, y_train, X_test, y_test)
