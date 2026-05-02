"""
Fraud / anomaly detection model — Isolation Forest.
Returns anomaly score; lower (more negative) = more suspicious.
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FRAUD_MODEL_PATH, FRAUD_THRESHOLD


def train(X_train, y_train, X_test, y_test) -> IsolationForest:
    fraud_rate = float((y_train == 1).mean())
    model = IsolationForest(
        n_estimators=200,
        contamination=fraud_rate,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    joblib.dump(model, FRAUD_MODEL_PATH)

    scores  = model.decision_function(X_test)
    y_pred  = (scores < FRAUD_THRESHOLD).astype(int)
    print(f"\n=== Fraud Model (Isolation Forest) ===")
    print(f"Decision function threshold: {FRAUD_THRESHOLD}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    return model


def load() -> IsolationForest:
    return joblib.load(FRAUD_MODEL_PATH)


def predict(model_or_tx, X: np.ndarray = None):
    """
    Two call modes:
      predict(model, X)   → original dict return (used internally)
      predict(tx_dict)    → tuple (raw_score: float, is_anomaly: bool)
    """
    if isinstance(model_or_tx, dict):
        from src.data.preprocessor import preprocess_transaction
        _model = load()
        X = preprocess_transaction(model_or_tx)
        raw_score  = float(_model.decision_function(X)[0])
        is_anomaly = _model.predict(X)[0] == -1
        return raw_score, is_anomaly

    model      = model_or_tx
    score      = float(model.decision_function(X)[0])
    label      = model.predict(X)[0]
    is_anomaly = label == -1
    anomaly_score = round(max(0.0, min(1.0, (-score + 0.5) / 0.5)), 4)
    patterns = []
    if is_anomaly:
        patterns.append("statistical_anomaly")
    if score < FRAUD_THRESHOLD - 0.1:
        patterns.append("high_confidence_fraud")
    return {
        "anomaly_score":     anomaly_score,
        "raw_score":         round(score, 4),
        "is_anomaly":        is_anomaly,
        "detected_patterns": patterns,
    }
