"""
Risk scoring model — XGBoost classifier.
Returns probability of a transaction being fraudulent/high-risk.
"""

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RISK_MODEL_PATH, RISK_THRESHOLD


def train(X_train, y_train, X_test, y_test) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    joblib.dump(model, RISK_MODEL_PATH)

    y_pred  = (model.predict_proba(X_test)[:, 1] >= RISK_THRESHOLD).astype(int)
    auc     = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n=== Risk Model (XGBoost) ===")
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    return model


def load() -> XGBClassifier:
    return joblib.load(RISK_MODEL_PATH)


def predict(model_or_tx, X: np.ndarray = None) -> "float | dict":
    """
    Two call modes:
      predict(model, X)   → original dict return (used internally)
      predict(tx_dict)    → float risk_score (used by orchestrator)
    """
    if isinstance(model_or_tx, dict):
        from src.data.preprocessor import preprocess_transaction
        _model = load()
        X = preprocess_transaction(model_or_tx)
        prob = float(_model.predict_proba(X)[0, 1])
        return prob
    model = model_or_tx
    prob  = float(model.predict_proba(X)[0, 1])
    label = "high_risk" if prob >= RISK_THRESHOLD else "low_risk"
    fi    = model.feature_importances_.tolist()
    return {"risk_score": round(prob, 4), "risk_label": label, "feature_importances": fi}
