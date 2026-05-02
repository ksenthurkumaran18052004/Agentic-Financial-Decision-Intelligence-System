"""
Feature engineering + train/test split + scaling.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SCALER_PATH

MERCHANT_RISK = {
    "grocery": 0.1, "electronics": 0.4, "restaurant": 0.1, "restaurants": 0.1,
    "online retail": 0.3, "travel": 0.5, "gas_station": 0.15, "gas station": 0.15,
    "pharmacy": 0.1, "atm": 0.35, "luxury_goods": 0.6, "luxury goods": 0.6,
    "casino": 0.7,
    # title-case aliases kept for backward compatibility
    "Grocery": 0.1, "Electronics": 0.4, "Restaurants": 0.1,
    "Online Retail": 0.3, "Travel": 0.5, "Gas Station": 0.15,
    "Pharmacy": 0.1, "ATM": 0.35, "Luxury Goods": 0.6, "Casino": 0.7,
}

FEATURES = [
    "amount", "hour", "day_of_week", "is_weekend", "is_late_night",
    "amount_vs_avg", "is_home_city", "is_known_merchant", "merchant_risk_score",
    "amount_log",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["merchant_risk_score"] = df["merchant_type"].map(MERCHANT_RISK).fillna(0.3)
    df["amount_log"] = np.log1p(df["amount"])
    return df


def load_and_prepare(path=None):
    if path is None:
        path = RAW_DATA_DIR / "transactions.csv"
    df = pd.read_csv(path)
    df = engineer_features(df)

    X = df[FEATURES].values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved → {SCALER_PATH}")

    return X_train_s, X_test_s, y_train, y_test, scaler, df


def preprocess_transaction(tx: dict, scaler: StandardScaler | None = None) -> np.ndarray:
    """Preprocess a single transaction dict into a feature vector."""
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)

    merchant_risk = MERCHANT_RISK.get(tx.get("merchant_type", ""), 0.3)
    amount = float(tx.get("amount", 0))

    vec = np.array([[
        amount,
        int(tx.get("hour", 12)),
        int(tx.get("day_of_week", 0)),
        int(tx.get("is_weekend", 0)),
        int(tx.get("is_late_night", 0)),
        float(tx.get("amount_vs_avg", 1.0)),
        int(tx.get("is_home_city", 1)),
        int(tx.get("is_known_merchant", 1)),
        merchant_risk,
        np.log1p(amount),
    ]])
    return scaler.transform(vec)
