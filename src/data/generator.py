"""
Synthetic financial transaction data generator.
Produces realistic normal behaviour + fraud scenarios + edge cases.
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, SYNTHETIC_SAMPLES, FRAUD_RATE

fake = Faker()
rng = np.random.default_rng(42)

MERCHANTS = {
    "Grocery":       {"avg": 80,   "std": 40,  "risk": 0.1},
    "Electronics":   {"avg": 500,  "std": 300, "risk": 0.4},
    "Restaurants":   {"avg": 45,   "std": 25,  "risk": 0.1},
    "Online Retail": {"avg": 120,  "std": 80,  "risk": 0.3},
    "Travel":        {"avg": 800,  "std": 500, "risk": 0.5},
    "Gas Station":   {"avg": 60,   "std": 20,  "risk": 0.15},
    "Pharmacy":      {"avg": 40,   "std": 30,  "risk": 0.1},
    "ATM":           {"avg": 200,  "std": 100, "risk": 0.35},
    "Luxury Goods":  {"avg": 2000, "std": 1500,"risk": 0.6},
    "Casino":        {"avg": 500,  "std": 400, "risk": 0.7},
}

LOCATIONS = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
             "New York", "London", "Dubai", "Singapore", "Tokyo"]

FRAUD_PATTERNS = ["card_not_present", "account_takeover", "velocity_fraud",
                  "location_mismatch", "unusual_amount", "late_night_high_value"]


def _user_profile(user_id: str) -> dict:
    home_city = random.choice(LOCATIONS[:5])
    avg_spend  = rng.lognormal(mean=4.5, sigma=0.8)
    return {
        "user_id":    user_id,
        "home_city":  home_city,
        "avg_spend":  float(avg_spend),
        "usual_merchants": random.sample(list(MERCHANTS.keys()), k=random.randint(3, 6)),
        "active_hours": (random.randint(7, 10), random.randint(20, 23)),
    }


def _normal_transaction(profile: dict, ts: datetime) -> dict:
    merchant = random.choice(profile["usual_merchants"])
    m = MERCHANTS[merchant]
    amount = max(1.0, rng.normal(m["avg"], m["std"]))
    hour = ts.hour
    location = profile["home_city"] if rng.random() < 0.85 else random.choice(LOCATIONS)
    return {
        "transaction_id": fake.uuid4(),
        "user_id":        profile["user_id"],
        "timestamp":      ts.isoformat(),
        "amount":         round(float(amount), 2),
        "merchant_type":  merchant,
        "location":       location,
        "hour":           hour,
        "day_of_week":    ts.weekday(),
        "is_weekend":     int(ts.weekday() >= 5),
        "is_late_night":  int(hour < 5 or hour > 23),
        "amount_vs_avg":  round(amount / max(profile["avg_spend"], 1), 3),
        "is_home_city":   int(location == profile["home_city"]),
        "is_known_merchant": 1,
        "fraud_pattern":  "none",
        "is_fraud":       0,
    }


def _fraud_transaction(profile: dict, ts: datetime) -> dict:
    pattern = random.choice(FRAUD_PATTERNS)
    merchant = random.choice(list(MERCHANTS.keys()))
    m = MERCHANTS[merchant]

    if pattern == "unusual_amount":
        amount = float(rng.uniform(profile["avg_spend"] * 5, profile["avg_spend"] * 20))
    elif pattern == "velocity_fraud":
        amount = float(rng.uniform(50, 300))
    elif pattern == "late_night_high_value":
        amount = float(rng.uniform(500, 5000))
        ts = ts.replace(hour=random.randint(1, 4))
    else:
        amount = max(1.0, rng.normal(m["avg"] * 2, m["std"]))

    if pattern == "location_mismatch":
        location = random.choice([l for l in LOCATIONS if l != profile["home_city"]])
    elif pattern == "account_takeover":
        location = random.choice(LOCATIONS)
    else:
        location = profile["home_city"]

    hour = ts.hour
    return {
        "transaction_id": fake.uuid4(),
        "user_id":        profile["user_id"],
        "timestamp":      ts.isoformat(),
        "amount":         round(float(amount), 2),
        "merchant_type":  merchant,
        "location":       location,
        "hour":           hour,
        "day_of_week":    ts.weekday(),
        "is_weekend":     int(ts.weekday() >= 5),
        "is_late_night":  int(hour < 5 or hour > 23),
        "amount_vs_avg":  round(amount / max(profile["avg_spend"], 1), 3),
        "is_home_city":   int(location == profile["home_city"]),
        "is_known_merchant": int(merchant in profile["usual_merchants"]),
        "fraud_pattern":  pattern,
        "is_fraud":       1,
    }


def generate(n_samples: int = SYNTHETIC_SAMPLES, fraud_rate: float = FRAUD_RATE,
             save: bool = True) -> pd.DataFrame:
    n_fraud  = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    n_users   = max(50, n_samples // 20)
    user_ids  = [f"U{i:04d}" for i in range(n_users)]
    profiles  = {uid: _user_profile(uid) for uid in user_ids}

    base_ts   = datetime(2024, 1, 1)
    records   = []

    for _ in range(n_normal):
        uid = random.choice(user_ids)
        ts  = base_ts + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(*profiles[uid]["active_hours"]),
            minutes=random.randint(0, 59),
        )
        records.append(_normal_transaction(profiles[uid], ts))

    for _ in range(n_fraud):
        uid = random.choice(user_ids)
        ts  = base_ts + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        records.append(_fraud_transaction(profiles[uid], ts))

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

    if save:
        path = RAW_DATA_DIR / "transactions.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} transactions → {path}")
        print(f"  Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")

    return df


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(df.dtypes)
