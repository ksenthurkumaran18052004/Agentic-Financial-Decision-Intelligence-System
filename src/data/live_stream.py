"""
Live transaction stream generator — produces one transaction at a time.
Used by the dashboard to simulate a real-time payment feed.
"""

import random
import uuid
from datetime import datetime

MERCHANTS = {
    "grocery":      {"avg": 80,   "std": 30,  "risk": 0.1},
    "electronics":  {"avg": 500,  "std": 300, "risk": 0.4},
    "restaurant":   {"avg": 45,   "std": 20,  "risk": 0.1},
    "travel":       {"avg": 800,  "std": 500, "risk": 0.5},
    "gas_station":  {"avg": 60,   "std": 20,  "risk": 0.15},
    "pharmacy":     {"avg": 40,   "std": 25,  "risk": 0.1},
    "atm":          {"avg": 200,  "std": 100, "risk": 0.35},
    "luxury_goods": {"avg": 2000, "std": 1500,"risk": 0.6},
    "casino":       {"avg": 500,  "std": 400, "risk": 0.7},
}

HOME_CITIES   = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"]
FOREIGN_CITIES = ["Moscow", "Lagos", "Bucharest", "Minsk", "Kyiv",
                  "New York", "London", "Dubai", "Singapore", "Tokyo"]
ALL_CITIES    = HOME_CITIES + FOREIGN_CITIES

FRAUD_PATTERNS = [
    "card_not_present", "account_takeover", "velocity_fraud",
    "location_mismatch", "unusual_amount", "late_night_high_value",
]

_USER_POOL = [
    {"user_id": f"U{i:04d}",
     "home_city": random.choice(HOME_CITIES),
     "avg_spend": random.uniform(50, 400),
     "usual_merchants": random.sample(list(MERCHANTS), k=random.randint(3, 5)),
    }
    for i in range(50)
]


def _normal(user: dict) -> dict:
    merchant = random.choice(user["usual_merchants"])
    m = MERCHANTS[merchant]
    amount = max(1.0, random.gauss(m["avg"], m["std"]))
    hour = random.randint(8, 22)
    location = user["home_city"] if random.random() < 0.85 else random.choice(HOME_CITIES)
    avg_spend = user["avg_spend"]
    return {
        "transaction_id":    str(uuid.uuid4())[:12],
        "user_id":           user["user_id"],
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "amount":            round(abs(amount), 2),
        "merchant_type":     merchant,
        "location":          location,
        "hour":              hour,
        "day_of_week":       datetime.now().weekday(),
        "is_weekend":        int(datetime.now().weekday() >= 5),
        "is_late_night":     0,
        "amount_vs_avg":     round(abs(amount) / max(avg_spend, 1), 2),
        "is_home_city":      int(location == user["home_city"]),
        "is_known_merchant": 1,
        "fraud_pattern":     "none",
        "is_fraud":          0,
    }


def _fraud(user: dict) -> dict:
    pattern  = random.choice(FRAUD_PATTERNS)
    merchant = random.choice(list(MERCHANTS))
    avg_spend = user["avg_spend"]

    if pattern == "unusual_amount":
        amount = avg_spend * random.uniform(5, 20)
        hour   = random.randint(8, 22)
        location = user["home_city"]
    elif pattern == "late_night_high_value":
        amount = random.uniform(500, 3000)
        hour   = random.randint(1, 4)
        location = user["home_city"]
    elif pattern in ("location_mismatch", "account_takeover"):
        amount = random.gauss(MERCHANTS[merchant]["avg"] * 2, MERCHANTS[merchant]["std"])
        hour   = random.randint(0, 23)
        location = random.choice(FOREIGN_CITIES)
    else:
        amount = random.uniform(50, 500)
        hour   = random.randint(0, 23)
        location = random.choice(FOREIGN_CITIES)

    return {
        "transaction_id":    str(uuid.uuid4())[:12],
        "user_id":           user["user_id"],
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "amount":            round(abs(amount), 2),
        "merchant_type":     merchant,
        "location":          location,
        "hour":              hour,
        "day_of_week":       datetime.now().weekday(),
        "is_weekend":        int(datetime.now().weekday() >= 5),
        "is_late_night":     int(hour < 5),
        "amount_vs_avg":     round(abs(amount) / max(avg_spend, 1), 2),
        "is_home_city":      int(location == user["home_city"]),
        "is_known_merchant": int(merchant in user["usual_merchants"]),
        "fraud_pattern":     pattern,
        "is_fraud":          1,
    }


def next_transaction(fraud_rate: float = 0.20) -> dict:
    """Generate the next transaction in the stream."""
    user = random.choice(_USER_POOL)
    if random.random() < fraud_rate:
        return _fraud(user)
    return _normal(user)
