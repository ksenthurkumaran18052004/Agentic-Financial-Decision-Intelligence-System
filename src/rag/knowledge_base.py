"""
Fraud domain knowledge base — static rules + pattern descriptions.
These get embedded into FAISS and retrieved at inference time.
"""

FRAUD_KNOWLEDGE = [
    {
        "id": "rule_001",
        "category": "velocity_fraud",
        "text": "Velocity fraud occurs when multiple transactions are made in rapid succession. "
                "If a user performs more than 5 transactions within 10 minutes, or more than 20 "
                "within an hour, this indicates possible card testing or stolen credentials.",
        "risk_level": "high",
    },
    {
        "id": "rule_002",
        "category": "location_mismatch",
        "text": "Geographic impossibility is a strong fraud indicator. Transactions in two "
                "geographically distant locations within a short timeframe (impossible travel) "
                "strongly suggest account compromise or card cloning.",
        "risk_level": "high",
    },
    {
        "id": "rule_003",
        "category": "unusual_amount",
        "text": "Transaction amounts significantly deviating from a user's historical average "
                "warrant investigation. An amount 5x or more above baseline is suspicious. "
                "Round-number transactions (exactly $500, $1000) can also indicate fraud.",
        "risk_level": "medium",
    },
    {
        "id": "rule_004",
        "category": "late_night_high_value",
        "text": "High-value transactions between 1 AM and 5 AM local time are statistically "
                "associated with fraud. Fraudsters often act at night when victims are asleep "
                "and less likely to notice suspicious activity immediately.",
        "risk_level": "medium",
    },
    {
        "id": "rule_005",
        "category": "merchant_risk",
        "text": "Certain merchant categories carry higher fraud risk: casinos, luxury goods, "
                "electronics, and travel agencies are common targets. First-time purchases at "
                "high-risk merchants after a long dormant period increase suspicion.",
        "risk_level": "medium",
    },
    {
        "id": "rule_006",
        "category": "card_not_present",
        "text": "Card-not-present (CNP) fraud is the most common type in online transactions. "
                "Indicators include: mismatched billing/shipping addresses, use of VPN or proxy, "
                "multiple failed attempts before success, and use of prepaid cards.",
        "risk_level": "high",
    },
    {
        "id": "rule_007",
        "category": "account_takeover",
        "text": "Account takeover (ATO) involves a fraudster gaining access to a legitimate "
                "account. Signs: recent password change followed by unusual activity, device "
                "fingerprint change, IP address from unusual geography, rapid spending increase.",
        "risk_level": "critical",
    },
    {
        "id": "rule_008",
        "category": "normal_pattern",
        "text": "Normal transactions exhibit: amounts consistent with merchant category averages, "
                "transactions during typical business hours (7 AM - 11 PM), familiar merchant "
                "types, home city or known travel destinations, consistent spending patterns.",
        "risk_level": "low",
    },
    {
        "id": "rule_009",
        "category": "electronics_fraud",
        "text": "Electronics stores are high-value fraud targets. Large electronics purchases "
                "($500+) at odd hours, especially for new accounts or after a long period of "
                "inactivity, should trigger additional verification. "
                "Gift card purchases at electronics stores are particularly suspicious.",
        "risk_level": "high",
    },
    {
        "id": "rule_010",
        "category": "atm_fraud",
        "text": "ATM fraud patterns include: multiple withdrawals at maximum daily limit, "
                "ATM use in locations inconsistent with user's home city, ATM skimming "
                "indicators such as unusual withdrawal amounts or foreign ATM usage.",
        "risk_level": "high",
    },
    {
        "id": "rule_011",
        "category": "statistical_baseline",
        "text": "Statistical anomaly detection identifies transactions that deviate significantly "
                "from a user's baseline behaviour. Key metrics: z-score of amount, hour entropy, "
                "merchant category diversity. Isolation Forest scores below -0.2 indicate "
                "significant deviation from normal spending behaviour.",
        "risk_level": "medium",
    },
    {
        "id": "rule_012",
        "category": "travel_fraud",
        "text": "Travel-related fraud often involves bookings to unusual destinations, "
                "last-minute high-value travel purchases, or booking without prior travel history. "
                "Legitimate travellers typically show gradual increase in travel spending.",
        "risk_level": "medium",
    },
]


def get_all_documents() -> list[dict]:
    return FRAUD_KNOWLEDGE


def get_texts() -> list[str]:
    return [doc["text"] for doc in FRAUD_KNOWLEDGE]
