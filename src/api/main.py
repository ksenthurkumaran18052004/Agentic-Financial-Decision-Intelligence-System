"""
FastAPI application — single endpoint to analyze a transaction.
Run: uvicorn src.api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import TransactionRequest, AnalysisResponse
from src.agents.orchestrator import analyze
from src.data.live_stream import next_transaction
from src.data.transaction_broker import get_transaction_broker

app = FastAPI(
    title="Agentic Financial Decision Intelligence System",
    description="Multi-agent AI that detects fraud and explains decisions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

broker = get_transaction_broker()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stream/health")
def stream_health():
    return broker.health()


@app.get("/sample")
def sample_transactions():
    """Return a few sample transactions for testing."""
    return {
        "suspicious": {
            "transaction_id": "txn_suspicious",
            "amount": 950.00,
            "merchant_type": "electronics",
            "location": "Lagos",
            "hour": 3,
            "is_home_city": 0,
            "is_known_merchant": 0,
            "is_late_night": 1,
            "amount_vs_avg": 9.2,
        },
        "normal": {
            "transaction_id": "txn_normal",
            "amount": 42.50,
            "merchant_type": "grocery",
            "location": "New York",
            "hour": 11,
            "is_home_city": 1,
            "is_known_merchant": 1,
            "is_late_night": 0,
            "amount_vs_avg": 0.9,
        },
        "borderline": {
            "transaction_id": "txn_borderline",
            "amount": 310.00,
            "merchant_type": "travel",
            "location": "London",
            "hour": 22,
            "is_home_city": 0,
            "is_known_merchant": 0,
            "is_late_night": 0,
            "amount_vs_avg": 3.5,
        },
    }


@app.post("/stream/publish")
def publish_transaction(request: TransactionRequest):
    try:
        return broker.publish(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stream/next")
def next_stream_transaction(fraud_rate: float = 0.2):
    try:
        record = broker.take_next(seed_fn=lambda: next_transaction(fraud_rate=fraud_rate))
        return {
            "stream_id": record.stream_id,
            "source": record.source,
            "transaction": record.transaction,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_transaction(request: TransactionRequest):
    try:
        tx = request.model_dump()
        result = analyze(tx)
        return AnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
