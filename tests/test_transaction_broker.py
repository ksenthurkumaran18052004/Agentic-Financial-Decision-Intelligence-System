from src.data.transaction_broker import TransactionBroker
from config import build_redis_url


def test_build_redis_url_from_upstash_vars(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "secret-token")

    url = build_redis_url()

    assert url == "rediss://default:secret-token@example.upstash.io:6379"


def test_transaction_broker_memory_fallback_round_trip():
    broker = TransactionBroker(redis_url="")
    tx = {"transaction_id": "txn-1", "amount": 12.5, "merchant_type": "grocery"}

    publish_meta = broker.publish(tx)
    record = broker.take_next()

    assert publish_meta["source"] == "memory"
    assert record.transaction["transaction_id"] == "txn-1"
    assert record.source.startswith("memory")
