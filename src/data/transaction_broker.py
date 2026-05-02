"""Redis Streams broker with a local in-memory fallback.

The broker is intentionally small:
- Redis/Upstash enabled: use Redis Streams as the durable queue.
- Redis unavailable: fall back to an in-memory deque so the app still runs.

The dashboard can pull transactions from the broker, and API routes can
publish transactions into the same stream.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

from config import REDIS_URL, REDIS_STREAM_CONSUMER, REDIS_STREAM_GROUP, REDIS_STREAM_KEY

try:
    import redis
    from redis.exceptions import ResponseError
except Exception:  # pragma: no cover - optional dependency fallback
    redis = None
    ResponseError = Exception


SeedFunction = Callable[[], dict]


@dataclass(frozen=True)
class StreamRecord:
    transaction: dict
    stream_id: str
    source: str
    acknowledged: bool = False


class TransactionBroker:
    def __init__(
        self,
        redis_url: str | None = None,
        stream_key: str = REDIS_STREAM_KEY,
        stream_group: str = REDIS_STREAM_GROUP,
        consumer_name: str = REDIS_STREAM_CONSUMER,
    ):
        self.redis_url = redis_url or REDIS_URL
        self.stream_key = stream_key
        self.stream_group = stream_group
        self.consumer_name = consumer_name
        self._client = None
        self._lock = threading.Lock()
        self._fallback_queue: Deque[dict] = deque()
        self._redis_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.redis_url and redis is not None)

    def _get_client(self):
        if not self.enabled:
            return None
        if self._client is None:
            try:
                self._client = redis.Redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=10,
                )
                self._client.ping()
            except Exception as exc:
                self._redis_error = f"Redis connection failed: {str(exc)}"
                self._client = None
        return self._client

    def ensure_group(self) -> None:
        if not self.enabled:
            return

        try:
            client = self._get_client()
            if client is None:
                self._redis_error = "Redis client initialization failed"
                return
            client.xgroup_create(name=self.stream_key, groupname=self.stream_group, id="$", mkstream=True)
        except Exception as exc:
            error_str = str(exc)
            if "BUSYGROUP" not in error_str:
                self._redis_error = f"Redis group creation failed: {error_str}"
                self._client = None

    def health(self) -> dict:
        if not self.enabled:
            return {"mode": "memory", "connected": True, "stream_key": self.stream_key}

        client = self._get_client()
        if client is None:
            return {
                "mode": "memory-fallback",
                "connected": False,
                "error": self._redis_error or "Redis unavailable",
                "stream_key": self.stream_key,
            }

        try:
            client.ping()
            return {
                "mode": "redis",
                "connected": True,
                "stream_key": self.stream_key,
                "redis_url": self.redis_url.rsplit("@", 1)[-1] if self.redis_url else "",
            }
        except Exception as exc:
            self._redis_error = str(exc)
            return {
                "mode": "memory-fallback",
                "connected": False,
                "error": str(exc),
                "stream_key": self.stream_key,
            }

    def publish(self, transaction: dict) -> dict:
        payload = dict(transaction)
        payload.setdefault("transaction_id", f"txn_{uuid.uuid4().hex[:12]}")

        if self.enabled:
            try:
                client = self._get_client()
                if client is not None:
                    self.ensure_group()
                    stream_id = client.xadd(self.stream_key, {"payload": json.dumps(payload, separators=(",", ":"))})
                    return {"stream_id": stream_id, "source": "redis"}
            except Exception as exc:
                self._redis_error = f"Publish failed: {str(exc)}"

        with self._lock:
            self._fallback_queue.append(payload)
        return {"stream_id": f"memory-{uuid.uuid4().hex[:12]}", "source": "memory-fallback"}

    def _decode_stream_fields(self, stream_id: str, fields: dict) -> StreamRecord:
        payload = fields.get("payload")
        if isinstance(payload, str):
            transaction = json.loads(payload)
        else:
            transaction = dict(fields)
        return StreamRecord(transaction=transaction, stream_id=stream_id, source="redis", acknowledged=False)

    def take_next(self, seed_fn: SeedFunction | None = None, block_ms: int = 1000) -> StreamRecord:
        """Read one transaction from the stream, seeding synthetic data if needed."""
        if self.enabled:
            try:
                client = self._get_client()
                if client is not None:
                    self.ensure_group()

                    messages = client.xreadgroup(
                        groupname=self.stream_group,
                        consumername=self.consumer_name,
                        streams={self.stream_key: ">"},
                        count=1,
                        block=block_ms,
                    )

                    if not messages and seed_fn is not None:
                        seed_tx = seed_fn()
                        publish_meta = self.publish(seed_tx)
                        messages = client.xreadgroup(
                            groupname=self.stream_group,
                            consumername=self.consumer_name,
                            streams={self.stream_key: ">"},
                            count=1,
                            block=0,
                        )
                        if not messages:
                            return StreamRecord(transaction=seed_tx, stream_id=publish_meta["stream_id"], source=publish_meta["source"])

                    if messages:
                        _, entries = messages[0]
                        stream_id, fields = entries[0]
                        return self._decode_stream_fields(stream_id, fields)

                    fallback_tx = seed_fn() if seed_fn is not None else {}
                    return StreamRecord(transaction=fallback_tx, stream_id=f"memory-{uuid.uuid4().hex[:12]}", source="redis-empty")
            except Exception as exc:
                self._redis_error = f"Read failed: {str(exc)}"
                self._client = None

        with self._lock:
            if self._fallback_queue:
                tx = self._fallback_queue.popleft()
                return StreamRecord(transaction=tx, stream_id=f"memory-{uuid.uuid4().hex[:12]}", source="memory")

        tx = seed_fn() if seed_fn is not None else {}
        return StreamRecord(transaction=tx, stream_id=f"memory-{uuid.uuid4().hex[:12]}", source="memory-seeded")

    def ack(self, stream_id: str) -> None:
        if not self.enabled or not stream_id:
            return
        client = self._get_client()
        client.xack(self.stream_key, self.stream_group, stream_id)


_BROKER: TransactionBroker | None = None


def get_transaction_broker() -> TransactionBroker:
    global _BROKER
    if _BROKER is None:
        _BROKER = TransactionBroker()
    return _BROKER
