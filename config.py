import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "src" / "models"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw in (None, "") else raw


def build_redis_url() -> str:
    """Return a Redis TLS URL from either REDIS_URL or Upstash REST vars.

    Upstash free-tier databases expose both a REST endpoint and a Redis TCP
    endpoint. This app uses the Redis protocol when available because Redis
    Streams are a native fit for queue-style workloads.
    """
    redis_url = _env_str("REDIS_URL")
    if redis_url:
        return redis_url

    upstash_rest_url = _env_str("UPSTASH_REDIS_REST_URL")
    upstash_rest_token = _env_str("UPSTASH_REDIS_REST_TOKEN")
    if upstash_rest_url and upstash_rest_token:
        host = upstash_rest_url.replace("https://", "").replace("http://", "").rstrip("/")
        return f"rediss://default:{upstash_rest_token}@{host}:6379"

    return ""


REDIS_URL = build_redis_url()
REDIS_STREAM_KEY = _env_str("REDIS_STREAM_KEY", "afdis:transactions")
REDIS_STREAM_GROUP = _env_str("REDIS_STREAM_GROUP", "afdis-consumers")
REDIS_STREAM_CONSUMER = _env_str("REDIS_STREAM_CONSUMER", "dashboard")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got: {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an int, got: {raw!r}") from exc

# Model IDs (Together AI — OpenAI-compatible)
LLM_SMART = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Planner, Insight, Evaluator
LLM_FAST  = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # same model — smaller ones not on serverless tier

# Agent determinism controls
AGENT_TEMPERATURE = _env_float("AGENT_TEMPERATURE", 0.2)
AGENT_TOP_P = _env_float("AGENT_TOP_P", 0.9)
AGENT_MAX_RETRIES = _env_int("AGENT_MAX_RETRIES", 2)
AGENT_TIMEOUT_S = _env_float("AGENT_TIMEOUT_S", 30.0)

# ML model paths
RISK_MODEL_PATH  = PROCESSED_DATA_DIR / "risk_model.joblib"
FRAUD_MODEL_PATH = PROCESSED_DATA_DIR / "fraud_model.joblib"
SCALER_PATH      = PROCESSED_DATA_DIR / "scaler.joblib"
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "faiss_index.bin"
METADATA_PATH    = PROCESSED_DATA_DIR / "faiss_metadata.pkl"

# Thresholds
RISK_THRESHOLD  = 0.6
FRAUD_THRESHOLD = -0.2   # Isolation Forest decision function; lower = more anomalous

# MCTS
MCTS_SIMULATIONS = 50
MCTS_EXPLORATION = 1.41

# Data generation
SYNTHETIC_SAMPLES = 5000
FRAUD_RATE = 0.08
