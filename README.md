# Multi-Agent Financial Fraud Intelligence System (AFDIS)

AFDIS is a multi-agent financial fraud detection system that analyzes transactions, scores risk, searches a rules knowledge base, reasons over the evidence, and produces a final decision with an explanation. It includes both a FastAPI backend and a Streamlit dashboard so you can inspect live transaction analysis in real time.

## What it does

AFDIS helps detect suspicious financial activity by combining machine learning, retrieval, and agent reasoning.

It can:

- score transactions for fraud risk and anomaly behavior
- search a knowledge base of fraud patterns and rules
- run a multi-agent reasoning pipeline for planning, risk interpretation, fraud analysis, insight synthesis, and final evaluation
- stream live transaction events through Redis Streams with a safe in-memory fallback
- show the full decision process in a browser dashboard
- expose API endpoints for health checks, sample transactions, streaming, publishing, and transaction analysis

## How it works

The system follows a transaction pipeline:

1. A transaction is received from the stream or generated as synthetic test data.
2. The ML models score it using a risk model and an anomaly detector.
3. The retriever searches the knowledge base for matching fraud rules.
4. MCTS proposes an action using the ML and retrieval signals.
5. The agent chain runs:
   - Planner: identifies what needs investigation
   - Risk Agent: explains the ML risk score
   - Fraud Agent: interprets fraud patterns and retrieved rules
   - Insight Agent: synthesizes the evidence
   - Evaluator Agent: makes the final approve, flag, or block decision
6. The dashboard displays the full reasoning step by step.
7. The API returns a structured JSON result with the decision, scores, reasons, and agent traces.

If Redis is unavailable, the broker falls back to memory so the app still runs locally.

## Tech behind it

### Backend

- FastAPI for the HTTP API
- Uvicorn for serving the backend
- Redis Streams / Upstash for live transaction queuing
- Pydantic for request/response schemas
- Python dotenv for configuration loading

### Intelligence layer

- XGBoost-based risk scoring
- Isolation Forest anomaly detection
- FAISS + sentence-transformers for retrieval over fraud knowledge
- MCTS for recommendation-style reasoning
- Multi-agent orchestration for planning, interpretation, synthesis, and final evaluation
- Together AI for LLM-powered agent reasoning

### Frontend

- Streamlit for the live dashboard
- Plotly for charts and transaction trend visualization
- Pandas for tabular event handling

### Deployment and tooling

- GitHub for source control
- Render or Railway for production deployment
- Docker files for container-based deployment
- Pytest for tests
- JSON Schema validation for agent input checks

## Project structure

- `src/api/main.py` - FastAPI app and endpoints
- `src/dashboard/app.py` - Streamlit live monitor
- `src/agents/` - planner, risk, fraud, insight, and evaluator agents
- `src/data/transaction_broker.py` - Redis Streams broker with fallback
- `src/rag/` - retrieval and knowledge base logic
- `src/models/` - fraud and risk models
- `src/reasoning/mcts.py` - MCTS logic
- `data/processed/` - trained model artifacts and vector index files
- `tests/` - validation and broker tests

## API endpoints

- `GET /health` - backend health check
- `GET /stream/health` - broker and Redis status
- `GET /sample` - sample transactions for testing
- `POST /stream/publish` - publish a transaction into the stream
- `GET /stream/next` - fetch the next transaction from the stream
- `POST /analyze` - run the full fraud analysis pipeline

## Dashboard

The Streamlit dashboard shows:

- live transaction feed
- ML risk and anomaly scores
- knowledge base matches
- agent reasoning output
- final decision and confidence
- charts for decision split and risk trends

## Setup

1. Clone the repository.
2. Create a virtual environment with Python 3.11 or newer.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file from `.env.example` and add your real values:

```dotenv
TOGETHER_API_KEY=your_real_together_key
REDIS_URL=rediss://default:your_upstash_token@your-upstash-host:6379
REDIS_STREAM_KEY=afdis:transactions
REDIS_STREAM_GROUP=afdis-consumers
REDIS_STREAM_CONSUMER=dashboard
AGENT_TEMPERATURE=0.2
AGENT_TOP_P=0.9
AGENT_MAX_RETRIES=2
AGENT_TIMEOUT_S=30
```

5. Run the API:

```bash
python run_api.py
```

6. Run the dashboard:

```bash
python run_dashboard.py
```

7. Open these URLs locally:

- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## Notes

- The dashboard is the UI layer; the API and dashboard can be deployed separately.
- The broker automatically falls back to memory when Redis is unavailable.
- The project is designed to work with live Redis on deployment, but still remain usable locally without it.
