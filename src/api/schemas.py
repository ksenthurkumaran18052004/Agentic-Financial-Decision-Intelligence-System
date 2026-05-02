from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = "txn_001"
    amount:          float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_type:   str   = Field(..., description="e.g. grocery, electronics, casino")
    location:        str   = Field(..., description="City or country of transaction")
    hour:            int   = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    is_home_city:    int   = Field(1, ge=0, le=1, description="1 if user's home city")
    is_known_merchant: int = Field(1, ge=0, le=1, description="1 if merchant seen before")
    is_late_night:   int   = Field(0, ge=0, le=1, description="1 if 1am-5am")
    amount_vs_avg:   float = Field(1.0, gt=0, description="Ratio vs user's average spend")

    model_config = {"json_schema_extra": {"example": {
        "transaction_id": "txn_demo_001",
        "amount": 899.99,
        "merchant_type": "electronics",
        "location": "Moscow",
        "hour": 3,
        "is_home_city": 0,
        "is_known_merchant": 0,
        "is_late_night": 1,
        "amount_vs_avg": 8.5,
    }}}


class RAGMatch(BaseModel):
    id:          str
    category:    str
    relevance:   float


class AgentReasoning(BaseModel):
    planner:   str
    risk:      str
    fraud:     str
    insight:   str
    evaluator: str


class AgentContractMetadata(BaseModel):
    agent_name: str
    version: str
    schema_version: str
    model: str


class AgentEvaluation(BaseModel):
    agent_name: str
    contract_version: str
    schema_version: str
    model: str
    latency_ms: int
    retries: int
    output_valid: bool
    validation_errors: list[str]


class AnalysisResponse(BaseModel):
    transaction_id:    str
    decision:          str   # APPROVE | FLAG | BLOCK
    confidence:        float
    reason:            str
    risk_score:        float
    anomaly_score:     float
    is_anomaly:        bool
    mcts_action:       str
    mcts_simulations:  int
    rag_rules_matched: list[RAGMatch]
    agent_reasoning:   AgentReasoning
    agent_contracts:   dict[str, AgentContractMetadata]
    agent_evaluations: dict[str, AgentEvaluation]
    processing_time_s: float
