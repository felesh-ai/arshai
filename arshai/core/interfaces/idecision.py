"""Decision models: closed questions in, probabilities out (e.g. TypeSafe Jev via an AI gateway)."""
from typing import Any, Dict, Optional, Protocol
from pydantic import BaseModel, Field


class IDecisionResult(BaseModel):
    answers: Dict[str, Dict[str, Any]] = Field(description="Per question id, the provider's answer as returned")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Provider/gateway usage, passed through untouched")


class IDecisionModel(Protocol):
    async def decide(self, state: Dict[str, Any], questions: Dict[str, Dict[str, Any]], *,
                     headers: Optional[Dict[str, str]] = None) -> IDecisionResult: ...
