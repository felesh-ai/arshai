"""Generic client for the System One decision wire: `POST {base_url}{path}` with `{model, state, questions}` ->
`{answers, usage}`. Every System One server speaks it (TypeSafe Jev, Kev and Laya servers), and so does an AI
gateway's pass-through route; only `base_url` and `path` differ. Named for the wire protocol, not the provider.

No retries here: the gateway owns its one retry and never retries a timeout, so a client retry would
push the worst case past a caller's turn budget. Every failure is one exception, so callers have one
fallback path."""
from typing import Any, Dict, Optional

import httpx
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, Field

from arshai.core.interfaces.idecision import IDecisionResult


class DecisionUnavailable(RuntimeError):
    """The decision call produced no usable answers (HTTP error, timeout, transport, malformed body)."""


class SystemOneConfig(BaseModel):
    base_url: str = Field(description="Server base URL ending in /v1")
    api_key: str
    model: str
    timeout: float = 6.0
    path: str = Field(default="/systemone", description="'/decisions' behind the Felesh ai-gateway")
    headers: Dict[str, str] = Field(default_factory=dict)


class SystemOneClient:
    def __init__(self, config: SystemOneConfig, *, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self.config = config
        self._client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, max_retries=0,
                                   timeout=config.timeout, default_headers=config.headers or None,
                                   http_client=http_client)

    async def decide(self, state: Dict[str, Any], questions: Dict[str, Dict[str, Any]], *,
                     headers: Optional[Dict[str, str]] = None) -> IDecisionResult:
        qs = {k: {kk: vv for kk, vv in q.items() if not (kk == "criteria" and not vv)} for k, q in questions.items()}
        try:
            resp = await self._client.post(self.config.path, cast_to=httpx.Response,
                                           body={"model": self.config.model, "state": state, "questions": qs},
                                           options={"headers": headers or {}})
            body = resp.json()
            return IDecisionResult(answers=body["answers"], usage=body.get("usage") or {})
        except (APIError, httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            raise DecisionUnavailable(f"{type(e).__name__}: {e}") from e
