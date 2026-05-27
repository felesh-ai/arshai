"""Qwen3 ASR client — gateway-fronted speech transcription.

The v1 ``ISpeechTranscriber`` implementation. Routes through the AI Gateway's
``POST /v1/audio/transcriptions`` route (URL-driven JSON; see AI-Gateway
ADR 0007), which translates to DashScope's native ``qwen3-asr-flash``.

URL-only on the input side: callers stage audio to object storage and pass
a signed URL. No S3 / staging dependency lives here — that's the caller's
problem and intentionally so.

The implementation is small by design: ASR is a one-shot call per audio
file, the gateway already handles upstream retries, and the calling layer
(``core/video_rag``) wraps each transcription in a ``pipeline_runs`` row
for resumability. No client-side retry loop is wired in here for v1; the
gateway's ``/v1/audio/transcriptions`` route returns plain HTTP errors on
upstream failure and callers can re-invoke.
"""

import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import Field, field_validator

from arshai.core.interfaces.idto import IDTO


class Qwen3ASRConfig(IDTO):
    """Configuration for ``Qwen3ASR``.

    Inherits ``IDTO`` rather than ``EmbeddingConfig`` / ``ILLMConfig``: ASR
    isn't really either of those, and inheriting LLM-shaped fields
    (``temperature``, ``max_tokens``) would be misleading. Credentials use
    the same env-fallback pattern as ``TongyiVisionEmbeddingConfig`` so a
    single gateway deployment can serve both.
    """

    gateway_base_url: Optional[str] = Field(
        default=None,
        validate_default=True,
        description="AI Gateway base URL. Falls back to GATEWAY_BASE_URL env var.",
    )
    gateway_token: Optional[str] = Field(
        default=None,
        validate_default=True,
        description="AI Gateway virtual API key. Falls back to GATEWAY_TOKEN env var.",
    )
    model: str = Field(
        default="qwen3-asr-flash",
        description="Upstream ASR model id as registered in the gateway.",
    )

    @field_validator("gateway_base_url", mode="before")
    @classmethod
    def _gateway_base_url_env_fallback(cls, v: Optional[str]) -> Optional[str]:
        return v if v is not None else os.environ.get("GATEWAY_BASE_URL")

    @field_validator("gateway_token", mode="before")
    @classmethod
    def _gateway_token_env_fallback(cls, v: Optional[str]) -> Optional[str]:
        return v if v is not None else os.environ.get("GATEWAY_TOKEN")


class Qwen3ASR:
    """Gateway-fronted ``ISpeechTranscriber`` for ``qwen3-asr-flash``.

    ``transcribe()`` returns a dict matching the ``ISpeechTranscriber``
    contract — ``segments`` is always ``[]`` for this provider (the
    upstream model returns a single transcript chunk; OpenAI's
    ``verbose_json`` shape is preserved structurally but unpopulated).
    """

    def __init__(
        self,
        config: Qwen3ASRConfig,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config
        self._http = http_client

    def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.config.gateway_base_url)
        return self._http

    async def transcribe(
        self,
        audio: str,
        language: Optional[str] = None,
        response_format: str = "json",
        hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # The gateway's request schema is extra=forbid; ``hints`` are not
        # part of v1. Drop them silently rather than forward and 422.
        # Callers that need biasing can re-issue at the application layer.
        _ = hints

        body: Dict[str, Any] = {
            "model": self.config.model,
            "audio": audio,
            "response_format": response_format,
        }
        if language is not None:
            body["language"] = language

        headers: Dict[str, str] = {}
        if self.config.gateway_token:
            headers["Authorization"] = f"Bearer {self.config.gateway_token}"

        http = self._ensure_http()
        resp = await http.post(
            "/v1/audio/transcriptions", json=body, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "text": data["text"],
            "language": data.get("language"),
            "segments": [],
            "upstream_request_id": data.get("upstream_request_id") or "",
        }
