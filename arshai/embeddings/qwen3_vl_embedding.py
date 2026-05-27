"""Tongyi vision embedding client (gateway-fronted).

The active v1 implementation backing the ``IMultimodalEmbedding`` protocol.
Routes through the AI Gateway's ``POST /v1/multimodal-embeddings`` endpoint,
which translates to DashScope's native ``tongyi-embedding-vision-plus`` (and
its successor ``qwen3-vl-embedding`` when activated upstream).

This module currently exposes the configuration class only (Phase 2 Task
2.3). The implementation class ``TongyiVisionEmbedding`` lands in Tasks 2.4
and 2.5 — input resolution + the async ``embed`` / ``embed_fused`` methods.

The file name is ``qwen3_vl_embedding.py`` (not ``tongyi_vision_embedding.py``)
intentionally: the upstream model swap (``qwen3-vl-embedding`` →
``tongyi-embedding-vision-plus``, see ADR 0001) was a one-config-line
runtime change, not a re-write — keeping the original module name avoids
churn for every future caller and matches the project's expectation that
the swap is reversible.
"""

import os
from typing import Optional

from pydantic import Field, field_validator

from arshai.core.interfaces.iembedding import EmbeddingConfig


class TongyiVisionEmbeddingConfig(EmbeddingConfig):
    """Configuration for the Tongyi vision embedding gateway client.

    Credentials (``gateway_base_url``, ``gateway_token``) fall back to the
    ``GATEWAY_BASE_URL`` and ``GATEWAY_TOKEN`` environment variables when not
    supplied explicitly. This mirrors the convention used by 12-factor app
    callers and avoids leaking secrets into config files.

    Field guide:

    * ``dimension``: target embedding dimensionality. Defaults to 1024 — the
      Matryoshka prefix truncation of the full 1152-d Tongyi output, verified
      ``cosine=1.000000`` against the v1024-direct path (Phase 0 Part 1).
    * ``batch_limit``: max items per upstream call. DashScope's multimodal
      route caps at 10; the default matches that.
    * ``max_concurrency``: in-flight batches under an ``asyncio.Semaphore``.
      Conservative default; bump after watching gateway-side rate limits.
    * ``staging_bucket`` / ``staging_url_ttl_seconds``: bucket name and signed-URL
      TTL used by the implementation when a caller passes ``video_path``
      (resolved to a signed URL before the upstream call). Optional — only
      required when callers actually use ``video_path``.
    """

    gateway_base_url: Optional[str] = Field(
        default=None,
        validate_default=True,
        description=(
            "AI Gateway base URL (e.g. https://ai-gateway.svc.cluster-prod...). "
            "Falls back to GATEWAY_BASE_URL env var."
        ),
    )
    gateway_token: Optional[str] = Field(
        default=None,
        validate_default=True,
        description=(
            "AI Gateway virtual API key. Falls back to GATEWAY_TOKEN env var."
        ),
    )

    dimension: int = Field(
        default=1024,
        description="Target embedding dimension (Matryoshka prefix when < model max).",
    )
    batch_limit: int = Field(
        default=10,
        description="Max items per upstream call; DashScope multimodal caps at 10.",
    )
    max_concurrency: int = Field(
        default=4,
        description="In-flight batches under asyncio.Semaphore.",
    )

    staging_bucket: Optional[str] = Field(
        default=None,
        description="Object-storage bucket used to stage video_path inputs.",
    )
    staging_url_ttl_seconds: int = Field(
        default=86400,
        description="TTL (seconds) for signed URLs minted for staged inputs.",
    )

    @field_validator("gateway_base_url", mode="before")
    @classmethod
    def _gateway_base_url_env_fallback(cls, v: Optional[str]) -> Optional[str]:
        return v if v is not None else os.environ.get("GATEWAY_BASE_URL")

    @field_validator("gateway_token", mode="before")
    @classmethod
    def _gateway_token_env_fallback(cls, v: Optional[str]) -> Optional[str]:
        return v if v is not None else os.environ.get("GATEWAY_TOKEN")
