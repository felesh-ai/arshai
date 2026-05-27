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

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, List, Optional

from pydantic import Field, field_validator

from arshai.core.interfaces.iembedding import EmbeddingConfig
from arshai.core.interfaces.imultimodal_embedding import MultimodalInput

logger = logging.getLogger(__name__)


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


class TongyiVisionEmbedding:
    """Gateway-fronted Tongyi vision embedding client.

    Implements ``IMultimodalEmbedding`` for the AI Gateway's
    ``/v1/multimodal-embeddings`` endpoint. This class is built up across
    Tasks 2.4 / 2.5: input resolution lives here; the async ``embed`` /
    ``embed_fused`` methods land in 2.5.

    Dependencies are injected, not constructed:

    * ``s3_client``: boto3-shaped client used to stage ``video_path`` inputs
      to ``config.staging_bucket`` and mint signed GET URLs. Required even
      when no caller currently uses ``video_path`` — keeps construction
      uniform and the resolution path testable.
    * ``http_client``: optional ``httpx.AsyncClient``. ``None`` means the
      implementation will own its own client (created lazily in Task 2.5).
    """

    def __init__(
        self,
        config: TongyiVisionEmbeddingConfig,
        s3_client: Any,
        http_client: Optional[Any] = None,
    ) -> None:
        self.config = config
        self._s3 = s3_client
        self._http = http_client

    @property
    def dimension(self) -> int:
        return self.config.dimension

    async def _resolve_items(
        self, items: List[MultimodalInput]
    ) -> List[MultimodalInput]:
        """Return a list parallel to ``items`` with ``video_path`` inputs
        resolved to signed ``video_url`` references.

        Items whose only modality fields are passthrough types (``text``,
        ``image_base64``, ``image_url``, ``video_url``) are returned
        unchanged. The s3 client is invoked exactly once per ``video_path``.
        """
        resolved: List[MultimodalInput] = []
        for item in items:
            if item.video_path is None:
                resolved.append(item)
                continue

            if not self.config.staging_bucket:
                raise ValueError(
                    "TongyiVisionEmbedding received a video_path input but "
                    "config.staging_bucket is not set."
                )

            key = self._make_staging_key(item.video_path)
            await asyncio.to_thread(
                self._s3.upload_file,
                item.video_path,
                self.config.staging_bucket,
                key,
            )
            signed_url: str = await asyncio.to_thread(
                self._s3.generate_presigned_url,
                "get_object",
                Params={"Bucket": self.config.staging_bucket, "Key": key},
                ExpiresIn=self.config.staging_url_ttl_seconds,
            )

            # Build a new immutable item: keep every other modality field,
            # swap video_path → video_url. IDTO sets validate_assignment so
            # constructing afresh is the safe path.
            resolved.append(
                MultimodalInput(
                    text=item.text,
                    image_base64=item.image_base64,
                    image_url=item.image_url,
                    video_url=signed_url,
                    video_path=None,
                )
            )
        return resolved

    @staticmethod
    def _make_staging_key(path: str) -> str:
        suffix = Path(path).suffix or ".bin"
        return f"video-rag-staging/{uuid.uuid4().hex}{suffix}"
