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
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import Field, field_validator

from arshai.core.interfaces.iembedding import EmbeddingConfig
from arshai.core.interfaces.imultimodal_embedding import MultimodalInput

logger = logging.getLogger(__name__)

RETRY_STATUSES = {429, 502, 503, 504}
_MAX_ATTEMPTS = 4


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

    # -- gateway request shape -------------------------------------------------

    @staticmethod
    def _item_to_payload(item: MultimodalInput) -> Dict[str, str]:
        """Convert a (resolved) ``MultimodalInput`` to the gateway's
        ``MultiModalItem`` shape — only the three URL/text fields the
        gateway accepts. ``image_base64`` is not supported in v1 and is
        rejected here with a clear error so the failure mode is
        client-side, not a 422 from upstream.
        """
        if item.image_base64 is not None and item.image_url is None:
            raise NotImplementedError(
                "TongyiVisionEmbedding v1 does not stage image_base64 inputs. "
                "Encode the image as a fetchable image_url (e.g. via R2 "
                "presigned URL) before calling embed()."
            )
        payload: Dict[str, str] = {}
        if item.text is not None:
            payload["text"] = item.text
        if item.image_url is not None:
            payload["image_url"] = item.image_url
        if item.video_url is not None:
            payload["video_url"] = item.video_url
        return payload

    # -- HTTP plumbing ---------------------------------------------------------

    def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.config.gateway_base_url)
        return self._http

    async def _post(self, payload: dict) -> httpx.Response:
        """POST with exponential backoff on the retryable status set.

        Honors ``Retry-After`` when present. Returns the last response
        (even if retryable) after ``_MAX_ATTEMPTS`` tries — caller is
        responsible for ``raise_for_status`` on the final return.
        """
        http = self._ensure_http()
        headers: Dict[str, str] = {}
        if self.config.gateway_token:
            headers["Authorization"] = f"Bearer {self.config.gateway_token}"

        delay = 1.0
        resp: Optional[httpx.Response] = None
        for attempt in range(_MAX_ATTEMPTS):
            resp = await http.post(
                "/v1/multimodal-embeddings", json=payload, headers=headers
            )
            if resp.status_code not in RETRY_STATUSES:
                return resp
            retry_after = resp.headers.get("Retry-After")
            await asyncio.sleep(float(retry_after) if retry_after else delay)
            delay *= 2
        assert resp is not None  # loop runs at least once.
        return resp

    async def _embed_batch(
        self, items: List[MultimodalInput], enable_fusion: bool
    ) -> Tuple[List[List[float]], str]:
        """One upstream call. Returns (vectors, upstream_request_id)."""
        payload = {
            "model": self.config.model_name,
            "input": [self._item_to_payload(item) for item in items],
            "dimension": self.config.dimension,
            "enable_fusion": enable_fusion,
        }
        resp = await self._post(payload)
        resp.raise_for_status()
        body = resp.json()
        # `data` is index-ordered; sort defensively in case upstream returns
        # out of order (cheap; len == batch_limit at most).
        data = sorted(body["data"], key=lambda d: d["index"])
        vectors = [d["embedding"] for d in data]
        upstream_id = body.get("upstream_request_id") or ""
        return vectors, upstream_id

    # -- IMultimodalEmbedding --------------------------------------------------

    async def embed(self, items: List[MultimodalInput]) -> Dict[str, Any]:
        """Embed a list of items; one vector per item."""
        if not items:
            return {"dense": [], "upstream_request_ids": []}

        resolved = await self._resolve_items(items)

        # Chunk into batches of size <= batch_limit.
        batch_limit = self.config.batch_limit
        batches: List[List[MultimodalInput]] = [
            resolved[i : i + batch_limit]
            for i in range(0, len(resolved), batch_limit)
        ]

        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def _run(batch: List[MultimodalInput]) -> Tuple[List[List[float]], str]:
            async with sem:
                return await self._embed_batch(batch, enable_fusion=False)

        # Intentionally NOT return_exceptions=True — partial failure is the
        # caller's problem (resumability lives in pipeline_runs).
        results = await asyncio.gather(*(_run(b) for b in batches))

        all_vectors: List[List[float]] = []
        upstream_ids: List[str] = []
        for vectors, upstream_id in results:
            all_vectors.extend(vectors)
            upstream_ids.append(upstream_id)

        return {"dense": all_vectors, "upstream_request_ids": upstream_ids}

    async def embed_fused(self, item: MultimodalInput) -> Dict[str, Any]:
        """Fuse the modality fields of a single item into one vector.

        The active v1 upstream (``tongyi-embedding-vision-plus``) returns
        a 4xx for ``enable_fusion=true``. The protocol method is preserved
        so a swap to ``qwen3-vl-embedding`` (which supports fusion) is a
        one-config-line change; this implementation simply forwards.
        """
        resolved = await self._resolve_items([item])
        vectors, upstream_id = await self._embed_batch(
            resolved, enable_fusion=True
        )
        if not vectors:
            raise RuntimeError("embed_fused upstream returned no vectors")
        return {"dense": vectors[0], "upstream_request_id": upstream_id}
