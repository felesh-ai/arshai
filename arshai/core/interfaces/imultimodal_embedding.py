"""Multimodal embedding interface + input DTO.

Adds Arshai-level support for "embed something that isn't only text" — surveillance
image frames, short video clips, mixed text+image captions, etc. Created as part
of the forensic video-RAG project (Phase 2 Task 2.1 of
`core/video_rag/plans/2026-05-21-forensic-video-rag.md`).

Why this lives next to `IEmbedding` instead of extending it:

* `IEmbedding` is text-only by contract (`embed_documents(texts: List[str])`).
  Bolting multimodality onto it would either widen the input type to `Any` (loss
  of static checking) or break every existing caller.
* The return contract diverges intentionally: `embed()` here returns *one* vector
  per `MultimodalInput`, while a multimodal `embed_fused()` fuses *multiple*
  modality fields of a single `MultimodalInput` into one vector. Two distinct
  shapes, one protocol.

The active v1 implementation backing this protocol is `TongyiVisionEmbedding`,
which calls the AI Gateway's `/v1/multimodal-embeddings` route.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import Field, model_validator

from .idto import IDTO


class MultimodalInput(IDTO):
    """A single item to embed. Exactly one input modality is enough; mixing
    several is allowed (the implementation decides how to combine them, e.g.
    via the upstream model's fusion mode).

    At least one of ``text`` / ``image_base64`` / ``image_url`` / ``video_url``
    / ``video_path`` must be supplied; passing all-None raises
    ``pydantic.ValidationError``.

    ``video_path`` is the only non-URL/inline field — implementations accepting
    it are expected to stage the file to object storage and substitute a signed
    URL before calling the upstream model.
    """

    text: Optional[str] = Field(default=None, description="Caption / query text.")
    image_base64: Optional[str] = Field(
        default=None, description="Inline base64-encoded image bytes."
    )
    image_url: Optional[str] = Field(
        default=None, description="Externally-fetchable image URL."
    )
    video_url: Optional[str] = Field(
        default=None, description="Externally-fetchable video URL."
    )
    video_path: Optional[str] = Field(
        default=None,
        description=(
            "Local filesystem path to a video. Implementations resolve this to a "
            "signed URL by uploading to object storage before calling the model."
        ),
    )

    @model_validator(mode="after")
    def _at_least_one_modality(self) -> "MultimodalInput":
        if not any(
            v is not None
            for v in (
                self.text,
                self.image_base64,
                self.image_url,
                self.video_url,
                self.video_path,
            )
        ):
            raise ValueError(
                "MultimodalInput requires at least one of: text, image_base64, "
                "image_url, video_url, video_path"
            )
        return self


@runtime_checkable
class IMultimodalEmbedding(Protocol):
    """Interface for multimodal embedding services.

    **Return contract** — both methods return ``Dict[str, Any]``, with a
    ``"dense"`` key holding the vector(s). The shape under ``"dense"`` differs
    between the two methods (see below). The mirror of `IEmbedding`'s
    ``{"dense": ...}`` convention is intentional so that downstream code can
    treat text-only and multimodal embeddings uniformly when it doesn't care.

    Dense-key asymmetry
    -------------------
    * ``embed()`` returns ``{"dense": List[List[float]], ...}`` — one vector
      per input item.
    * ``embed_fused()`` returns ``{"dense": List[float], ...}`` — a single
      vector fused from one item's multiple modality fields.

    Fusion semantics
    ----------------
    * ``embed()`` produces one vector per ``MultimodalInput``. If an item has
      both ``text`` and ``image_url``, the implementation chooses how to encode
      them — typically by sending each modality separately and not fusing.
    * ``embed_fused()`` fuses multiple modality fields of a *single*
      ``MultimodalInput`` into one combined vector, using the upstream model's
      fusion mode (e.g. ``enable_fusion=True`` on DashScope multimodal).

    Provider capability note
    ------------------------
    The active v1 implementation (``TongyiVisionEmbedding`` backed by
    ``tongyi-embedding-vision-plus``) raises ``NotImplementedError`` from
    ``embed_fused()`` — that upstream model does not support fusion. The method
    is kept on the protocol so a swap to ``qwen3-vl-embedding`` (which does
    support fusion) is a one-config-line change. No v1 caller uses
    ``embed_fused()``.

    Partial-failure contract
    ------------------------
    ``embed()`` raises on the first batch failure — implementations use
    ``asyncio.gather`` *without* ``return_exceptions=True``. Resumability /
    retry of partial batches is the caller's responsibility (in the forensic
    video-RAG project this lives in the ``pipeline_runs`` table).

    Upstream request-id passthrough
    -------------------------------
    Both methods surface the gateway's ``X-Upstream-Request-ID`` header so that
    forensic audit logs can correlate to upstream provider traces:

    * ``embed()`` → ``"upstream_request_ids": List[str]`` (one per batch).
    * ``embed_fused()`` → ``"upstream_request_id": str`` (scalar; single call).
    """

    @property
    def dimension(self) -> int:
        """Embedding output dimensionality."""
        ...

    async def embed(self, items: List[MultimodalInput]) -> Dict[str, Any]:
        """Embed multiple items, one vector per item.

        Returns a dict with at minimum::

            {"dense": List[List[float]], "upstream_request_ids": List[str]}
        """
        ...

    async def embed_fused(self, item: MultimodalInput) -> Dict[str, Any]:
        """Embed a single item, fusing its modality fields into one vector.

        Returns a dict with at minimum::

            {"dense": List[float], "upstream_request_id": str}

        Implementations whose upstream model does not support fusion may raise
        ``NotImplementedError``.
        """
        ...
