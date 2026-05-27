"""Tests for IMultimodalEmbedding protocol and MultimodalInput DTO.

Per Phase 2 Task 2.1 of the forensic video-RAG plan
(`core/video_rag/plans/2026-05-21-forensic-video-rag.md`).
"""

from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from arshai.core.interfaces.imultimodal_embedding import (
    IMultimodalEmbedding,
    MultimodalInput,
)


class TestMultimodalInput:
    def test_accepts_text_only(self):
        item = MultimodalInput(text="a person walking")
        assert item.text == "a person walking"
        assert item.image_url is None
        assert item.video_url is None

    def test_accepts_image_url_only(self):
        item = MultimodalInput(image_url="https://example.com/frame.jpg")
        assert item.image_url == "https://example.com/frame.jpg"

    def test_accepts_video_url_only(self):
        item = MultimodalInput(video_url="https://example.com/clip.mp4")
        assert item.video_url == "https://example.com/clip.mp4"

    def test_accepts_image_base64_only(self):
        item = MultimodalInput(image_base64="aGVsbG8=")
        assert item.image_base64 == "aGVsbG8="

    def test_accepts_video_path_only(self):
        item = MultimodalInput(video_path="/tmp/clip.mp4")
        assert item.video_path == "/tmp/clip.mp4"

    def test_accepts_multiple_modalities(self):
        item = MultimodalInput(text="caption", image_url="https://x/y.jpg")
        assert item.text == "caption"
        assert item.image_url == "https://x/y.jpg"

    def test_rejects_all_none(self):
        with pytest.raises(ValidationError):
            MultimodalInput()

    def test_rejects_all_explicit_none(self):
        with pytest.raises(ValidationError):
            MultimodalInput(
                text=None,
                image_base64=None,
                image_url=None,
                video_url=None,
                video_path=None,
            )


class _DummyMultimodalEmbedding:
    """Minimal implementation that satisfies the IMultimodalEmbedding protocol."""

    @property
    def dimension(self) -> int:
        return 1024

    async def embed(self, items: List[MultimodalInput]) -> Dict[str, Any]:
        return {
            "dense": [[0.0] * self.dimension for _ in items],
            "upstream_request_ids": ["req-1"],
        }

    async def embed_fused(self, item: MultimodalInput) -> Dict[str, Any]:
        return {
            "dense": [0.0] * self.dimension,
            "upstream_request_id": "req-1",
        }


class TestIMultimodalEmbeddingProtocol:
    def test_dummy_is_protocol_instance(self):
        client = _DummyMultimodalEmbedding()
        # Runtime-checkable protocol structural check.
        assert isinstance(client, IMultimodalEmbedding)

    def test_dimension_returns_int(self):
        client = _DummyMultimodalEmbedding()
        assert isinstance(client.dimension, int)
        assert client.dimension > 0

    @pytest.mark.asyncio
    async def test_embed_returns_dense_list_of_lists(self):
        client = _DummyMultimodalEmbedding()
        out = await client.embed([MultimodalInput(text="a"), MultimodalInput(text="b")])
        assert "dense" in out
        assert isinstance(out["dense"], list)
        assert len(out["dense"]) == 2
        assert isinstance(out["dense"][0], list)
        assert "upstream_request_ids" in out
        assert isinstance(out["upstream_request_ids"], list)

    @pytest.mark.asyncio
    async def test_embed_fused_returns_flat_dense(self):
        client = _DummyMultimodalEmbedding()
        out = await client.embed_fused(MultimodalInput(text="a", image_url="https://x/y.jpg"))
        assert "dense" in out
        assert isinstance(out["dense"], list)
        # Flat vector, not list of vectors.
        assert all(isinstance(v, (int, float)) for v in out["dense"])
        assert "upstream_request_id" in out
        assert isinstance(out["upstream_request_id"], str)
