"""Tests for the ISpeechTranscriber protocol.

Per Phase 2 Task 2.2 of the forensic video-RAG plan
(`core/video_rag/plans/2026-05-21-forensic-video-rag.md`).
"""

from typing import Any, Dict, List, Optional

import pytest

from arshai.core.interfaces.ispeech_transcriber import ISpeechTranscriber


class _DummyTranscriber:
    """Minimal implementation that satisfies ISpeechTranscriber."""

    async def transcribe(
        self,
        audio: str,
        language: Optional[str] = None,
        response_format: str = "json",
        hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "text": "hello world",
            "language": language or "en",
            "segments": [],
            "upstream_request_id": "req-1",
        }


class TestISpeechTranscriberProtocol:
    def test_dummy_is_protocol_instance(self):
        assert isinstance(_DummyTranscriber(), ISpeechTranscriber)

    @pytest.mark.asyncio
    async def test_transcribe_url_only(self):
        client = _DummyTranscriber()
        out = await client.transcribe("https://example.com/audio.mp3")
        assert isinstance(out, dict)
        assert "text" in out
        assert isinstance(out["text"], str)

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self):
        client = _DummyTranscriber()
        out = await client.transcribe(
            "https://example.com/audio.mp3", language="fa"
        )
        assert out["language"] == "fa"

    @pytest.mark.asyncio
    async def test_transcribe_with_hints(self):
        client = _DummyTranscriber()
        out = await client.transcribe(
            "https://example.com/audio.mp3",
            language="en",
            response_format="json",
            hints=["car", "garage"],
        )
        assert "text" in out
        assert "upstream_request_id" in out

    @pytest.mark.asyncio
    async def test_transcribe_returns_upstream_request_id(self):
        out = await _DummyTranscriber().transcribe("https://example.com/audio.mp3")
        assert "upstream_request_id" in out
        assert isinstance(out["upstream_request_id"], str)
