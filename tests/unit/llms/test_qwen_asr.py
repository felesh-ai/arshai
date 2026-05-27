"""Tests for Qwen3ASR — the v1 ISpeechTranscriber implementation.

Per Phase 2 Task 2.6 of the forensic video-RAG plan
(`core/video_rag/plans/2026-05-21-forensic-video-rag.md`).
"""

import json

import httpx
import pytest

from arshai.core.interfaces.ispeech_transcriber import ISpeechTranscriber
from arshai.llms.qwen_asr import Qwen3ASR, Qwen3ASRConfig


def _ok_response(
    text: str = "hello world",
    language: str = "en",
    upstream_id: str = "asr-req-1",
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "text": text,
            "language": language,
            "duration": 1.23,
            "upstream_request_id": upstream_id,
        },
    )


def _client(handler) -> Qwen3ASR:
    cfg = Qwen3ASRConfig(
        gateway_base_url="https://gw.example",
        gateway_token="t",
        model="qwen3-asr-flash",
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://gw.example"
    )
    return Qwen3ASR(cfg, http_client=http)


class TestConfigDefaults:
    def test_model_default(self):
        cfg = Qwen3ASRConfig(gateway_base_url="https://x", gateway_token="t")
        assert cfg.model == "qwen3-asr-flash"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_BASE_URL", "https://env.example")
        monkeypatch.setenv("GATEWAY_TOKEN", "env-tok")
        cfg = Qwen3ASRConfig()
        assert cfg.gateway_base_url == "https://env.example"
        assert cfg.gateway_token == "env-tok"


class TestProtocolCompliance:
    def test_is_speech_transcriber(self):
        cfg = Qwen3ASRConfig(gateway_base_url="https://x", gateway_token="t")
        c = Qwen3ASR(cfg)
        assert isinstance(c, ISpeechTranscriber)


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_url_only_call(self):
        captured: list[httpx.Request] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            return _ok_response()

        c = _client(handler)
        out = await c.transcribe("https://r2.example/audio.mp3")

        assert out["text"] == "hello world"
        assert out["language"] == "en"
        assert out["segments"] == []  # DashScope returns no segments.
        assert out["upstream_request_id"] == "asr-req-1"

        assert len(captured) == 1
        body = json.loads(captured[0].content)
        assert body["model"] == "qwen3-asr-flash"
        assert body["audio"] == "https://r2.example/audio.mp3"
        assert body["response_format"] == "json"
        assert "language" not in body  # omitted when None
        assert captured[0].headers["authorization"] == "Bearer t"

    @pytest.mark.asyncio
    async def test_language_hint_is_forwarded(self):
        captured: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(json.loads(req.content))
            return _ok_response(language="fa")

        c = _client(handler)
        out = await c.transcribe(
            "https://r2.example/audio.mp3", language="fa"
        )
        assert out["language"] == "fa"
        assert captured[0]["language"] == "fa"

    @pytest.mark.asyncio
    async def test_response_format_forwarded(self):
        captured: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(json.loads(req.content))
            return _ok_response()

        c = _client(handler)
        await c.transcribe(
            "https://r2.example/audio.mp3", response_format="verbose_json"
        )
        assert captured[0]["response_format"] == "verbose_json"

    @pytest.mark.asyncio
    async def test_hints_are_dropped_silently(self):
        # Gateway schema is extra=forbid; we drop hints rather than 422.
        captured: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(json.loads(req.content))
            return _ok_response()

        c = _client(handler)
        await c.transcribe(
            "https://r2.example/audio.mp3", hints=["car", "garage"]
        )
        assert "hints" not in captured[0]

    @pytest.mark.asyncio
    async def test_4xx_raises(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad audio url"})

        c = _client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await c.transcribe("https://r2.example/audio.mp3")
