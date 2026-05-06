"""
Unit tests for per-call extra_headers forwarding in AIGatewayLLM.

These tests verify that ILLMInput.extra_headers is forwarded to the underlying
OpenAI client's chat.completions.create() call so that gateways can read
request-scoped identity headers (e.g. X-Auth-User).

Mocks the OpenAI client; no network or credentials required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arshai.core.interfaces.illm import ILLMInput
from arshai.llms.ai_gateway import AIGatewayConfig, AIGatewayLLM


def _build_client_with_mock(monkeypatch, capture: dict) -> AIGatewayLLM:
    """Construct AIGatewayLLM with its OpenAI client replaced by a capturing mock."""

    config = AIGatewayConfig(
        base_url="https://gateway.example.test/v1",
        gateway_token="test-token",
        model="test-model",
        temperature=0.0,
    )

    # Bypass the real _initialize_client so we don't open a network client.
    monkeypatch.setattr(
        AIGatewayLLM,
        "_initialize_client",
        lambda self: MagicMock(),
    )

    llm = AIGatewayLLM(config)

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
        id="resp-1",
    )

    def fake_create(**kwargs):
        capture["kwargs"] = kwargs
        return fake_response

    llm._client.chat.completions.create = fake_create
    return llm


@pytest.mark.asyncio
async def test_chat_simple_forwards_extra_headers(monkeypatch):
    capture: dict = {}
    llm = _build_client_with_mock(monkeypatch, capture)

    await llm.chat(
        ILLMInput(
            system_prompt="sys",
            user_message="hi",
            extra_headers={"X-Auth-User": "abc-base64", "X-Trace-Id": "t-1"},
        )
    )

    assert "kwargs" in capture, "OpenAI client was not invoked"
    assert capture["kwargs"].get("extra_headers") == {
        "X-Auth-User": "abc-base64",
        "X-Trace-Id": "t-1",
    }


@pytest.mark.asyncio
async def test_chat_simple_omits_extra_headers_when_unset(monkeypatch):
    capture: dict = {}
    llm = _build_client_with_mock(monkeypatch, capture)

    await llm.chat(ILLMInput(system_prompt="sys", user_message="hi"))

    assert "kwargs" in capture
    # Absent rather than None — we never want to pass an explicit None to the SDK.
    assert "extra_headers" not in capture["kwargs"]


@pytest.mark.asyncio
async def test_stream_simple_forwards_extra_headers(monkeypatch):
    capture: dict = {}
    llm = _build_client_with_mock(monkeypatch, capture)

    # Replace create() with a generator-returning callable to model streaming.
    def fake_create(**kwargs):
        capture["kwargs"] = kwargs
        # Return an empty iterable; _stream_simple tolerates that and yields a final usage row.
        return iter([])

    llm._client.chat.completions.create = fake_create

    async for _ in llm.stream(
        ILLMInput(
            system_prompt="sys",
            user_message="hi",
            extra_headers={"X-Auth-User": "stream-user"},
        )
    ):
        pass

    assert capture["kwargs"].get("extra_headers") == {"X-Auth-User": "stream-user"}
    assert capture["kwargs"].get("stream") is True
