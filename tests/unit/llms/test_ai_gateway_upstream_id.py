"""Tests for AIGatewayLLM upstream_request_id passthrough.

Per Phase 2 Task 2.7 of the forensic video-RAG plan
(`core/video_rag/plans/2026-05-21-forensic-video-rag.md`).

The change is twofold:

* ``_extract_and_standardize_usage`` accepts a keyword-only
  ``upstream_request_id`` and includes it in the returned usage dict.
* The ``_chat_simple`` call site uses ``with_raw_response.create(...)``
  to capture the ``X-Upstream-Request-ID`` response header and pass it
  through.

This test focuses on the helper-level contract because:

1. Regression risk lives in the helper — every existing usage key must
   still appear.
2. The call-site wiring is structurally trivial; the value passes
   through unchanged.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arshai.llms.ai_gateway import AIGatewayConfig, AIGatewayLLM


_EXISTING_USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "thinking_tokens",
    "tool_calling_tokens",
    "provider",
    "model",
    "request_id",
}


def _make_llm() -> AIGatewayLLM:
    cfg = AIGatewayConfig(
        model="gpt-4o-mini",
        base_url="https://gw.example",
        api_key="t",
    )
    # Patch the sync OpenAI client so __init__ doesn't try to reach the network.
    llm = AIGatewayLLM.__new__(AIGatewayLLM)
    llm.config = cfg
    llm.logger = MagicMock()
    llm._client = MagicMock()
    llm._async_client = MagicMock()
    return llm


def _response_with_usage(usage_obj, response_id: str = "resp-123"):
    return SimpleNamespace(id=response_id, usage=usage_obj)


class TestExtractAndStandardizeUsage:
    """Regression guard + new-behavior tests for the helper."""

    def test_regression_existing_keys_present_when_upstream_id_absent(self):
        llm = _make_llm()
        usage_obj = SimpleNamespace(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        response = _response_with_usage(usage_obj)

        result = llm._extract_and_standardize_usage(response)

        # Every pre-existing key is still present.
        for key in _EXISTING_USAGE_KEYS:
            assert key in result, f"existing key {key!r} dropped from usage dict"

        # Values are unchanged from before the patch.
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30
        assert result["thinking_tokens"] == 0
        assert result["tool_calling_tokens"] == 0
        assert result["provider"] == "gateway"
        assert result["model"] == "gpt-4o-mini"
        assert result["request_id"] == "resp-123"

        # New key: present but None when no upstream id supplied.
        assert "upstream_request_id" in result
        assert result["upstream_request_id"] is None

    def test_upstream_request_id_surfaced_when_provided(self):
        llm = _make_llm()
        usage_obj = SimpleNamespace(
            prompt_tokens=5, completion_tokens=7, total_tokens=12
        )
        response = _response_with_usage(usage_obj)

        result = llm._extract_and_standardize_usage(
            response, upstream_request_id="dashscope-abc-789"
        )

        assert result["upstream_request_id"] == "dashscope-abc-789"
        # Other keys still correct.
        assert result["input_tokens"] == 5
        assert result["request_id"] == "resp-123"

    def test_upstream_id_present_on_missing_usage_branch(self):
        """The early-return path (no usage attr) must also surface the id."""
        llm = _make_llm()
        # Response with no usage attribute / falsy usage.
        response = SimpleNamespace(id="r-x", usage=None)

        result = llm._extract_and_standardize_usage(
            response, upstream_request_id="up-1"
        )

        assert "upstream_request_id" in result
        assert result["upstream_request_id"] == "up-1"
        # Zero-valued tokens (the early-return shape) still present.
        assert result["input_tokens"] == 0
        assert result["request_id"] == "r-x"

    def test_kw_only_argument(self):
        """Passing upstream_request_id positionally must fail — kw-only."""
        llm = _make_llm()
        response = _response_with_usage(SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        ))
        with pytest.raises(TypeError):
            llm._extract_and_standardize_usage(response, "some-id")  # type: ignore[misc]
