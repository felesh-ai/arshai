"""
Unit tests for sampling control parameters across all LLM clients.

Tests cover:
- ILLMConfig new fields
- _build_sampling_kwargs() on AIGatewayLLM, OpenRouterClient, OpenAIClient, AzureClient
- Gemini GenerateContentConfig wiring
"""

import pytest
from unittest.mock import MagicMock, patch
from arshai.core.interfaces.illm import ILLMConfig


# ---------------------------------------------------------------------------
# 7.1 ILLMConfig accepts and stores all new fields
# ---------------------------------------------------------------------------

class TestILLMConfigNewFields:
    """Task 7.1: ILLMConfig accepts and stores all new fields."""

    def test_new_fields_default_to_none(self):
        config = ILLMConfig(model="test-model")
        assert config.top_k is None
        assert config.reasoning_max_tokens is None
        assert config.reasoning_effort is None
        assert config.extra_body is None

    def test_new_fields_can_be_set(self):
        config = ILLMConfig(
            model="test-model",
            top_k=40,
            reasoning_max_tokens=4000,
            reasoning_effort="high",
            extra_body={"custom": "value"},
        )
        assert config.top_k == 40
        assert config.reasoning_max_tokens == 4000
        assert config.reasoning_effort == "high"
        assert config.extra_body == {"custom": "value"}

    def test_existing_fields_still_work(self):
        config = ILLMConfig(
            model="test-model",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2


# ---------------------------------------------------------------------------
# Helper: build a minimal client instance without calling __init__ network setup
# ---------------------------------------------------------------------------

def _make_ai_gateway_client(config):
    from arshai.llms.ai_gateway import AIGatewayLLM
    client = object.__new__(AIGatewayLLM)
    client.config = config
    client._client = MagicMock()
    import logging
    client.logger = logging.getLogger("test")
    client._background_tasks = set()
    return client


def _make_openrouter_client(config):
    from arshai.llms.openrouter import OpenRouterClient
    client = object.__new__(OpenRouterClient)
    client.config = config
    client._client = MagicMock()
    client._provider_name = "openrouter"
    import logging
    client.logger = logging.getLogger("test")
    client._background_tasks = set()
    return client


def _make_openai_client(config):
    from arshai.llms.openai import OpenAIClient
    client = object.__new__(OpenAIClient)
    client.config = config
    client._client = MagicMock()
    client._provider_name = "openai"
    import logging
    client.logger = logging.getLogger("test")
    client._background_tasks = set()
    return client


def _make_azure_client(config):
    from arshai.llms.azure import AzureClient
    client = object.__new__(AzureClient)
    client.config = config
    client._client = MagicMock()
    client._provider_name = "azure"
    import logging
    client.logger = logging.getLogger("test")
    client._background_tasks = set()
    return client


# ---------------------------------------------------------------------------
# 7.2 AIGatewayLLM._build_sampling_kwargs() correct extra_body
# ---------------------------------------------------------------------------

class TestAIGatewayBuildSamplingKwargs:
    """Task 7.2 & 7.3: AIGatewayLLM._build_sampling_kwargs()."""

    def _config(self, **kwargs):
        from arshai.llms.ai_gateway import AIGatewayConfig
        return AIGatewayConfig(
            model="test-model",
            base_url="https://example.com",
            gateway_token="test-token",
            **kwargs,
        )

    def test_basic_fields_included(self):
        config = self._config(temperature=0.5, max_tokens=512)
        client = _make_ai_gateway_client(config)
        messages = [{"role": "user", "content": "hi"}]
        result = client._build_sampling_kwargs(messages)
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 512
        assert result["messages"] is messages

    def test_top_k_in_extra_body(self):
        config = self._config(top_k=50)
        client = _make_ai_gateway_client(config)
        result = client._build_sampling_kwargs([])
        assert result["extra_body"]["top_k"] == 50

    def test_reasoning_effort_in_extra_body(self):
        config = self._config(top_k=50, reasoning_effort="medium", extra_body={"custom": 1})
        client = _make_ai_gateway_client(config)
        result = client._build_sampling_kwargs([])
        eb = result["extra_body"]
        assert eb["top_k"] == 50
        assert eb["reasoning"] == {"effort": "medium"}
        assert eb["custom"] == 1

    def test_reasoning_effort_takes_precedence(self):
        """Task 7.3: reasoning_effort takes precedence over reasoning_max_tokens."""
        config = self._config(reasoning_effort="low", reasoning_max_tokens=4000)
        client = _make_ai_gateway_client(config)
        result = client._build_sampling_kwargs([])
        assert result["extra_body"]["reasoning"] == {"effort": "low"}
        assert "max_tokens" not in result["extra_body"].get("reasoning", {})

    def test_reasoning_max_tokens_used_when_no_effort(self):
        config = self._config(reasoning_max_tokens=4000)
        client = _make_ai_gateway_client(config)
        result = client._build_sampling_kwargs([])
        assert result["extra_body"]["reasoning"] == {"max_tokens": 4000}

    def test_no_extra_body_when_nothing_set(self):
        """Task 7.6: empty/None fields are omitted."""
        config = self._config()
        client = _make_ai_gateway_client(config)
        result = client._build_sampling_kwargs([])
        assert "extra_body" not in result

    def test_overrides_applied(self):
        config = self._config()
        client = _make_ai_gateway_client(config)
        result = client._build_sampling_kwargs([], stream=True, tools=[{"type": "function"}])
        assert result["stream"] is True
        assert result["tools"] == [{"type": "function"}]

    def test_optional_fields_omitted_when_none(self):
        """Task 7.6: max_tokens, top_p etc not in result when not set."""
        config = self._config()
        client = _make_ai_gateway_client(config)
        result = client._build_sampling_kwargs([])
        assert "max_tokens" not in result
        assert "top_p" not in result
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result


# ---------------------------------------------------------------------------
# 7.4 OpenRouterClient._build_sampling_kwargs() uses max_tokens
# ---------------------------------------------------------------------------

class TestOpenRouterBuildSamplingKwargs:
    """Task 7.4: OpenRouterClient uses max_tokens (Chat Completions API)."""

    def _config(self, **kwargs):
        return ILLMConfig(model="test-model", **kwargs)

    def test_uses_max_tokens_not_max_output_tokens(self):
        config = self._config(max_tokens=1000)
        client = _make_openrouter_client(config)
        result = client._build_sampling_kwargs([])
        assert result["max_tokens"] == 1000
        assert "max_output_tokens" not in result

    def test_top_k_in_extra_body(self):
        config = self._config(top_k=40)
        client = _make_openrouter_client(config)
        result = client._build_sampling_kwargs([])
        assert result["extra_body"]["top_k"] == 40

    def test_no_extra_body_when_nothing_set(self):
        config = self._config()
        client = _make_openrouter_client(config)
        result = client._build_sampling_kwargs([])
        assert "extra_body" not in result


# ---------------------------------------------------------------------------
# 7.5 OpenAIClient._build_sampling_kwargs() uses max_output_tokens
# ---------------------------------------------------------------------------

class TestOpenAIBuildSamplingKwargs:
    """Task 7.5: OpenAIClient uses max_output_tokens (Responses API)."""

    def _config(self, **kwargs):
        return ILLMConfig(model="test-model", **kwargs)

    def test_uses_max_output_tokens_not_max_tokens(self):
        config = self._config(max_tokens=1000)
        client = _make_openai_client(config)
        result = client._build_sampling_kwargs([])
        assert result["max_output_tokens"] == 1000
        assert "max_tokens" not in result

    def test_uses_input_key_not_messages(self):
        config = self._config()
        client = _make_openai_client(config)
        response_input = [{"type": "message", "role": "user", "content": "hi"}]
        result = client._build_sampling_kwargs(response_input)
        assert result["input"] is response_input
        assert "messages" not in result

    def test_no_extra_body_when_nothing_set(self):
        config = self._config()
        client = _make_openai_client(config)
        result = client._build_sampling_kwargs([])
        assert "extra_body" not in result


# ---------------------------------------------------------------------------
# 7.6 AzureClient._build_sampling_kwargs() uses max_output_tokens
# ---------------------------------------------------------------------------

class TestAzureBuildSamplingKwargs:
    """AzureClient same as OpenAI — Responses API."""

    def _config(self, **kwargs):
        return ILLMConfig(model="test-model", **kwargs)

    def test_uses_max_output_tokens(self):
        config = self._config(max_tokens=2048)
        client = _make_azure_client(config)
        result = client._build_sampling_kwargs([])
        assert result["max_output_tokens"] == 2048
        assert "max_tokens" not in result

    def test_reasoning_effort_takes_precedence(self):
        config = self._config(reasoning_effort="high", reasoning_max_tokens=8000)
        client = _make_azure_client(config)
        result = client._build_sampling_kwargs([])
        assert result["extra_body"]["reasoning"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# 7.7 Gemini GenerateContentConfig receives top_k and penalty fields
# ---------------------------------------------------------------------------

class TestGeminiSamplingFields:
    """Task 7.7: Gemini _create_generation_config wires top_k and penalties."""

    def _make_gemini_client(self, config):
        from arshai.llms.google_genai import GeminiClient
        client = object.__new__(GeminiClient)
        client.config = config
        client._client = MagicMock()
        client._provider_name = "gemini"
        client.model_config = {}
        import logging
        client.logger = logging.getLogger("test")
        client._background_tasks = set()
        return client

    def test_top_k_passed_to_generation_config(self):
        config = ILLMConfig(model="gemini-pro", top_k=40)
        client = self._make_gemini_client(config)
        gen_config = client._create_generation_config()
        assert gen_config.top_k == 40

    def test_penalties_passed_to_generation_config(self):
        config = ILLMConfig(model="gemini-pro", presence_penalty=0.5, frequency_penalty=0.3)
        client = self._make_gemini_client(config)
        gen_config = client._create_generation_config()
        assert gen_config.presence_penalty == 0.5
        assert gen_config.frequency_penalty == 0.3

    def test_reasoning_and_extra_body_not_passed(self):
        config = ILLMConfig(
            model="gemini-pro",
            reasoning_effort="high",
            extra_body={"key": "val"},
        )
        client = self._make_gemini_client(config)
        gen_config = client._create_generation_config()
        assert not hasattr(gen_config, "reasoning_effort") or gen_config.reasoning_effort is None
        assert not hasattr(gen_config, "extra_body") or gen_config.extra_body is None

    def test_omitted_fields_not_in_config(self):
        config = ILLMConfig(model="gemini-pro")
        client = self._make_gemini_client(config)
        gen_config = client._create_generation_config()
        # top_k, presence_penalty, frequency_penalty should not be set (None or absent)
        top_k = getattr(gen_config, "top_k", None)
        assert top_k is None
