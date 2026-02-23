"""
Unit tests for PDF message-building logic across all LLM providers.
Tests the internal _create_*_messages helpers in isolation — no API calls.
"""

import base64
from unittest.mock import MagicMock, patch

import pytest

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput

# Minimal fake PDF base64 (b"%PDF-1.4\ntest")
FAKE_PDF_RAW = base64.b64encode(b"%PDF-1.4\ntest").decode()
FAKE_PDF_DATA_URL = f"data:application/pdf;base64,{FAKE_PDF_RAW}"
FAKE_IMG_RAW = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def _make_config(model="test-model"):
    return ILLMConfig(model=model, temperature=0.0)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class TestOpenAIMessageBuilding:

    def _make_client(self):
        from arshai.llms.openai import OpenAIClient
        config = _make_config()
        with patch.object(OpenAIClient, '_initialize_client', return_value=MagicMock()):
            return OpenAIClient(config)

    def test_text_only_unchanged(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hello")
        msgs = client._create_multimodal_messages(inp)
        assert msgs[1]["content"] == "hello"

    def test_pdf_raw_gets_data_prefix(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hi", pdfs_base64=[FAKE_PDF_RAW])
        msgs = client._create_multimodal_messages(inp)
        content = msgs[1]["content"]
        pdf_block = next(b for b in content if b.get("type") == "input_file")
        assert pdf_block["file_data"].startswith("data:application/pdf;base64,")
        assert pdf_block["filename"] == "document.pdf"

    def test_pdf_data_url_kept_as_is(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hi", pdfs_base64=[FAKE_PDF_DATA_URL])
        msgs = client._create_multimodal_messages(inp)
        content = msgs[1]["content"]
        pdf_block = next(b for b in content if b.get("type") == "input_file")
        assert pdf_block["file_data"] == FAKE_PDF_DATA_URL

    def test_image_and_pdf_together(self):
        client = self._make_client()
        inp = ILLMInput(
            system_prompt="sys", user_message="hi",
            images_base64=[FAKE_IMG_RAW],
            pdfs_base64=[FAKE_PDF_RAW]
        )
        msgs = client._create_multimodal_messages(inp)
        content = msgs[1]["content"]
        types = [b["type"] for b in content]
        assert "text" in types
        assert "image_url" in types
        assert "input_file" in types


# ---------------------------------------------------------------------------
# Azure
# ---------------------------------------------------------------------------

class TestAzureMessageBuilding:

    def _make_client(self):
        from arshai.llms.azure import AzureClient
        config = _make_config()
        with patch.object(AzureClient, '_initialize_client', return_value=MagicMock()):
            return AzureClient(config, azure_deployment="test-deployment", api_version="2024-02-01")

    def test_text_only_unchanged(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hello")
        msgs = client._create_multimodal_response_input(inp)
        user_msg = next(m for m in msgs if m.get("role") == "user")
        assert user_msg["content"] == "hello"

    def test_pdf_raw_gets_data_prefix(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hi", pdfs_base64=[FAKE_PDF_RAW])
        msgs = client._create_multimodal_response_input(inp)
        user_msg = next(m for m in msgs if m.get("role") == "user")
        pdf_block = next(b for b in user_msg["content"] if b.get("type") == "input_file")
        assert pdf_block["file_data"].startswith("data:application/pdf;base64,")

    def test_image_and_pdf_together(self):
        client = self._make_client()
        inp = ILLMInput(
            system_prompt="sys", user_message="hi",
            images_base64=[FAKE_IMG_RAW],
            pdfs_base64=[FAKE_PDF_RAW]
        )
        msgs = client._create_multimodal_response_input(inp)
        user_msg = next(m for m in msgs if m.get("role") == "user")
        types = [b["type"] for b in user_msg["content"]]
        assert "image_url" in types
        assert "input_file" in types


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

class TestOpenRouterMessageBuilding:

    def _make_client(self):
        from arshai.llms.openrouter import OpenRouterClient
        config = _make_config()
        with patch.object(OpenRouterClient, '_initialize_client', return_value=MagicMock()):
            return OpenRouterClient(config)

    def test_text_only_unchanged(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hello")
        msgs = client._create_openai_messages(inp)
        assert msgs[1]["content"] == "hello"

    def test_pdf_uses_file_block_format(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hi", pdfs_base64=[FAKE_PDF_RAW])
        msgs = client._create_openai_messages(inp)
        content = msgs[1]["content"]
        file_block = next(b for b in content if b.get("type") == "file")
        assert "file" in file_block
        assert file_block["file"]["filename"] == "document.pdf"
        assert file_block["file"]["file_data"].startswith("data:application/pdf;base64,")

    def test_image_and_pdf_together(self):
        client = self._make_client()
        inp = ILLMInput(
            system_prompt="sys", user_message="hi",
            images_base64=[FAKE_IMG_RAW],
            pdfs_base64=[FAKE_PDF_RAW]
        )
        msgs = client._create_openai_messages(inp)
        content = msgs[1]["content"]
        types = [b["type"] for b in content]
        assert "image_url" in types
        assert "file" in types


# ---------------------------------------------------------------------------
# AIGateway
# ---------------------------------------------------------------------------

class TestAIGatewayMessageBuilding:

    def _make_client(self):
        from arshai.llms.ai_gateway import AIGatewayLLM
        config = _make_config()
        with patch.object(AIGatewayLLM, '_initialize_client', return_value=MagicMock()):
            return AIGatewayLLM(config)

    def test_text_only_unchanged(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hello")
        msgs = client._create_openai_messages(inp)
        assert msgs[1]["content"] == "hello"

    def test_pdf_raw_gets_data_prefix(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hi", pdfs_base64=[FAKE_PDF_RAW])
        msgs = client._create_openai_messages(inp)
        content = msgs[1]["content"]
        pdf_block = next(b for b in content if b.get("type") == "file")
        assert pdf_block["file"]["file_data"].startswith("data:application/pdf;base64,")
        assert pdf_block["file"]["filename"] == "document.pdf"

    def test_image_and_pdf_together(self):
        client = self._make_client()
        inp = ILLMInput(
            system_prompt="sys", user_message="hi",
            images_base64=[FAKE_IMG_RAW],
            pdfs_base64=[FAKE_PDF_RAW]
        )
        msgs = client._create_openai_messages(inp)
        content = msgs[1]["content"]
        types = [b["type"] for b in content]
        assert "image_url" in types
        assert "file" in types


# ---------------------------------------------------------------------------
# Gemini (prepare_base_context)
# ---------------------------------------------------------------------------

class TestGeminiPdfPreparation:

    def _make_client(self):
        from arshai.llms.google_genai import GeminiClient
        config = _make_config("gemini-2.0-flash")
        with patch.object(GeminiClient, '_initialize_client', return_value=MagicMock()):
            return GeminiClient(config)

    def test_text_only_returns_string(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hello")
        result = client._prepare_base_context(inp)
        assert isinstance(result, str)
        assert "hello" in result

    def test_pdf_only_returns_list(self):
        client = self._make_client()
        inp = ILLMInput(system_prompt="sys", user_message="hi", pdfs_base64=[FAKE_PDF_RAW])
        with patch.object(client, '_convert_pdfs_to_parts', return_value=["<pdf_part>"]) as mock_convert:
            result = client._prepare_base_context(inp)
        assert isinstance(result, list)
        mock_convert.assert_called_once_with([FAKE_PDF_RAW])

    def test_images_and_pdfs_both_included(self):
        client = self._make_client()
        inp = ILLMInput(
            system_prompt="sys", user_message="hi",
            images_base64=[FAKE_IMG_RAW],
            pdfs_base64=[FAKE_PDF_RAW]
        )
        with patch.object(client, '_convert_base64_to_parts', return_value=["<img_part>"]) as mock_img, \
             patch.object(client, '_convert_pdfs_to_parts', return_value=["<pdf_part>"]) as mock_pdf:
            result = client._prepare_base_context(inp)
        assert isinstance(result, list)
        assert "<img_part>" in result
        assert "<pdf_part>" in result
        mock_img.assert_called_once()
        mock_pdf.assert_called_once()

    def test_convert_pdfs_strips_data_prefix(self):
        client = self._make_client()
        from google.genai.types import Part
        with patch.object(Part, 'from_bytes', return_value=MagicMock()) as mock_part:
            client._convert_pdfs_to_parts([FAKE_PDF_DATA_URL])
        call_kwargs = mock_part.call_args[1]
        assert call_kwargs["mime_type"] == "application/pdf"
        # Verify the bytes decoded are from the raw portion (no prefix)
        assert call_kwargs["data"] == base64.b64decode(FAKE_PDF_RAW)
