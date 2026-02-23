"""
Unit tests for core interfaces, focusing on backward compatibility and multimodal support.
"""

import pytest
from arshai.core.interfaces.illm import ILLMInput


class TestILLMInputBackwardCompatibility:
    """Tests ensuring backward compatibility with existing code."""

    def test_create_illm_input_without_images_field(self):
        """Task 1.3: Verify existing code creates ILLMInput without errors."""
        # This is how existing code creates ILLMInput - should work unchanged
        input_data = ILLMInput(
            system_prompt="You are a helpful assistant",
            user_message="Hello, world!"
        )

        assert input_data.system_prompt == "You are a helpful assistant"
        assert input_data.user_message == "Hello, world!"
        assert input_data.images_base64 == []  # Default to empty list

    def test_illm_input_with_empty_images_default(self):
        """Task 1.4: Test ILLMInput with empty images_base64 (default)."""
        input_data = ILLMInput(
            system_prompt="Test system",
            user_message="Test message",
            images_base64=[]
        )

        assert input_data.images_base64 == []
        assert isinstance(input_data.images_base64, list)

    def test_illm_input_omitting_images_field(self):
        """Test that omitting images_base64 field works (defaults to empty list)."""
        input_data = ILLMInput(
            system_prompt="System",
            user_message="Message"
        )

        assert hasattr(input_data, 'images_base64')
        assert input_data.images_base64 == []


class TestILLMInputWithImages:
    """Tests for new multimodal image support."""

    def test_illm_input_with_single_image(self):
        """Task 1.5: Test ILLMInput with single image."""
        fake_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        input_data = ILLMInput(
            system_prompt="Analyze this image",
            user_message="What do you see?",
            images_base64=[fake_base64]
        )

        assert len(input_data.images_base64) == 1
        assert input_data.images_base64[0] == fake_base64

    def test_illm_input_with_multiple_images(self):
        """Task 1.6: Test ILLMInput with multiple images."""
        image1 = "base64_image_1_data"
        image2 = "base64_image_2_data"
        image3 = "base64_image_3_data"

        input_data = ILLMInput(
            system_prompt="Compare images",
            user_message="What are the differences?",
            images_base64=[image1, image2, image3]
        )

        assert len(input_data.images_base64) == 3
        assert input_data.images_base64[0] == image1
        assert input_data.images_base64[1] == image2
        assert input_data.images_base64[2] == image3

    def test_illm_input_accepts_raw_base64_format(self):
        """Task 1.7: Test ILLMInput accepting raw base64 format."""
        raw_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        input_data = ILLMInput(
            system_prompt="Test",
            user_message="Test",
            images_base64=[raw_base64]
        )

        assert input_data.images_base64[0] == raw_base64

    def test_illm_input_accepts_data_url_format(self):
        """Task 1.7: Test ILLMInput accepting data URL format."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        input_data = ILLMInput(
            system_prompt="Test",
            user_message="Test",
            images_base64=[data_url]
        )

        assert input_data.images_base64[0] == data_url
        assert input_data.images_base64[0].startswith("data:image")

    def test_illm_input_accepts_mixed_formats(self):
        """Task 1.7: Test ILLMInput accepting both raw base64 and data URL formats in same request."""
        raw_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA=="

        input_data = ILLMInput(
            system_prompt="Test",
            user_message="Test",
            images_base64=[raw_base64, data_url]
        )

        assert len(input_data.images_base64) == 2
        assert input_data.images_base64[0] == raw_base64
        assert input_data.images_base64[1] == data_url


class TestILLMInputWithPdfs:
    """Tests for new PDF support via pdfs_base64."""

    def test_pdfs_base64_defaults_to_empty_list(self):
        """Verify pdfs_base64 defaults to empty list when omitted."""
        input_data = ILLMInput(
            system_prompt="You are a document analyst.",
            user_message="Hello"
        )
        assert input_data.pdfs_base64 == []
        assert isinstance(input_data.pdfs_base64, list)

    def test_illm_input_with_single_pdf(self):
        """Test ILLMInput with a single base64 PDF."""
        fake_pdf_b64 = "JVBERi0xLjQKdGVzdA=="  # b"%PDF-1.4\ntest" base64

        input_data = ILLMInput(
            system_prompt="Summarize documents.",
            user_message="What does this say?",
            pdfs_base64=[fake_pdf_b64]
        )

        assert len(input_data.pdfs_base64) == 1
        assert input_data.pdfs_base64[0] == fake_pdf_b64

    def test_illm_input_with_multiple_pdfs(self):
        """Test ILLMInput with multiple PDF documents."""
        pdf1 = "JVBERi0xLjQ="
        pdf2 = "JVBERi0xLjU="

        input_data = ILLMInput(
            system_prompt="Compare these documents.",
            user_message="What are the differences?",
            pdfs_base64=[pdf1, pdf2]
        )

        assert len(input_data.pdfs_base64) == 2
        assert input_data.pdfs_base64[0] == pdf1
        assert input_data.pdfs_base64[1] == pdf2

    def test_illm_input_accepts_raw_pdf_base64(self):
        """Test ILLMInput accepts raw base64 string (no data URL prefix)."""
        raw_b64 = "JVBERi0xLjQKdGVzdA=="

        input_data = ILLMInput(
            system_prompt="Test",
            user_message="Test",
            pdfs_base64=[raw_b64]
        )

        assert not input_data.pdfs_base64[0].startswith("data:")

    def test_illm_input_accepts_pdf_data_url(self):
        """Test ILLMInput accepts data URL format for PDFs."""
        data_url = "data:application/pdf;base64,JVBERi0xLjQKdGVzdA=="

        input_data = ILLMInput(
            system_prompt="Test",
            user_message="Test",
            pdfs_base64=[data_url]
        )

        assert input_data.pdfs_base64[0].startswith("data:application/pdf")

    def test_illm_input_combined_images_and_pdfs(self):
        """Test ILLMInput with both images and PDFs simultaneously."""
        img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        pdf_b64 = "JVBERi0xLjQKdGVzdA=="

        input_data = ILLMInput(
            system_prompt="Analyze everything.",
            user_message="Describe all content.",
            images_base64=[img_b64],
            pdfs_base64=[pdf_b64]
        )

        assert len(input_data.images_base64) == 1
        assert len(input_data.pdfs_base64) == 1

    def test_images_unaffected_when_pdfs_empty(self):
        """Ensure images_base64 behaviour is unchanged when pdfs_base64 is empty."""
        img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        input_data = ILLMInput(
            system_prompt="Vision assistant",
            user_message="Describe the image.",
            images_base64=[img_b64]
        )

        assert len(input_data.images_base64) == 1
        assert input_data.pdfs_base64 == []


class TestILLMInputValidation:
    """Tests for ILLMInput validation and Pydantic behavior."""

    def test_illm_input_requires_system_prompt(self):
        """Verify system_prompt is still required."""
        with pytest.raises(ValueError, match="system_prompt is required"):
            ILLMInput(
                user_message="Test message"
            )

    def test_illm_input_requires_user_message(self):
        """Verify user_message is still required."""
        with pytest.raises(ValueError, match="user_message is required"):
            ILLMInput(
                system_prompt="Test system"
            )

    def test_illm_input_with_all_fields(self):
        """Test ILLMInput with all fields including optional ones."""
        def sample_function():
            return "result"

        input_data = ILLMInput(
            system_prompt="System",
            user_message="Message",
            images_base64=["base64_image"],
            regular_functions={"func": sample_function},
            background_tasks={},
            max_turns=5
        )

        assert input_data.system_prompt == "System"
        assert input_data.user_message == "Message"
        assert len(input_data.images_base64) == 1
        assert "func" in input_data.regular_functions
        assert input_data.max_turns == 5
