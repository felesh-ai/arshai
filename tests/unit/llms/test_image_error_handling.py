"""
Error handling tests for multimodal (image) support.

These tests document expected behavior when invalid or problematic
image data is provided to LLM clients.
"""

import base64

import pytest

from arshai.core.interfaces.illm import ILLMInput


class TestInvalidBase64Handling:
    """Test handling of invalid base64 strings."""

    def test_empty_images_list_text_only_behavior(self):
        """Test that empty images_base64 list results in text-only behavior."""
        # This should work fine - empty list is valid
        input_data = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Hello, how are you?",
            images_base64=[]  # Empty list - text-only mode
        )

        assert input_data.images_base64 == []
        assert len(input_data.images_base64) == 0

    def test_images_base64_default_is_empty_list(self):
        """Test that images_base64 defaults to empty list when not provided."""
        input_data = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Hello, how are you?"
            # images_base64 not provided - should default to []
        )

        assert input_data.images_base64 == []
        assert isinstance(input_data.images_base64, list)


class TestProviderErrorHandling:
    """
    Tests for provider-specific error handling.

    Note: Actual error handling (invalid base64, unsupported formats,
    oversized images) is provider-specific and tested at the provider level.
    The framework passes data through to providers who handle validation.
    """

    def test_invalid_base64_string_format(self):
        """
        Test that invalid base64 strings are accepted by ILLMInput.

        Validation happens at the provider level, not in ILLMInput.
        Providers will return appropriate errors when they attempt to decode.
        """
        # ILLMInput accepts any string - providers validate
        input_data = ILLMInput(
            system_prompt="You are a vision assistant.",
            user_message="Describe this image.",
            images_base64=["this_is_not_valid_base64!!!"]
        )

        # ILLMInput creation succeeds
        assert len(input_data.images_base64) == 1
        assert input_data.images_base64[0] == "this_is_not_valid_base64!!!"

        # Note: Provider will raise error when attempting to decode

    def test_mixed_valid_and_data_url_formats(self):
        """Test that mixed base64 formats (raw and data URL) are accepted."""
        # Sample 1x1 PNG
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        sample_image_raw = base64.b64encode(png_bytes).decode("utf-8")
        sample_image_data_url = f"data:image/png;base64,{sample_image_raw}"

        input_data = ILLMInput(
            system_prompt="You are a vision assistant.",
            user_message="Compare these images.",
            images_base64=[sample_image_raw, sample_image_data_url]
        )

        assert len(input_data.images_base64) == 2
        # First is raw base64
        assert not input_data.images_base64[0].startswith("data:")
        # Second is data URL format
        assert input_data.images_base64[1].startswith("data:image/")


class TestImageDataValidation:
    """Tests for image data validation at ILLMInput level."""

    def test_images_base64_must_be_list_of_strings(self):
        """Test that images_base64 must be a list of strings."""
        # Valid: list of strings
        input_data = ILLMInput(
            system_prompt="You are a vision assistant.",
            user_message="Describe this image.",
            images_base64=["base64string1", "base64string2"]
        )
        assert len(input_data.images_base64) == 2

    def test_very_long_base64_string_accepted(self):
        """
        Test that very long base64 strings are accepted by ILLMInput.

        Size limits are provider-specific:
        - Gemini: 20MB
        - OpenAI/Azure: ~4MB (varies by model)
        - OpenRouter: Varies by underlying provider

        Providers handle size validation and return errors if exceeded.
        """
        # Create a reasonably long base64 string (simulating a larger image)
        large_data = "A" * 100000  # 100KB of 'A' characters
        large_base64 = base64.b64encode(large_data.encode()).decode()

        input_data = ILLMInput(
            system_prompt="You are a vision assistant.",
            user_message="Describe this image.",
            images_base64=[large_base64]
        )

        # ILLMInput accepts it - provider will enforce size limits
        assert len(input_data.images_base64) == 1
        assert len(input_data.images_base64[0]) > 100000


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing code."""

    def test_existing_text_only_code_unchanged(self):
        """Test that existing text-only code works without modification."""
        # This is how code worked before images_base64 was added
        input_data = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="What is 2+2?"
        )

        # Should work exactly as before
        assert input_data.system_prompt == "You are a helpful assistant."
        assert input_data.user_message == "What is 2+2?"
        assert input_data.images_base64 == []  # Defaults to empty list

    def test_all_existing_parameters_still_work(self):
        """Test that all pre-existing parameters continue to work."""
        def example_tool(query: str) -> str:
            """Example tool for testing."""
            return f"Result for: {query}"

        input_data = ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Search for Python",
            regular_functions={"example_tool": example_tool},
            max_turns=5
        )

        # All existing functionality intact
        assert input_data.system_prompt == "You are a helpful assistant."
        assert input_data.user_message == "Search for Python"
        assert "example_tool" in input_data.regular_functions
        assert input_data.max_turns == 5
        assert input_data.images_base64 == []  # New field has safe default
