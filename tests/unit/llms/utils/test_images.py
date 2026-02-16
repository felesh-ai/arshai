"""Unit tests for image utility helpers."""

import base64
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from arshai.llms.utils.images import image_file_to_base64, image_url_to_base64


class TestImageFileToBase64:
    """Tests for image_file_to_base64 function."""

    def test_image_file_to_base64_success(self):
        """Test successful conversion of image file to base64."""
        # Create a temporary test image file (1x1 PNG)
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            f.write(png_bytes)
            temp_path = f.name

        try:
            # Convert file to base64
            result = image_file_to_base64(temp_path)

            # Verify result is valid base64 string
            assert isinstance(result, str)
            assert len(result) > 0

            # Verify it can be decoded back to the original bytes
            decoded = base64.b64decode(result)
            assert decoded == png_bytes

        finally:
            # Cleanup
            os.unlink(temp_path)

    def test_image_file_to_base64_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            image_file_to_base64("/nonexistent/path/image.png")

    def test_image_file_to_base64_returns_string(self):
        """Test that function returns a string type."""
        # Create a minimal valid file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            f.write(b"test data")
            temp_path = f.name

        try:
            result = image_file_to_base64(temp_path)
            assert isinstance(result, str)
            # Verify it's valid base64
            base64.b64decode(result)  # Should not raise exception

        finally:
            os.unlink(temp_path)


class TestImageUrlToBase64:
    """Tests for image_url_to_base64 function."""

    @patch('urllib.request.urlopen')
    def test_image_url_to_base64_success(self, mock_urlopen):
        """Test successful conversion of image URL to base64."""
        # Mock HTTP response with PNG data
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )

        mock_response = Mock()
        mock_response.read.return_value = png_bytes
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        # Convert URL to base64
        result = image_url_to_base64("https://example.com/image.png")

        # Verify result
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify it can be decoded back to the original bytes
        decoded = base64.b64decode(result)
        assert decoded == png_bytes

        # Verify urlopen was called
        mock_urlopen.assert_called_once()

    @patch('urllib.request.urlopen')
    def test_image_url_to_base64_http_error(self, mock_urlopen):
        """Test error handling when URL returns HTTP error."""
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="https://example.com/image.png",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None
        )

        with pytest.raises(HTTPError):
            image_url_to_base64("https://example.com/image.png")

    @patch('urllib.request.urlopen')
    def test_image_url_to_base64_with_custom_timeout(self, mock_urlopen):
        """Test that custom timeout is passed to urlopen."""
        mock_response = Mock()
        mock_response.read.return_value = b"test data"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        # Call with custom timeout
        image_url_to_base64("https://example.com/image.png", timeout=30)

        # Verify urlopen was called with timeout
        call_args = mock_urlopen.call_args
        assert call_args[1].get('timeout') == 30 or call_args[0][1] == 30

    @patch('urllib.request.urlopen')
    def test_image_url_to_base64_returns_string(self, mock_urlopen):
        """Test that function returns a string type."""
        mock_response = Mock()
        mock_response.read.return_value = b"any data"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = image_url_to_base64("https://example.com/image.png")

        assert isinstance(result, str)
        # Verify it's valid base64
        base64.b64decode(result)  # Should not raise exception
