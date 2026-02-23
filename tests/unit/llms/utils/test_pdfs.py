"""Unit tests for PDF utility helpers."""

import base64
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from arshai.llms.utils.pdfs import pdf_file_to_base64, pdf_url_to_base64


class TestPdfFileToBase64:
    """Tests for pdf_file_to_base64 function."""

    def test_pdf_file_to_base64_success(self):
        """Test successful conversion of a PDF file to base64."""
        pdf_bytes = b"%PDF-1.4 minimal test content"

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            f.write(pdf_bytes)
            temp_path = f.name

        try:
            result = pdf_file_to_base64(temp_path)

            assert isinstance(result, str)
            assert len(result) > 0
            assert base64.b64decode(result) == pdf_bytes
        finally:
            os.unlink(temp_path)

    def test_pdf_file_to_base64_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            pdf_file_to_base64("/nonexistent/path/document.pdf")

    def test_pdf_file_to_base64_returns_no_prefix(self):
        """Test that the returned string has no data URL prefix."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            f.write(b"pdf content")
            temp_path = f.name

        try:
            result = pdf_file_to_base64(temp_path)
            assert not result.startswith("data:")
            base64.b64decode(result)  # Must be valid base64
        finally:
            os.unlink(temp_path)


class TestPdfUrlToBase64:
    """Tests for pdf_url_to_base64 function."""

    @patch('urllib.request.urlopen')
    def test_pdf_url_to_base64_success(self, mock_urlopen):
        """Test successful conversion of a PDF URL to base64."""
        pdf_bytes = b"%PDF-1.4 test content from url"

        mock_response = Mock()
        mock_response.read.return_value = pdf_bytes
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = pdf_url_to_base64("https://example.com/document.pdf")

        assert isinstance(result, str)
        assert base64.b64decode(result) == pdf_bytes
        mock_urlopen.assert_called_once()

    @patch('urllib.request.urlopen')
    def test_pdf_url_to_base64_http_error(self, mock_urlopen):
        """Test error propagation on HTTP error."""
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="https://example.com/document.pdf",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None
        )

        with pytest.raises(HTTPError):
            pdf_url_to_base64("https://example.com/document.pdf")

    @patch('urllib.request.urlopen')
    def test_pdf_url_to_base64_custom_timeout(self, mock_urlopen):
        """Test that custom timeout is forwarded to urlopen."""
        mock_response = Mock()
        mock_response.read.return_value = b"pdf data"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        pdf_url_to_base64("https://example.com/document.pdf", timeout=30)

        call_args = mock_urlopen.call_args
        assert call_args[1].get('timeout') == 30 or call_args[0][1] == 30

    @patch('urllib.request.urlopen')
    def test_pdf_url_to_base64_returns_no_prefix(self, mock_urlopen):
        """Test that returned string has no data URL prefix."""
        mock_response = Mock()
        mock_response.read.return_value = b"any pdf data"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = pdf_url_to_base64("https://example.com/document.pdf")

        assert isinstance(result, str)
        assert not result.startswith("data:")
        base64.b64decode(result)  # Must be valid base64
