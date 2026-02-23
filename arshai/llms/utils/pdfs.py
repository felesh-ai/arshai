"""
Optional utility helpers for converting PDFs to base64.
Basic helpers using stdlib only. Developers are encouraged to
implement their own logic for production use (async loading,
caching, preprocessing, error handling, etc.).
"""
import base64
import urllib.request


def pdf_file_to_base64(file_path: str) -> str:
    """Convert local PDF file to base64 string (no data URL prefix)."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def pdf_url_to_base64(url: str, timeout: int = 10) -> str:
    """Fetch PDF from URL and convert to base64 string (no data URL prefix)."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return base64.b64encode(response.read()).decode('utf-8')
