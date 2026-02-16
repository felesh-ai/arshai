"""
Optional utility helpers for converting images to base64.

These are basic helpers using stdlib only. Developers are encouraged to
implement their own conversion logic with their preferred libraries,
optimization strategies, caching, and error handling for production use.

Tasks 7.1-7.5
"""

import base64
import urllib.request
from typing import Optional


def image_file_to_base64(file_path: str) -> str:
    """
    Convert local image file to base64 string.

    This is a BASIC helper using stdlib only. For production use, developers
    should implement their own logic with:
    - Image preprocessing (resize, compress, format conversion)
    - Async file I/O if needed
    - Better error handling
    - Libraries like PIL/Pillow for advanced processing

    Args:
        file_path: Path to the image file

    Returns:
        Base64-encoded string (without data URL prefix)

    Example:
        >>> img_b64 = image_file_to_base64("photo.jpg")
        >>> # Use with framework
        >>> input = ILLMInput(
        ...     system_prompt="Analyze this",
        ...     user_message="What's in the image?",
        ...     images_base64=[img_b64]
        ... )
    """
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def image_url_to_base64(url: str, timeout: int = 10) -> str:
    """
    Fetch image from URL and convert to base64 string.

    This is a BASIC helper with NO retries, caching, or advanced error handling.
    For production use, developers should implement their own logic with:
    - Async HTTP client (httpx, aiohttp)
    - Retry logic with exponential backoff
    - Caching (memory, Redis, disk)
    - Better error handling and logging
    - Support for authenticated endpoints

    Args:
        url: HTTP/HTTPS URL of the image
        timeout: Request timeout in seconds (default: 10)

    Returns:
        Base64-encoded string (without data URL prefix)

    Raises:
        urllib.error.URLError: If URL fetch fails

    Example:
        >>> img_b64 = image_url_to_base64("https://example.com/image.jpg")
        >>> # Use with framework
        >>> input = ILLMInput(
        ...     system_prompt="Analyze this",
        ...     user_message="What's in the image?",
        ...     images_base64=[img_b64]
        ... )
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return base64.b64encode(response.read()).decode('utf-8')
