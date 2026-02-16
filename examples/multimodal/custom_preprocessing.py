"""
Example: Custom Image Preprocessing - Production-ready image handling.

This example demonstrates how to implement custom image preprocessing
using PIL/Pillow for production use cases including:
- Image resizing and optimization
- Format conversion
- Compression to reduce API costs
- Batch processing
- Caching strategies
"""

import asyncio
import base64
import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.google_genai import GeminiClient

try:
    from PIL import Image
except ImportError:
    print("❌ This example requires Pillow. Install it with: pip install Pillow")
    exit(1)


class ImagePreprocessor:
    """
    Production-ready image preprocessor with optimization and caching.

    Features:
    - Automatic resizing to stay within provider limits
    - Format conversion (PNG, JPEG, WebP)
    - Compression to reduce API costs
    - Memory-based caching to avoid re-encoding
    """

    def __init__(
        self,
        max_size: int = 2048,
        format: str = "JPEG",
        quality: int = 85,
        cache_enabled: bool = True
    ):
        """
        Initialize preprocessor.

        Args:
            max_size: Maximum dimension (width or height) in pixels
            format: Output format (JPEG, PNG, WebP)
            quality: Compression quality (1-100, only for JPEG/WebP)
            cache_enabled: Whether to cache encoded images
        """
        self.max_size = max_size
        self.format = format
        self.quality = quality
        self.cache_enabled = cache_enabled
        self._cache = {}

    def preprocess_image(self, image_path: str) -> str:
        """
        Preprocess an image file to optimized base64.

        Args:
            image_path: Path to image file

        Returns:
            Base64-encoded optimized image
        """
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(image_path)
            if cache_key in self._cache:
                print(f"   ✓ Cache hit for {Path(image_path).name}")
                return self._cache[cache_key]

        # Load image
        img = Image.open(image_path)

        # Convert RGBA to RGB if needed (for JPEG)
        if img.mode in ("RGBA", "LA") and self.format == "JPEG":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode != "RGB" and self.format == "JPEG":
            img = img.convert("RGB")

        # Resize if needed
        if max(img.size) > self.max_size:
            img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
            print(f"   ✓ Resized {Path(image_path).name} to {img.size}")

        # Encode to bytes
        buffer = BytesIO()
        save_kwargs = {"format": self.format}
        if self.format in ("JPEG", "WEBP"):
            save_kwargs["quality"] = self.quality
            save_kwargs["optimize"] = True

        img.save(buffer, **save_kwargs)
        image_bytes = buffer.getvalue()

        # Encode to base64
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = base64_str

        print(f"   ✓ Encoded {Path(image_path).name}: {len(base64_str)} chars")
        return base64_str

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key based on path and settings."""
        key_data = f"{image_path}:{self.max_size}:{self.format}:{self.quality}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def clear_cache(self):
        """Clear the preprocessing cache."""
        self._cache.clear()


async def main():
    """Demonstrate custom image preprocessing."""
    print("=" * 70)
    print("Custom Image Preprocessing Example")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ Error: GOOGLE_API_KEY environment variable not set")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        return

    # Initialize preprocessor with custom settings
    print("\n🔧 Initializing Image Preprocessor")
    print(f"   - Max size: 2048px")
    print(f"   - Format: JPEG")
    print(f"   - Quality: 85%")
    print(f"   - Caching: Enabled")

    preprocessor = ImagePreprocessor(
        max_size=2048,      # Resize large images
        format="JPEG",      # Use JPEG for smaller size
        quality=85,         # Good quality vs size balance
        cache_enabled=True  # Cache encoded images
    )

    # Initialize LLM client
    config = ILLMConfig(
        model="gemini-1.5-flash",
        temperature=0.5
    )
    llm = GeminiClient(config)

    # Example 1: Single image with preprocessing
    print("\n" + "=" * 70)
    print("Example 1: Single Image Preprocessing")
    print("=" * 70)

    # For demo purposes, create a sample image
    # In production, you'd use actual image files
    print("\n📸 Creating sample image...")
    sample_img = Image.new("RGB", (3000, 2000), color=(255, 100, 100))
    sample_path = "/tmp/sample_large_image.png"
    sample_img.save(sample_path)
    print(f"   Created {sample_path} (3000x2000px)")

    # Preprocess the image
    print("\n⚙️  Preprocessing image...")
    optimized_image = preprocessor.preprocess_image(sample_path)

    # Compare sizes
    with open(sample_path, "rb") as f:
        original_size = len(base64.b64encode(f.read()))
    optimized_size = len(optimized_image)

    print(f"\n📊 Size Comparison:")
    print(f"   - Original: {original_size:,} chars")
    print(f"   - Optimized: {optimized_size:,} chars")
    print(f"   - Reduction: {((1 - optimized_size/original_size) * 100):.1f}%")

    # Analyze with LLM
    print("\n🤖 Analyzing preprocessed image...")
    input_data = ILLMInput(
        system_prompt="You are a vision assistant. Analyze images briefly.",
        user_message="Describe this image concisely.",
        images_base64=[optimized_image]
    )

    response = await llm.chat(input_data)
    print(f"\n✅ Response: {response['llm_response']}")

    # Example 2: Batch processing with cache
    print("\n" + "=" * 70)
    print("Example 2: Batch Processing with Caching")
    print("=" * 70)

    # Create multiple sample images
    print("\n📸 Creating batch of images...")
    image_paths = []
    for i in range(3):
        img = Image.new("RGB", (1024, 768), color=(50 * i, 100, 200 - 50 * i))
        path = f"/tmp/batch_image_{i}.png"
        img.save(path)
        image_paths.append(path)
    print(f"   Created {len(image_paths)} images")

    # First pass - preprocessing
    print("\n⚙️  First pass (preprocessing)...")
    preprocessed_images = []
    for path in image_paths:
        preprocessed_images.append(preprocessor.preprocess_image(path))

    # Second pass - using cache
    print("\n⚙️  Second pass (using cache)...")
    for path in image_paths:
        preprocessor.preprocess_image(path)  # Should hit cache

    # Analyze batch
    print(f"\n🤖 Analyzing {len(preprocessed_images)} images...")
    input_data_batch = ILLMInput(
        system_prompt="You are a vision assistant comparing multiple images.",
        user_message=f"I'm providing {len(preprocessed_images)} images. Briefly describe the differences you observe.",
        images_base64=preprocessed_images
    )

    response_batch = await llm.chat(input_data_batch)
    print(f"\n✅ Batch Analysis: {response_batch['llm_response']}")

    # Cleanup
    print("\n🧹 Cleaning up temporary files...")
    os.unlink(sample_path)
    for path in image_paths:
        os.unlink(path)

    print("\n" + "=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
✅ Production Best Practices Demonstrated:

1. **Image Optimization**
   - Resize large images to stay within provider limits
   - Use JPEG for photos (smaller size than PNG)
   - Adjust quality for cost/quality balance

2. **Caching**
   - Cache preprocessed images to avoid re-encoding
   - Use hash-based keys for cache invalidation

3. **Batch Processing**
   - Process multiple images efficiently
   - Reuse preprocessed results from cache

4. **Format Conversion**
   - Convert RGBA to RGB for JPEG compatibility
   - Handle different input formats uniformly

5. **Error Handling**
   - Graceful handling of corrupt/missing files
   - Clear error messages for debugging

For your production use:
- Adjust max_size based on your provider (Gemini: 20MB, OpenAI: 4MB)
- Use Redis/disk cache for persistent caching across runs
- Add async processing for better performance
- Implement rate limiting and retry logic
- Monitor API costs and optimize quality settings
    """)

    print("=" * 70)
    print("✅ Custom Preprocessing Example Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
