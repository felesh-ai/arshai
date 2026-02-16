# Multimodal Examples

This directory contains examples demonstrating multimodal (vision) capabilities in the Arshai framework.

## Overview

All examples use base64-encoded images following Arshai's developer-first approach:
- **Framework provides the interface** (`images_base64` field)
- **You control the implementation** (file loading, preprocessing, optimization)
- **Universal provider support** (Gemini, OpenAI, Azure, OpenRouter)

## Prerequisites

1. **Install Arshai**: `poetry install`
2. **Set up API credentials**:
   - For Gemini: `export GOOGLE_API_KEY="your-key"`
   - For OpenAI: `export OPENAI_API_KEY="your-key"`
   - For Azure: Set Azure-specific environment variables
3. **Prepare image files**: Have image files ready to test with

## Examples

### 1. `basic_image_analysis.py`
**What it shows:**
- Loading and encoding a single image
- Basic image analysis with Gemini
- Asking specific questions about images

**Run:**
```bash
python basic_image_analysis.py
```

### 2. `multiple_images_comparison.py`
**What it shows:**
- Comparing multiple images in one request
- Analyzing image sequences (before/after, time-lapse)
- Finding common elements across images

**Run:**
```bash
python multiple_images_comparison.py
```

### 3. `image_with_tools.py` *(Coming Soon)*
**What it shows:**
- Combining vision with function calling
- Using tools based on image analysis
- Multi-turn conversations with images

### 4. `streaming_with_images.py` *(Coming Soon)*
**What it shows:**
- Streaming responses with images
- Progressive text generation
- Real-time analysis feedback

### 5. `cross_provider_example.py` *(Coming Soon)*
**What it shows:**
- Same code working across all providers
- Provider-agnostic multimodal code
- Swapping providers with minimal changes

### 6. `custom_preprocessing.py` *(Coming Soon)*
**What it shows:**
- Custom image preprocessing with PIL/Pillow
- Resizing, compressing, format conversion
- Optimization strategies for production

## Developer-First Approach

These examples use the simplest possible approach (stdlib `base64` module). For production, you should implement:

### Image Preprocessing
```python
from PIL import Image
import base64
from io import BytesIO

def preprocess_image(path: str, max_size: int = 1024) -> str:
    img = Image.open(path)

    # Resize
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))

    # Convert format
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Compress
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)

    return base64.b64encode(buffer.getvalue()).decode('utf-8')
```

### Async Loading
```python
import aiofiles
import base64

async def load_image_async(path: str) -> str:
    async with aiofiles.open(path, 'rb') as f:
        data = await f.read()
        return base64.b64encode(data).decode('utf-8')
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def load_image_cached(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')
```

## Tips & Best Practices

### Image Optimization
- **Resize** images to 2048x2048 max before encoding
- **Compress** with appropriate quality settings
- **Choose format** wisely (JPEG for photos, PNG for graphics)

### Provider Limits
- **Gemini**: 20MB inline base64 images
- **OpenAI/Azure**: ~4MB typical
- **OpenRouter**: Varies by underlying model

### Cost Optimization
- Smaller images = fewer tokens = lower cost
- Use "low detail" mode for OpenAI when high resolution isn't needed
- Cache processed images to avoid repeated encoding

### Error Handling
```python
try:
    with open(image_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
except FileNotFoundError:
    print(f"Image not found: {image_path}")
except Exception as e:
    print(f"Error loading image: {e}")
```

## Contributing

Have a useful multimodal pattern? Submit a PR with:
1. Working example code
2. Clear documentation
3. Real-world use case explanation

## Questions?

- **Framework Docs**: See `CLAUDE.md` → Multimodal Support section
- **API Reference**: Check `arshai/core/interfaces/illm.py` for `ILLMInput`
- **Issues**: Report at https://github.com/yourusername/arshai/issues
