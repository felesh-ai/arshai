"""
Example: Vision + Tool Calling - Analyzing images with function calling.

This example demonstrates combining image analysis with tool calling,
allowing the LLM to both see images and execute functions to perform
structured analysis or take actions based on what it sees.
"""

import asyncio
import base64
import os
from pathlib import Path

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.google_genai import GeminiClient


def save_analysis_result(description: str, object_count: int, primary_color: str) -> str:
    """
    Save image analysis results to a structured format.

    Args:
        description: Text description of the image
        object_count: Number of objects detected
        primary_color: Primary color identified

    Returns:
        Confirmation message
    """
    result = {
        "description": description,
        "object_count": object_count,
        "primary_color": primary_color
    }
    print(f"\n📊 Analysis Result Saved:")
    print(f"   - Description: {result['description']}")
    print(f"   - Objects: {result['object_count']}")
    print(f"   - Primary Color: {result['primary_color']}")

    return f"Analysis saved successfully with {object_count} objects detected"


def classify_image_type(category: str, confidence: str) -> str:
    """
    Classify the image into a category.

    Args:
        category: The category (e.g., 'photo', 'illustration', 'diagram', 'screenshot')
        confidence: Confidence level ('high', 'medium', 'low')

    Returns:
        Confirmation message
    """
    print(f"\n🏷️  Image Classification:")
    print(f"   - Category: {category}")
    print(f"   - Confidence: {confidence}")

    return f"Image classified as '{category}' with {confidence} confidence"


async def main():
    """Demonstrate vision + tool calling."""
    print("=" * 70)
    print("Vision + Tool Calling Example")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ Error: GOOGLE_API_KEY environment variable not set")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        return

    # Initialize Gemini client with vision model
    config = ILLMConfig(
        model="gemini-1.5-flash",  # Vision-capable model
        temperature=0.3
    )
    llm = GeminiClient(config)

    # Sample image (1x1 red pixel for demonstration)
    # In production, load your own images
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    sample_image = base64.b64encode(png_bytes).decode("utf-8")

    print("\n📸 Analyzing image with tool calling...")

    # Prepare input with image and tools
    input_data = ILLMInput(
        system_prompt=(
            "You are an expert image analyst with access to analysis tools. "
            "When analyzing images:\n"
            "1. First describe what you see in detail\n"
            "2. Count any distinct objects or elements\n"
            "3. Identify the primary color\n"
            "4. Use the save_analysis_result tool to save your findings\n"
            "5. Use the classify_image_type tool to categorize the image\n"
            "Always use the tools to structure your analysis."
        ),
        user_message=(
            "Please analyze this image thoroughly. "
            "Describe what you see, count the objects, identify colors, "
            "and use the tools to save and classify your findings."
        ),
        images_base64=[sample_image],
        regular_functions={
            "save_analysis_result": save_analysis_result,
            "classify_image_type": classify_image_type
        },
        max_turns=10  # Allow multiple tool calls
    )

    # Process with vision + tools
    response = await llm.chat(input_data)

    print("\n" + "=" * 70)
    print("🤖 Final Response:")
    print("=" * 70)
    print(response["llm_response"])

    # Show usage stats
    if "usage" in response:
        usage = response["usage"]
        print(f"\n📊 Usage Statistics:")
        print(f"   - Input tokens: {usage.get('input_tokens', 0)}")
        print(f"   - Output tokens: {usage.get('output_tokens', 0)}")
        print(f"   - Total tokens: {usage.get('total_tokens', 0)}")

    print("\n" + "=" * 70)
    print("✅ Vision + Tool Calling Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
