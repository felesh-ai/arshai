"""
Example: Streaming with Images - Progressive responses for vision tasks.

This example demonstrates streaming responses when analyzing images,
allowing for real-time feedback as the LLM generates its analysis.
"""

import asyncio
import base64
import os
import sys

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.google_genai import GeminiClient


async def main():
    """Demonstrate streaming responses with image analysis."""
    print("=" * 70)
    print("Streaming with Images Example")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ Error: GOOGLE_API_KEY environment variable not set")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        return

    # Initialize Gemini client
    config = ILLMConfig(
        model="gemini-1.5-flash",  # Vision-capable model
        temperature=0.7
    )
    llm = GeminiClient(config)

    # Sample image (1x1 red pixel for demonstration)
    # In production, load your own images
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    sample_image = base64.b64encode(png_bytes).decode("utf-8")

    print("\n📸 Analyzing image with streaming response...")
    print("\n🤖 LLM Response (streaming):")
    print("-" * 70)

    # Prepare input with image
    input_data = ILLMInput(
        system_prompt=(
            "You are an expert image analyst. "
            "Provide detailed, structured analysis of images including:\n"
            "- Visual description\n"
            "- Color analysis\n"
            "- Composition and layout\n"
            "- Notable features or patterns\n"
            "- Suggested use cases or context\n"
            "Be thorough and descriptive."
        ),
        user_message=(
            "Please provide a comprehensive analysis of this image. "
            "Include all visual details, colors, composition, and any "
            "interesting observations you can make."
        ),
        images_base64=[sample_image]
    )

    # Stream the response
    full_response = ""
    chunk_count = 0
    final_usage = None

    try:
        async for chunk in llm.stream(input_data):
            if "llm_response" in chunk and chunk["llm_response"]:
                # Print new content as it arrives
                sys.stdout.write(chunk["llm_response"])
                sys.stdout.flush()
                full_response = chunk["llm_response"]
                chunk_count += 1

            # Capture final usage information
            if "usage" in chunk:
                final_usage = chunk["usage"]

    except Exception as e:
        print(f"\n\n❌ Error during streaming: {e}")
        return

    print("\n" + "-" * 70)

    # Show streaming statistics
    print(f"\n📊 Streaming Statistics:")
    print(f"   - Chunks received: {chunk_count}")
    print(f"   - Total characters: {len(full_response)}")

    if final_usage:
        print(f"\n📊 Usage Statistics:")
        print(f"   - Input tokens: {final_usage.get('input_tokens', 0)}")
        print(f"   - Output tokens: {final_usage.get('output_tokens', 0)}")
        print(f"   - Total tokens: {final_usage.get('total_tokens', 0)}")

    print("\n" + "=" * 70)
    print("✅ Streaming Analysis Complete")
    print("=" * 70)

    # Additional example: Stream with multiple images
    print("\n\n" + "=" * 70)
    print("Streaming with Multiple Images")
    print("=" * 70)

    # Create second image (same for demo)
    sample_image_2 = sample_image

    print("\n📸📸 Analyzing multiple images with streaming...")
    print("\n🤖 LLM Response (streaming):")
    print("-" * 70)

    input_data_multi = ILLMInput(
        system_prompt="You are an expert image analyst specializing in comparative analysis.",
        user_message=(
            "I'm providing you with two images. "
            "Compare and contrast them. Describe similarities and differences."
        ),
        images_base64=[sample_image, sample_image_2]
    )

    full_response_2 = ""

    try:
        async for chunk in llm.stream(input_data_multi):
            if "llm_response" in chunk and chunk["llm_response"]:
                sys.stdout.write(chunk["llm_response"])
                sys.stdout.flush()
                full_response_2 = chunk["llm_response"]

    except Exception as e:
        print(f"\n\n❌ Error during streaming: {e}")
        return

    print("\n" + "-" * 70)
    print("\n✅ Multi-image streaming complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
