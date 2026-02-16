"""
Basic Image Analysis Example

Demonstrates single image analysis using Gemini with the Arshai framework.
This example shows the simplest way to analyze an image using vision-capable models.

Requirements:
- Set GOOGLE_API_KEY environment variable
- Have an image file to analyze
"""

import base64
import asyncio
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.google_genai import GeminiClient


async def analyze_single_image():
    """Analyze a single image with Gemini"""

    # Step 1: Load and encode your image
    # (Use any method you prefer - this is the simplest approach)
    image_path = "your_image.jpg"  # Replace with your image path

    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Step 2: Configure the LLM client
    config = ILLMConfig(
        model="gemini-2.0-flash-exp",  # Use a vision-capable model
        temperature=0.7
    )

    client = GeminiClient(config)

    # Step 3: Create input with image
    input_data = ILLMInput(
        system_prompt="You are a helpful assistant that analyzes images.",
        user_message="Describe this image in detail. What do you see?",
        images_base64=[img_base64]  # Add your base64-encoded image
    )

    # Step 4: Get response
    response = await client.chat(input_data)

    print("Image Analysis:")
    print("-" * 50)
    print(response["llm_response"])
    print("-" * 50)
    print(f"\nUsage: {response['usage']}")


async def analyze_with_specific_question():
    """Analyze an image with a specific question"""

    # Load image
    with open("your_image.jpg", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Configure client
    config = ILLMConfig(model="gemini-2.0-flash-exp")
    client = GeminiClient(config)

    # Ask a specific question about the image
    input_data = ILLMInput(
        system_prompt="You are an expert at analyzing images and answering questions about them.",
        user_message="Is there any text visible in this image? If so, what does it say?",
        images_base64=[img_base64]
    )

    response = await client.chat(input_data)

    print("Question: Is there text in the image?")
    print("Answer:", response["llm_response"])


async def main():
    """Run examples"""
    print("Example 1: General Image Analysis")
    print("=" * 50)
    await analyze_single_image()

    print("\n\n")

    print("Example 2: Specific Question About Image")
    print("=" * 50)
    await analyze_with_specific_question()


if __name__ == "__main__":
    # Make sure to set GOOGLE_API_KEY environment variable
    # export GOOGLE_API_KEY="your-api-key-here"

    asyncio.run(main())
