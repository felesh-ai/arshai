"""
Multiple Images Comparison Example

Demonstrates comparing multiple images in a single request using vision-capable models.
Shows how to analyze relationships, differences, and similarities across images.

Requirements:
- Set GOOGLE_API_KEY environment variable
- Have multiple image files to compare
"""

import base64
import asyncio
from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.google_genai import GeminiClient


async def compare_two_images():
    """Compare two images and identify differences"""

    # Load multiple images
    image_paths = ["image1.jpg", "image2.jpg"]  # Replace with your images
    images_base64 = []

    for path in image_paths:
        with open(path, "rb") as f:
            images_base64.append(base64.b64encode(f.read()).decode('utf-8'))

    # Configure client
    config = ILLMConfig(
        model="gemini-2.0-flash-exp",
        temperature=0.5
    )
    client = GeminiClient(config)

    # Create input with multiple images
    input_data = ILLMInput(
        system_prompt="You are an expert at comparing images and identifying differences.",
        user_message=(
            "I've provided two images. Please analyze them and:\n"
            "1. Describe what each image shows\n"
            "2. List the key differences between them\n"
            "3. Identify any similarities"
        ),
        images_base64=images_base64  # Multiple images
    )

    response = await client.chat(input_data)

    print("Comparison Analysis:")
    print("=" * 60)
    print(response["llm_response"])
    print("=" * 60)
    print(f"\nUsage: {response['usage']}")


async def analyze_image_sequence():
    """Analyze a sequence of images (e.g., time-lapse, before/after)"""

    # Load sequence of images
    image_paths = ["before.jpg", "during.jpg", "after.jpg"]
    images_base64 = []

    for path in image_paths:
        with open(path, "rb") as f:
            images_base64.append(base64.b64encode(f.read()).decode('utf-8'))

    config = ILLMConfig(model="gemini-2.0-flash-exp")
    client = GeminiClient(config)

    input_data = ILLMInput(
        system_prompt="You analyze sequences of images to understand progression and changes over time.",
        user_message=(
            "I've provided a sequence of 3 images showing a progression. "
            "Please describe the changes you observe from the first to the last image. "
            "What story do these images tell?"
        ),
        images_base64=images_base64
    )

    response = await client.chat(input_data)

    print("\nSequence Analysis:")
    print("=" * 60)
    print(response["llm_response"])
    print("=" * 60)


async def find_common_elements():
    """Find common elements across multiple images"""

    # Load images
    image_paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg", "photo4.jpg"]
    images_base64 = [
        base64.b64encode(open(path, "rb").read()).decode('utf-8')
        for path in image_paths
    ]

    config = ILLMConfig(model="gemini-2.0-flash-exp", temperature=0.3)
    client = GeminiClient(config)

    input_data = ILLMInput(
        system_prompt="You identify patterns and common elements across multiple images.",
        user_message=(
            "I've provided 4 different images. Please identify:\n"
            "1. Common objects or elements present in all or most images\n"
            "2. Common themes or patterns\n"
            "3. Any notable outliers"
        ),
        images_base64=images_base64
    )

    response = await client.chat(input_data)

    print("\nCommon Elements Analysis:")
    print("=" * 60)
    print(response["llm_response"])
    print("=" * 60)


async def main():
    """Run all examples"""

    print("Example 1: Compare Two Images")
    print("=" * 60)
    await compare_two_images()

    print("\n\n")

    print("Example 2: Analyze Image Sequence")
    print("=" * 60)
    await analyze_image_sequence()

    print("\n\n")

    print("Example 3: Find Common Elements")
    print("=" * 60)
    await find_common_elements()


if __name__ == "__main__":
    # Set your API key: export GOOGLE_API_KEY="your-api-key-here"
    asyncio.run(main())
