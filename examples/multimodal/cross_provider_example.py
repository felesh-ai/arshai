"""
Example: Cross-Provider Vision - Same code works across all LLM providers.

This example demonstrates that the same image analysis code works
seamlessly across Gemini, OpenAI, Azure, OpenRouter, and custom gateways.
"""

import asyncio
import base64
import os

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.google_genai import GeminiClient
from arshai.llms.openai import OpenAIClient
from arshai.llms.azure import AzureClient
from arshai.llms.openrouter import OpenRouterClient
from arshai.llms.ai_gateway import AIGatewayLLM


# Sample image (1x1 red pixel for demonstration)
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
SAMPLE_IMAGE = base64.b64encode(PNG_BYTES).decode("utf-8")


async def analyze_with_provider(provider_name: str, llm_client, include_image: bool = True):
    """
    Analyze an image with any provider using the unified interface.

    Args:
        provider_name: Name of the provider (for display)
        llm_client: Any ILLM-compatible client
        include_image: Whether to include image in analysis

    Returns:
        Analysis response text
    """
    print(f"\n{'=' * 70}")
    print(f"🔍 Analyzing with {provider_name}")
    print(f"{'=' * 70}")

    # Create input - same format for all providers
    input_data = ILLMInput(
        system_prompt="You are a vision assistant. Analyze images concisely.",
        user_message="Describe what you see in this image. Be brief but accurate.",
        images_base64=[SAMPLE_IMAGE] if include_image else []
    )

    try:
        response = await llm_client.chat(input_data)

        print(f"\n✅ {provider_name} Response:")
        print(f"   {response['llm_response'][:200]}...")

        if "usage" in response:
            usage = response["usage"]
            print(f"\n📊 Token Usage:")
            print(f"   - Input: {usage.get('input_tokens', 0)}")
            print(f"   - Output: {usage.get('output_tokens', 0)}")

        return response["llm_response"]

    except Exception as e:
        print(f"\n❌ {provider_name} Error: {e}")
        return None


async def main():
    """Demonstrate cross-provider compatibility."""
    print("=" * 70)
    print("Cross-Provider Vision Example")
    print("Demonstrating unified image analysis across all providers")
    print("=" * 70)

    # Track which providers are configured
    providers = []

    # 1. Gemini
    if os.getenv("GOOGLE_API_KEY"):
        gemini_config = ILLMConfig(
            model="gemini-1.5-flash",
            temperature=0.3
        )
        gemini_client = GeminiClient(gemini_config)
        providers.append(("Gemini", gemini_client))
    else:
        print("\n⚠️  Gemini skipped: GOOGLE_API_KEY not set")

    # 2. OpenAI
    if os.getenv("OPENAI_API_KEY"):
        openai_config = ILLMConfig(
            model="gpt-4o",  # Vision-capable model
            temperature=0.3
        )
        openai_client = OpenAIClient(openai_config)
        providers.append(("OpenAI", openai_client))
    else:
        print("⚠️  OpenAI skipped: OPENAI_API_KEY not set")

    # 3. Azure OpenAI
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        azure_config = ILLMConfig(
            model="gpt-4o",  # Your Azure deployment name
            temperature=0.3
        )
        azure_client = AzureClient(azure_config)
        providers.append(("Azure OpenAI", azure_client))
    else:
        print("⚠️  Azure skipped: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")

    # 4. OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        openrouter_config = ILLMConfig(
            model="openai/gpt-4o-mini",  # Vision-capable model
            temperature=0.3
        )
        openrouter_client = OpenRouterClient(openrouter_config)
        providers.append(("OpenRouter", openrouter_client))
    else:
        print("⚠️  OpenRouter skipped: OPENROUTER_API_KEY not set")

    # 5. Custom Gateway (if configured)
    if os.getenv("GATEWAY_BASE_URL") and os.getenv("GATEWAY_TOKEN"):
        from arshai.llms.ai_gateway import AIGatewayConfig

        gateway_config = AIGatewayConfig(
            base_url=os.getenv("GATEWAY_BASE_URL"),
            gateway_token=os.getenv("GATEWAY_TOKEN"),
            model=os.getenv("GATEWAY_MODEL", "gpt-4o-mini"),
            temperature=0.3
        )
        gateway_client = AIGatewayLLM(gateway_config)
        providers.append(("Custom Gateway", gateway_client))
    else:
        print("⚠️  Custom Gateway skipped: GATEWAY_BASE_URL or GATEWAY_TOKEN not set")

    # Check if any providers are available
    if not providers:
        print("\n" + "=" * 70)
        print("❌ No providers configured!")
        print("=" * 70)
        print("\nPlease set at least one provider's API key:")
        print("  - GOOGLE_API_KEY for Gemini")
        print("  - OPENAI_API_KEY for OpenAI")
        print("  - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT for Azure")
        print("  - OPENROUTER_API_KEY for OpenRouter")
        print("  - GATEWAY_BASE_URL + GATEWAY_TOKEN for Custom Gateway")
        return

    # Analyze with each configured provider
    print(f"\n✅ Found {len(providers)} configured provider(s)\n")

    responses = {}
    for provider_name, client in providers:
        response = await analyze_with_provider(provider_name, client)
        if response:
            responses[provider_name] = response

        # Small delay between providers to avoid rate limits
        await asyncio.sleep(1)

    # Summary
    print("\n" + "=" * 70)
    print("📊 Cross-Provider Analysis Summary")
    print("=" * 70)

    if len(responses) > 1:
        print("\n✅ Successfully analyzed image with multiple providers!")
        print(f"   Providers tested: {', '.join(responses.keys())}")
        print("\n🔑 Key Insight:")
        print("   The SAME code works across all providers.")
        print("   Only the client initialization differs.")
        print("   The ILLMInput interface is universal.")
    elif len(responses) == 1:
        print(f"\n✅ Successfully analyzed image with {list(responses.keys())[0]}")
        print("   Configure additional providers to test cross-compatibility")
    else:
        print("\n❌ No successful analyses")
        print("   Check your API keys and try again")

    print("\n" + "=" * 70)
    print("✅ Cross-Provider Example Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
