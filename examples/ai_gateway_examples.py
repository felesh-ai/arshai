"""
AI Gateway Examples

This file demonstrates how to use the generic OpenAI-compatible gateway client
with various gateway providers including Cloudflare, LiteLLM, and custom gateways.

It also shows the migration path from the deprecated CloudflareGatewayLLM.
"""

import os
import asyncio
from arshai.llms.ai_gateway import (
    AIGatewayLLM,
    AIGatewayConfig,
)
from arshai.core.interfaces.illm import ILLMInput


# =============================================================================
# EXAMPLE 1: Cloudflare AI Gateway (New Recommended Way)
# =============================================================================

async def cloudflare_gateway_example():
    """
    Using Cloudflare AI Gateway with the generic OpenAI-compatible client.

    This is the recommended way to use Cloudflare AI Gateway.
    """
    print("\n=== Cloudflare AI Gateway Example ===")

    config = AIGatewayConfig(
        # Build the full Cloudflare gateway URL
        base_url=f"https://gateway.ai.cloudflare.com/v1/{os.getenv('CLOUDFLARE_ACCOUNT_ID')}/{os.getenv('CLOUDFLARE_GATEWAY_ID')}/compat",
        api_key=os.getenv("CLOUDFLARE_GATEWAY_TOKEN"),
        model="anthropic/claude-sonnet-4-5",  # Format: {provider}/{model}
        headers={
            "HTTP-Referer": "https://myapp.com",
            "X-Title": "My Application"
        }
    )

    llm = AIGatewayLLM(config)

    response = await llm.chat(ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="What is Cloudflare AI Gateway?"
    ))

    print(f"Response: {response['llm_response']}")
    print(f"Usage: {response['usage']}")


# =============================================================================
# EXAMPLE 2: Cloudflare Gateway with Regional Endpoint
# =============================================================================

async def cloudflare_regional_example():
    """Using Cloudflare AI Gateway with a regional endpoint (China)."""
    print("\n=== Cloudflare Regional Gateway Example ===")

    config = AIGatewayConfig(
        # Use China regional endpoint
        base_url=f"https://gateway.ai.cloudflare.cn/v1/{os.getenv('CLOUDFLARE_ACCOUNT_ID')}/{os.getenv('CLOUDFLARE_GATEWAY_ID')}/compat",
        api_key=os.getenv("CLOUDFLARE_GATEWAY_TOKEN"),
        model="openai/gpt-4o",
    )

    llm = AIGatewayLLM(config)

    response = await llm.chat(ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="Hello from China!"
    ))

    print(f"Response: {response['llm_response']}")


# =============================================================================
# EXAMPLE 3: LiteLLM Proxy
# =============================================================================

async def litellm_proxy_example():
    """
    Using LiteLLM proxy with the generic client.

    LiteLLM is a popular proxy that provides a unified interface to 100+ LLMs.
    """
    print("\n=== LiteLLM Proxy Example ===")

    config = AIGatewayConfig(
        base_url="http://localhost:4000",  # Default LiteLLM proxy URL
        api_key=os.getenv("LITELLM_API_KEY", "sk-1234"),
        model="gpt-4o",  # Model name as configured in LiteLLM
    )

    llm = AIGatewayLLM(config)

    response = await llm.chat(ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="Hello from LiteLLM!"
    ))

    print(f"Response: {response['llm_response']}")


# =============================================================================
# EXAMPLE 4: Custom Enterprise Gateway
# =============================================================================

async def custom_enterprise_gateway_example():
    """Using a custom enterprise gateway with authentication headers."""
    print("\n=== Custom Enterprise Gateway Example ===")

    config = AIGatewayConfig(
        base_url="https://api.mycompany.com/v1",
        api_key=os.getenv("ENTERPRISE_API_KEY"),
        model="my-fine-tuned-model",
        headers={
            "X-Organization-ID": "org-123",
            "X-Department": "engineering",
            "X-Cost-Center": "ai-research"
        }
    )

    llm = AIGatewayLLM(config)

    response = await llm.chat(ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="What is the company policy?"
    ))

    print(f"Response: {response['llm_response']}")


# =============================================================================
# EXAMPLE 5: Azure OpenAI (using custom endpoint)
# =============================================================================

async def azure_openai_gateway_example():
    """Using Azure OpenAI deployment with the generic client."""
    print("\n=== Azure OpenAI Gateway Example ===")

    config = AIGatewayConfig(
        base_url=f"https://{os.getenv('AZURE_RESOURCE_NAME')}.openai.azure.com/openai/deployments",
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        model="gpt-4-deployment",  # Your deployment name
        headers={
            "api-version": "2024-02-01"
        }
    )

    llm = AIGatewayLLM(config)

    response = await llm.chat(ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="Hello from Azure!"
    ))

    print(f"Response: {response['llm_response']}")


# =============================================================================
# EXAMPLE 6: Environment Variable Configuration
# =============================================================================

async def environment_config_example():
    """
    Using environment variables for configuration.

    Set these environment variables:
    - GATEWAY_BASE_URL: Your gateway base URL
    - GATEWAY_API_KEY: Your API key/token
    """
    print("\n=== Environment Variable Configuration Example ===")

    # These will automatically fall back to environment variables
    config = AIGatewayConfig(
        model="gpt-4o",
        # base_url falls back to GATEWAY_BASE_URL
        # api_key falls back to GATEWAY_API_KEY
    )

    llm = AIGatewayLLM(config)

    response = await llm.chat(ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="Configuration from environment!"
    ))

    print(f"Response: {response['llm_response']}")


# =============================================================================
# EXAMPLE 7: Function Calling with Gateway
# =============================================================================

def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 72°F"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

async def function_calling_example():
    """Using function calling with the gateway client."""
    print("\n=== Function Calling Example ===")

    config = AIGatewayConfig(
        base_url=os.getenv("GATEWAY_BASE_URL", "http://localhost:4000"),
        api_key=os.getenv("GATEWAY_API_KEY", "test-key"),
        model="gpt-4o",
    )

    llm = AIGatewayLLM(config)

    response = await llm.chat(ILLMInput(
        system_prompt="You are a helpful assistant with access to tools.",
        user_message="What's the weather in San Francisco?",
        regular_functions={
            "get_weather": get_weather,
            "search_web": search_web
        }
    ))

    print(f"Response: {response['llm_response']}")


# =============================================================================
# EXAMPLE 8: Streaming with Gateway
# =============================================================================

async def streaming_example():
    """Using streaming responses with the gateway client."""
    print("\n=== Streaming Example ===")

    config = AIGatewayConfig(
        base_url=os.getenv("GATEWAY_BASE_URL", "http://localhost:4000"),
        api_key=os.getenv("GATEWAY_API_KEY", "test-key"),
        model="gpt-4o",
    )

    llm = AIGatewayLLM(config)

    print("Streaming response: ", end="", flush=True)

    async for chunk in llm.stream(ILLMInput(
        system_prompt="You are a helpful assistant.",
        user_message="Tell me a short story about AI."
    )):
        if chunk.get('llm_response'):
            print(chunk['llm_response'], end="", flush=True)

    print("\n")


# =============================================================================
# MIGRATION EXAMPLE: Old vs New
# =============================================================================

async def migration_comparison():
    """
    Comparison showing the deprecated Cloudflare client vs the new generic client.

    Note: The old way will show deprecation warnings but still works.
    """
    print("\n=== Migration Comparison ===")

    print("\n--- OLD WAY (Deprecated) ---")
    try:
        from arshai.llms.cloudflare_gateway import (
            CloudflareGatewayLLM,
            CloudflareGatewayLLMConfig
        )

        # This will trigger deprecation warnings
        old_config = CloudflareGatewayLLMConfig(
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID", "test-account"),
            gateway_id=os.getenv("CLOUDFLARE_GATEWAY_ID", "test-gateway"),
            gateway_token=os.getenv("CLOUDFLARE_GATEWAY_TOKEN", "test-token"),
            provider="anthropic",
            model="claude-sonnet-4-5",
        )

        print(f"Old config base_url: {old_config.compat_base_url}")
        print(f"Old config model: {old_config.full_model_name}")

    except Exception as e:
        print(f"Old way error (expected if env vars not set): {e}")

    print("\n--- NEW WAY (Recommended) ---")
    new_config = AIGatewayConfig(
        base_url=f"https://gateway.ai.cloudflare.com/v1/{os.getenv('CLOUDFLARE_ACCOUNT_ID', 'test-account')}/{os.getenv('CLOUDFLARE_GATEWAY_ID', 'test-gateway')}/compat",
        api_key=os.getenv("CLOUDFLARE_GATEWAY_TOKEN", "test-token"),
        model="anthropic/claude-sonnet-4-5",
    )

    print(f"New config base_url: {new_config.base_url}")
    print(f"New config model: {new_config.model}")

    print("\n✅ Same functionality, more flexible, no deprecation warnings!")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all examples."""
    print("=" * 80)
    print("AI Gateway Examples")
    print("=" * 80)

    # Run migration comparison first (doesn't require real API calls)
    await migration_comparison()

    # Uncomment to run live examples (requires real gateway configuration)
    # await cloudflare_gateway_example()
    # await cloudflare_regional_example()
    # await litellm_proxy_example()
    # await custom_enterprise_gateway_example()
    # await azure_openai_gateway_example()
    # await environment_config_example()
    # await function_calling_example()
    # await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())
