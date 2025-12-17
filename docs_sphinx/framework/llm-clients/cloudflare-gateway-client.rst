Cloudflare AI Gateway Client
=============================

The Cloudflare AI Gateway client provides standardized access to multiple language model providers through Cloudflare's unified AI Gateway. Using the BYOK (Bring Your Own Key) mode, provider API keys are securely stored in the Cloudflare gateway, and you only need the gateway token to authenticate.

.. note::
   **BYOK Mode (Bring Your Own Key)**: This client operates exclusively in BYOK mode where provider API keys are stored securely in your Cloudflare AI Gateway configuration. Only the gateway token is needed in your application code.

Benefits
--------

**Centralized Key Management**
   Provider API keys are stored in Cloudflare AI Gateway, not in your code or environment variables.

**Multi-Provider Access**
   Access OpenAI, Anthropic, Google, OpenRouter, and many other providers through a single gateway.

**Unified Observability**
   Centralized caching, rate limiting, logging, and analytics across all providers.

**Easy Provider Switching**
   Change providers by modifying ``provider`` and ``model`` - no code changes required.

**Cost Optimization**
   Monitor and control costs across multiple providers from one dashboard.

Configuration
-------------

Basic Setup:

.. code-block:: python

   from arshai.llms import CloudflareGatewayLLM, CloudflareGatewayLLMConfig

   # Configure the client (BYOK mode)
   config = CloudflareGatewayLLMConfig(
       account_id="your-cloudflare-account-id",
       gateway_id="your-gateway-id",
       gateway_token="your-gateway-token",  # Or set CLOUDFLARE_GATEWAY_TOKEN env var
       provider="openrouter",               # Provider name
       model="openai/gpt-4o-mini",          # Model name
       temperature=0.7,
       max_tokens=500,
   )

   # Create client
   client = CloudflareGatewayLLM(config)

Environment Variables:

.. code-block:: bash

   # Gateway token (required if not passed in config)
   export CLOUDFLARE_GATEWAY_TOKEN="your-gateway-token"

.. important::
   **Gateway Token**: The gateway token authenticates with Cloudflare AI Gateway. Cloudflare then uses the provider API keys stored in your gateway configuration to call the actual providers.

Setting Up Your Gateway
-----------------------

Before using this client, configure your Cloudflare AI Gateway:

1. **Create an AI Gateway** in your Cloudflare dashboard
2. **Store Provider API Keys** in the gateway's Provider Keys section (BYOK)
3. **Generate a Gateway Token** for authentication
4. **Enable Authenticated Gateway** (recommended) for security

Supported Providers
-------------------

The unified ``/compat`` endpoint supports these providers:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Provider
     - Example Models
   * - ``openai``
     - gpt-4o, gpt-4o-mini, gpt-4-turbo
   * - ``anthropic``
     - claude-sonnet-4-5, claude-opus-4, claude-3-haiku
   * - ``google-ai-studio``
     - gemini-2.0-flash, gemini-1.5-pro
   * - ``openrouter``
     - openai/gpt-4o-mini, anthropic/claude-3.5-sonnet
   * - ``groq``
     - llama-3-70b, mixtral-8x7b
   * - ``mistral``
     - mistral-large, mistral-medium
   * - ``cohere``
     - command-r-plus, command-r
   * - ``xai``
     - grok-2
   * - ``deepseek``
     - deepseek-chat, deepseek-coder
   * - ``perplexity``
     - sonar-medium-online

.. note::
   **Model Format**: The unified endpoint uses ``{provider}/{model}`` format internally. For example, ``openrouter`` provider with ``openai/gpt-4o-mini`` model becomes ``openrouter/openai/gpt-4o-mini``.

Basic Usage
-----------

Simple conversation:

.. code-block:: python

   from arshai.core.interfaces.illm import ILLMInput

   input_data = ILLMInput(
       system_prompt="You are a helpful AI assistant.",
       user_message="What is the capital of France?"
   )

   response = await client.chat(input_data)
   print(response["llm_response"])  # "The capital of France is Paris."
   print(response["usage"])  # Token usage information

Streaming responses:

.. code-block:: python

   async for chunk in client.stream(input_data):
       if chunk.get("llm_response"):
           print(chunk["llm_response"], end="", flush=True)
       if chunk.get("usage"):
           print(f"\nTotal tokens: {chunk['usage']['total_tokens']}")

Switching Providers
-------------------

Switch between providers without changing your application code:

.. code-block:: python

   # OpenAI through OpenRouter
   config_openai = CloudflareGatewayLLMConfig(
       account_id="xxx", gateway_id="my-gw", gateway_token="xxx",
       provider="openrouter",
       model="openai/gpt-4o-mini",
   )

   # Anthropic direct
   config_anthropic = CloudflareGatewayLLMConfig(
       account_id="xxx", gateway_id="my-gw", gateway_token="xxx",
       provider="anthropic",
       model="claude-sonnet-4-5",
   )

   # Google AI Studio
   config_google = CloudflareGatewayLLMConfig(
       account_id="xxx", gateway_id="my-gw", gateway_token="xxx",
       provider="google-ai-studio",
       model="gemini-2.0-flash",
   )

   # Same code works with any provider
   client = CloudflareGatewayLLM(config_anthropic)
   response = await client.chat(input_data)

Function Calling
----------------

Regular functions that return results to the conversation:

.. code-block:: python

   def get_weather(city: str) -> str:
       """Get current weather for a city."""
       return f"Weather in {city}: Sunny, 22C"

   def calculate(expression: str) -> float:
       """Calculate a mathematical expression."""
       return eval(expression)  # Use safe evaluation in production

   input_data = ILLMInput(
       system_prompt="You are a helpful assistant with access to tools.",
       user_message="What's the weather in Tokyo and what is 15 * 7?",
       regular_functions={
           "get_weather": get_weather,
           "calculate": calculate
       },
       max_turns=5
   )

   response = await client.chat(input_data)

Background Tasks
----------------

Fire-and-forget tasks that run independently:

.. code-block:: python

   def log_interaction(user_id: str, action: str) -> None:
       """BACKGROUND TASK: Log user interaction for analytics."""
       print(f"Logged: {user_id} performed {action}")

   def send_notification(event: str, details: str = "") -> None:
       """BACKGROUND TASK: Send notification to admin."""
       print(f"Notification: {event} - {details}")

   input_data = ILLMInput(
       system_prompt="You are a helpful assistant. Log important interactions.",
       user_message="Hello, I need help with Python!",
       background_tasks={
           "log_interaction": log_interaction,
           "send_notification": send_notification
       }
   )

   response = await client.chat(input_data)
   # Background tasks execute automatically but don't affect the response

Structured Output
-----------------

Get type-safe responses using Pydantic models:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List

   class SentimentAnalysis(BaseModel):
       """Structured sentiment analysis result."""
       sentiment: str = Field(description="positive, negative, or neutral")
       confidence: float = Field(description="Confidence score 0.0 to 1.0")
       key_phrases: List[str] = Field(description="Key phrases that indicate sentiment")

   config = CloudflareGatewayLLMConfig(
       account_id="xxx", gateway_id="my-gw", gateway_token="xxx",
       provider="openrouter",
       model="openai/gpt-4o-mini",
       temperature=0.3  # Lower temperature for structured output
   )
   client = CloudflareGatewayLLM(config)

   input_data = ILLMInput(
       system_prompt="Analyze the sentiment of user messages.",
       user_message="I absolutely love this product! Best purchase ever!",
       structure_type=SentimentAnalysis
   )

   response = await client.chat(input_data)
   analysis = response["llm_response"]  # SentimentAnalysis instance

   print(f"Sentiment: {analysis.sentiment}")
   print(f"Confidence: {analysis.confidence}")
   print(f"Key phrases: {', '.join(analysis.key_phrases)}")

Usage Tracking
--------------

All calls return standardized usage information:

.. code-block:: python

   response = await client.chat(input_data)
   usage = response["usage"]

   print(f"Input tokens: {usage['input_tokens']}")
   print(f"Output tokens: {usage['output_tokens']}")
   print(f"Total tokens: {usage['total_tokens']}")
   print(f"Provider: {usage['provider']}")
   print(f"Model: {usage['model']}")
   print(f"Request ID: {usage['request_id']}")

   # For reasoning models (when available)
   if usage.get('thinking_tokens'):
       print(f"Thinking tokens: {usage['thinking_tokens']}")

Error Handling
--------------

.. code-block:: python

   async def chat_with_retry(client, input_data, max_retries=3):
       """Chat with exponential backoff retry."""
       import asyncio

       for attempt in range(max_retries):
           try:
               return await client.chat(input_data)
           except Exception as e:
               error_str = str(e).lower()
               if "429" in error_str or "rate" in error_str:
                   wait_time = 2 ** attempt
                   print(f"Rate limited, waiting {wait_time}s...")
                   await asyncio.sleep(wait_time)
                   continue
               elif "401" in error_str:
                   print("Authentication failed - check your gateway token")
                   raise
               elif "402" in error_str:
                   print("Insufficient credits - check provider balance")
                   raise
               else:
                   raise

   # Configuration validation
   try:
       client = CloudflareGatewayLLM(config)
   except ValueError as e:
       if "gateway_token" in str(e).lower():
           print("Set CLOUDFLARE_GATEWAY_TOKEN environment variable")
       else:
           print(f"Configuration error: {e}")

Comparison with Direct Provider Access
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Feature
     - Direct Provider
     - Cloudflare Gateway
   * - API Key Management
     - Multiple env vars
     - Single gateway token
   * - Provider Switching
     - Code changes
     - Config change only
   * - Caching
     - DIY
     - Built-in
   * - Rate Limiting
     - Per-provider
     - Unified
   * - Analytics
     - Provider dashboards
     - Unified dashboard
   * - Latency
     - Direct
     - +1 hop (minimal)
   * - Cost Monitoring
     - Multiple sources
     - Unified billing

Best Practices
--------------

1. **Store Provider Keys Securely**: Configure provider API keys in Cloudflare dashboard, not in code.

2. **Use Environment Variables**: Store gateway token in ``CLOUDFLARE_GATEWAY_TOKEN`` for production.

3. **Enable Authenticated Gateway**: Require gateway token for all requests to prevent unauthorized access.

4. **Monitor in Dashboard**: Use Cloudflare AI Gateway dashboard for real-time monitoring.

5. **Configure Caching**: Enable caching in gateway settings to reduce costs and latency.

6. **Set Rate Limits**: Configure rate limits per provider to prevent cost overruns.

Next Steps
----------

- :doc:`extending-llm-clients` - Creating custom LLM clients
- :doc:`../agents/index` - Building agents with LLM clients
- Visit https://developers.cloudflare.com/ai-gateway/ for gateway documentation
