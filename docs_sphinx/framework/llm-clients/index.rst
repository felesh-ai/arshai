LLM Clients (Layer 1)
====================

Layer 1 provides standardized access to different language model providers through a unified interface. This is the foundation of the Arshai framework - giving you consistent, reliable access to LLMs without vendor lock-in.

.. toctree::
   :maxdepth: 2
   :caption: LLM Client Implementation

   interface-overview
   openai-client
   azure-client
   google-gemini-client
   openrouter-client
   cloudflare-gateway-client
   extending-llm-clients

Core Philosophy
---------------

LLM clients in Arshai follow these principles:

**Standardized Interface**
   All LLM clients implement the ILLM interface, providing consistent methods regardless of the underlying provider.

**Direct Configuration**
   You create and configure clients explicitly using ILLMConfig. No hidden settings or magic configuration.

**Full Feature Support**
   Clients support streaming, structured output, function calling, and background tasks across all providers.

**Provider Abstraction**
   Switch between providers by changing the client instance - your application code remains the same.

Basic Usage Pattern
-------------------

All LLM clients follow this consistent pattern:

.. code-block:: python

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
   
   # 1. Configure the client
   config = ILLMConfig(
       model="gpt-4o-mini",
       temperature=0.7,
       max_tokens=500
   )
   
   # 2. Create the client
   llm_client = OpenAIClient(config)
   
   # 3. Prepare input
   input_data = ILLMInput(
       system_prompt="You are a helpful assistant",
       user_message="What is the capital of France?"
   )
   
   # 4. Get response
   response = await llm_client.chat(input_data)
   print(response["llm_response"])

Advanced Features
-----------------

**Streaming Responses**:

.. code-block:: python

   async for chunk in llm_client.stream(input_data):
       if chunk.get("llm_response"):
           print(chunk["llm_response"], end="")

**Function Calling**:

.. code-block:: python

   def get_weather(city: str) -> str:
       return f"Weather in {city}: Sunny, 22°C"
   
   input_data = ILLMInput(
       system_prompt="You can check weather for users",
       user_message="What's the weather in Paris?",
       regular_functions={"get_weather": get_weather}
   )

**Structured Output**:

.. code-block:: python

   from pydantic import BaseModel
   
   class Analysis(BaseModel):
       sentiment: str
       confidence: float
   
   input_data = ILLMInput(
       system_prompt="Analyze sentiment",
       user_message="I love this product!",
       structure_type=Analysis
   )

**Background Tasks**:

.. code-block:: python

   def log_interaction(action: str, user_id: str = "anonymous"):
       # Runs in background, doesn't return to conversation
       print(f"Logged: {action} by {user_id}")
   
   input_data = ILLMInput(
       system_prompt="You are a helpful assistant",
       user_message="Hello!",
       background_tasks={"log_interaction": log_interaction}
   )

Available Providers
-------------------

**OpenAI** (``arshai.llms.openai.OpenAIClient``)
   - Models: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
   - Features: Chat, streaming, function calling, structured output
   - Configuration: API key via environment or direct config

**Azure OpenAI** (``arshai.llms.azure.AzureClient``)
   - Models: Azure-hosted OpenAI models
   - Features: Same as OpenAI with Azure-specific configuration
   - Configuration: Azure endpoint, API key, deployment names

**Google Gemini** (``arshai.llms.google_genai.GeminiClient``)
   - Models: Gemini Pro, Gemini Pro Vision
   - Features: Chat, streaming, function calling (reference implementation)
   - Configuration: Google AI API key

**OpenRouter** (``arshai.llms.openrouter.OpenRouterClient``)
   - Models: Access to multiple providers through OpenRouter
   - Features: Unified access to Claude, GPT, Llama, and more
   - Configuration: OpenRouter API key

**Cloudflare AI Gateway** (``arshai.llms.CloudflareGatewayLLM``)
   - Models: Access to multiple providers through Cloudflare's unified endpoint
   - Features: BYOK mode, centralized caching, unified analytics
   - Configuration: Gateway token (provider keys stored in Cloudflare)

Interface Details
-----------------

All clients implement the ``ILLM`` interface with these core methods:

``async chat(input: ILLMInput) -> Dict[str, Any]``
   Single-turn conversation that returns the complete response.

``async stream(input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]``
   Streaming conversation that yields response chunks as they arrive.

The ``ILLMInput`` contains:

- ``system_prompt``: Instructions defining the AI's behavior
- ``user_message``: The user's input message  
- ``regular_functions``: Dict of functions the LLM can call
- ``background_tasks``: Dict of fire-and-forget functions
- ``structure_type``: Pydantic model for structured output
- ``max_turns``: Maximum conversation turns for function calling

Testing and Reliability
------------------------

All LLM clients are tested with identical test scenarios to ensure consistent behavior:

- Simple knowledge queries
- Structured output generation
- Function calling (sequential and parallel)
- Background task execution
- Streaming capabilities
- Usage tracking

This comprehensive testing ensures that switching between providers won't break your application logic.

Error Handling
--------------

LLM clients implement defensive error handling:

- Rate limiting with automatic retries
- Graceful degradation when features aren't available
- Safe handling of provider-specific errors
- Comprehensive logging for debugging

Next Steps
----------

- :doc:`interface-overview` - Detailed interface documentation
- :doc:`openai-client` - OpenAI-specific implementation details  
- :doc:`extending-llm-clients` - Adding support for new providers