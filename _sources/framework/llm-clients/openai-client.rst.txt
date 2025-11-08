OpenAI Client
=============

The OpenAI client provides standardized access to OpenAI's language models through the Arshai framework. It implements the full ILLM interface with support for chat, streaming, function calling, structured output, and background tasks.

.. note::
   This documentation reflects the actual implementation based on tested functionality. All examples are verified through the framework's test suite.

Configuration
-------------

Basic Setup:

.. code-block:: python

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig
   
   # Configure the client
   config = ILLMConfig(
       model="gpt-4o-mini",    # Any OpenAI model name
       temperature=0.7,        # 0.0 = deterministic, 1.0 = creative
       max_tokens=500,         # Response length limit
       top_p=1.0,             # Nucleus sampling parameter
       frequency_penalty=0.0,  # Reduce repetition
       presence_penalty=0.0    # Encourage topic diversity
   )
   
   # Create client
   client = OpenAIClient(config)

Environment Variables:

.. code-block:: bash

   # Required
   export OPENAI_API_KEY="your-openai-api-key"
   
   # Optional - for organization usage
   export OPENAI_ORG_ID="your-organization-id"
   
   # Optional - for custom endpoints (e.g., Azure OpenAI)
   export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"

Supported Models
----------------

The OpenAI client supports **all models available through OpenAI's chat completions API**. The client works with any model name that OpenAI's API accepts, including:

**Current Popular Models** (as examples):
   - ``gpt-4o``: Latest GPT-4 optimized model
   - ``gpt-4o-mini``: Fast and cost-effective option
   - ``gpt-4-turbo``: Previous generation high-performance model
   - ``gpt-3.5-turbo``: Legacy but efficient model

**Model Compatibility**
   The client dynamically works with OpenAI's chat completions endpoint, so any model supported by that API will work. This includes:
   
   - All current GPT models
   - Future models as they become available
   - Custom fine-tuned models in your organization
   - Regional model variants

.. note::
   **No Model Restrictions**: The framework doesn't enforce which models you can use. Simply specify any valid OpenAI model name in your configuration.
   
   **Current Models**: Check OpenAI's documentation for the most up-to-date list of available models, pricing, and capabilities.

Basic Usage
-----------

Simple conversation:

.. code-block:: python

   from arshai.core.interfaces.illm import ILLMInput
   
   # Prepare input
   input_data = ILLMInput(
       system_prompt="You are a helpful travel assistant with expertise in Japanese culture and tourism.",
       user_message="I'm planning a trip to Japan. What should I know about Tokyo?"
   )
   
   # Get response
   response = await client.chat(input_data)
   print(response["llm_response"])
   print(f"Tokens used: {response['usage']['total_tokens']}")

Streaming responses:

.. code-block:: python

   async for chunk in client.stream(input_data):
       if chunk.get("llm_response"):
           print(chunk["llm_response"], end="", flush=True)
       if chunk.get("usage"):
           print(f"\nTotal tokens: {chunk['usage']['total_tokens']}")

Function Calling
----------------

The OpenAI client supports both regular functions and background tasks:

**Regular Functions** (results integrated into conversation):

.. code-block:: python

   def calculate_power(base: float, exponent: float) -> float:
       """Calculate base raised to the power of exponent."""
       return base ** exponent
   
   def multiply_numbers(a: float, b: float) -> float:
       """Multiply two numbers together."""
       return a * b
   
   input_data = ILLMInput(
       system_prompt="You are a mathematics assistant. Use the provided tools for calculations.",
       user_message="Calculate 5 to the power of 2, then multiply the result by 3. Show each step.",
       regular_functions={
           "calculate_power": calculate_power,
           "multiply_numbers": multiply_numbers
       },
       max_turns=10  # Allow multiple function calls
   )
   
   response = await client.chat(input_data)
   # LLM will call calculate_power(5, 2), get 25, then call multiply_numbers(25, 3)

**Background Tasks** (fire-and-forget execution):

.. code-block:: python

   def log_user_interaction(action: str, details: str = "User interaction"):
       """Log user interactions for analytics (background task)."""
       import datetime
       timestamp = datetime.datetime.now().isoformat()
       print(f"[{timestamp}] ANALYTICS: {action} - {details}")
   
   input_data = ILLMInput(
       system_prompt="You are a helpful assistant. For every user interaction, log it for analytics.",
       user_message="What is the capital of France?",
       background_tasks={
           "log_user_interaction": log_user_interaction
       }
   )
   
   response = await client.chat(input_data)
   # LLM answers the question AND calls log_user_interaction in background

**Parallel Function Calling**:

.. code-block:: python

   input_data = ILLMInput(
       system_prompt="You can perform multiple calculations simultaneously.",
       user_message="Calculate: 3^2, 4^2, and 6*7. You can call multiple functions at once.",
       regular_functions={
           "calculate_power": calculate_power,
           "multiply_numbers": multiply_numbers
       }
   )
   # LLM can execute multiple function calls in parallel for efficiency

Structured Output
-----------------

Generate structured data using Pydantic models:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List
   
   class SentimentAnalysis(BaseModel):
       """Structured sentiment analysis result."""
       topic: str = Field(description="Main topic being analyzed")
       sentiment: str = Field(description="Overall sentiment (positive/negative/neutral)")
       confidence: float = Field(description="Confidence score between 0.0 and 1.0")
       key_points: List[str] = Field(description="List of key points identified")
   
   input_data = ILLMInput(
       system_prompt="You are an expert sentiment analyst. Analyze the provided text thoroughly.",
       user_message="The new renewable energy project is fantastic! It will create thousands of jobs and reduce emissions significantly.",
       structure_type=SentimentAnalysis
   )
   
   response = await client.chat(input_data)
   analysis = response["llm_response"]  # Returns SentimentAnalysis instance
   
   print(f"Topic: {analysis.topic}")
   print(f"Sentiment: {analysis.sentiment}")
   print(f"Confidence: {analysis.confidence}")
   print(f"Key points: {', '.join(analysis.key_points)}")

Streaming with structured output:

.. code-block:: python

   # For streaming, use dict-based models with schema method
   from typing import TypedDict
   
   class StreamingSentiment(TypedDict):
       topic: str
       sentiment: str
       confidence: float
       key_points: List[str]
       
       @classmethod
       def model_json_schema(cls):
           return {
               "type": "object",
               "properties": {
                   "topic": {"type": "string", "description": "Main topic"},
                   "sentiment": {"type": "string", "description": "Sentiment"},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                   "key_points": {"type": "array", "items": {"type": "string"}}
               },
               "required": ["topic", "sentiment", "confidence", "key_points"]
           }
   
   input_data = ILLMInput(
       system_prompt="Analyze sentiment and return structured data.",
       user_message="I love this product! It's amazing and works perfectly.",
       structure_type=StreamingSentiment
   )
   
   async for chunk in client.stream(input_data):
       if chunk.get("llm_response") and isinstance(chunk["llm_response"], dict):
           result = chunk["llm_response"]
           if "sentiment" in result:
               print(f"Sentiment: {result['sentiment']}")

Advanced Features
-----------------

**Custom Base URL Support**
   The client supports custom endpoints for Azure OpenAI or other compatible services:

.. code-block:: python

   # Via environment variable
   export OPENAI_BASE_URL="https://your-azure-instance.openai.azure.com/openai/deployments/your-deployment/v1"
   
   # The client automatically uses the custom endpoint

**Safe HTTP Configuration**
   The client includes enhanced HTTP safety features when available:

.. code-block:: python

   # Automatically uses SafeHttpClientFactory if available
   # Falls back to standard OpenAI client if not
   client = OpenAIClient(config)  # Safe by default

**Context Management**
   Use the client as a context manager for proper resource cleanup:

.. code-block:: python

   async with OpenAIClient(config) as client:
       response = await client.chat(input_data)
       # Client automatically closes connections when exiting

Error Handling
--------------

The OpenAI client implements comprehensive error handling:

**Rate Limiting**:

.. code-block:: python

   import asyncio
   
   async def chat_with_retry(client, input_data, max_retries=3):
       """Example retry logic for rate limiting."""
       for attempt in range(max_retries):
           try:
               return await client.chat(input_data)
           except Exception as e:
               if "429" in str(e) and attempt < max_retries - 1:
                   wait_time = 2 ** attempt  # Exponential backoff
                   await asyncio.sleep(wait_time)
                   continue
               raise

**Configuration Validation**:

.. code-block:: python

   # Invalid configuration will raise errors during client creation
   try:
       config = ILLMConfig(model="invalid-model")
       client = OpenAIClient(config)
   except ValueError as e:
       print(f"Configuration error: {e}")

**Network and API Errors**:

.. code-block:: python

   try:
       response = await client.chat(input_data)
   except Exception as e:
       if "authentication" in str(e).lower():
           print("Check your OpenAI API key")
       elif "quota" in str(e).lower():
           print("API quota exceeded")
       else:
           print(f"Unexpected error: {e}")

Usage Tracking
--------------

The client provides detailed usage information compatible with OpenAI's latest API format:

.. code-block:: python

   response = await client.chat(input_data)
   
   if response["usage"]:
       usage = response["usage"]
       print(f"Input tokens: {usage['input_tokens']}")
       print(f"Output tokens: {usage['output_tokens']}")
       print(f"Total tokens: {usage['total_tokens']}")
       print(f"Thinking tokens: {usage['thinking_tokens']}")  # For reasoning models
       print(f"Tool calling tokens: {usage['tool_calling_tokens']}")  # Function calls
       
       # Provider information
       print(f"Provider: {usage['provider']}")
       print(f"Model: {usage['model']}")
       print(f"Request ID: {usage['request_id']}")

Performance Optimization
------------------------

**Model Selection**:

.. code-block:: python

   # For simple tasks
   config_fast = ILLMConfig(model="gpt-4o-mini", temperature=0.3)
   
   # For complex reasoning
   config_powerful = ILLMConfig(model="gpt-4o", temperature=0.7)
   
   # For specific capabilities (check OpenAI docs for latest models)
   config_latest = ILLMConfig(model="gpt-4o", temperature=0.5)

**Token Management**:

.. code-block:: python

   # Limit response length for cost control
   config = ILLMConfig(
       model="gpt-4o-mini",
       max_tokens=200,  # Shorter responses
       temperature=0.2  # More focused responses
   )

**Streaming for Better UX**:

.. code-block:: python

   # Use streaming for real-time user interfaces
   async def stream_to_user(client, input_data):
       response_text = ""
       async for chunk in client.stream(input_data):
           if chunk.get("llm_response"):
               new_text = chunk["llm_response"]
               # Display incremental text to user
               print(new_text[len(response_text):], end="", flush=True)
               response_text = new_text

Testing Integration
-------------------

The OpenAI client is thoroughly tested with scenarios including:

- Simple knowledge queries with pattern validation
- Structured output generation and validation
- Sequential function calling with step-by-step execution
- Parallel function calling for efficiency
- Background task execution with verification
- Streaming behavior validation
- Usage tracking accuracy

These tests ensure reliable behavior across different use cases and can serve as examples for your own implementations.

Implementation Notes
--------------------

**OpenAI Responses API**
   The client uses OpenAI's latest Responses API for structured output and enhanced function calling capabilities.

**Progressive Function Execution**
   Functions execute immediately when detected during streaming, providing real-time responsiveness.

**Safe HTTP Handling**
   Enhanced HTTP client configuration when SafeHttpClientFactory is available, with graceful fallback.

**Resource Management**
   Proper connection cleanup through context managers and destructor methods.

Limitations and Considerations
------------------------------

**Rate Limits**
   OpenAI enforces rate limits based on your plan. Implement retry logic for production use.

**Cost Management**
   Monitor token usage carefully, especially with advanced models. Consider using max_tokens limits.

**Model Availability**
   Model names and availability change. The client works with any valid OpenAI model.

**Function Calling Limits**
   Complex function calling scenarios may hit context length limits. Design functions to be concise.

**Streaming Consistency**
   Streaming behavior may vary based on response length and complexity. Test thoroughly for your use cases.

Next Steps
----------

- :doc:`azure-client` - Azure OpenAI integration
- :doc:`extending-llm-clients` - Creating custom LLM clients
- :doc:`../agents/index` - Building agents with LLM clients