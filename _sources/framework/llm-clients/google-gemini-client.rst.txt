Google Gemini Client
====================

The Google Gemini client provides standardized access to Google's Gemini models through the Arshai framework. It implements the full ILLM interface with support for chat, streaming, function calling, structured output, and background tasks.

.. note::
   **Reference Implementation**: According to the Arshai codebase, this client serves as the reference implementation for LLM clients, demonstrating best practices for the framework architecture.

Configuration
-------------

Basic Setup:

.. code-block:: python

   from arshai.llms.google_genai import GeminiClient
   from arshai.core.interfaces.illm import ILLMConfig
   
   # Configure the client
   config = ILLMConfig(
       model="gemini-2.0-flash-exp",  # Latest Gemini model
       temperature=0.7,               # 0.0 = deterministic, 1.0 = creative
       max_tokens=500,                # Response length limit
       config={                       # Gemini-specific configuration
           "thinking_config": {
               "include_thoughts": True
           }
       }
   )
   
   # Create client
   client = GeminiClient(config)

Authentication Methods:

The Gemini client supports two authentication methods:

**Method 1: API Key (Simpler)**:

.. code-block:: bash

   # Set API key from Google AI Studio
   export GOOGLE_API_KEY="your-google-api-key"

.. code-block:: python

   # Client automatically uses API key
   client = GeminiClient(config)

**Method 2: Service Account (Enterprise)**:

.. code-block:: bash

   # Set Vertex AI configuration
   export VERTEX_AI_SERVICE_ACCOUNT_PATH="/path/to/service-account.json"
   export VERTEX_AI_PROJECT_ID="your-gcp-project-id"
   export VERTEX_AI_LOCATION="us-central1"

.. code-block:: python

   # Client automatically detects and uses service account
   client = GeminiClient(config)

Supported Models
----------------

The Gemini client supports **all models available through Google's Gemini API**. The client dynamically works with any model name that Google's API accepts, including:

**Current Gemini Models** (examples):
   - ``gemini-2.0-flash-exp``: Latest experimental flash model
   - ``gemini-1.5-pro``: High-capability model for complex tasks
   - ``gemini-1.5-flash``: Fast model for quick responses
   - ``gemini-pro-vision``: Multimodal model with vision capabilities

**Model Capabilities**:
   - **Text Generation**: All models support text-based conversations
   - **Function Calling**: Native support for tool integration
   - **Reasoning**: Models with thinking capabilities (Gemini 2.0+)
   - **Multimodal**: Some models support images, audio, and video
   - **Long Context**: Extended context windows for complex tasks

**Model Selection Guidelines**:
   - **gemini-2.0-flash-exp**: Latest features and performance
   - **gemini-1.5-pro**: Best for complex reasoning and analysis
   - **gemini-1.5-flash**: Optimal for speed and cost efficiency
   - **gemini-pro-vision**: When working with visual content

.. note::
   **No Model Restrictions**: The framework doesn't enforce which models you can use. Simply specify any valid Gemini model name.
   
   **Current Models**: Check Google's documentation for the most up-to-date list of available models and their capabilities.

Basic Usage
-----------

Simple conversation:

.. code-block:: python

   from arshai.core.interfaces.illm import ILLMInput
   
   # Prepare input
   input_data = ILLMInput(
       system_prompt="You are a helpful AI assistant with expertise in Google Cloud and AI technologies.",
       user_message="Explain the differences between Gemini models and when to use each one."
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

Advanced Configuration
----------------------

Gemini-specific features:

.. code-block:: python

   from google.genai.types import ThinkingConfig, SpeechConfig
   
   # Advanced configuration with thinking capabilities
   config = ILLMConfig(
       model="gemini-2.0-flash-exp",
       temperature=0.7,
       config={
           "thinking_config": {
               "include_thoughts": True  # Enable reasoning traces
           },
           "speech_config": {
               "voice_config": {
                   "prebuilt_voice_config": {
                       "voice_name": "en-US-Journey-D"
                   }
               }
           },
           "max_output_tokens": 2048,
           "top_p": 0.8,
           "top_k": 40
       }
   )

Function Calling
----------------

The Gemini client supports auto-generation of function declarations:

**Regular Functions with Auto-Discovery**:

.. code-block:: python

   def search_google_cloud_docs(query: str, product: str = "all") -> dict:
       """Search Google Cloud documentation for specific information."""
       # Mock implementation
       return {
           "query": query,
           "product": product,
           "results": [
               {
                   "title": f"Documentation for {query}",
                   "url": f"https://cloud.google.com/docs/{product}",
                   "summary": f"Comprehensive guide for {query} in {product}"
               }
           ]
       }
   
   def analyze_gcp_costs(project_id: str, service: str, days: int = 30) -> dict:
       """Analyze Google Cloud Platform costs for a specific service."""
       # Mock implementation
       return {
           "project_id": project_id,
           "service": service,
           "period_days": days,
           "total_cost": 125.50,
           "average_daily": 4.18,
           "currency": "USD"
       }
   
   input_data = ILLMInput(
       system_prompt="You are a Google Cloud expert. Use the provided tools to help with GCP questions.",
       user_message="Help me find documentation about Cloud Functions and analyze costs for my project 'my-gcp-project'.",
       regular_functions={
           "search_google_cloud_docs": search_google_cloud_docs,
           "analyze_gcp_costs": analyze_gcp_costs
       },
       max_turns=10
   )
   
   response = await client.chat(input_data)
   # Gemini automatically generates function schemas using introspection

**Background Tasks** (fire-and-forget execution):

.. code-block:: python

   def log_gemini_interaction(operation: str, model_used: str, user_id: str = "anonymous"):
       """BACKGROUND TASK: Log Gemini API interactions for monitoring."""
       import datetime
       timestamp = datetime.datetime.now().isoformat()
       print(f"[GEMINI_LOG] {timestamp} - Operation: {operation}, Model: {model_used}, User: {user_id}")
   
   input_data = ILLMInput(
       system_prompt="You are a helpful assistant. Log all interactions for monitoring purposes.",
       user_message="What are the best practices for using Gemini models?",
       background_tasks={
           "log_gemini_interaction": log_gemini_interaction
       }
   )
   
   response = await client.chat(input_data)
   # Function executes in background without affecting conversation

Structured Output
-----------------

Generate structured data with Gemini's native structured output:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List
   
   class AIModelComparison(BaseModel):
       """Structured comparison of AI models."""
       model_name: str = Field(description="Name of the AI model")
       strengths: List[str] = Field(description="Key strengths of this model")
       weaknesses: List[str] = Field(description="Limitations of this model")
       best_use_cases: List[str] = Field(description="Ideal scenarios for using this model")
       performance_score: int = Field(description="Performance rating from 1-10")
       cost_efficiency: str = Field(description="Cost efficiency level (low/medium/high)")
   
   input_data = ILLMInput(
       system_prompt="You are an AI expert. Provide detailed model comparisons with structured analysis.",
       user_message="Compare Gemini 2.0 Flash with GPT-4 for business applications.",
       structure_type=AIModelComparison
   )
   
   response = await client.chat(input_data)
   comparison = response["llm_response"]  # Returns AIModelComparison instance
   
   print(f"Model: {comparison.model_name}")
   print(f"Performance: {comparison.performance_score}/10")
   print(f"Strengths: {', '.join(comparison.strengths)}")
   print(f"Best for: {', '.join(comparison.best_use_cases)}")

Streaming with structured output:

.. code-block:: python

   async for chunk in client.stream(input_data):
       if chunk.get("llm_response") and isinstance(chunk["llm_response"], AIModelComparison):
           result = chunk["llm_response"]
           print(f"Structured response received: {result.model_name}")

Thinking and Reasoning
----------------------

Enable reasoning traces with Gemini 2.0+ models:

.. code-block:: python

   config = ILLMConfig(
       model="gemini-2.0-flash-exp",
       temperature=0.3,
       config={
           "thinking_config": {
               "include_thoughts": True  # Enable reasoning traces
           }
       }
   )
   
   client = GeminiClient(config)
   
   input_data = ILLMInput(
       system_prompt="You are a logical reasoning expert. Show your thinking process step by step.",
       user_message="If a train travels 300 miles in 4 hours, and then speeds up by 25% for the next 2 hours, how far does it travel in total?"
   )
   
   response = await client.chat(input_data)
   # Response includes both the answer and the reasoning process

Error Handling
--------------

Gemini-specific error handling:

.. code-block:: python

   import asyncio
   
   async def gemini_chat_with_retry(client, input_data, max_retries=3):
       """Example retry logic for Gemini-specific errors."""
       for attempt in range(max_retries):
           try:
               return await client.chat(input_data)
           except Exception as e:
               error_str = str(e).lower()
               if "quota" in error_str or "429" in error_str:
                   # Rate limiting or quota exceeded
                   wait_time = 2 ** attempt
                   await asyncio.sleep(wait_time)
                   continue
               elif "authentication" in error_str or "401" in error_str:
                   print("Check your GOOGLE_API_KEY or service account configuration")
                   break
               elif "model" in error_str or "not found" in error_str:
                   print("Check if the specified Gemini model is available")
                   break
               else:
                   raise

**Configuration Validation**:

.. code-block:: python

   try:
       client = GeminiClient(config)
   except ValueError as e:
       if "authentication" in str(e):
           print("Set GOOGLE_API_KEY or configure Vertex AI service account")
       else:
           print(f"Configuration error: {e}")

**Network and API Errors**:

.. code-block:: python

   try:
       response = await client.chat(input_data)
   except Exception as e:
       if "timeout" in str(e).lower():
           print("Request timeout - try reducing input length or complexity")
       elif "safety" in str(e).lower():
           print("Content filtered by Gemini safety systems")
       elif "resource" in str(e).lower():
           print("Resource exhausted - check quotas and billing")

Usage Tracking
--------------

Gemini-specific usage information:

.. code-block:: python

   response = await client.chat(input_data)
   
   if response["usage"]:
       usage = response["usage"]
       print(f"Input tokens: {usage['input_tokens']}")        # prompt_token_count
       print(f"Output tokens: {usage['output_tokens']}")      # candidates_token_count
       print(f"Total tokens: {usage['total_tokens']}")        # total_token_count
       print(f"Thinking tokens: {usage['thinking_tokens']}")  # thoughts_token_count (2.0+)
       
       # Provider information
       print(f"Provider: {usage['provider']}")  # Will be "gemini"
       print(f"Model: {usage['model']}")        # Your specified model
       print(f"Request ID: {usage['request_id']}")

Performance Optimization
------------------------

**Model Selection for Performance**:

.. code-block:: python

   # For speed-critical applications
   config_fast = ILLMConfig(
       model="gemini-1.5-flash",
       temperature=0.3,
       max_tokens=500
   )
   
   # For complex reasoning tasks
   config_powerful = ILLMConfig(
       model="gemini-1.5-pro", 
       temperature=0.7,
       config={"thinking_config": {"include_thoughts": True}}
   )
   
   # For latest features
   config_experimental = ILLMConfig(
       model="gemini-2.0-flash-exp",
       temperature=0.5
   )

**Context Management**:

.. code-block:: python

   # Optimize for long conversations
   config = ILLMConfig(
       model="gemini-1.5-pro",
       config={
           "max_output_tokens": 1024,  # Reasonable response length
           "top_p": 0.8,                # Focused sampling
           "top_k": 40                  # Controlled vocabulary
       }
   )

**Streaming for Responsiveness**:

.. code-block:: python

   # Use streaming for real-time applications
   async def stream_response(client, input_data):
       response_text = ""
       async for chunk in client.stream(input_data):
           if chunk.get("llm_response"):
               new_text = chunk["llm_response"]
               if isinstance(new_text, str):
                   # Display progressive text
                   print(new_text[len(response_text):], end="", flush=True)
                   response_text = new_text

Enterprise Features
-------------------

**Vertex AI Integration**:

.. code-block:: python

   # Configure for Vertex AI (enterprise)
   import os
   os.environ.update({
       "VERTEX_AI_SERVICE_ACCOUNT_PATH": "/path/to/service-account.json",
       "VERTEX_AI_PROJECT_ID": "your-project",
       "VERTEX_AI_LOCATION": "us-central1"
   })
   
   client = GeminiClient(config)  # Automatically uses Vertex AI

**Content Safety and Compliance**:

.. code-block:: python

   # Gemini includes built-in safety filtering
   # Configure prompts to work with safety systems
   input_data = ILLMInput(
       system_prompt="You are a helpful, safe, and responsible AI assistant.",
       user_message="Help me create appropriate content for my business application."
   )

**Cost Management**:

.. code-block:: python

   # Monitor and control costs
   config = ILLMConfig(
       model="gemini-1.5-flash",  # Cost-effective model
       max_tokens=200,            # Limit response length
       temperature=0.3            # More focused responses
   )

Limitations and Considerations
------------------------------

**Model Availability**
   Model names and availability change as Google updates their offerings. Check Google's documentation for current models.

**Rate Limits**
   Google enforces rate limits on API usage. Implement retry logic for production applications.

**Content Filtering**
   Gemini applies safety filtering that may affect certain types of content. Design prompts accordingly.

**Regional Availability**
   Some models may have regional restrictions. Check availability for your target regions.

**Authentication Methods**
   Choose between API key (simpler) and service account (enterprise) based on your security requirements.

Next Steps
----------

- :doc:`openrouter-client` - Multi-provider access via OpenRouter
- :doc:`extending-llm-clients` - Creating custom LLM clients
- :doc:`../agents/index` - Building agents with LLM clients