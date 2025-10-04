OpenRouter Client
=================

The OpenRouter client provides standardized access to multiple language model providers through a single API gateway. OpenRouter acts as a proxy service that gives you access to models from OpenAI, Anthropic, Meta, Google, and many other providers through one unified interface.

.. note::
   This documentation reflects the actual implementation. OpenRouter allows you to access dozens of different models through a single API, making it ideal for testing different models or avoiding vendor lock-in.

Configuration
-------------

Basic Setup:

.. code-block:: python

   from arshai.llms.openrouter import OpenRouterClient
   from arshai.core.interfaces.illm import ILLMConfig
   
   # Configure the client
   config = ILLMConfig(
       model="openai/gpt-4o-mini",    # Provider/model format
       temperature=0.7,               # 0.0 = deterministic, 1.0 = creative
       max_tokens=500,                # Response length limit
       top_p=1.0,                     # Nucleus sampling parameter
       frequency_penalty=0.0,         # Reduce repetition
       presence_penalty=0.0           # Encourage topic diversity
   )
   
   # Create client
   client = OpenRouterClient(config)

Environment Variables:

.. code-block:: bash

   # Required
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   
   # Optional - for OpenRouter analytics and identification
   export OPENROUTER_SITE_URL="https://yoursite.com"
   export OPENROUTER_APP_NAME="your-app-name"
   
   # Optional - custom endpoint (usually not needed)
   export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

Supported Models
----------------

OpenRouter provides access to **dozens of models from multiple providers**. The client works with any model available through OpenRouter's platform:

**OpenAI Models**:
   - ``openai/gpt-4o``: Latest GPT-4 optimized
   - ``openai/gpt-4o-mini``: Fast and cost-effective
   - ``openai/gpt-4-turbo``: High-performance GPT-4
   - ``openai/gpt-3.5-turbo``: Efficient legacy model

**Anthropic Models**:
   - ``anthropic/claude-3.5-sonnet``: Latest Claude model
   - ``anthropic/claude-3-haiku``: Fast Claude model
   - ``anthropic/claude-3-opus``: Most capable Claude

**Google Models**:
   - ``google/gemini-2.0-flash-exp``: Latest Gemini
   - ``google/gemini-pro-1.5``: High-capability Gemini
   - ``google/gemini-flash-1.5``: Fast Gemini

**Meta Models**:
   - ``meta-llama/llama-3.2-90b-vision-instruct``: Latest Llama with vision
   - ``meta-llama/llama-3.1-405b-instruct``: Largest Llama model
   - ``meta-llama/llama-3.1-70b-instruct``: Balanced Llama model

**Specialized Models**:
   - ``perplexity/llama-3.1-sonar-large-128k-online``: Web-search enabled
   - ``mistralai/mistral-large``: Mistral's flagship model
   - ``cohere/command-r-plus``: Cohere's enterprise model
   - ``x-ai/grok-beta``: xAI's Grok model

**Model Discovery**:
   Check https://openrouter.ai/models for the complete, up-to-date list of available models, pricing, and capabilities.

.. note::
   **Model Format**: Use the ``provider/model-name`` format (e.g., ``openai/gpt-4o-mini``).
   
   **Dynamic Availability**: OpenRouter regularly adds new models. The client works with any model they support.

Basic Usage
-----------

Simple conversation with model selection:

.. code-block:: python

   from arshai.core.interfaces.illm import ILLMInput
   
   # Try different models easily
   models_to_test = [
       "openai/gpt-4o-mini",
       "anthropic/claude-3.5-sonnet", 
       "google/gemini-2.0-flash-exp"
   ]
   
   for model in models_to_test:
       config = ILLMConfig(model=model, temperature=0.7)
       client = OpenRouterClient(config)
       
       input_data = ILLMInput(
           system_prompt="You are a helpful AI assistant. Identify yourself and your capabilities.",
           user_message="What model are you and what can you help me with?"
       )
       
       response = await client.chat(input_data)
       print(f"Model: {model}")
       print(f"Response: {response['llm_response']}")
       print(f"Tokens: {response['usage']['total_tokens']}")
       print("-" * 50)

Streaming responses:

.. code-block:: python

   async for chunk in client.stream(input_data):
       if chunk.get("llm_response"):
           print(chunk["llm_response"], end="", flush=True)
       if chunk.get("usage"):
           print(f"\nTotal tokens: {chunk['usage']['total_tokens']}")

Model Comparison
----------------

Compare different models on the same task:

.. code-block:: python

   async def compare_models(prompt, models):
       """Compare responses from different models."""
       results = []
       
       for model in models:
           config = ILLMConfig(model=model, temperature=0.3)
           client = OpenRouterClient(config)
           
           input_data = ILLMInput(
               system_prompt="You are an expert problem solver.",
               user_message=prompt
           )
           
           response = await client.chat(input_data)
           results.append({
               "model": model,
               "response": response["llm_response"],
               "tokens": response["usage"]["total_tokens"]
           })
       
       return results
   
   # Compare models on a reasoning task
   models = [
       "openai/gpt-4o",
       "anthropic/claude-3.5-sonnet",
       "meta-llama/llama-3.1-405b-instruct"
   ]
   
   results = await compare_models(
       "Solve this logic puzzle: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
       models
   )
   
   for result in results:
       print(f"Model: {result['model']}")
       print(f"Response: {result['response'][:200]}...")
       print(f"Tokens: {result['tokens']}")
       print()

Function Calling
----------------

OpenRouter supports function calling for compatible models:

**Regular Functions**:

.. code-block:: python

   def get_model_info(model_name: str) -> dict:
       """Get information about an OpenRouter model."""
       # Mock implementation - in practice, query OpenRouter API
       model_info = {
           "openai/gpt-4o-mini": {
               "provider": "OpenAI",
               "context_length": 128000,
               "pricing_per_1k_tokens": {"prompt": 0.00015, "completion": 0.0006}
           },
           "anthropic/claude-3.5-sonnet": {
               "provider": "Anthropic", 
               "context_length": 200000,
               "pricing_per_1k_tokens": {"prompt": 0.003, "completion": 0.015}
           }
       }
       return model_info.get(model_name, {"error": "Model not found"})
   
   def calculate_cost(tokens: int, model: str, type: str = "completion") -> float:
       """Calculate cost for a given number of tokens."""
       rates = {
           "openai/gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
           "anthropic/claude-3.5-sonnet": {"prompt": 0.003, "completion": 0.015}
       }
       rate = rates.get(model, {}).get(type, 0.001)
       return (tokens / 1000) * rate
   
   input_data = ILLMInput(
       system_prompt="You are an OpenRouter expert. Use the provided tools to help with model selection and cost analysis.",
       user_message="What's the cost difference between GPT-4o-mini and Claude 3.5 Sonnet for a 1000-token response?",
       regular_functions={
           "get_model_info": get_model_info,
           "calculate_cost": calculate_cost
       },
       max_turns=10
   )
   
   response = await client.chat(input_data)

**Background Tasks** (analytics, logging):

.. code-block:: python

   def log_model_usage(model: str, tokens: int, cost: float, user_id: str = "anonymous"):
       """BACKGROUND TASK: Log model usage for analytics."""
       import datetime
       timestamp = datetime.datetime.now().isoformat()
       print(f"[USAGE_LOG] {timestamp} - Model: {model}, Tokens: {tokens}, Cost: ${cost:.4f}, User: {user_id}")
   
   input_data = ILLMInput(
       system_prompt="You are a helpful assistant. Log all usage for cost tracking.",
       user_message="Explain the benefits of using multiple AI models through OpenRouter.",
       background_tasks={
           "log_model_usage": log_model_usage
       }
   )
   
   response = await client.chat(input_data)
   # Usage is automatically logged in the background

Structured Output
-----------------

Generate structured data (for compatible models):

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List
   
   class ModelRecommendation(BaseModel):
       """Structured AI model recommendation."""
       recommended_model: str = Field(description="Full model name (provider/model)")
       reasoning: str = Field(description="Why this model is recommended")
       cost_estimate: float = Field(description="Estimated cost per 1K tokens in USD")
       strengths: List[str] = Field(description="Key strengths for this use case")
       limitations: List[str] = Field(description="Potential limitations to consider")
       alternative_models: List[str] = Field(description="Other models to consider")
   
   # Use a model that supports structured output
   config = ILLMConfig(
       model="openai/gpt-4o-mini",  # Ensure compatibility
       temperature=0.3
   )
   client = OpenRouterClient(config)
   
   input_data = ILLMInput(
       system_prompt="You are an AI model expert. Provide detailed model recommendations based on requirements.",
       user_message="I need an AI model for a customer service chatbot that handles 10,000 conversations per day. Budget is $500/month.",
       structure_type=ModelRecommendation
   )
   
   response = await client.chat(input_data)
   recommendation = response["llm_response"]  # Returns ModelRecommendation instance
   
   print(f"Recommended: {recommendation.recommended_model}")
   print(f"Cost: ${recommendation.cost_estimate}/1K tokens")
   print(f"Reasoning: {recommendation.reasoning}")
   print(f"Alternatives: {', '.join(recommendation.alternative_models)}")

Model-Specific Features
-----------------------

**Web-Enhanced Models**:

.. code-block:: python

   # Use models with web search capabilities
   config = ILLMConfig(model="perplexity/llama-3.1-sonar-large-128k-online")
   client = OpenRouterClient(config)
   
   input_data = ILLMInput(
       system_prompt="You have access to real-time web information. Use it to provide current data.",
       user_message="What are the latest developments in AI model releases this month?"
   )
   
   response = await client.chat(input_data)

**Vision-Capable Models**:

.. code-block:: python

   # Use models that can process images
   config = ILLMConfig(model="meta-llama/llama-3.2-90b-vision-instruct")
   client = OpenRouterClient(config)
   
   # Note: Actual image processing would require additional setup
   input_data = ILLMInput(
       system_prompt="You can analyze images and provide detailed descriptions.",
       user_message="Describe the architectural features in the uploaded image."
   )

**Reasoning Models**:

.. code-block:: python

   # Use models optimized for complex reasoning
   config = ILLMConfig(model="openai/o1-preview", temperature=0.1)
   client = OpenRouterClient(config)
   
   input_data = ILLMInput(
       system_prompt="You excel at step-by-step reasoning and problem solving.",
       user_message="Design an algorithm to efficiently sort a billion numbers with limited memory."
   )

Error Handling
--------------

OpenRouter-specific error handling:

.. code-block:: python

   import asyncio
   
   async def openrouter_chat_with_retry(client, input_data, max_retries=3):
       """Example retry logic for OpenRouter-specific errors."""
       for attempt in range(max_retries):
           try:
               return await client.chat(input_data)
           except Exception as e:
               error_str = str(e).lower()
               if "429" in error_str or "rate" in error_str:
                   # Rate limiting
                   wait_time = 2 ** attempt
                   await asyncio.sleep(wait_time)
                   continue
               elif "402" in error_str or "insufficient" in error_str:
                   print("Insufficient credits - check your OpenRouter balance")
                   break
               elif "model" in error_str or "not found" in error_str:
                   print("Model not available - check OpenRouter model list")
                   break
               elif "context" in error_str or "too long" in error_str:
                   print("Input too long for model context window")
                   break
               else:
                   raise

**Configuration Validation**:

.. code-block:: python

   try:
       client = OpenRouterClient(config)
   except ValueError as e:
       if "api_key" in str(e).lower():
           print("Set your OPENROUTER_API_KEY environment variable")
       else:
           print(f"Configuration error: {e}")

Usage Tracking and Cost Management
-----------------------------------

OpenRouter provides detailed usage information:

.. code-block:: python

   response = await client.chat(input_data)
   
   if response["usage"]:
       usage = response["usage"]
       print(f"Input tokens: {usage['input_tokens']}")
       print(f"Output tokens: {usage['output_tokens']}")
       print(f"Total tokens: {usage['total_tokens']}")
       print(f"Thinking tokens: {usage['thinking_tokens']}")  # For reasoning models
       
       # Provider information
       print(f"Provider: {usage['provider']}")  # Will be "openrouter"
       print(f"Model: {usage['model']}")        # Your specified model
       print(f"Request ID: {usage['request_id']}")

**Cost Calculation**:

.. code-block:: python

   def estimate_cost(usage_data, model):
       """Estimate cost based on usage and model pricing."""
       # These rates change - check OpenRouter for current pricing
       pricing = {
           "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
           "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
           "meta-llama/llama-3.1-405b-instruct": {"input": 0.005, "output": 0.015}
       }
       
       rates = pricing.get(model, {"input": 0.001, "output": 0.002})
       input_cost = (usage_data["input_tokens"] / 1000) * rates["input"]
       output_cost = (usage_data["output_tokens"] / 1000) * rates["output"]
       
       return {
           "input_cost": input_cost,
           "output_cost": output_cost,
           "total_cost": input_cost + output_cost
       }
   
   # Use after API call
   cost_estimate = estimate_cost(response["usage"], config.model)
   print(f"Estimated cost: ${cost_estimate['total_cost']:.6f}")

Performance Optimization
------------------------

**Model Selection Strategy**:

.. code-block:: python

   # Choose models based on requirements
   def select_model(use_case, budget_sensitive=False):
       if use_case == "simple_qa" and budget_sensitive:
           return "openai/gpt-4o-mini"
       elif use_case == "complex_reasoning":
           return "anthropic/claude-3.5-sonnet"
       elif use_case == "web_search":
           return "perplexity/llama-3.1-sonar-large-128k-online"
       elif use_case == "vision":
           return "meta-llama/llama-3.2-90b-vision-instruct"
       else:
           return "openai/gpt-4o"  # Default balanced choice

**Fallback Strategy**:

.. code-block:: python

   async def chat_with_fallback(input_data, preferred_models):
       """Try models in order of preference."""
       for model in preferred_models:
           try:
               config = ILLMConfig(model=model)
               client = OpenRouterClient(config)
               return await client.chat(input_data)
           except Exception as e:
               print(f"Model {model} failed: {str(e)}")
               continue
       raise Exception("All fallback models failed")
   
   # Use with fallback strategy
   preferred_models = [
       "anthropic/claude-3.5-sonnet",  # First choice
       "openai/gpt-4o",                # Second choice  
       "openai/gpt-4o-mini"            # Budget fallback
   ]
   
   response = await chat_with_fallback(input_data, preferred_models)

**Batch Processing**:

.. code-block:: python

   async def process_batch(inputs, model="openai/gpt-4o-mini"):
       """Process multiple inputs efficiently."""
       config = ILLMConfig(model=model, temperature=0.3)
       client = OpenRouterClient(config)
       
       tasks = []
       for input_data in inputs:
           tasks.append(client.chat(input_data))
       
       results = await asyncio.gather(*tasks, return_exceptions=True)
       return results

Benefits of OpenRouter
-----------------------

**1. Multi-Provider Access**
   Access dozens of models through one API without managing multiple provider accounts.

**2. Cost Optimization** 
   Compare pricing across providers and choose the most cost-effective model for each task.

**3. Reduced Vendor Lock-in**
   Easily switch between models and providers without changing your code.

**4. Unified Interface**
   Consistent API regardless of the underlying model provider.

**5. Model Discovery**
   Try new models as they become available without additional integration work.

**6. Transparent Pricing**
   Clear, competitive pricing with no hidden fees or minimum commitments.

Limitations and Considerations
------------------------------

**Model-Specific Features**
   Some provider-specific features may not be available through OpenRouter.

**Rate Limits**
   Rate limits are applied per provider and may vary from direct provider access.

**Latency**
   Additional network hop may introduce slight latency compared to direct provider access.

**Credit System**
   Requires pre-funding your OpenRouter account with credits.

**Model Availability**
   Model availability depends on provider relationships and may change.

Next Steps
----------

- :doc:`extending-llm-clients` - Creating custom LLM clients
- :doc:`../agents/index` - Building agents with LLM clients
- Visit https://openrouter.ai/models for current model listings and pricing