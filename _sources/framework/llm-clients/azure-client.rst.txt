Azure OpenAI Client
===================

The Azure OpenAI client provides standardized access to Azure-hosted OpenAI models through the Arshai framework. It implements the full ILLM interface with support for chat, streaming, function calling, structured output, and background tasks.

.. note::
   This documentation reflects the actual implementation based on tested functionality. The Azure client uses the same underlying OpenAI SDK with Azure-specific configuration.

Configuration
-------------

Basic Setup:

.. code-block:: python

   from arshai.llms.azure import AzureClient
   from arshai.core.interfaces.illm import ILLMConfig
   
   # Configure the client
   config = ILLMConfig(
       model="gpt-4o-mini",    # Your Azure deployment name
       temperature=0.7,        # 0.0 = deterministic, 1.0 = creative
       max_tokens=500,         # Response length limit
       top_p=1.0,             # Nucleus sampling parameter
       frequency_penalty=0.0,  # Reduce repetition
       presence_penalty=0.0    # Encourage topic diversity
   )
   
   # Create client with Azure-specific configuration
   client = AzureClient(
       config=config,
       azure_deployment="your-deployment-name",    # Optional if set in env
       api_version="2024-10-21"                    # Optional if set in env
   )

Environment Variables:

.. code-block:: bash

   # Required
   export AZURE_OPENAI_API_KEY="your-azure-api-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_DEPLOYMENT="your-deployment-name"
   export AZURE_API_VERSION="2024-10-21"
   
   # Optional - for organization tracking
   export AZURE_OPENAI_AD_TOKEN="your-ad-token"  # Alternative to API key

Azure-Specific Configuration:

.. code-block:: python

   # Initialize with explicit Azure parameters
   client = AzureClient(
       config=config,
       azure_deployment="my-gpt-4-deployment",
       api_version="2024-10-21"
   )
   
   # Or let it read from environment variables
   client = AzureClient(config=config)

Supported Models
----------------

The Azure OpenAI client supports **all models available in your Azure OpenAI resource**. The client works with any deployment name configured in your Azure OpenAI service, including:

**Deployment-Based Model Access**
   - Unlike direct OpenAI, Azure uses deployment names rather than model names
   - Each deployment maps to a specific model version
   - Specify your deployment name in the `model` parameter

**Common Azure Deployments** (examples):
   - ``gpt-4o``: Latest GPT-4 optimized model deployments
   - ``gpt-4o-mini``: Fast and cost-effective deployments
   - ``gpt-4-turbo``: High-performance model deployments
   - ``gpt-35-turbo``: Efficient model deployments

**Enterprise Features**
   - Private endpoints and VNet integration
   - Customer-managed keys and data residency
   - Azure Active Directory authentication
   - Compliance with enterprise security requirements

.. note::
   **Deployment Names**: The `model` parameter should contain your Azure deployment name, not the underlying model name.
   
   **Regional Availability**: Check your Azure region for model availability and update API versions for latest features.

Basic Usage
-----------

Simple conversation:

.. code-block:: python

   from arshai.core.interfaces.illm import ILLMInput
   
   # Prepare input
   input_data = ILLMInput(
       system_prompt="You are a helpful AI assistant specializing in Azure cloud services.",
       user_message="How do I set up Azure OpenAI with private endpoints?"
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

The Azure client supports identical function calling to the OpenAI client:

**Regular Functions**:

.. code-block:: python

   def check_azure_service_health(service_name: str, region: str = "eastus") -> dict:
       """Check the health status of an Azure service in a specific region."""
       # Mock implementation
       return {
           "service": service_name,
           "region": region,
           "status": "healthy",
           "last_updated": "2024-01-15T10:30:00Z"
       }
   
   def estimate_azure_costs(service: str, tier: str, hours: int = 24) -> dict:
       """Estimate Azure service costs for a given time period."""
       base_rates = {"basic": 0.10, "standard": 0.25, "premium": 0.50}
       hourly_rate = base_rates.get(tier.lower(), 0.25)
       return {
           "service": service,
           "tier": tier,
           "hours": hours,
           "estimated_cost": hourly_rate * hours,
           "currency": "USD"
       }
   
   input_data = ILLMInput(
       system_prompt="You are an Azure consultant. Use the provided tools to help with Azure questions.",
       user_message="Check the health of Azure OpenAI in East US and estimate costs for standard tier for 48 hours.",
       regular_functions={
           "check_azure_service_health": check_azure_service_health,
           "estimate_azure_costs": estimate_azure_costs
       },
       max_turns=10
   )
   
   response = await client.chat(input_data)

**Background Tasks** (logging, monitoring, etc.):

.. code-block:: python

   def log_azure_usage(service: str, operation: str, user_id: str = "system"):
       """Log Azure service usage for compliance tracking."""
       import datetime
       timestamp = datetime.datetime.now().isoformat()
       print(f"[AUDIT] {timestamp} - Service: {service}, Operation: {operation}, User: {user_id}")
   
   input_data = ILLMInput(
       system_prompt="You are an Azure AI assistant. Log all interactions for compliance.",
       user_message="Help me understand Azure OpenAI pricing models.",
       background_tasks={
           "log_azure_usage": log_azure_usage
       }
   )
   
   response = await client.chat(input_data)
   # Automatically logs the interaction in the background

Structured Output
-----------------

Generate structured data for Azure automation:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List
   
   class AzureResourceRecommendation(BaseModel):
       """Structured Azure resource recommendation."""
       resource_type: str = Field(description="Type of Azure resource (e.g., 'App Service', 'Virtual Machine')")
       tier: str = Field(description="Recommended service tier (Basic, Standard, Premium)")
       region: str = Field(description="Recommended Azure region")
       estimated_monthly_cost: float = Field(description="Estimated monthly cost in USD")
       justification: str = Field(description="Reason for this recommendation")
       configuration_steps: List[str] = Field(description="Steps to configure this resource")
   
   input_data = ILLMInput(
       system_prompt="You are an Azure architect. Provide detailed resource recommendations.",
       user_message="I need to host a Python web application with 1000 daily users. What Azure resources do I need?",
       structure_type=AzureResourceRecommendation
   )
   
   response = await client.chat(input_data)
   recommendation = response["llm_response"]  # Returns AzureResourceRecommendation instance
   
   print(f"Resource: {recommendation.resource_type}")
   print(f"Tier: {recommendation.tier}")
   print(f"Region: {recommendation.region}")
   print(f"Monthly Cost: ${recommendation.estimated_monthly_cost}")
   print(f"Steps: {', '.join(recommendation.configuration_steps)}")

Azure-Specific Features
-----------------------

**Private Endpoint Support**:

.. code-block:: python

   # Configure for private endpoint access
   client = AzureClient(
       config=config,
       azure_deployment="my-private-deployment"
   )
   # The client automatically uses your configured Azure endpoint

**Azure Active Directory Authentication**:

.. code-block:: bash

   # Use Azure AD token instead of API key
   export AZURE_OPENAI_AD_TOKEN="your-aad-token"
   # Don't set AZURE_OPENAI_API_KEY when using AD auth

**Multi-Region Deployments**:

.. code-block:: python

   # Configure for specific region/deployment
   us_client = AzureClient(
       config=ILLMConfig(model="us-east-gpt4"),
       azure_deployment="us-east-gpt4",
       api_version="2024-10-21"
   )
   
   eu_client = AzureClient(
       config=ILLMConfig(model="eu-west-gpt4"),
       azure_deployment="eu-west-gpt4", 
       api_version="2024-10-21"
   )

**Content Filtering Integration**:

.. code-block:: python

   # Azure's content filtering is automatically applied
   # No additional configuration needed - handled by Azure OpenAI service
   try:
       response = await client.chat(input_data)
   except Exception as e:
       if "content_filter" in str(e).lower():
           print("Content was filtered by Azure OpenAI safety systems")

Error Handling
--------------

Azure-specific error handling:

.. code-block:: python

   import asyncio
   
   async def azure_chat_with_retry(client, input_data, max_retries=3):
       """Example retry logic for Azure-specific errors."""
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
               elif "401" in error_str or "authentication" in error_str:
                   print("Check your Azure OpenAI API key or AD token")
                   break
               elif "403" in error_str or "forbidden" in error_str:
                   print("Check your Azure OpenAI resource permissions")
                   break
               elif "deployment" in error_str:
                   print("Check your Azure deployment name and model availability")
                   break
               else:
                   raise

**Configuration Validation**:

.. code-block:: python

   try:
       client = AzureClient(config)
   except ValueError as e:
       print(f"Azure configuration error: {e}")
       # Check AZURE_DEPLOYMENT and AZURE_API_VERSION environment variables

**Network and Regional Errors**:

.. code-block:: python

   try:
       response = await client.chat(input_data)
   except Exception as e:
       if "timeout" in str(e).lower():
           print("Network timeout - check Azure region connectivity")
       elif "ssl" in str(e).lower():
           print("SSL/TLS error - check certificate configuration")
       elif "dns" in str(e).lower():
           print("DNS resolution error - check Azure endpoint URL")

Usage Tracking
--------------

Azure-specific usage information:

.. code-block:: python

   response = await client.chat(input_data)
   
   if response["usage"]:
       usage = response["usage"]
       print(f"Input tokens: {usage['input_tokens']}")
       print(f"Output tokens: {usage['output_tokens']}")
       print(f"Total tokens: {usage['total_tokens']}")
       print(f"Thinking tokens: {usage['thinking_tokens']}")  # For reasoning models
       
       # Azure-specific metadata
       print(f"Provider: {usage['provider']}")  # Will be "azure"
       print(f"Deployment: {usage['model']}")   # Your deployment name
       print(f"Request ID: {usage['request_id']}")  # For Azure support tickets

Performance Optimization
------------------------

**Regional Deployment Selection**:

.. code-block:: python

   # Choose regions close to your users
   config_us = ILLMConfig(model="us-deployment", temperature=0.7)
   config_eu = ILLMConfig(model="eu-deployment", temperature=0.7)

**Tier-Based Cost Management**:

.. code-block:: python

   # Use different tiers for different use cases
   config_dev = ILLMConfig(
       model="dev-deployment",  # Lower tier for development
       max_tokens=100,
       temperature=0.3
   )
   
   config_prod = ILLMConfig(
       model="prod-deployment",  # Higher tier for production
       max_tokens=1000,
       temperature=0.7
   )

**Content Filtering Optimization**:

.. code-block:: python

   # Configure prompts to work well with Azure content filtering
   input_data = ILLMInput(
       system_prompt="You are a helpful, safe, and responsible AI assistant. Follow Azure content policies.",
       user_message="Help me create appropriate content for my application."
   )

Enterprise Integration
----------------------

**Azure Monitor Integration**:

.. code-block:: python

   # Usage data automatically flows to Azure Monitor
   # Configure alerting and monitoring in Azure portal
   response = await client.chat(input_data)
   # Metrics are automatically tracked

**Azure Key Vault Integration**:

.. code-block:: bash

   # Store API keys securely in Key Vault
   # Reference them in your application configuration
   export AZURE_OPENAI_API_KEY="@Microsoft.KeyVault(VaultName=myVault;SecretName=openai-key)"

**Virtual Network Integration**:

.. code-block:: python

   # Configure private endpoints in Azure
   # Client automatically uses configured networking
   client = AzureClient(config=config)  # Uses your VNet configuration

Limitations and Considerations
------------------------------

**Deployment Dependencies**
   Your Azure OpenAI resource must have the required model deployments configured.

**Regional Availability**
   Model availability varies by Azure region. Check Azure documentation for current regional support.

**Content Filtering**
   Azure applies content filtering that may affect certain use cases. Design prompts accordingly.

**Rate Limits**
   Rate limits are applied per deployment. Scale deployments for higher throughput requirements.

**API Version Updates**
   Azure regularly updates API versions. Keep your `api_version` parameter current for latest features.

Next Steps
----------

- :doc:`google-gemini-client` - Google Gemini integration
- :doc:`openrouter-client` - Multi-provider access via OpenRouter
- :doc:`extending-llm-clients` - Creating custom LLM clients
- :doc:`../agents/index` - Building agents with LLM clients