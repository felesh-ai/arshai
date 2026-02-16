"""
AI Gateway LLM implementation.

This client works with any gateway or proxy that implements the OpenAI API specification.
Supports custom base URLs, headers, and authentication methods.

Compatible gateways include:
- Cloudflare AI Gateway
- LiteLLM Proxy
- Custom OpenAI proxies
- Regional OpenAI endpoints
- Enterprise gateways
- Any service implementing OpenAI's /v1/chat/completions endpoint
"""

import os
import json
import time
import base64
import inspect
from typing import Dict, Any, TypeVar, AsyncGenerator, List, Callable, Optional
from pydantic import Field, field_validator

from openai import OpenAI

from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
from arshai.llms.base_llm_client import BaseLLMClient
from arshai.llms.utils import is_json_complete, parse_to_structure
from arshai.llms.utils.function_execution import (
    FunctionCall,
    FunctionExecutionInput,
    StreamingExecutionState,
)

T = TypeVar("T")

# Structure instructions template
STRUCTURE_INSTRUCTIONS_TEMPLATE = """

You MUST ALWAYS use the {function_name} tool/function to format your response.
Your response ALWAYS MUST be returned using the tool, independently of what the message or response are.
You MUST ALWAYS CALL TOOLS FOR RETURNING RESPONSE
The response Must be in JSON format"""


class AIGatewayConfig(ILLMConfig):
    """
    Configuration for OpenAI-compatible gateway/proxy services.

    This client works with any service that implements the OpenAI API specification,
    including custom gateways, proxies, and regional endpoints.

    Examples:
        # Cloudflare AI Gateway (BYOK mode)
        config = AIGatewayConfig(
            base_url="https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat",
            api_key=os.getenv("CLOUDFLARE_GATEWAY_TOKEN"),
            model="anthropic/claude-sonnet-4-5",
            headers={
                "HTTP-Referer": "https://myapp.com",
                "X-Title": "My Application"
            }
        )

        # LiteLLM Proxy
        config = AIGatewayConfig(
            base_url="http://localhost:4000",
            api_key="sk-litellm-key",
            model="gpt-4o",
        )

        # Custom Enterprise Gateway
        config = AIGatewayConfig(
            base_url="https://api.mycompany.com/v1",
            api_key=os.getenv("ENTERPRISE_API_KEY"),
            model="my-fine-tuned-model",
            headers={
                "X-Organization-ID": "org-123",
                "X-Department": "engineering"
            }
        )

        # Regional OpenAI Endpoint
        config = AIGatewayConfig(
            base_url="https://api.openai.azure.com/openai/deployments",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            model="gpt-4-deployment",
            headers={"api-version": "2024-02-01"}
        )

        # Environment variable fallback
        config = AIGatewayConfig(
            # Falls back to GATEWAY_BASE_URL env var
            # Falls back to GATEWAY_API_KEY env var
            model="gpt-4o"
        )
    """

    # Core gateway configuration
    base_url: str = Field(
        default=None,
        description="Gateway base URL (e.g., 'https://gateway.ai.cloudflare.com/v1/account/gateway'). "
                    "Falls back to GATEWAY_BASE_URL environment variable."
    )

    gateway_token: Optional[str] = Field(
        default=None,
        description="Gateway token for authentication. Falls back to GATEWAY_TOKEN environment variable."
    )

    model: str = Field(
        description="Model name as expected by the gateway (e.g., 'gpt-4o', 'anthropic/claude-sonnet-4-5', 'my-model')"
    )

    # Custom headers for gateway-specific requirements
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers to include in requests (e.g., organization ID, referrer, API version)"
    )

    # Standard LLM settings (inherited from ILLMConfig)
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)

    # Request settings
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Optional: Organization/project settings (OpenAI compatibility)
    organization: Optional[str] = Field(
        default=None,
        description="Organization ID for OpenAI-style organization routing"
    )

    project: Optional[str] = Field(
        default=None,
        description="Project ID for OpenAI-style project routing"
    )

    @field_validator('base_url', mode='before')
    @classmethod
    def validate_base_url(cls, v):
        """Validate and set base_url with environment variable fallback."""
        if v is None:
            v = os.environ.get("GATEWAY_BASE_URL")
        if not v:
            raise ValueError(
                "base_url is required. Set it directly or via GATEWAY_BASE_URL environment variable."
            )
        # Remove trailing slash for consistency
        return v.rstrip('/')

    @field_validator('gateway_token', mode='before')
    @classmethod
    def validate_gateway_token(cls, v):
        """Validate and set gateway_token with environment variable fallback."""
        if v is None:
            v = os.environ.get("GATEWAY_TOKEN")
        if not v:
            raise ValueError(
                "gateway_token is required. Set it directly or via GATEWAY_TOKEN environment variable."
            )
        return v

    def get_effective_headers(self) -> Dict[str, str]:
        """
        Get the complete headers dictionary including defaults and custom headers.

        Returns:
            Merged headers with custom headers taking precedence
        """
        default_headers = {
            "HTTP-Referer": "https://github.com/ArshaAI/arshai",
            "X-Title": "Arshai Framework",
        }
        # Custom headers override defaults
        return {**default_headers, **self.headers}


class AIGatewayLLM(BaseLLMClient):
    """
    OpenAI-compatible gateway/proxy LLM client.

    This client works with any service that implements the OpenAI API specification,
    providing maximum flexibility for different deployment scenarios.

    Benefits:
    - **Gateway Agnostic**: Works with Cloudflare, LiteLLM, custom proxies
    - **Flexible Authentication**: Supports various auth methods via headers and API keys
    - **Custom Headers**: Add organization IDs, referrers, versions, etc.
    - **Regional Support**: Connect to regional or custom endpoints
    - **Enterprise Ready**: Compatible with internal gateways and proxies

    Architecture:
    - Follows the reference implementation patterns from Google Gemini client
    - Implements full ILLM interface with chat() and stream() methods
    - Supports function calling (regular tools + background tasks)
    - Handles structured outputs via tool calling
    - Provides comprehensive usage tracking

    Examples:
        # Basic usage with Cloudflare
        config = AIGatewayConfig(
            base_url="https://gateway.ai.cloudflare.com/v1/account/gateway/compat",
            api_key=os.getenv("CLOUDFLARE_GATEWAY_TOKEN"),
            model="anthropic/claude-sonnet-4-5"
        )
        llm = AIGatewayLLM(config)

        # Usage with custom headers
        config = AIGatewayConfig(
            base_url="https://api.mycompany.com/v1",
            api_key=os.getenv("COMPANY_API_KEY"),
            model="custom-model",
            headers={
                "X-Organization": "org-123",
                "X-Department": "engineering"
            }
        )
        llm = AIGatewayLLM(config)

        # Make a request
        response = await llm.chat(ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Hello!"
        ))
    """

    config: AIGatewayConfig

    def __del__(self) -> None:
        """Cleanup connections when the client is destroyed."""
        self.close()

    def close(self) -> None:
        """Close the client and cleanup connections."""
        try:
            if hasattr(self._client, '_client') and hasattr(self._client._client, 'close'):
                self._client._client.close()
                self.logger.info("Closed gateway httpx client")
            elif hasattr(self._client, 'close'):
                self._client.close()
                self.logger.info("Closed gateway client")
        except Exception as e:
            self.logger.warning(f"Error closing gateway client: {e}")

    def _initialize_client(self) -> Any:
        """
        Initialize the OpenAI client configured for the gateway.

        Returns:
            OpenAI client instance configured for the gateway

        Raises:
            ValueError: If required configuration is missing
        """
        # Get effective headers (defaults + custom)
        headers = self.config.get_effective_headers()

        self.logger.info(f"Creating OpenAI-compatible gateway client")
        self.logger.info(f"Gateway endpoint: {self.config.base_url}")
        self.logger.info(f"Model: {self.config.model}")

        try:
            from arshai.clients.safe_http_client import SafeHttpClientFactory

            import httpx
            httpx_version = getattr(httpx, '__version__', '0.0.0')

            limits_config = SafeHttpClientFactory._get_safe_limits_config(httpx_version)
            timeout_config = SafeHttpClientFactory._get_safe_timeout_config(httpx_version)
            additional_config = SafeHttpClientFactory._get_additional_httpx_config(httpx_version)

            safe_http_client = httpx.Client(
                limits=limits_config,
                timeout=timeout_config,
                **additional_config
            )

            # Build OpenAI client with all configuration
            # gateway_token will be sent as Authorization: Bearer <token>
            client_kwargs = {
                "api_key": self.config.gateway_token,
                "base_url": self.config.base_url,
                "default_headers": headers,
                "http_client": safe_http_client,
                "max_retries": self.config.max_retries,
            }

            # Add organization/project if specified (OpenAI compatibility)
            if self.config.organization:
                client_kwargs["organization"] = self.config.organization
            if self.config.project:
                client_kwargs["project"] = self.config.project

            client = OpenAI(**client_kwargs)

            self.logger.info("Gateway client initialized successfully")
            return client

        except ImportError as e:
            self.logger.warning(f"Safe HTTP client not available: {e}, using default client")
            client_kwargs = {
                "api_key": self.config.gateway_token,
                "base_url": self.config.base_url,
                "default_headers": headers,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            if self.config.organization:
                client_kwargs["organization"] = self.config.organization
            if self.config.project:
                client_kwargs["project"] = self.config.project

            return OpenAI(**client_kwargs)

        except Exception as e:
            self.logger.error(f"Failed to create gateway client: {e}")
            # Fallback to basic client
            client_kwargs = {
                "api_key": self.config.gateway_token,
                "base_url": self.config.base_url,
                "default_headers": headers,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            return OpenAI(**client_kwargs)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _extract_and_standardize_usage(self, response: Any) -> Dict[str, Any]:
        """Extract and standardize usage metadata from response."""
        if not hasattr(response, 'usage') or not response.usage:
            return {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "thinking_tokens": 0, "tool_calling_tokens": 0,
                "provider": "gateway", "model": self.config.model,
                "request_id": getattr(response, 'id', None)
            }

        usage = response.usage

        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', 0)

        # Extract reasoning tokens if available
        thinking_tokens = 0
        if hasattr(usage, 'completion_tokens_details'):
            details = usage.completion_tokens_details
            thinking_tokens = getattr(details, 'reasoning_tokens', 0) or 0

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "thinking_tokens": thinking_tokens,
            "tool_calling_tokens": 0,
            "provider": "gateway",
            "model": self.config.model,
            "request_id": getattr(response, 'id', None)
        }

    def _create_openai_messages(self, input: ILLMInput) -> List[Dict[str, Any]]:
        """
        Create OpenAI-compatible messages with images support for gateway passthrough.

        Tasks 6.1-6.4: Build OpenAI-compatible content array for gateway passthrough,
        ensure data URL format, maintain backward compatibility.

        Args:
            input: ILLMInput with optional images_base64

        Returns:
            List of OpenAI-formatted messages with multimodal content
        """
        messages = [
            {"role": "system", "content": input.system_prompt}
        ]

        # Task 6.2: Check for images_base64
        if input.images_base64:
            # Build content array with text and image_url objects
            content = [{"type": "text", "text": input.user_message}]

            # Task 6.3: Ensure data URL format
            for img_data in input.images_base64:
                if not img_data.startswith('data:'):
                    img_data = f"data:image/jpeg;base64,{img_data}"

                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_data}
                })

            messages.append({"role": "user", "content": content})
        else:
            # Task 6.4: Text-only path for backward compatibility
            messages.append({"role": "user", "content": input.user_message})

        return messages

    def _convert_callables_to_provider_format(self, functions: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """Convert python callables to OpenAI-compatible function declarations."""
        openai_tools = []

        for name, func in functions.items():
            try:
                sig = inspect.signature(func)
                description = func.__doc__ or f"Execute {name} function"
                description = description.strip()

                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue

                    param_type = "string"
                    if param.annotation != inspect.Parameter.empty:
                        param_type = self._python_type_to_json_schema_type(param.annotation)

                    param_def = {
                        "type": param_type,
                        "description": f"{param_name} parameter"
                    }

                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)
                    else:
                        param_def["description"] += f" (default: {param.default})"

                    properties[param_name] = param_def

                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False
                        }
                    }
                })

            except Exception as e:
                self.logger.warning(f"Failed to convert function {name}: {e}")
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Execute {name} function",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False
                        }
                    }
                })

        return openai_tools

    def _python_type_to_json_schema_type(self, python_type) -> str:
        """Convert Python type annotations to JSON schema types."""
        if python_type == str:
            return "string"
        elif python_type == int:
            return "integer"
        elif python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        elif python_type == list or (hasattr(python_type, '__origin__') and python_type.__origin__ == list):
            return "array"
        elif python_type == dict or (hasattr(python_type, '__origin__') and python_type.__origin__ == dict):
            return "object"
        else:
            return "string"

    def _create_structure_function_openai(self, structure_type) -> Dict[str, Any]:
        """Create OpenAI function definition for structured output."""
        function_name = structure_type.__name__.lower()
        description = structure_type.__doc__ or f"Create a {structure_type.__name__} response"

        if hasattr(structure_type, 'model_json_schema'):
            schema = structure_type.model_json_schema()
        elif hasattr(structure_type, '__annotations__'):
            properties = {}
            required = []
            for field_name, field_type in structure_type.__annotations__.items():
                properties[field_name] = {
                    "type": self._python_type_to_json_schema_type(field_type),
                    "description": f"{field_name} field"
                }
                required.append(field_name)
            schema = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            }
        else:
            schema = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }

        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description,
                "parameters": schema
            }
        }

    def _extract_function_calls_from_response(self, response) -> List:
        """Extract function calls from OpenAI response."""
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return message.tool_calls
        return []

    def _process_function_calls_for_orchestrator(self, tool_calls, input: ILLMInput) -> tuple:
        """Process function calls and prepare them for the orchestrator."""
        function_calls = []
        structured_response = None

        for i, tool_call in enumerate(tool_calls):
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse function arguments for {function_name}")
                function_args = {}

            # Check if it's the structure function
            if input.structure_type and function_name == input.structure_type.__name__.lower():
                try:
                    structured_response = input.structure_type(**function_args)
                    self.logger.info(f"Created structured response: {function_name}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error creating structured response: {e}")
                    continue

            call_id = f"{function_name}_{i}"

            if function_name in (input.background_tasks or {}):
                function_calls.append(FunctionCall(
                    name=function_name,
                    args=function_args,
                    call_id=call_id,
                    is_background=True
                ))
            elif function_name in (input.regular_functions or {}):
                function_calls.append(FunctionCall(
                    name=function_name,
                    args=function_args,
                    call_id=call_id,
                    is_background=False
                ))
            else:
                self.logger.warning(f"Function {function_name} not found in available functions")

        return function_calls, structured_response

    # ========================================================================
    # FRAMEWORK-REQUIRED ABSTRACT METHODS
    # ========================================================================

    async def _chat_simple(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle simple chat without tools or background tasks."""
        messages = self._create_openai_messages(input)

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
        }

        if input.structure_type:
            structure_function = self._create_structure_function_openai(input.structure_type)
            kwargs["tools"] = [structure_function]
            function_name = input.structure_type.__name__.lower()
            kwargs["messages"][0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE.format(function_name=function_name)

        response = self._client.chat.completions.create(**kwargs)
        usage = self._extract_and_standardize_usage(response)

        if input.structure_type:
            tool_calls = self._extract_function_calls_from_response(response)
            if tool_calls:
                _, structured_response = self._process_function_calls_for_orchestrator(tool_calls, input)
                if structured_response is not None:
                    return {"llm_response": structured_response, "usage": usage}
            return {"llm_response": f"Failed to generate structured response", "usage": usage}

        message = response.choices[0].message
        return {"llm_response": message.content, "usage": usage}

    async def _chat_with_functions(self, input: ILLMInput) -> Dict[str, Any]:
        """Handle complex chat with tools and/or background tasks."""
        messages = self._create_openai_messages(input)

        openai_tools = []

        if input.structure_type:
            structure_function = self._create_structure_function_openai(input.structure_type)
            openai_tools.append(structure_function)
            function_name = input.structure_type.__name__.lower()
            messages[0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE.format(function_name=function_name)

        all_functions = {}
        if input.regular_functions:
            all_functions.update(input.regular_functions)
        if input.background_tasks:
            all_functions.update(input.background_tasks)

        if all_functions:
            openai_tools.extend(self._convert_callables_to_provider_format(all_functions))

        current_turn = 0
        accumulated_usage = None

        while current_turn < input.max_turns:
            self.logger.info(f"Function calling turn: {current_turn}")

            try:
                start_time = time.time()

                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
                    "tools": openai_tools if openai_tools else None,
                }

                response = self._client.chat.completions.create(**kwargs)
                self.logger.info(f"Response time: {time.time() - start_time:.2f}s")

                if hasattr(response, "usage") and response.usage:
                    current_usage = self._extract_and_standardize_usage(response)
                    accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)

                message = response.choices[0].message

                tool_calls = self._extract_function_calls_from_response(response)
                if tool_calls:
                    self.logger.info(f"Turn {current_turn}: Found {len(tool_calls)} function calls")

                    function_calls, structured_response = self._process_function_calls_for_orchestrator(tool_calls, input)

                    if structured_response is not None:
                        return {"llm_response": structured_response, "usage": accumulated_usage}

                    if function_calls:
                        execution_input = FunctionExecutionInput(
                            function_calls=function_calls,
                            available_functions=input.regular_functions or {},
                            available_background_tasks=input.background_tasks or {}
                        )

                        execution_result = await self._execute_functions_with_orchestrator(execution_input)
                        self._add_function_results_to_messages(execution_result, messages)

                        regular_function_calls = [call for call in function_calls if not call.is_background]
                        if regular_function_calls:
                            current_turn += 1
                            continue

                if input.structure_type:
                    return {"llm_response": "Failed to generate structured response", "usage": accumulated_usage}

                if message.content:
                    return {"llm_response": message.content, "usage": accumulated_usage}

            except Exception as e:
                self.logger.error(f"Error in chat_with_functions turn {current_turn}: {e}")
                return {"llm_response": f"An error occurred: {e}", "usage": accumulated_usage}

        return {"llm_response": "Maximum function calling turns reached", "usage": accumulated_usage}

    def _add_function_results_to_messages(self, execution_result: Dict, messages: List[Dict]) -> None:
        """Add function execution results to messages."""
        for result in execution_result.get('regular_results', []):
            messages.append({
                "role": "function",
                "name": result['name'],
                "content": f"Function '{result['name']}' returned: {result['result']}"
            })

        for bg_message in execution_result.get('background_initiated', []):
            messages.append({
                "role": "user",
                "content": f"Background task initiated: {bg_message}"
            })

        if execution_result.get('regular_results'):
            messages.append({
                "role": "user",
                "content": f"All {len(execution_result['regular_results'])} function(s) completed."
            })

    async def _stream_simple(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle simple streaming without tools."""
        messages = self._create_openai_messages(input)

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
            "stream": True,
        }

        if input.structure_type:
            structure_function = self._create_structure_function_openai(input.structure_type)
            kwargs["tools"] = [structure_function]
            function_name = input.structure_type.__name__.lower()
            kwargs["messages"][0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE.format(function_name=function_name)

        accumulated_usage = None
        collected_text = ""
        collected_tool_calls = []

        for chunk in self._client.chat.completions.create(**kwargs):
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                current_usage = self._extract_and_standardize_usage(chunk)
                accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, 'content') and delta.content is not None:
                collected_text += delta.content
                if not input.structure_type:
                    yield {"llm_response": collected_text}

            if hasattr(delta, 'tool_calls') and delta.tool_calls and input.structure_type:
                for i, tool_delta in enumerate(delta.tool_calls):
                    if i >= len(collected_tool_calls):
                        collected_tool_calls.append({
                            "id": tool_delta.id or "",
                            "function": {"name": "", "arguments": ""}
                        })

                    current_tool_call = collected_tool_calls[i]

                    if tool_delta.id:
                        current_tool_call["id"] = tool_delta.id

                    if hasattr(tool_delta, 'function'):
                        if tool_delta.function.name:
                            current_tool_call["function"]["name"] = tool_delta.function.name
                        if tool_delta.function.arguments:
                            current_tool_call["function"]["arguments"] += tool_delta.function.arguments

                            if current_tool_call["function"]["name"] == input.structure_type.__name__.lower():
                                is_complete, fixed_json = is_json_complete(current_tool_call["function"]["arguments"])
                                if is_complete:
                                    try:
                                        structured_response = parse_to_structure(fixed_json, input.structure_type)
                                        yield {"llm_response": structured_response}
                                    except Exception:
                                        continue

        yield {"llm_response": None, "usage": accumulated_usage}

    async def _stream_with_functions(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle complex streaming with tools and/or background tasks."""
        messages = self._create_openai_messages(input)

        openai_tools = []

        if input.structure_type:
            structure_function = self._create_structure_function_openai(input.structure_type)
            openai_tools.append(structure_function)
            function_name = input.structure_type.__name__.lower()
            messages[0]["content"] += STRUCTURE_INSTRUCTIONS_TEMPLATE.format(function_name=function_name)

        all_functions = {}
        if input.regular_functions:
            all_functions.update(input.regular_functions)
        if input.background_tasks:
            all_functions.update(input.background_tasks)

        if all_functions:
            openai_tools.extend(self._convert_callables_to_provider_format(all_functions))

        current_turn = 0
        accumulated_usage = None

        while current_turn < input.max_turns:
            self.logger.info(f"Stream function calling turn: {current_turn}")

            try:
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens if self.config.max_tokens else None,
                    "tools": openai_tools if openai_tools else None,
                    "stream": True,
                }

                streaming_state = StreamingExecutionState()
                collected_text = ""
                tool_calls_in_progress = {}

                for chunk in self._client.chat.completions.create(**kwargs):
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        current_usage = self._extract_and_standardize_usage(chunk)
                        accumulated_usage = self._accumulate_usage_safely(current_usage, accumulated_usage)

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    if hasattr(delta, 'content') and delta.content is not None:
                        collected_text += delta.content
                        if not input.structure_type:
                            yield {"llm_response": collected_text}

                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_delta in delta.tool_calls:
                            tool_index = tool_delta.index

                            if tool_index not in tool_calls_in_progress:
                                tool_calls_in_progress[tool_index] = {
                                    "id": "",
                                    "function": {"name": "", "arguments": ""}
                                }

                            current_tool_call = tool_calls_in_progress[tool_index]

                            if tool_delta.id:
                                current_tool_call["id"] = tool_delta.id

                            if hasattr(tool_delta, 'function'):
                                if tool_delta.function.name:
                                    current_tool_call["function"]["name"] = tool_delta.function.name
                                if tool_delta.function.arguments:
                                    current_tool_call["function"]["arguments"] += tool_delta.function.arguments

                            # Handle structured output
                            if (input.structure_type and
                                current_tool_call["function"]["name"] and
                                current_tool_call["function"]["name"].lower() == input.structure_type.__name__.lower()):
                                is_complete, fixed_json = is_json_complete(current_tool_call["function"]["arguments"])
                                if is_complete:
                                    try:
                                        structured_response = parse_to_structure(fixed_json, input.structure_type)
                                        yield {"llm_response": structured_response, "usage": accumulated_usage}
                                    except Exception as e:
                                        self.logger.error(f"Failed to parse structure: {e}")

                            # Progressive function execution
                            if self._is_function_complete(current_tool_call["function"]):
                                function_name = current_tool_call["function"]["name"]

                                if (input.structure_type and
                                    function_name.lower() == input.structure_type.__name__.lower()):
                                    continue

                                function_call = FunctionCall(
                                    name=function_name,
                                    args=json.loads(current_tool_call["function"]["arguments"]) if current_tool_call["function"]["arguments"] else {},
                                    call_id=current_tool_call["id"] or f"{function_name}_{tool_index}",
                                    is_background=function_name in (input.background_tasks or {})
                                )

                                if not streaming_state.is_already_executed(function_call):
                                    self.logger.info(f"Executing function progressively: {function_call.name}")
                                    try:
                                        task = await self._execute_function_progressively(function_call, input)
                                        streaming_state.add_function_task(task, function_call)
                                    except Exception as e:
                                        self.logger.error(f"Progressive execution failed: {e}")

                # Gather progressive results
                if streaming_state.active_function_tasks:
                    execution_result = await self._gather_progressive_results(streaming_state.active_function_tasks)
                    self._add_function_results_to_messages(execution_result, messages)

                    regular_results = execution_result.get('regular_results', [])
                    if regular_results:
                        current_turn += 1
                        continue

                break

            except Exception as e:
                self.logger.error(f"Error in stream_with_functions turn {current_turn}: {e}")
                yield {"llm_response": f"An error occurred: {e}", "usage": accumulated_usage}
                return

        if current_turn >= input.max_turns:
            yield {"llm_response": "Maximum function calling turns reached", "usage": accumulated_usage}
        else:
            yield {"llm_response": None, "usage": accumulated_usage}
