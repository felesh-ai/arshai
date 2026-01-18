"""
Cloudflare AI Gateway LLM implementation.

This client routes LLM requests through Cloudflare AI Gateway's unified
/compat/chat/completions endpoint, enabling multi-provider support with
a single gateway for caching, rate limiting, and observability.

Supported providers through the unified endpoint:
- OpenAI (gpt-4o, gpt-4-turbo, etc.)
- Anthropic (claude-sonnet-4-5, claude-opus-4, etc.)
- Google AI Studio (gemini-2.0-flash, gemini-1.5-pro, etc.)
- Groq (llama-3-70b, mixtral-8x7b, etc.)
- Mistral (mistral-large, mistral-medium, etc.)
- Cohere (command-r-plus, command-r, etc.)
- xAI (grok-2, etc.)
- DeepSeek (deepseek-chat, deepseek-coder, etc.)
- OpenRouter (any model via openrouter provider)
- And more...
"""

import os
import json
import time
import inspect
from typing import Dict, Any, TypeVar, AsyncGenerator, List, Callable, Optional
from pydantic import Field

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


class CloudflareGatewayLLMConfig(ILLMConfig):
    """
    Configuration for Cloudflare AI Gateway LLM client (BYOK mode).

    This client uses Cloudflare AI Gateway with stored provider API keys (BYOK).
    Only the gateway token is required - provider keys are managed in the gateway.

    The gateway routes requests to multiple providers through a unified endpoint.
    Users specify provider and model separately for clarity.

    Examples:
        # Basic usage
        config = CloudflareGatewayLLMConfig(
            account_id="your-account-id",
            gateway_id="your-gateway-id",
            gateway_token="your-gateway-token",  # or set CLOUDFLARE_GATEWAY_TOKEN env var
            provider="anthropic",
            model="claude-sonnet-4-5",
        )

        # Custom Cloudflare endpoint (e.g., regional)
        config = CloudflareGatewayLLMConfig(
            account_id="your-account-id",
            gateway_id="your-gateway-id",
            gateway_token="your-gateway-token",
            provider="openai",
            model="gpt-4o",
            cloudflare_base_url="https://gateway.ai.cloudflare.cn",
        )
    """

    account_id: str = Field(
        description="Cloudflare account ID"
    )
    gateway_id: str = Field(
        description="Cloudflare AI Gateway ID"
    )
    provider: str = Field(
        description="LLM provider (openai, anthropic, google-ai-studio, groq, openrouter, etc.)"
    )
    model: str = Field(
        description="Model name (gpt-4o, claude-sonnet-4-5, gemini-2.0-flash, etc.)"
    )

    # Gateway authentication (BYOK mode - provider keys stored in gateway)
    gateway_token: Optional[str] = Field(
        default=None,
        description="Cloudflare AI Gateway token. Falls back to CLOUDFLARE_GATEWAY_TOKEN env var."
    )

    # Cloudflare Gateway base URL (configurable for custom/regional endpoints)
    cloudflare_base_url: str = Field(
        default="https://gateway.ai.cloudflare.com",
        description="Cloudflare AI Gateway base URL. Override for custom or regional endpoints."
    )

    # Standard LLM settings (inherited from ILLMConfig)
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)

    # Request settings
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    @property
    def base_url(self) -> str:
        """Get the base URL for Cloudflare AI Gateway."""
        return f"{self.cloudflare_base_url}/v1/{self.account_id}/{self.gateway_id}"

    @property
    def compat_base_url(self) -> str:
        """Get the unified /compat base URL for OpenAI client."""
        return f"{self.base_url}/compat"

    @property
    def provider_base_url(self) -> str:
        """Get the provider-specific base URL for OpenAI client."""
        return f"{self.base_url}/{self.provider}/v1"

    @property
    def full_model_name(self) -> str:
        """Get the full model name in {provider}/{model} format for /compat endpoint."""
        return f"{self.provider}/{self.model}"


class CloudflareGatewayLLM(BaseLLMClient):
    """
    Cloudflare AI Gateway LLM client using the unified /compat endpoint (BYOK mode).

    This client uses Cloudflare AI Gateway with stored provider API keys (BYOK).
    Only the gateway token is required - provider keys are managed in the gateway.

    Benefits:
    - Centralized caching and rate limiting
    - Unified logging and analytics
    - Easy provider switching by changing provider/model
    - Secure API key management (keys stored in gateway, not in code)

    The /compat endpoint uses model format: "{provider}/{model}"
    Example: "openrouter/openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"

    Examples:
        config = CloudflareGatewayLLMConfig(
            account_id="xxx",
            gateway_id="my-gateway",
            gateway_token="your-gateway-token",
            provider="anthropic",
            model="claude-sonnet-4-5",
        )

        llm = CloudflareGatewayLLM(config)

        response = await llm.chat(ILLMInput(
            system_prompt="You are a helpful assistant.",
            user_message="Hello!"
        ))
    """

    config: CloudflareGatewayLLMConfig

    def __del__(self) -> None:
        """Cleanup connections when the client is destroyed."""
        self.close()

    def close(self) -> None:
        """Close the client and cleanup connections."""
        try:
            if hasattr(self._client, '_client') and hasattr(self._client._client, 'close'):
                self._client._client.close()
                self.logger.info("Closed Cloudflare Gateway httpx client")
            elif hasattr(self._client, 'close'):
                self._client.close()
                self.logger.info("Closed Cloudflare Gateway client")
        except Exception as e:
            self.logger.warning(f"Error closing Cloudflare Gateway client: {e}")

    def _initialize_client(self) -> Any:
        """
        Initialize the OpenAI client configured for Cloudflare AI Gateway (BYOK mode).

        Uses gateway token for authentication. Provider API keys must be stored
        in the Cloudflare AI Gateway configuration.

        Returns:
            OpenAI client instance configured for Cloudflare Gateway

        Raises:
            ValueError: If gateway token is not available
        """
        # Get gateway token (required for BYOK mode)
        gateway_token = self.config.gateway_token or os.environ.get("CLOUDFLARE_GATEWAY_TOKEN")
        if not gateway_token:
            raise ValueError(
                "Gateway token is required for Cloudflare AI Gateway (BYOK mode). "
                "Set gateway_token in config or CLOUDFLARE_GATEWAY_TOKEN environment variable."
            )

        # Use gateway token as API key - Cloudflare recognizes it and uses stored provider keys
        api_key = gateway_token

        # Build headers (no cf-aig-authorization needed when using gateway token as api_key)
        default_headers = {
            "HTTP-Referer": "https://github.com/ArshaAI/arshai",
            "X-Title": "Arshai Framework",
        }

        try:
            from arshai.clients.safe_http_client import SafeHttpClientFactory

            self.logger.info(f"Creating Cloudflare Gateway client for {self.config.provider}")

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

            # Use unified /compat endpoint for multi-provider support
            client = OpenAI(
                api_key=api_key,
                base_url=self.config.compat_base_url,
                default_headers=default_headers,
                http_client=safe_http_client,
                max_retries=self.config.max_retries,
            )

            self.logger.info(
                f"Cloudflare Gateway client initialized (BYOK mode) - "
                f"Endpoint: /compat, Model: {self.config.full_model_name}"
            )
            return client

        except ImportError as e:
            self.logger.warning(f"Safe HTTP client not available: {e}, using default client")
            return OpenAI(
                api_key=api_key,
                base_url=self.config.compat_base_url,
                default_headers=default_headers,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        except Exception as e:
            self.logger.error(f"Failed to create Cloudflare Gateway client: {e}")
            return OpenAI(
                api_key=api_key,
                base_url=self.config.compat_base_url,
                default_headers=default_headers,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )

    # ========================================================================
    # PROVIDER-SPECIFIC HELPER METHODS
    # ========================================================================

    def _extract_and_standardize_usage(self, response: Any) -> Dict[str, Any]:
        """Extract and standardize usage metadata from response."""
        if not hasattr(response, 'usage') or not response.usage:
            return {
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "thinking_tokens": 0, "tool_calling_tokens": 0,
                "provider": self.config.provider, "model": self.config.full_model_name,
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
            "provider": self.config.provider,
            "model": self.config.full_model_name,
            "request_id": getattr(response, 'id', None)
        }

    def _create_openai_messages(self, input: ILLMInput) -> List[Dict[str, Any]]:
        """Create OpenAI-compatible messages from input."""
        return [
            {"role": "system", "content": input.system_prompt},
            {"role": "user", "content": input.user_message}
        ]

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
            "model": self.config.full_model_name,  # Use {provider}/{model} for /compat
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
                    "model": self.config.full_model_name,
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
            "model": self.config.full_model_name,
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
                    "model": self.config.full_model_name,
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
