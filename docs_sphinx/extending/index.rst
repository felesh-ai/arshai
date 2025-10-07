Extending Arshai
================

Comprehensive guides for extending the framework with custom components and contributing to the Arshai project.

.. toctree::
   :maxdepth: 2
   :caption: Extension Guides

   custom-agents
   custom-llm-clients

Overview
--------

Arshai is designed to be extended and customized for your specific needs. The framework provides clear extension points through:

- **Protocol-Based Interfaces**: Implement interfaces without inheritance constraints
- **Base Classes**: Extend foundation classes for common patterns
- **Factory Pattern**: Plugin your custom implementations seamlessly
- **Direct Instantiation**: Full control over component creation

**Extension Philosophy:**

The framework follows the principle of **developer authority** - giving you complete control over:

- Response formats and data structures
- Tool integration patterns
- Memory management strategies
- LLM provider selection and configuration
- System orchestration approaches

What You Can Extend
-------------------

**Agents**
   Create custom agents with specialized behaviors, tool integrations, and response patterns.

   - Extend ``BaseAgent`` for standard patterns
   - Implement ``IAgent`` protocol for full flexibility
   - Integrate with tools, memory, and workflows
   - See :doc:`custom-agents` for complete guide

**LLM Providers**
   Add support for new LLM providers or customize existing ones.

   - Extend ``BaseLLMClient`` for framework integration
   - Implement ``ILLM`` protocol for provider-specific logic
   - Support function calling, streaming, and structured output
   - See :doc:`custom-llm-clients` for complete guide

**Memory Backends**
   Implement custom memory storage solutions.

   - Implement ``IMemoryManager`` protocol
   - Support different memory types (short-term, long-term, working)
   - Integrate with various storage systems (Redis, PostgreSQL, etc.)
   - See :doc:`../framework/memory/index` for patterns

**Vector Databases**
   Add support for new vector database providers.

   - Implement ``IVectorDBClient`` protocol
   - Support dense and sparse vector operations
   - Enable hybrid search capabilities
   - See :doc:`../implementations/components/vector-databases` for examples

**Embeddings**
   Integrate new embedding models.

   - Implement ``IEmbedding`` protocol
   - Support batch processing
   - Handle dense and sparse embeddings
   - See :doc:`../implementations/components/embeddings` for patterns

**Workflows**
   Build custom orchestration systems.

   - Create custom workflow nodes
   - Implement routing logic
   - Manage state across agents
   - See :doc:`../implementations/orchestration/building-your-own` for patterns

Extension Patterns
------------------

Protocol Implementation
~~~~~~~~~~~~~~~~~~~~~~~

All Arshai interfaces are protocols, enabling duck-typed implementations:

.. code-block:: python

   from arshai.core.interfaces import IAgent, IAgentInput

   class MyCustomAgent:
       """Custom agent without inheriting from BaseAgent"""

       async def process(self, input: IAgentInput) -> dict:
           """Implement the required process method"""
           return {"response": "Custom logic here"}

   # MyCustomAgent automatically satisfies IAgent protocol

Base Class Extension
~~~~~~~~~~~~~~~~~~~~

Extend base classes for common infrastructure:

.. code-block:: python

   from arshai.agents.base import BaseAgent

   class SpecializedAgent(BaseAgent):
       """Extend BaseAgent for common patterns"""

       def __init__(self, llm_client, system_prompt, **kwargs):
           super().__init__(llm_client, system_prompt)
           self.custom_config = kwargs.get('custom_config')

       async def process(self, input: IAgentInput) -> dict:
           # Use inherited self.llm_client
           # Use inherited self.system_prompt
           # Implement custom logic
           pass

Factory Integration
~~~~~~~~~~~~~~~~~~~

Register custom implementations with the factory system:

.. code-block:: python

   from arshai.config import Settings

   # Create custom component
   class MyLLMClient(BaseLLMClient):
       # Implementation
       pass

   # Use via settings
   settings = Settings()
   llm = MyLLMClient(ILLMConfig(model="custom-model"))

   # Or direct instantiation
   from arshai.core.interfaces import ILLMConfig
   llm = MyLLMClient(ILLMConfig(model="custom-model"))

Quick Start Guides
------------------

Creating a Custom Agent
~~~~~~~~~~~~~~~~~~~~~~~

**Simple Example:**

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces import IAgentInput, ILLMInput

   class SentimentAgent(BaseAgent):
       """Agent that analyzes sentiment"""

       async def process(self, input: IAgentInput) -> dict:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Analyze sentiment: {input.message}"
           )

           result = await self.llm_client.chat(llm_input)

           return {
               "sentiment": self._extract_sentiment(result['llm_response']),
               "original_message": input.message
           }

       def _extract_sentiment(self, response: str) -> str:
           # Custom sentiment extraction logic
           return "positive"  # Simplified

See :doc:`custom-agents` for comprehensive agent extension guide.

Adding an LLM Provider
~~~~~~~~~~~~~~~~~~~~~~

**Minimal Example:**

.. code-block:: python

   from arshai.llms.base_llm_client import BaseLLMClient
   from arshai.core.interfaces import ILLMConfig, ILLMInput

   class MyProviderClient(BaseLLMClient):
       """Custom LLM provider implementation"""

       def _initialize_client(self):
           """Initialize provider client"""
           import my_provider
           return my_provider.Client(api_key="...")

       def _convert_callables_to_provider_format(self, functions):
           """Convert functions to provider format"""
           return [{"name": name, "function": func} for name, func in functions.items()]

       async def _chat_simple(self, input: ILLMInput):
           """Handle simple chat"""
           response = await self._client.chat(
               system=input.system_prompt,
               user=input.user_message
           )
           return {"llm_response": response.text, "usage": {...}}

       async def _chat_with_functions(self, input: ILLMInput):
           """Handle function calling"""
           # Multi-turn function calling implementation
           pass

       async def _stream_simple(self, input: ILLMInput):
           """Handle streaming"""
           async for chunk in self._client.stream(...):
               yield {"llm_response": chunk.text}

       async def _stream_with_functions(self, input: ILLMInput):
           """Handle streaming with functions"""
           # Streaming function calling implementation
           pass

See :doc:`custom-llm-clients` for comprehensive LLM provider integration guide.

Best Practices
--------------

**Follow Framework Patterns**

1. **Use Protocols**: Implement interfaces through duck typing
2. **Extend Base Classes**: Leverage common infrastructure when appropriate
3. **Direct Instantiation**: Create components explicitly
4. **Type Safety**: Use type hints throughout your code
5. **Error Handling**: Implement robust error handling

**Code Quality**

1. **Testing**: Write comprehensive tests for your extensions
2. **Documentation**: Document your custom components
3. **Logging**: Use appropriate logging for debugging
4. **Performance**: Optimize for your use case
5. **Maintainability**: Keep code clean and well-organized

**Integration**

1. **Factory Support**: Make components discoverable
2. **Configuration**: Support configuration via DTOs
3. **Compatibility**: Ensure compatibility with existing components
4. **Versioning**: Handle version compatibility

Testing Your Extensions
-----------------------

**Unit Testing Agents:**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock
   from arshai.core.interfaces import IAgentInput

   @pytest.mark.asyncio
   async def test_custom_agent():
       # Mock LLM client
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Test response",
           "usage": {"total_tokens": 100}
       }

       # Test agent
       agent = MyCustomAgent(mock_llm, "Test prompt")
       result = await agent.process(IAgentInput(message="Test"))

       assert result is not None
       assert "response" in result
       mock_llm.chat.assert_called_once()

**Integration Testing:**

.. code-block:: python

   @pytest.mark.asyncio
   async def test_agent_with_real_llm():
       from arshai.llms.openai_client import OpenAIClient
       from arshai.core.interfaces import ILLMConfig

       llm = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))
       agent = MyCustomAgent(llm, "You are a helpful assistant")

       result = await agent.process(IAgentInput(message="Hello"))

       assert result is not None
       assert isinstance(result, dict)

**Testing LLM Clients:**

.. code-block:: python

   @pytest.mark.asyncio
   async def test_llm_client():
       from arshai.core.interfaces import ILLMInput

       client = MyProviderClient(ILLMConfig(model="test-model"))

       result = await client.chat(ILLMInput(
           system_prompt="You are helpful",
           user_message="Hello"
       ))

       assert "llm_response" in result
       assert "usage" in result

Contributing to Arshai
----------------------

We welcome contributions to the Arshai framework! Here's how you can help:

**Code Contributions**

1. **Fork the Repository**: https://github.com/felesh-ai/arshai
2. **Create a Branch**: ``git checkout -b feature/your-feature``
3. **Write Code**: Follow the code standards below
4. **Write Tests**: Ensure comprehensive test coverage
5. **Submit PR**: Create a pull request with clear description

**Code Standards**

**Style Guide:**

- **Black**: Use Black for code formatting (``poetry run black .``)
- **isort**: Sort imports with isort (``poetry run isort .``)
- **Type Hints**: Use type hints throughout
- **Docstrings**: Document all public methods and classes

**Testing Requirements:**

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Coverage**: Aim for >80% code coverage
- **Commands**: Run ``poetry run pytest --cov=arshai``

**Quality Checks:**

.. code-block:: bash

   # Format code
   poetry run black .
   poetry run isort .

   # Type checking
   poetry run mypy arshai/

   # Security analysis
   poetry run bandit -r arshai/

   # Run tests
   poetry run pytest --cov=arshai

**Documentation Contributions**

1. **API Documentation**: Add/update docstrings
2. **Guides**: Improve or add documentation guides
3. **Examples**: Contribute working examples
4. **Tutorials**: Create tutorial content

**Issue Reporting**

Report bugs or request features:

- **GitHub Issues**: https://github.com/felesh-ai/arshai/issues
- **Include Details**: Provide code samples, error messages, environment info
- **Reproducible**: Provide steps to reproduce issues

**Community Guidelines**

- **Be Respectful**: Treat all community members with respect
- **Be Constructive**: Provide constructive feedback
- **Be Patient**: Maintainers are volunteers
- **Share Knowledge**: Help others in the community

Development Setup
-----------------

**Prerequisites:**

- Python 3.9 or higher
- Poetry for dependency management
- Git

**Setup Steps:**

1. **Clone Repository:**

   .. code-block:: bash

      git clone https://github.com/felesh-ai/arshai.git
      cd arshai

2. **Install Dependencies:**

   .. code-block:: bash

      # Install all dependencies including dev tools
      poetry install -E all

3. **Set Up Pre-commit Hooks:**

   .. code-block:: bash

      poetry run pre-commit install

4. **Run Tests:**

   .. code-block:: bash

      poetry run pytest

5. **Build Documentation:**

   .. code-block:: bash

      cd docs_sphinx && make html

**Project Structure:**

.. code-block:: text

   arshai/
   ├── arshai/                  # Main package
   │   ├── agents/              # Agent implementations
   │   ├── core/                # Core interfaces and types
   │   ├── llms/                # LLM client implementations
   │   ├── memory/              # Memory implementations
   │   ├── workflows/           # Workflow system
   │   └── ...
   ├── tests/                   # Test suite
   │   ├── unit/                # Unit tests
   │   └── integration/         # Integration tests
   ├── docs_sphinx/             # Sphinx documentation
   ├── examples/                # Example applications
   └── pyproject.toml           # Poetry configuration

Resources
---------

**Documentation:**

- :doc:`../framework/index` - Framework guides
- :doc:`../reference/index` - API reference
- :doc:`../tutorials/index` - Complete tutorials
- :doc:`../implementations/index` - Implementation guides

**Example Code:**

- :doc:`../framework/agents/examples/index` - Agent examples
- ``examples/`` directory in repository
- Tutorial code in documentation

**GitHub:**

- **Repository**: https://github.com/felesh-ai/arshai
- **Issues**: https://github.com/felesh-ai/arshai/issues
- **Pull Requests**: https://github.com/felesh-ai/arshai/pulls

**Architecture Documents:**

- ``CLAUDE.md`` - Development guidelines for Claude Code
- LLM client architecture standards
- Interface-driven design principles

Next Steps
----------

Ready to extend Arshai? Choose your path:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Guide
     - Description
   * - :doc:`custom-agents`
     - Build custom agents with specialized behaviors
   * - :doc:`custom-llm-clients`
     - Add support for new LLM providers
   * - :doc:`../framework/memory/index`
     - Implement custom memory backends
   * - :doc:`../implementations/orchestration/building-your-own`
     - Create custom orchestration systems

Happy extending! 🚀
