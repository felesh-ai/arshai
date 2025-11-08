Agent Examples
==============

This section provides comprehensive documentation for all agent examples in the Arshai framework. Each example demonstrates specific concepts and patterns for building agents.

.. note::
   The quickstart and comprehensive guides have moved to :doc:`/getting-started/index` for better discoverability.

.. toctree::
   :maxdepth: 2
   :caption: Focused Learning Examples
   
   01-basic-usage
   02-custom-agents
   03-memory-patterns
   04-tool-integration

.. note::
   Agent composition patterns (multi-agent systems) have moved to :doc:`/framework/building-systems/composition-patterns` as they demonstrate Layer 3 system building concepts.

Learning Path
-------------

**For Beginners**:

1. **Getting Started**: See :doc:`/getting-started/quickstart` for 5-minute interactive demo
2. :doc:`01-basic-usage` - Foundation concepts and simple patterns (234 lines)
3. :doc:`02-custom-agents` - Specialized agents with custom return types (337 lines)

**For Intermediate Developers**:

4. :doc:`03-memory-patterns` - Conversation context and memory management (283 lines)
5. :doc:`04-tool-integration` - External function integration and background tasks (529 lines)

**For Advanced Users**:

6. **Agent Composition**: See :doc:`/framework/building-systems/composition-patterns` for multi-agent systems

**For Complete Reference**:

- See :doc:`/getting-started/comprehensive-guide` - All patterns in one comprehensive tutorial

Example Comparison
------------------

.. list-table:: Example Overview
   :header-rows: 1
   :widths: 20 10 20 50

   * - Example
     - Lines
     - Focus
     - Best For
   * - **01-basic-usage**
     - 234
     - Foundation concepts
     - Understanding core patterns
   * - **02-custom-agents**
     - 337
     - Specialized agents
     - Custom implementations
   * - **03-memory-patterns**
     - 283
     - Memory management
     - Stateful conversations
   * - **04-tool-integration**
     - 529
     - Tool patterns
     - External integrations

Key Concepts Covered
--------------------

**Basic Concepts** (Examples 1-2):
- Extending ``BaseAgent``
- Implementing the ``process()`` method
- Working with ``IAgentInput`` and ``ILLMInput``
- Direct instantiation patterns
- Custom return types and error handling

**Intermediate Concepts** (Examples 3-4):
- Memory management with ``WorkingMemoryAgent``
- Conversation context and metadata usage
- Tool integration with regular functions
- Background tasks for system coordination
- Dynamic tool selection patterns

**Advanced Concepts** (Examples 5-6):
- Multi-agent orchestration patterns
- Agent composition and communication
- Pipeline and mesh architectures
- Comprehensive testing strategies
- Performance and load testing

Running the Examples
--------------------

**Prerequisites**:

.. code-block:: bash

   # Required for all examples
   export OPENROUTER_API_KEY=your_key_here
   
   # Optional: For testing examples
   pip install pytest psutil

**Quick Start**:

.. code-block:: bash

   cd examples/agents
   
   # Interactive getting started
   python agent_quickstart.py
   
   # Comprehensive tutorial
   python agents_comprehensive_guide.py

**Focused Learning**:

.. code-block:: bash

   # Run specific examples
   python 01_basic_usage.py
   python 02_custom_agents.py
   python 03_memory_patterns.py
   python 04_tool_integration.py
   python 05_agent_composition.py
   python 06_testing_agents.py

Common Patterns Demonstrated
-----------------------------

**1. Agent Creation Patterns**:

.. code-block:: python

   # Basic agent
   class SimpleAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           return "response"
   
   # Specialized agent with custom output
   class AnalysisAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           return {"analysis": "result", "confidence": 0.95}

**2. Tool Integration Patterns**:

.. code-block:: python

   # Regular functions (results return to conversation)
   def search_data(query: str) -> List[Dict]:
       return search_results
   
   # Background tasks (fire-and-forget)
   def log_interaction(action: str, user_id: str):
       print(f"Logged: {action} by {user_id}")
   
   # Usage in agent
   llm_input = ILLMInput(
       system_prompt=self.system_prompt,
       user_message=input.message,
       regular_functions={"search_data": search_data},
       background_tasks={"log_interaction": log_interaction}
   )

**3. Memory Integration Patterns**:

.. code-block:: python

   # Using WorkingMemoryAgent
   memory_agent = WorkingMemoryAgent(
       llm_client=llm_client,
       memory_manager=memory_manager
   )
   
   # Update memory with conversation context
   result = await memory_agent.process(IAgentInput(
       message="User discussed machine learning interest",
       metadata={"conversation_id": "user_123"}
   ))

**4. Composition Patterns**:

.. code-block:: python

   # Orchestrator pattern
   class OrchestratorAgent(BaseAgent):
       def __init__(self, llm_client, specialized_agents):
           self.agents = specialized_agents
       
       async def process(self, input: IAgentInput):
           # Coordinate multiple agents
           return orchestrated_result
   
   # Pipeline pattern
   class PipelineAgent(BaseAgent):
       def __init__(self, llm_client, pipeline_agents):
           self.pipeline = pipeline_agents
       
       async def process(self, input: IAgentInput):
           # Sequential processing through agents
           return final_result

**5. Testing Patterns**:

.. code-block:: python

   # Unit testing with mocks
   @pytest.mark.asyncio
   async def test_agent():
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {"llm_response": "test"}
       
       agent = MyAgent(mock_llm, "prompt")
       result = await agent.process(IAgentInput(message="test"))
       
       assert result == expected
   
   # Integration testing
   async def test_with_real_llm():
       llm_client = OpenRouterClient(config)
       agent = MyAgent(llm_client, "prompt")
       result = await agent.process(IAgentInput(message="test"))
       assert len(result) > 0

Framework vs Examples
---------------------

**Framework Core** (What IS the framework):
- ``BaseAgent`` class in ``arshai/agents/base.py``
- ``IAgent`` interface and ``IAgentInput`` structure
- Tool integration patterns with functions and background tasks
- Direct instantiation and dependency injection patterns

**Reference Implementations** (What are examples):
- ``WorkingMemoryAgent`` in ``arshai/agents/hub/working_memory.py``
- All example agents (SentimentAgent, TranslationAgent, etc.)
- Orchestration patterns and composition examples
- Testing utilities and mock implementations

**Key Principle**: The framework provides building blocks. Examples show "our experience" with those blocks, not "the way" to use them.

Next Steps
----------

After working through the examples:

1. **Read the Implementation Guide**: See ``arshai/agents/README.md`` for critical implementation notes
2. **Explore the Architecture**: See ``docs/technical/agent_architecture.md`` for design decisions
3. **Create Your Own**: Start with ``BaseAgent`` and implement your domain-specific logic
4. **Integration**: Use agents in workflows, applications, and larger systems

**Support Resources**:

- **Issues**: https://github.com/MobileTechLab/ArsHai/issues
- **Architecture Documentation**: Framework design principles and decisions
- **Implementation Guide**: Critical patterns and best practices