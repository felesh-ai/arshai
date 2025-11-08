Agent Reference Implementations
================================

This section documents the reference agent implementations provided with Arshai. These are working examples that demonstrate how to build specialized agents using the framework's building blocks.

.. toctree::
   :maxdepth: 2
   :caption: Reference Agent Implementations
   
   working-memory-agent

.. note::
   **Reference Implementation Philosophy**
   
   These agent implementations are **not part of the core framework**. They are working examples that show how we've used the framework to solve specific problems. You can:
   
   - Use them as-is if they meet your needs
   - Modify them for your specific requirements
   - Learn from them to build your own agents
   - Ignore them completely and build from scratch

Available Reference Implementations
-----------------------------------

**WorkingMemoryAgent** (:doc:`working-memory-agent`)
   A specialized agent for managing conversation memory. Demonstrates memory integration, context management, and persistent storage patterns.

**Future Implementations**
   Additional reference agents will be added here as they're developed. Each represents a different pattern or use case.

Common Patterns Demonstrated
-----------------------------

**Memory Integration**
   How agents can work with memory managers to maintain conversation context across interactions.

**Error Handling**
   Robust error handling patterns that gracefully handle failures without crashing the system.

**Metadata Usage**
   How agents can use input metadata to coordinate with other system components.

**Background Processing**
   Patterns for agents that update system state as a side effect of processing.

**Specialized Behavior**
   How to create agents with specific responsibilities that complement general-purpose agents.

Using Reference Implementations
--------------------------------

**Direct Import and Usage**

.. code-block:: python

   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
   from arshai.llms.openai import OpenAIClient
   
   # Create components
   llm_client = OpenAIClient(config)
   memory_manager = RedisMemoryManager(redis_client)
   
   # Use reference implementation directly
   memory_agent = WorkingMemoryAgent(
       llm_client=llm_client,
       memory_manager=memory_manager
   )
   
   # Process memory updates
   result = await memory_agent.process(IAgentInput(
       message="User discussed pricing concerns",
       metadata={"conversation_id": "user_123"}
   ))

**Adaptation Pattern**

.. code-block:: python

   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   
   class CustomMemoryAgent(WorkingMemoryAgent):
       """Extended version with custom behavior"""
       
       def __init__(self, llm_client, memory_manager, custom_config):
           # Use custom system prompt
           custom_prompt = "Your specialized memory management prompt..."
           super().__init__(llm_client, custom_prompt, memory_manager)
           self.custom_config = custom_config
       
       async def process(self, input: IAgentInput) -> str:
           # Add pre-processing
           if self.should_apply_custom_logic(input):
               return await self.custom_memory_handling(input)
           
           # Use parent implementation
           return await super().process(input)
       
       def should_apply_custom_logic(self, input: IAgentInput) -> bool:
           # Your custom decision logic
           return "urgent" in input.metadata.get("flags", [])

**Learning Pattern**

.. code-block:: python

   # Study WorkingMemoryAgent, then build your own approach
   class MyMemoryApproach(BaseAgent):
       """My own memory management approach"""
       
       def __init__(self, llm_client, my_storage):
           super().__init__(llm_client, "My memory prompt")
           self.storage = my_storage
       
       async def process(self, input: IAgentInput) -> str:
           # Your own implementation inspired by reference
           conversation_id = input.metadata.get("conversation_id")
           
           # Your approach to memory management
           current_state = await self.storage.get(conversation_id)
           updated_state = await self.generate_update(input, current_state)
           await self.storage.save(conversation_id, updated_state)
           
           return "updated"

Framework Integration Patterns
-------------------------------

**Memory Manager Integration**
   Reference implementations show how agents integrate with different memory backends through the IMemoryManager interface.

**Metadata-Driven Coordination**
   How agents use input metadata to coordinate with other system components and maintain shared context.

**Error Recovery**
   Patterns for handling failures gracefully, including partial failures that don't break the entire system.

**Async Processing**
   How agents handle asynchronous operations like storage updates and external API calls.

**Tool Integration**
   Some reference implementations demonstrate how agents can also serve as tools for other agents.

Key Design Principles
---------------------

**Single Responsibility**
   Each reference agent focuses on one specific capability, making them composable and testable.

**Interface Compliance**
   All reference agents properly implement the IAgent interface and work seamlessly with the framework.

**Configuration Flexibility**
   Reference implementations accept configuration through constructor parameters, making them adaptable.

**Graceful Degradation**
   Agents handle missing dependencies or configuration gracefully, often with reduced functionality rather than failure.

**Observable Behavior**
   Reference implementations include logging and status reporting to help with debugging and monitoring.

Testing Reference Implementations
----------------------------------

**Unit Testing**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock
   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   
   @pytest.mark.asyncio
   async def test_working_memory_agent():
       # Mock dependencies
       mock_llm = AsyncMock()
       mock_memory_manager = AsyncMock()
       
       # Configure mocks
       mock_llm.chat.return_value = {"llm_response": "Updated memory content"}
       mock_memory_manager.retrieve.return_value = []
       mock_memory_manager.store.return_value = None
       
       # Create agent
       agent = WorkingMemoryAgent(
           llm_client=mock_llm,
           memory_manager=mock_memory_manager
       )
       
       # Test agent
       result = await agent.process(IAgentInput(
           message="Test message",
           metadata={"conversation_id": "test_123"}
       ))
       
       assert result == "success"
       mock_memory_manager.store.assert_called_once()

**Integration Testing**

.. code-block:: python

   @pytest.mark.asyncio
   async def test_memory_agent_integration():
       # Test with real components
       llm_client = OpenAIClient(config)
       memory_manager = InMemoryManager()
       
       agent = WorkingMemoryAgent(llm_client, memory_manager=memory_manager)
       
       # Test full flow
       result = await agent.process(IAgentInput(
           message="User wants to know about pricing",
           metadata={"conversation_id": "integration_test"}
       ))
       
       assert result == "success"
       
       # Verify memory was stored
       memory_data = await memory_manager.retrieve({"conversation_id": "integration_test"})
       assert len(memory_data) > 0

Best Practices from Reference Implementations
----------------------------------------------

**Configuration Management**
   Accept configuration through constructor parameters rather than global settings.

**Dependency Injection**
   Accept dependencies (LLM clients, memory managers) as constructor parameters for testability.

**Error Handling**
   Handle errors gracefully and return meaningful status information.

**Logging**
   Include appropriate logging for debugging and monitoring without overwhelming logs.

**Metadata Usage**
   Use input metadata for coordination while maintaining agent independence.

**Interface Compliance**
   Strictly follow the IAgent interface contract for seamless framework integration.

Contributing Reference Implementations
---------------------------------------

If you've built agents that might be useful as reference implementations:

1. **Follow Framework Patterns**: Use the same patterns demonstrated in existing reference implementations
2. **Include Documentation**: Provide clear documentation of what the agent does and how to use it
3. **Add Tests**: Include unit and integration tests
4. **Handle Errors**: Implement robust error handling
5. **Share Your Experience**: Consider contributing your implementation to help other developers

Remember: Reference implementations are about sharing proven patterns and working code that demonstrates the framework's capabilities.