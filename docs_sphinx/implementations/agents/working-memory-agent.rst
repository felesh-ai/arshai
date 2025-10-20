WorkingMemoryAgent Reference Implementation
===========================================

The ``WorkingMemoryAgent`` is a specialized agent that manages conversation working memory. It demonstrates how to build agents that maintain context across interactions using the framework's memory management capabilities.

.. note::
   **Reference Implementation**
   
   This is a reference implementation showing how we've built memory-enabled agents. It's not part of the core framework - use it as-is, modify it, or build your own memory management approach.

Overview
--------

**Purpose**
   Maintains and updates working memory for conversations, ensuring context is preserved across multiple interactions.

**Location**
   ``arshai.agents.hub.working_memory.WorkingMemoryAgent``

**Key Capabilities**
   - Retrieves existing conversation memory
   - Incorporates new interaction context
   - Generates updated memory summaries
   - Stores updated memory persistently
   - Handles missing dependencies gracefully

**Integration Points**
   - Works with any ``IMemoryManager`` implementation
   - Optional chat history integration
   - Uses metadata for conversation identification
   - Returns status information for system coordination

Basic Usage
-----------

**Simple Usage**

.. code-block:: python

   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   from arshai.memory.working_memory.in_memory_manager import InMemoryManager
   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.iagent import IAgentInput
   
   # Create dependencies
   llm_client = OpenAIClient(config)
   memory_manager = InMemoryManager()
   
   # Create memory agent
   memory_agent = WorkingMemoryAgent(
       llm_client=llm_client,
       memory_manager=memory_manager
   )
   
   # Update memory with new interaction
   result = await memory_agent.process(IAgentInput(
       message="User asked about product pricing and mentioned budget concerns",
       metadata={"conversation_id": "user_123"}
   ))
   
   print(result)  # "success" if successful

**With Redis Storage**

.. code-block:: python

   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
   from arshai.clients.redis_client import RedisClient
   
   # Create Redis-backed memory manager
   redis_client = RedisClient(host="localhost", port=6379)
   memory_manager = RedisMemoryManager(redis_client)
   
   # Create memory agent with persistent storage
   memory_agent = WorkingMemoryAgent(
       llm_client=llm_client,
       memory_manager=memory_manager
   )
   
   # Memory will persist across application restarts
   result = await memory_agent.process(IAgentInput(
       message="User completed purchase and requested support",
       metadata={"conversation_id": "user_123"}
   ))

**Custom System Prompt**

.. code-block:: python

   custom_prompt = """You are a specialized memory manager for customer service conversations.

   Focus on:
   - Customer preferences and history
   - Previous issues and resolutions
   - Current context and needs
   - Action items and follow-ups
   
   Keep memory concise but actionable for support agents."""
   
   memory_agent = WorkingMemoryAgent(
       llm_client=llm_client,
       system_prompt=custom_prompt,
       memory_manager=memory_manager
   )

Configuration Options
---------------------

**Constructor Parameters**

.. code-block:: python

   WorkingMemoryAgent(
       llm_client: ILLM,                    # Required: LLM for memory generation
       system_prompt: str = None,           # Optional: Custom memory prompt
       memory_manager: IMemoryManager = None, # Optional: Storage backend
       chat_history_client: Any = None,     # Optional: Conversation history source
       **kwargs                             # Additional BaseAgent parameters
   )

**Default System Prompt**
   The agent includes a comprehensive default prompt optimized for memory management:

.. code-block:: text

   You are a memory management assistant responsible for maintaining conversation context.

   Your tasks:
   1. Analyze conversation history and current interaction
   2. Extract key information, facts, and context
   3. Generate a concise working memory summary
   4. Focus on information relevant for future interactions

   Keep the memory:
   - Concise but comprehensive
   - Focused on actionable information
   - Updated with latest context
   - Free from redundancy

Processing Flow
---------------

The ``WorkingMemoryAgent`` follows this processing flow:

**1. Conversation ID Extraction**
   
.. code-block:: python

   conversation_id = input.metadata.get("conversation_id")
   if not conversation_id:
       return "error: no conversation_id provided"

**2. Current Memory Retrieval**
   
.. code-block:: python

   if self.memory_manager:
       memory_data = await self.memory_manager.retrieve({"conversation_id": conversation_id})
       current_memory = memory_data[0].working_memory if memory_data else ""

**3. Conversation History Retrieval** (Optional)
   
.. code-block:: python

   if self.chat_history:
       history = await self.chat_history.get(conversation_id)
       conversation_history = str(history) if history else ""

**4. Context Preparation**
   
.. code-block:: python

   context = f"""
   Current Working Memory:
   {current_memory if current_memory else "No existing memory"}

   Conversation History:
   {conversation_history if conversation_history else "No previous history"}

   New Interaction:
   {input.message}

   Please generate an updated working memory that incorporates the new information...
   """

**5. Memory Generation**
   
.. code-block:: python

   llm_input = ILLMInput(
       system_prompt=self.system_prompt,
       user_message=context
   )
   result = await self.llm_client.chat(llm_input)
   updated_memory = result.get('llm_response', '')

**6. Memory Storage**
   
.. code-block:: python

   if self.memory_manager:
       await self.memory_manager.store({
           "conversation_id": conversation_id,
           "working_memory": updated_memory,
           "metadata": input.metadata
       })

**7. Status Return**
   
.. code-block:: python

   return "success"  # or "error: <description>"

Response Format
---------------

The agent returns status strings to indicate processing results:

**Success Response**
   ``"success"`` - Memory was successfully updated and stored

**Error Responses**
   - ``"error: no conversation_id provided"`` - No conversation ID in metadata
   - ``"error: empty memory response"`` - LLM returned empty response
   - ``"error: storage failed - <details>"`` - Memory storage failed
   - ``"error: <exception message>"`` - Other processing errors

Integration Patterns
---------------------

**As Background Task**
   Use WorkingMemoryAgent as a background task for automatic memory updates:

.. code-block:: python

   def update_memory(message: str, conversation_id: str):
       """Background task for memory updates"""
       memory_input = IAgentInput(
           message=message,
           metadata={"conversation_id": conversation_id}
       )
       # This runs in background, doesn't block conversation
       asyncio.create_task(memory_agent.process(memory_input))
   
   # In your main agent
   background_tasks = {"update_memory": update_memory}
   
   llm_input = ILLMInput(
       system_prompt=main_prompt,
       user_message=user_message,
       background_tasks=background_tasks
   )
   
   # LLM can trigger memory updates automatically
   result = await llm_client.chat(llm_input)

**In Multi-Agent Systems**
   Coordinate memory updates across multiple agents:

.. code-block:: python

   class CustomerServiceSystem:
       def __init__(self, llm_client, memory_manager):
           self.memory_agent = WorkingMemoryAgent(llm_client, memory_manager=memory_manager)
           self.support_agent = SupportAgent(llm_client)
           self.escalation_agent = EscalationAgent(llm_client)
       
       async def handle_request(self, message: str, conversation_id: str):
           # Update memory with new interaction
           await self.memory_agent.process(IAgentInput(
               message=f"User interaction: {message}",
               metadata={"conversation_id": conversation_id}
           ))
           
           # Handle request with appropriate agent
           if self.needs_escalation(message):
               return await self.escalation_agent.process(IAgentInput(
                   message=message,
                   metadata={"conversation_id": conversation_id}
               ))
           else:
               return await self.support_agent.process(IAgentInput(
                   message=message,
                   metadata={"conversation_id": conversation_id}
               ))

**With Workflow Systems**
   Integrate memory updates into workflow steps:

.. code-block:: python

   class MemoryUpdateNode(BaseNode):
       def __init__(self, memory_agent):
           self.memory_agent = memory_agent
       
       async def execute(self, context: dict) -> dict:
           # Update memory as part of workflow
           result = await self.memory_agent.process(IAgentInput(
               message=context.get("interaction_summary"),
               metadata={"conversation_id": context.get("conversation_id")}
           ))
           
           context["memory_update_status"] = result
           return context

Error Handling
--------------

The agent implements comprehensive error handling:

**Graceful Degradation**
   - Continues processing even if memory retrieval fails
   - Handles missing chat history gracefully
   - Works without memory manager (logs warnings)

**Error Recovery**
   - Catches and logs all exceptions
   - Returns descriptive error messages
   - Doesn't crash on partial failures

**Logging**
   - Debug logs for successful operations
   - Warning logs for missing dependencies
   - Error logs for failures with details

.. code-block:: python

   # Example error handling in the implementation
   try:
       memory_data = await self.memory_manager.retrieve({"conversation_id": conversation_id})
       # ... process memory data
   except Exception as e:
       # Log warning but continue without current memory
       logger.warning(f"Failed to retrieve memory: {e}")
       current_memory = ""

Testing Patterns
----------------

**Unit Testing with Mocks**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock
   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   
   @pytest.mark.asyncio
   async def test_memory_agent_success():
       # Mock dependencies
       mock_llm = AsyncMock()
       mock_memory_manager = AsyncMock()
       
       # Configure mock responses
       mock_llm.chat.return_value = {"llm_response": "Updated memory content"}
       mock_memory_manager.retrieve.return_value = []
       mock_memory_manager.store.return_value = None
       
       # Create and test agent
       agent = WorkingMemoryAgent(mock_llm, memory_manager=mock_memory_manager)
       
       result = await agent.process(IAgentInput(
           message="Test interaction",
           metadata={"conversation_id": "test_123"}
       ))
       
       assert result == "success"
       mock_memory_manager.store.assert_called_once()

**Integration Testing**

.. code-block:: python

   @pytest.mark.asyncio
   async def test_memory_agent_integration():
       # Test with real components
       llm_client = OpenAIClient(test_config)
       memory_manager = InMemoryManager()
       
       agent = WorkingMemoryAgent(llm_client, memory_manager=memory_manager)
       
       # First interaction
       result1 = await agent.process(IAgentInput(
           message="User wants to buy a laptop",
           metadata={"conversation_id": "integration_test"}
       ))
       assert result1 == "success"
       
       # Second interaction - should include previous context
       result2 = await agent.process(IAgentInput(
           message="User asked about warranty options",
           metadata={"conversation_id": "integration_test"}
       ))
       assert result2 == "success"
       
       # Verify memory persistence
       memory_data = await memory_manager.retrieve({"conversation_id": "integration_test"})
       assert len(memory_data) > 0
       assert "laptop" in memory_data[0].working_memory
       assert "warranty" in memory_data[0].working_memory

**Error Handling Tests**

.. code-block:: python

   @pytest.mark.asyncio
   async def test_memory_agent_error_handling():
       mock_llm = AsyncMock()
       mock_memory_manager = AsyncMock()
       
       # Test missing conversation ID
       agent = WorkingMemoryAgent(mock_llm, memory_manager=mock_memory_manager)
       result = await agent.process(IAgentInput(message="test"))
       assert result == "error: no conversation_id provided"
       
       # Test storage failure
       mock_llm.chat.return_value = {"llm_response": "Updated memory"}
       mock_memory_manager.retrieve.return_value = []
       mock_memory_manager.store.side_effect = Exception("Storage failed")
       
       result = await agent.process(IAgentInput(
           message="test",
           metadata={"conversation_id": "test"}
       ))
       assert result.startswith("error: storage failed")

Customization Patterns
-----------------------

**Custom Memory Processing**

.. code-block:: python

   class DomainSpecificMemoryAgent(WorkingMemoryAgent):
       """Memory agent specialized for e-commerce conversations"""
       
       def __init__(self, llm_client, memory_manager, product_catalog):
           custom_prompt = """You are an e-commerce memory manager.
           
           Focus on:
           - Product interests and preferences
           - Purchase history and patterns
           - Budget constraints and concerns
           - Support interactions and resolutions"""
           
           super().__init__(llm_client, custom_prompt, memory_manager)
           self.product_catalog = product_catalog
       
       async def process(self, input: IAgentInput) -> str:
           # Add product context before processing
           if self.should_add_product_context(input.message):
               enhanced_message = await self.add_product_context(input.message)
               enhanced_input = IAgentInput(
                   message=enhanced_message,
                   metadata=input.metadata
               )
               return await super().process(enhanced_input)
           
           return await super().process(input)

**Memory Validation**

.. code-block:: python

   class ValidatingMemoryAgent(WorkingMemoryAgent):
       """Memory agent with content validation"""
       
       async def process(self, input: IAgentInput) -> str:
           # Process memory normally
           result = await super().process(input)
           
           if result == "success":
               # Validate stored memory
               conversation_id = input.metadata.get("conversation_id")
               if await self.validate_memory_quality(conversation_id):
                   return "success"
               else:
                   return "error: memory validation failed"
           
           return result
       
       async def validate_memory_quality(self, conversation_id: str) -> bool:
           # Custom validation logic
           memory_data = await self.memory_manager.retrieve({"conversation_id": conversation_id})
           if not memory_data:
               return False
           
           memory_content = memory_data[0].working_memory
           return len(memory_content) > 10 and not self.contains_sensitive_data(memory_content)

Best Practices
--------------

**Memory Content**
   - Keep memory concise but comprehensive
   - Focus on actionable information for future interactions
   - Remove redundant or outdated information
   - Include context that affects future decisions

**Error Handling**
   - Always check for conversation_id in metadata
   - Handle storage failures gracefully
   - Log appropriate information for debugging
   - Return meaningful status information

**Performance**
   - Consider memory size limits for large conversations
   - Implement memory cleanup for old conversations
   - Use appropriate storage backends for your scale
   - Monitor memory update frequency

**Security**
   - Avoid storing sensitive information in working memory
   - Implement access controls for memory storage
   - Consider encryption for persistent storage
   - Validate and sanitize memory content

Limitations and Considerations
------------------------------

**Current Limitations**
   - No built-in memory size limits
   - No automatic cleanup of old memories
   - Single conversation context only
   - No memory versioning or history

**Performance Considerations**
   - Memory updates are synchronous operations
   - Large conversation histories may slow processing
   - Storage backend performance affects agent performance
   - Memory retrieval happens on every update

**Scaling Considerations**
   - Consider memory storage patterns for high-volume systems
   - Implement memory archiving for long-running conversations
   - Monitor storage backend performance and capacity
   - Consider distributed memory storage for multi-instance deployments

The WorkingMemoryAgent demonstrates a practical approach to conversation memory management using Arshai's building blocks. Use it as a starting point for your own memory management needs or as inspiration for different approaches.