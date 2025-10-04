Example 3: Memory Patterns
==========================

This example demonstrates memory patterns with agents, showcasing the ``WorkingMemoryAgent`` and memory management strategies for maintaining conversation context.

**File**: ``examples/agents/03_memory_patterns.py`` (283 lines)

**Focus**: Memory management and conversation context

**Best For**: Understanding stateful conversations and working memory patterns

Overview
--------

This example shows how to:

- Use the ``WorkingMemoryAgent`` for conversation context
- Implement custom memory managers for development and testing
- Handle multiple concurrent conversations
- Manage conversation state and context across interactions
- Test error conditions and edge cases with memory systems
- Simulate real-world conversation patterns

Key Concepts Demonstrated
-------------------------

**Working Memory Agent**
   Reference implementation that manages conversation context automatically.

**Memory Manager Interface**
   How agents work with external storage systems for persistence.

**Conversation Context**
   Building and maintaining context across multiple interactions.

**Multiple Conversations**
   Handling concurrent conversation threads with separate context.

**Error Handling**
   Graceful degradation when memory systems are unavailable.

Code Walkthrough
----------------

**1. Simple Memory Manager Implementation**

The example starts with an in-memory storage system for demonstration:

.. code-block:: python

   class InMemoryManager:
       """Simple in-memory storage for demonstration."""
       
       def __init__(self):
           self.memories = {}
           self.access_count = {}
       
       async def store(self, data: Dict[str, Any]):
           """Store memory for a conversation."""
           conv_id = data.get("conversation_id")
           if conv_id:
               self.memories[conv_id] = data.get("working_memory", "")
               self.access_count[conv_id] = self.access_count.get(conv_id, 0) + 1
       
       async def retrieve(self, query: Dict[str, Any]):
           """Retrieve memory for a conversation."""
           conv_id = query.get("conversation_id")
           if conv_id and conv_id in self.memories:
               return [type('obj', (), {'working_memory': self.memories[conv_id]})()]
           return None

**Key Points:**
- Simple dictionary-based storage for development/testing
- Tracks access patterns for debugging
- Returns structured objects matching the expected interface
- Handles missing conversations gracefully

**2. Conversation Simulator**

Demonstrates how to build conversation context over time:

.. code-block:: python

   class ConversationSimulator:
       """Simulates a conversation to demonstrate memory patterns."""
       
       async def add_interaction(self, conversation_id: str, interaction: str) -> str:
           """Add an interaction and update memory."""
           input_data = IAgentInput(
               message=interaction,
               metadata={"conversation_id": conversation_id}
           )
           
           # Process memory update
           result = await self.memory_agent.process(input_data)
           return result

**Key Points:**
- Shows how conversation_id ties interactions together
- Demonstrates the metadata pattern for passing context
- Illustrates memory accumulation over multiple turns

**3. Basic Memory Operations Test**

Shows fundamental memory patterns:

.. code-block:: python

   conversation_id = "user_alice_session_001"
   
   interactions = [
       "My name is Alice and I work as a software engineer at TechCorp",
       "I'm interested in learning about machine learning for my current project",
       "Specifically, I need help with natural language processing techniques",
       "I prefer practical examples over theoretical explanations"
   ]
   
   for i, interaction in enumerate(interactions, 1):
       await simulator.add_interaction(conversation_id, interaction)
       memory_manager.show_memory(conversation_id)

**Key Points:**
- Progressive context building across multiple interactions
- Shows how memory accumulates user preferences and context
- Demonstrates memory inspection for debugging

**4. Multiple Conversations**

Tests conversation isolation:

.. code-block:: python

   # Second conversation with different user
   conversation_2 = "user_bob_session_002"
   bob_interactions = [
       "I'm Bob, a product manager looking to understand AI capabilities",
       "My team is considering implementing chatbots for customer support"
   ]
   
   for interaction in bob_interactions:
       await simulator.add_interaction(conversation_2, interaction)

**Key Points:**
- Shows how different conversation IDs maintain separate context
- Demonstrates memory isolation between users
- Tests concurrent conversation handling

**5. Error Handling Tests**

Covers edge cases and error conditions:

.. code-block:: python

   # Test without conversation_id
   result = await memory_agent.process(IAgentInput(
       message="This message has no conversation ID",
       metadata={}
   ))
   
   # Test with None metadata
   result = await memory_agent.process(IAgentInput(
       message="This message has None metadata",
       metadata=None
   ))
   
   # Test agent without memory manager
   standalone_agent = WorkingMemoryAgent(
       llm_client=llm_client,
       memory_manager=None
   )

**Key Points:**
- Tests graceful handling of missing conversation IDs
- Shows behavior when metadata is None or missing
- Demonstrates agent operation without external storage

**6. Advanced Memory Patterns**

Shows real-world usage patterns:

.. code-block:: python

   # Customer support conversation simulation
   support_conversation = "support_ticket_12345"
   
   support_interactions = [
       "Customer Jane Smith called about billing issue with invoice #4567",
       "Issue: Charged twice for the same service in March 2024",
       "Customer provided transaction IDs: TXN001, TXN002",
       "Resolution: Refunded duplicate charge of $99.99",
       "Customer satisfied, case closed"
   ]

**Key Points:**
- Demonstrates structured conversation tracking
- Shows how memory captures important details over time
- Illustrates business context accumulation

Running the Example
--------------------

**Prerequisites:**

.. code-block:: bash

   export OPENROUTER_API_KEY=your_key_here

**Run the example:**

.. code-block:: bash

   cd examples/agents
   python 03_memory_patterns.py

**Expected Output:**

The example runs through six test scenarios:

1. **Basic Memory Operations** - Shows memory building across 4 interactions
2. **Multiple Conversations** - Demonstrates isolated conversation contexts
3. **Error Handling** - Tests edge cases and missing data
4. **Agent without Memory Manager** - Shows graceful degradation
5. **Advanced Memory Patterns** - Customer support scenario
6. **Memory Retrieval Simulation** - Returning customer context

Key Takeaways
-------------

**1. Memory Manager Interface**

The ``WorkingMemoryAgent`` expects memory managers to implement:

- ``store(data: Dict[str, Any])`` - Save conversation memory
- ``retrieve(query: Dict[str, Any])`` - Get conversation memory

**2. Conversation ID Pattern**

Always pass conversation_id in metadata for memory continuity:

.. code-block:: python

   input_data = IAgentInput(
       message="User message",
       metadata={"conversation_id": "unique_conversation_id"}
   )

**3. Memory Status Responses**

The ``WorkingMemoryAgent`` returns status indicators:

- ``"success"`` - Memory updated successfully
- ``"error: description"`` - Error occurred with details

**4. External Storage Flexibility**

The agent works with any storage backend:

- In-memory dictionaries (development)
- Redis (production caching)
- Databases (persistent storage)
- Cloud storage (distributed systems)

**5. Graceful Degradation**

The agent handles missing components gracefully:

- No memory manager: generates memory but doesn't store
- No conversation_id: returns error status
- Storage failures: continues operation with reduced functionality

Real-World Implementation Patterns
-----------------------------------

**Production Memory Manager Example:**

.. code-block:: python

   import redis.asyncio as redis
   import json
   
   class RedisMemoryManager:
       def __init__(self, redis_url: str):
           self.redis = redis.from_url(redis_url)
       
       async def store(self, data: Dict[str, Any]):
           conv_id = data.get("conversation_id")
           if conv_id:
               memory = data.get("working_memory", "")
               await self.redis.set(f"memory:{conv_id}", memory, ex=86400)  # 24h TTL
       
       async def retrieve(self, query: Dict[str, Any]):
           conv_id = query.get("conversation_id")
           if conv_id:
               memory = await self.redis.get(f"memory:{conv_id}")
               if memory:
                   return [type('obj', (), {'working_memory': memory.decode()})()]
           return None

**Database Memory Manager Example:**

.. code-block:: python

   class DatabaseMemoryManager:
       def __init__(self, db_connection):
           self.db = db_connection
       
       async def store(self, data: Dict[str, Any]):
           conv_id = data.get("conversation_id")
           memory = data.get("working_memory", "")
           
           async with self.db.cursor() as cursor:
               await cursor.execute("""
                   INSERT INTO conversation_memory (conversation_id, memory, updated_at)
                   VALUES (%s, %s, NOW())
                   ON DUPLICATE KEY UPDATE
                   memory = VALUES(memory), updated_at = NOW()
               """, (conv_id, memory))
       
       async def retrieve(self, query: Dict[str, Any]):
           conv_id = query.get("conversation_id")
           
           async with self.db.cursor() as cursor:
               await cursor.execute(
                   "SELECT memory FROM conversation_memory WHERE conversation_id = %s",
                   (conv_id,)
               )
               result = await cursor.fetchone()
               if result:
                   return [type('obj', (), {'working_memory': result['memory']})()]
           return None

**Configuration-Driven Setup:**

.. code-block:: python

   def create_memory_agent(config: dict):
       """Factory function for memory-enabled agents."""
       
       # Create LLM client
       llm_config = ILLMConfig(
           model=config["model"],
           temperature=config.get("temperature", 0.7)
       )
       llm_client = OpenRouterClient(llm_config)
       
       # Create memory manager based on configuration
       memory_type = config.get("memory_type", "memory")
       
       if memory_type == "redis":
           memory_manager = RedisMemoryManager(config["redis_url"])
       elif memory_type == "database":
           memory_manager = DatabaseMemoryManager(config["db_connection"])
       else:
           memory_manager = InMemoryManager()
       
       # Create memory agent
       return WorkingMemoryAgent(
           llm_client=llm_client,
           memory_manager=memory_manager
       )

Testing Memory Patterns
------------------------

**Unit Test Example:**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock

   @pytest.mark.asyncio
   async def test_memory_agent_with_context():
       """Test memory agent retrieves and uses context."""
       
       # Mock LLM client
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Memory updated with user context",
           "usage": {"total_tokens": 50}
       }
       
       # Mock memory manager
       mock_memory = AsyncMock()
       mock_memory.retrieve.return_value = [
           type('obj', (), {'working_memory': 'User likes technical details'})()
       ]
       
       # Create agent
       agent = WorkingMemoryAgent(mock_llm, mock_memory)
       
       # Test with existing memory
       result = await agent.process(IAgentInput(
           message="Tell me about APIs",
           metadata={"conversation_id": "test_123"}
       ))
       
       # Verify memory was retrieved and used
       mock_memory.retrieve.assert_called_once()
       mock_memory.store.assert_called_once()
       assert "success" in result

Common Use Cases
----------------

**1. Customer Support**

Track customer issues and context across multiple interactions:

.. code-block:: python

   # Each support ticket gets a unique conversation ID
   conversation_id = f"support_ticket_{ticket_id}"
   
   # Agent builds context of customer issues, solutions tried, etc.
   await memory_agent.process(IAgentInput(
       message="Customer reporting login issues after password reset",
       metadata={"conversation_id": conversation_id}
   ))

**2. Educational Tutoring**

Maintain student learning context and progress:

.. code-block:: python

   # Track student's learning journey
   conversation_id = f"student_{student_id}_session_{session_id}"
   
   # Agent remembers topics covered, learning style, progress
   await memory_agent.process(IAgentInput(
       message="Student struggled with recursion concepts",
       metadata={"conversation_id": conversation_id}
   ))

**3. Personal Assistants**

Remember user preferences and context across sessions:

.. code-block:: python

   # User's ongoing conversation with personal assistant
   conversation_id = f"user_{user_id}_assistant"
   
   # Agent remembers preferences, past requests, context
   await memory_agent.process(IAgentInput(
       message="User prefers evening meetings and detailed summaries",
       metadata={"conversation_id": conversation_id}
   ))

**4. Project Management**

Track project context and decisions:

.. code-block:: python

   # Project-specific conversation context
   conversation_id = f"project_{project_id}_discussions"
   
   # Agent builds context of decisions, requirements, issues
   await memory_agent.process(IAgentInput(
       message="Team decided to use microservices architecture",
       metadata={"conversation_id": conversation_id}
   ))

Next Steps
----------

After understanding memory patterns:

1. **Implement Your Memory Manager**: Create storage backend for your use case
2. **Integrate with Your Application**: Use memory agents in your conversation flows
3. **Handle Memory Lifecycle**: Implement cleanup and retention policies
4. **Monitor Memory Usage**: Track storage and performance metrics
5. **Test Memory Patterns**: Ensure reliable memory behavior in your tests

**Related Examples:**
- :doc:`04-tool-integration` - Combine memory with external tools
- :doc:`05-agent-composition` - Use memory agents in larger systems
- :doc:`06-testing-agents` - Test memory-enabled agents thoroughly