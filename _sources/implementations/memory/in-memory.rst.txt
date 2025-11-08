In-Memory Manager
=================

The ``InMemoryManager`` is a reference implementation of the ``IMemoryManager`` interface that stores conversation memory in local process memory. This is **one way** to implement memory management in Arshai - ideal for development, testing, and single-process applications.

.. important::

   **This is a Reference Implementation**

   The in-memory manager is provided as an example. You can:

   - Use it for development and testing
   - Use it for single-process applications
   - Extend it for your specific needs
   - Build your own memory manager implementing ``IMemoryManager``

   For production multi-process deployments, consider :doc:`redis-memory` or build a custom solution.

Overview
--------

The ``InMemoryManager`` provides:

- **Local Storage**: Stores memory in Python dictionaries
- **TTL Support**: Automatic expiration of old memories
- **Simple API**: Standard ``IMemoryManager`` interface
- **Zero Dependencies**: No external services required
- **Fast Access**: Direct memory access with no network latency

**Use Cases:**

- Development and testing
- Single-process applications
- Prototyping
- Short-lived conversations
- Non-distributed systems

Installation
------------

The in-memory manager is included in the core Arshai package:

.. code-block:: bash

   pip install arshai

No additional dependencies required.

Basic Usage
-----------

Simple Example
~~~~~~~~~~~~~~

.. code-block:: python

   from arshai.memory.working_memory.in_memory_manager import InMemoryManager
   from arshai.core.interfaces.imemorymanager import IMemoryInput, IWorkingMemory
   from arshai.memory.memory_types import ConversationMemoryType

   # Create memory manager
   memory_manager = InMemoryManager()

   # Store memory
   memory_data = [IWorkingMemory(working_memory="User's name is Alice")]

   memory_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=memory_data
   )

   key = memory_manager.store(memory_input)
   print(f"Stored memory with key: {key}")

   # Retrieve memory
   retrieve_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY
   )

   memories = memory_manager.retrieve(retrieve_input)
   print(f"Retrieved: {memories[0].working_memory}")
   # Output: Retrieved: User's name is Alice

With Agent
~~~~~~~~~~

.. code-block:: python

   import asyncio
   from arshai.agents.working_memory import WorkingMemoryAgent
   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.memory.working_memory.in_memory_manager import InMemoryManager

   async def main():
       # Create components
       llm_client = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))
       memory_manager = InMemoryManager()

       # Create memory-enabled agent
       agent = WorkingMemoryAgent(
           llm_client=llm_client,
           memory_manager=memory_manager
       )

       # First interaction
       response1 = await agent.process(IAgentInput(
           message="My name is Bob and I like Python",
           metadata={"conversation_id": "conv_123"}
       ))
       print(f"Agent: {response1}")

       # Second interaction (agent remembers)
       response2 = await agent.process(IAgentInput(
           message="What's my name and what do I like?",
           metadata={"conversation_id": "conv_123"}
       ))
       print(f"Agent: {response2}")
       # Agent will remember Bob and Python

   asyncio.run(main())

Configuration
-------------

TTL (Time To Live)
~~~~~~~~~~~~~~~~~~

Configure automatic memory expiration:

.. code-block:: python

   # Default TTL: 12 hours (43200 seconds)
   memory_manager = InMemoryManager()

   # Custom TTL: 1 hour
   memory_manager = InMemoryManager(ttl=3600)

   # Custom TTL: 24 hours
   memory_manager = InMemoryManager(ttl=86400)

   # Disable TTL (memories never expire automatically)
   memory_manager = InMemoryManager(ttl=None)

The TTL is checked on each ``store()`` operation, and expired memories are cleaned up automatically.

Memory Types
~~~~~~~~~~~~

The manager supports different memory types:

.. code-block:: python

   from arshai.memory.memory_types import ConversationMemoryType

   # Working memory (default)
   working_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=[IWorkingMemory(working_memory="Important context")]
   )

   # Different memory types are stored separately
   memory_manager.store(working_input)

CRUD Operations
---------------

Store Memory
~~~~~~~~~~~~

.. code-block:: python

   from arshai.core.interfaces.imemorymanager import IWorkingMemory

   # Create memory data
   memory_data = [IWorkingMemory(
       working_memory="User prefers email communication"
   )]

   # Store with metadata
   memory_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=memory_data,
       metadata={"source": "user_preference"}
   )

   key = memory_manager.store(memory_input)

Retrieve Memory
~~~~~~~~~~~~~~~

.. code-block:: python

   # Retrieve by conversation ID
   retrieve_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY
   )

   memories = memory_manager.retrieve(retrieve_input)

   if memories:
       for memory in memories:
           print(f"Memory: {memory.working_memory}")
   else:
       print("No memories found")

Update Memory
~~~~~~~~~~~~~

.. code-block:: python

   # Update existing memory
   updated_data = [IWorkingMemory(
       working_memory="User prefers SMS communication (updated)"
   )]

   update_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=updated_data
   )

   memory_manager.update(update_input)

Delete Memory
~~~~~~~~~~~~~

.. code-block:: python

   # Delete specific conversation memory
   delete_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY
   )

   memory_manager.delete(delete_input)

Advanced Patterns
-----------------

Multi-Conversation Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ConversationManager:
       """Manage multiple conversations with in-memory storage."""

       def __init__(self):
           self.memory_manager = InMemoryManager(ttl=3600)
           self.active_conversations = {}

       async def process_message(
           self,
           user_id: str,
           conversation_id: str,
           message: str
       ) -> str:
           """Process message with conversation-specific memory."""

           # Create agent for this conversation if not exists
           if conversation_id not in self.active_conversations:
               llm_client = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))
               agent = WorkingMemoryAgent(
                   llm_client=llm_client,
                   memory_manager=self.memory_manager
               )
               self.active_conversations[conversation_id] = agent

           # Get agent and process
           agent = self.active_conversations[conversation_id]
           response = await agent.process(IAgentInput(
               message=message,
               metadata={"conversation_id": conversation_id, "user_id": user_id}
           ))

           return response

   # Usage
   async def multi_conversation_example():
       manager = ConversationManager()

       # Conversation 1
       await manager.process_message("user_1", "conv_1", "I'm Alice")
       await manager.process_message("user_1", "conv_1", "What's my name?")

       # Conversation 2 (separate context)
       await manager.process_message("user_2", "conv_2", "I'm Bob")
       await manager.process_message("user_2", "conv_2", "What's my name?")

Custom Memory Structure
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from dataclasses import dataclass
   from typing import Dict, Any

   @dataclass
   class RichMemory:
       """Custom memory structure with additional fields."""
       content: str
       importance: int  # 1-10 scale
       tags: list
       metadata: Dict[str, Any]

   class EnhancedMemoryManager:
       """Extended in-memory manager with custom structure."""

       def __init__(self):
           self.base_manager = InMemoryManager()
           self.rich_storage = {}

       def store_rich_memory(
           self,
           conversation_id: str,
           memory: RichMemory
       ):
           """Store rich memory with additional metadata."""

           # Store base memory
           base_input = IMemoryInput(
               conversation_id=conversation_id,
               memory_type=ConversationMemoryType.WORKING_MEMORY,
               data=[IWorkingMemory(working_memory=memory.content)]
           )
           self.base_manager.store(base_input)

           # Store rich metadata separately
           if conversation_id not in self.rich_storage:
               self.rich_storage[conversation_id] = []
           self.rich_storage[conversation_id].append(memory)

       def get_important_memories(
           self,
           conversation_id: str,
           min_importance: int = 7
       ) -> list:
           """Retrieve memories above importance threshold."""
           if conversation_id not in self.rich_storage:
               return []

           return [
               m for m in self.rich_storage[conversation_id]
               if m.importance >= min_importance
           ]

Memory Cleanup
~~~~~~~~~~~~~~

.. code-block:: python

   class ManagedInMemory:
       """In-memory manager with explicit cleanup controls."""

       def __init__(self, ttl: int = 3600):
           self.memory_manager = InMemoryManager(ttl=ttl)

       def cleanup_expired(self):
           """Manually trigger cleanup of expired memories."""
           # The InMemoryManager does this automatically on store,
           # but you can trigger it explicitly
           self.memory_manager._clear_expired_memory()

       def cleanup_conversation(self, conversation_id: str):
           """Remove all memories for a specific conversation."""
           delete_input = IMemoryInput(
               conversation_id=conversation_id,
               memory_type=ConversationMemoryType.WORKING_MEMORY
           )
           self.memory_manager.delete(delete_input)

       def get_storage_stats(self) -> Dict[str, Any]:
           """Get statistics about current memory usage."""
           return {
               "total_entries": len(self.memory_manager.storage),
               "memory_keys": list(self.memory_manager.storage.keys())
           }

Limitations
-----------

**Single Process Only**

The in-memory manager stores data in the current process memory. If you:

- Scale to multiple processes/servers
- Restart the application
- Use distributed systems

...all memories will be lost. For these scenarios, use :doc:`redis-memory` or build a persistent storage solution.

**Memory Usage**

All data is stored in RAM. For large-scale applications with many conversations:

- Monitor memory usage
- Implement aggressive TTL
- Consider using persistent storage

**No Persistence**

Memories are lost on:

- Application restart
- Process crash
- Server failure

Testing
-------

Unit Testing
~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from arshai.memory.working_memory.in_memory_manager import InMemoryManager
   from arshai.core.interfaces.imemorymanager import IMemoryInput, IWorkingMemory
   from arshai.memory.memory_types import ConversationMemoryType

   def test_store_and_retrieve():
       """Test basic store and retrieve."""
       manager = InMemoryManager()

       # Store
       memory_input = IMemoryInput(
           conversation_id="test_conv",
           memory_type=ConversationMemoryType.WORKING_MEMORY,
           data=[IWorkingMemory(working_memory="test data")]
       )
       key = manager.store(memory_input)

       # Retrieve
       retrieve_input = IMemoryInput(
           conversation_id="test_conv",
           memory_type=ConversationMemoryType.WORKING_MEMORY
       )
       memories = manager.retrieve(retrieve_input)

       assert len(memories) == 1
       assert memories[0].working_memory == "test data"

   def test_ttl_expiration():
       """Test TTL expiration."""
       manager = InMemoryManager(ttl=1)  # 1 second TTL

       # Store memory
       memory_input = IMemoryInput(
           conversation_id="test_conv",
           memory_type=ConversationMemoryType.WORKING_MEMORY,
           data=[IWorkingMemory(working_memory="test data")]
       )
       manager.store(memory_input)

       # Wait for expiration
       import time
       time.sleep(2)

       # Trigger cleanup by storing new memory
       manager.store(IMemoryInput(
           conversation_id="other_conv",
           memory_type=ConversationMemoryType.WORKING_MEMORY,
           data=[IWorkingMemory(working_memory="other data")]
       ))

       # Original memory should be expired
       retrieve_input = IMemoryInput(
           conversation_id="test_conv",
           memory_type=ConversationMemoryType.WORKING_MEMORY
       )
       memories = manager.retrieve(retrieve_input)
       assert len(memories) == 0

Integration Testing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @pytest.mark.asyncio
   async def test_with_agent():
       """Test in-memory manager with actual agent."""
       from unittest.mock import AsyncMock

       # Create mock LLM
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {"llm_response": "I remember you!"}

       # Create components
       memory_manager = InMemoryManager()
       agent = WorkingMemoryAgent(
           llm_client=mock_llm,
           memory_manager=memory_manager
       )

       # Process message
       await agent.process(IAgentInput(
           message="Remember this",
           metadata={"conversation_id": "test_conv"}
       ))

       # Verify memory was stored
       retrieve_input = IMemoryInput(
           conversation_id="test_conv",
           memory_type=ConversationMemoryType.WORKING_MEMORY
       )
       memories = memory_manager.retrieve(retrieve_input)
       assert len(memories) > 0

Best Practices
--------------

1. **Set Appropriate TTL**
   Configure TTL based on your use case (shorter for high-volume, longer for better UX).

2. **Monitor Memory Usage**
   Track memory consumption in production environments.

3. **Use for Development**
   Perfect for local development and testing.

4. **Transition to Persistent Storage**
   Plan migration to Redis or database for production.

5. **Handle Missing Memories Gracefully**
   Always check if memories exist before using them.

6. **Clean Up Inactive Conversations**
   Implement logic to remove old conversation data.

Migration to Production
-----------------------

When moving to production, consider migrating to :doc:`redis-memory`:

.. code-block:: python

   # Development (in-memory)
   if os.getenv("ENVIRONMENT") == "development":
       memory_manager = InMemoryManager(ttl=3600)
   else:
       # Production (Redis)
       from arshai.memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager
       memory_manager = RedisWorkingMemoryManager(
           storage_url=os.getenv("REDIS_URL")
       )

   # Same interface, different implementation
   agent = WorkingMemoryAgent(
       llm_client=llm_client,
       memory_manager=memory_manager
   )

Next Steps
----------

- **Production Memory**: See :doc:`redis-memory` for distributed memory
- **Custom Implementation**: Implement ``IMemoryManager`` for your database
- **Agent Integration**: See :doc:`../../framework/agents/examples/03-memory-patterns`

Remember: This is **one way** to implement memory in Arshai. The framework provides the interface - you choose the implementation that fits your needs.
