Memory Reference Implementations
================================

This section documents the reference memory management implementations provided with Arshai. These demonstrate different approaches to storing and managing conversation memory using various storage backends.

.. toctree::
   :maxdepth: 2
   :caption: Memory Implementations
   
   in-memory
   redis-memory

.. note::
   **Reference Implementation Philosophy**
   
   These memory implementations are **not part of the core framework**. They represent working examples of how to implement the ``IMemoryManager`` interface for different storage needs. You can:
   
   - Use them directly if they meet your requirements
   - Modify them for your specific storage needs
   - Learn patterns to build your own memory implementations
   - Combine multiple approaches for different use cases

Available Reference Implementations
-----------------------------------

**InMemoryManager** (:doc:`in-memory`)
   Simple in-memory storage with automatic cleanup. Perfect for development, testing, and simple applications that don't require persistence.

**RedisMemoryManager** (:doc:`redis-memory`)
   Redis-backed persistent storage with TTL support. Ideal for production applications that need scalable, persistent memory storage.

Memory Management Patterns
---------------------------

**TTL-Based Cleanup**
   Automatic expiration of old memory entries to prevent unlimited growth.

**Key-Based Organization**
   Structured key patterns for efficient storage and retrieval of conversation memories.

**Metadata Handling**
   Storing additional context and metadata alongside working memory content.

**Error Recovery**
   Graceful handling of storage failures and missing data scenarios.

**Async Operations**
   Non-blocking memory operations that integrate seamlessly with agent processing.

Interface Compliance
---------------------

All reference memory implementations follow the ``IMemoryManager`` interface:

.. code-block:: python

   from arshai.core.interfaces.imemorymanager import IMemoryManager, IMemoryInput
   
   class IMemoryManager:
       """Interface for memory management implementations."""
       
       def store(self, input: IMemoryInput) -> str:
           """Store memory data and return storage key."""
           pass
       
       def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
           """Retrieve memory data matching the input criteria."""
           pass
       
       def update(self, input: IMemoryInput) -> None:
           """Update existing memory data."""
           pass
       
       def delete(self, input: IMemoryInput) -> None:
           """Delete memory data."""
           pass

**Memory Input Structure**

.. code-block:: python

   from arshai.core.interfaces.imemorymanager import IMemoryInput, IWorkingMemory
   from arshai.memory.memory_types import ConversationMemoryType
   
   # Creating memory input for storage
   memory_input = IMemoryInput(
       conversation_id="user_123",
       memory_type=ConversationMemoryType.WORKING,
       data=[IWorkingMemory(working_memory="User prefers technical details")],
       metadata={"user_type": "developer", "session_id": "abc123"}
   )

Usage Patterns
---------------

**Basic Memory Operations**

.. code-block:: python

   from arshai.memory.working_memory.in_memory_manager import InMemoryManager
   from arshai.core.interfaces.imemorymanager import IMemoryInput, IWorkingMemory
   from arshai.memory.memory_types import ConversationMemoryType
   
   # Create memory manager
   memory_manager = InMemoryManager(ttl=3600)  # 1 hour TTL
   
   # Store memory
   memory_data = IMemoryInput(
       conversation_id="conversation_123",
       memory_type=ConversationMemoryType.WORKING,
       data=[IWorkingMemory(working_memory="User is interested in machine learning")],
       metadata={"user_id": "user_456"}
   )
   
   key = memory_manager.store(memory_data)
   print(f"Stored memory with key: {key}")
   
   # Retrieve memory
   query = IMemoryInput(
       conversation_id="conversation_123",
       memory_type=ConversationMemoryType.WORKING
   )
   
   memories = memory_manager.retrieve(query)
   if memories:
       print(f"Retrieved memory: {memories[0].working_memory}")

**Agent Integration**

.. code-block:: python

   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
   
   # Create Redis-backed memory manager
   memory_manager = RedisMemoryManager(
       storage_url="redis://localhost:6379/1",
       ttl=60*60*24  # 24 hours
   )
   
   # Create memory-enabled agent
   memory_agent = WorkingMemoryAgent(
       llm_client=llm_client,
       memory_manager=memory_manager
   )
   
   # Agent automatically manages memory
   result = await memory_agent.process(IAgentInput(
       message="User mentioned they're a Python developer",
       metadata={"conversation_id": "dev_chat_123"}
   ))

**Custom Memory Implementation**

.. code-block:: python

   from arshai.core.interfaces.imemorymanager import IMemoryManager, IMemoryInput, IWorkingMemory
   from typing import List
   import sqlite3
   import json
   from datetime import datetime
   
   class SQLiteMemoryManager(IMemoryManager):
       """Custom SQLite-based memory implementation."""
       
       def __init__(self, db_path: str = "memory.db"):
           self.db_path = db_path
           self._init_db()
       
       def _init_db(self):
           """Initialize SQLite database."""
           conn = sqlite3.connect(self.db_path)
           cursor = conn.cursor()
           cursor.execute("""
               CREATE TABLE IF NOT EXISTS memories (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   conversation_id TEXT NOT NULL,
                   memory_type TEXT NOT NULL,
                   working_memory TEXT NOT NULL,
                   metadata TEXT,
                   created_at TEXT NOT NULL,
                   last_update TEXT NOT NULL,
                   UNIQUE(conversation_id, memory_type)
               )
           """)
           conn.commit()
           conn.close()
       
       def store(self, input: IMemoryInput) -> str:
           """Store memory in SQLite."""
           conn = sqlite3.connect(self.db_path)
           cursor = conn.cursor()
           
           for data in input.data:
               cursor.execute("""
                   INSERT OR REPLACE INTO memories 
                   (conversation_id, memory_type, working_memory, metadata, created_at, last_update)
                   VALUES (?, ?, ?, ?, ?, ?)
               """, (
                   input.conversation_id,
                   str(input.memory_type),
                   data.working_memory,
                   json.dumps(input.metadata or {}),
                   datetime.now().isoformat(),
                   datetime.now().isoformat()
               ))
           
           conn.commit()
           conn.close()
           
           return f"{input.conversation_id}:{input.memory_type}"
       
       def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
           """Retrieve memory from SQLite."""
           conn = sqlite3.connect(self.db_path)
           cursor = conn.cursor()
           
           cursor.execute("""
               SELECT working_memory FROM memories 
               WHERE conversation_id = ? AND memory_type = ?
           """, (input.conversation_id, str(input.memory_type)))
           
           result = cursor.fetchone()
           conn.close()
           
           if result:
               return [IWorkingMemory(working_memory=result[0])]
           return []
       
       def update(self, input: IMemoryInput) -> None:
           """Update memory in SQLite."""
           # For SQLite, update is the same as store due to INSERT OR REPLACE
           self.store(input)
       
       def delete(self, input: IMemoryInput) -> None:
           """Delete memory from SQLite."""
           conn = sqlite3.connect(self.db_path)
           cursor = conn.cursor()
           
           cursor.execute("""
               DELETE FROM memories 
               WHERE conversation_id = ? AND memory_type = ?
           """, (input.conversation_id, str(input.memory_type)))
           
           conn.commit()
           conn.close()

Memory Storage Strategies
-------------------------

**Choosing Storage Backend**

**In-Memory Storage**
   - **Use when**: Development, testing, simple applications
   - **Pros**: Fast, no external dependencies, automatic cleanup
   - **Cons**: Not persistent, limited by application memory
   - **Best for**: Prototypes, single-instance applications

**Redis Storage**
   - **Use when**: Production applications, multiple instances, scalability needs
   - **Pros**: Persistent, scalable, shared across instances, built-in TTL
   - **Cons**: Requires Redis server, network dependency
   - **Best for**: Production systems, multi-instance deployments

**Database Storage**
   - **Use when**: Complex queries, reporting, data analysis needs
   - **Pros**: Rich queries, transactions, data integrity, reporting
   - **Cons**: More complex setup, potential performance overhead
   - **Best for**: Enterprise applications, complex memory requirements

**Hybrid Approaches**
   - **Use when**: Different memory types have different requirements
   - **Pattern**: In-memory for short-term, persistent for long-term
   - **Example**: Cache recent memories in-memory, archive to database

Memory Optimization Patterns
-----------------------------

**TTL Management**

.. code-block:: python

   class TTLOptimizedMemoryManager:
       """Memory manager with intelligent TTL management."""
       
       def __init__(self, base_manager):
           self.base_manager = base_manager
           self.ttl_strategies = {
               "active_user": 60 * 60 * 24,      # 24 hours for active users
               "inactive_user": 60 * 60 * 2,     # 2 hours for inactive users
               "premium_user": 60 * 60 * 24 * 7,  # 7 days for premium users
           }
       
       def _get_ttl(self, metadata: dict) -> int:
           """Determine TTL based on user characteristics."""
           user_type = metadata.get("user_type", "active_user")
           return self.ttl_strategies.get(user_type, 60 * 60 * 12)
       
       def store(self, input: IMemoryInput) -> str:
           # Set appropriate TTL
           ttl = self._get_ttl(input.metadata or {})
           self.base_manager.ttl = ttl
           return self.base_manager.store(input)

**Memory Compression**

.. code-block:: python

   import gzip
   import base64
   from typing import List
   
   class CompressedMemoryWrapper:
       """Wrapper that compresses memory content."""
       
       def __init__(self, base_manager):
           self.base_manager = base_manager
       
       def _compress(self, text: str) -> str:
           """Compress text using gzip."""
           compressed = gzip.compress(text.encode('utf-8'))
           return base64.b64encode(compressed).decode('utf-8')
       
       def _decompress(self, compressed_text: str) -> str:
           """Decompress text."""
           compressed = base64.b64decode(compressed_text.encode('utf-8'))
           return gzip.decompress(compressed).decode('utf-8')
       
       def store(self, input: IMemoryInput) -> str:
           # Compress memory content
           compressed_data = []
           for data in input.data:
               compressed_memory = self._compress(data.working_memory)
               compressed_data.append(IWorkingMemory(working_memory=compressed_memory))
           
           compressed_input = IMemoryInput(
               conversation_id=input.conversation_id,
               memory_type=input.memory_type,
               data=compressed_data,
               metadata={**input.metadata, "compressed": True}
           )
           
           return self.base_manager.store(compressed_input)
       
       def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
           memories = self.base_manager.retrieve(input)
           
           # Decompress if needed
           decompressed_memories = []
           for memory in memories:
               if input.metadata and input.metadata.get("compressed"):
                   decompressed_content = self._decompress(memory.working_memory)
                   decompressed_memories.append(IWorkingMemory(working_memory=decompressed_content))
               else:
                   decompressed_memories.append(memory)
           
           return decompressed_memories

**Memory Versioning**

.. code-block:: python

   from datetime import datetime
   from typing import List, Dict, Any
   
   class VersionedMemoryManager:
       """Memory manager with version tracking."""
       
       def __init__(self, base_manager):
           self.base_manager = base_manager
           self.versions: Dict[str, List[Dict[str, Any]]] = {}
       
       def store(self, input: IMemoryInput) -> str:
           # Create version entry
           version_key = f"{input.conversation_id}:{input.memory_type}"
           
           if version_key not in self.versions:
               self.versions[version_key] = []
           
           # Store version metadata
           version_info = {
               "timestamp": datetime.now().isoformat(),
               "content": input.data[0].working_memory if input.data else "",
               "metadata": input.metadata or {}
           }
           
           self.versions[version_key].append(version_info)
           
           # Limit version history (keep last 10)
           if len(self.versions[version_key]) > 10:
               self.versions[version_key] = self.versions[version_key][-10:]
           
           return self.base_manager.store(input)
       
       def get_memory_history(self, conversation_id: str, memory_type) -> List[Dict[str, Any]]:
           """Get version history for a memory."""
           version_key = f"{conversation_id}:{memory_type}"
           return self.versions.get(version_key, [])

Testing Memory Implementations
-------------------------------

**Unit Testing**

.. code-block:: python

   import pytest
   from arshai.memory.working_memory.in_memory_manager import InMemoryManager
   from arshai.core.interfaces.imemorymanager import IMemoryInput, IWorkingMemory
   from arshai.memory.memory_types import ConversationMemoryType
   
   @pytest.fixture
   def memory_manager():
       return InMemoryManager(ttl=60)  # 1 minute TTL for testing
   
   @pytest.fixture
   def sample_memory_input():
       return IMemoryInput(
           conversation_id="test_conversation",
           memory_type=ConversationMemoryType.WORKING,
           data=[IWorkingMemory(working_memory="Test memory content")],
           metadata={"test": True}
       )
   
   def test_store_and_retrieve(memory_manager, sample_memory_input):
       # Store memory
       key = memory_manager.store(sample_memory_input)
       assert key is not None
       
       # Retrieve memory
       query = IMemoryInput(
           conversation_id="test_conversation",
           memory_type=ConversationMemoryType.WORKING
       )
       
       memories = memory_manager.retrieve(query)
       assert len(memories) == 1
       assert memories[0].working_memory == "Test memory content"
   
   def test_update_memory(memory_manager, sample_memory_input):
       # Store initial memory
       memory_manager.store(sample_memory_input)
       
       # Update memory
       update_input = IMemoryInput(
           conversation_id="test_conversation",
           memory_type=ConversationMemoryType.WORKING,
           data=[IWorkingMemory(working_memory="Updated memory content")]
       )
       
       memory_manager.update(update_input)
       
       # Verify update
       query = IMemoryInput(
           conversation_id="test_conversation",
           memory_type=ConversationMemoryType.WORKING
       )
       
       memories = memory_manager.retrieve(query)
       assert memories[0].working_memory == "Updated memory content"
   
   def test_delete_memory(memory_manager, sample_memory_input):
       # Store memory
       memory_manager.store(sample_memory_input)
       
       # Delete memory
       memory_manager.delete(sample_memory_input)
       
       # Verify deletion
       query = IMemoryInput(
           conversation_id="test_conversation",
           memory_type=ConversationMemoryType.WORKING
       )
       
       memories = memory_manager.retrieve(query)
       assert len(memories) == 0

**Integration Testing**

.. code-block:: python

   import pytest
   import time
   from arshai.memory.working_memory.redis_memory_manager import RedisMemoryManager
   
   @pytest.mark.integration
   def test_redis_memory_integration():
       # Test with real Redis (requires Redis server)
       memory_manager = RedisMemoryManager(
           storage_url="redis://localhost:6379/15"  # Use test database
       )
       
       # Test basic operations
       memory_input = IMemoryInput(
           conversation_id="integration_test",
           memory_type=ConversationMemoryType.WORKING,
           data=[IWorkingMemory(working_memory="Integration test content")],
           metadata={"integration": True}
       )
       
       # Store and retrieve
       key = memory_manager.store(memory_input)
       assert key is not None
       
       memories = memory_manager.retrieve(memory_input)
       assert len(memories) == 1
       assert memories[0].working_memory == "Integration test content"
       
       # Cleanup
       memory_manager.delete(memory_input)
   
   @pytest.mark.integration
   def test_memory_ttl():
       memory_manager = InMemoryManager(ttl=1)  # 1 second TTL
       
       memory_input = IMemoryInput(
           conversation_id="ttl_test",
           memory_type=ConversationMemoryType.WORKING,
           data=[IWorkingMemory(working_memory="TTL test content")]
       )
       
       # Store memory
       memory_manager.store(memory_input)
       
       # Should be available immediately
       memories = memory_manager.retrieve(memory_input)
       assert len(memories) == 1
       
       # Wait for TTL expiration
       time.sleep(2)
       
       # Should be expired
       memories = memory_manager.retrieve(memory_input)
       assert len(memories) == 0

**Performance Testing**

.. code-block:: python

   import time
   import asyncio
   from concurrent.futures import ThreadPoolExecutor
   
   def test_memory_performance():
       memory_manager = InMemoryManager()
       
       # Test storage performance
       start_time = time.time()
       
       for i in range(1000):
           memory_input = IMemoryInput(
               conversation_id=f"perf_test_{i}",
               memory_type=ConversationMemoryType.WORKING,
               data=[IWorkingMemory(working_memory=f"Performance test content {i}")]
           )
           memory_manager.store(memory_input)
       
       storage_time = time.time() - start_time
       print(f"Stored 1000 memories in {storage_time:.2f} seconds")
       
       # Test retrieval performance
       start_time = time.time()
       
       for i in range(1000):
           query = IMemoryInput(
               conversation_id=f"perf_test_{i}",
               memory_type=ConversationMemoryType.WORKING
           )
           memories = memory_manager.retrieve(query)
           assert len(memories) == 1
       
       retrieval_time = time.time() - start_time
       print(f"Retrieved 1000 memories in {retrieval_time:.2f} seconds")

Best Practices
---------------

**Design Principles**
   - Implement the IMemoryManager interface completely
   - Handle errors gracefully without throwing exceptions
   - Use appropriate logging for debugging and monitoring
   - Implement proper cleanup for resource management

**Performance Optimization**
   - Use connection pooling for database/Redis connections
   - Implement appropriate caching strategies
   - Batch operations when possible
   - Monitor memory usage and implement cleanup

**Error Handling**
   - Handle storage backend failures gracefully
   - Implement retry logic for transient failures
   - Provide meaningful error messages and logging
   - Consider fallback strategies for critical failures

**Security Considerations**
   - Validate input data to prevent injection attacks
   - Implement access controls for memory storage
   - Consider encryption for sensitive memory content
   - Audit memory access for compliance requirements

The reference memory implementations provide solid foundations for different memory management needs. Choose the implementation that best fits your requirements or use them as starting points for custom solutions.