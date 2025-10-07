Redis Memory Manager
===================

The ``RedisWorkingMemoryManager`` is a reference implementation of the ``IMemoryManager`` interface that stores conversation memory in Redis. This enables distributed, persistent memory storage for production multi-process deployments.

.. important::

   **This is a Reference Implementation**

   The Redis memory manager is provided as an example for production use. You can:

   - Use it as-is for Redis-based deployments
   - Extend it for your specific needs
   - Build your own memory manager for other databases (PostgreSQL, MongoDB, etc.)
   - Implement ``IMemoryManager`` for any storage backend

   The framework provides the interface - you choose the implementation.

Overview
--------

The ``RedisWorkingMemoryManager`` provides:

- **Distributed Storage**: Share memory across multiple processes/servers
- **Persistence**: Memories survive application restarts
- **TTL Support**: Automatic expiration using Redis TTL
- **High Performance**: Fast Redis access with connection pooling
- **Production Ready**: Battle-tested for production workloads

**Use Cases:**

- Production multi-process deployments
- Distributed systems
- Microservices architectures
- Scalable applications
- Long-running conversations
- Applications requiring persistence

Installation
------------

Install Arshai with Redis support:

.. code-block:: bash

   # Install with Redis extras
   pip install arshai[redis]

   # Or install Redis separately
   pip install arshai redis

Redis Setup
-----------

Local Development
~~~~~~~~~~~~~~~~~

Using Docker:

.. code-block:: bash

   # Start Redis container
   docker run --name arshai-redis -p 6379:6379 -d redis:7

   # Verify it's running
   docker ps

   # Test connection
   redis-cli ping
   # Should return: PONG

Using Redis directly:

.. code-block:: bash

   # Install Redis (Ubuntu/Debian)
   sudo apt-get install redis-server

   # Start Redis
   sudo systemctl start redis-server

   # Verify
   redis-cli ping

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

**Redis Cloud (Managed)**

.. code-block:: bash

   # Use managed Redis URL
   export REDIS_URL="redis://username:password@redis-host:port/db"

**AWS ElastiCache**

.. code-block:: bash

   export REDIS_URL="redis://your-elasticache-endpoint:6379/0"

**Self-Hosted Redis Cluster**

.. code-block:: bash

   export REDIS_URL="redis://redis-master:6379/0"

Environment Configuration
-------------------------

Set Redis connection URL:

.. code-block:: bash

   # Default (localhost)
   export REDIS_URL="redis://localhost:6379/1"

   # With password
   export REDIS_URL="redis://:password@localhost:6379/1"

   # With username and password
   export REDIS_URL="redis://username:password@localhost:6379/1"

   # Redis with SSL
   export REDIS_URL="rediss://username:password@redis-host:6380/1"

Basic Usage
-----------

Simple Example
~~~~~~~~~~~~~~

.. code-block:: python

   import os
   from arshai.memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager
   from arshai.core.interfaces.imemorymanager import IMemoryInput, IWorkingMemory
   from arshai.memory.memory_types import ConversationMemoryType

   # Create memory manager
   memory_manager = RedisWorkingMemoryManager(
       storage_url=os.getenv("REDIS_URL", "redis://localhost:6379/1")
   )

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

With Agent
~~~~~~~~~~

.. code-block:: python

   import asyncio
   import os
   from arshai.agents.working_memory import WorkingMemoryAgent
   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager

   async def main():
       # Create components
       llm_client = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))
       memory_manager = RedisWorkingMemoryManager()

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

       # Even if you restart the application, memory persists!
       # Second interaction (agent remembers from Redis)
       response2 = await agent.process(IAgentInput(
           message="What's my name and what do I like?",
           metadata={"conversation_id": "conv_123"}
       ))
       print(f"Agent: {response2}")

   asyncio.run(main())

Configuration
-------------

Connection Options
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Explicit URL
   memory_manager = RedisWorkingMemoryManager(
       storage_url="redis://localhost:6379/1"
   )

   # From environment variable (recommended)
   memory_manager = RedisWorkingMemoryManager()  # Uses REDIS_URL env var

   # With custom TTL (default: 12 hours)
   memory_manager = RedisWorkingMemoryManager(
       storage_url="redis://localhost:6379/1",
       ttl=86400  # 24 hours
   )

TTL Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1 hour TTL
   memory_manager = RedisWorkingMemoryManager(ttl=3600)

   # 24 hours TTL
   memory_manager = RedisWorkingMemoryManager(ttl=86400)

   # 7 days TTL
   memory_manager = RedisWorkingMemoryManager(ttl=604800)

   # Custom TTL based on use case
   if os.getenv("ENVIRONMENT") == "production":
       ttl = 86400 * 7  # 7 days
   else:
       ttl = 3600  # 1 hour for development

   memory_manager = RedisWorkingMemoryManager(ttl=ttl)

CRUD Operations
---------------

Store Memory
~~~~~~~~~~~~

.. code-block:: python

   # Store with metadata
   memory_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=[IWorkingMemory(working_memory="User prefers email")],
       metadata={"source": "user_settings", "timestamp": "2024-01-01"}
   )

   key = memory_manager.store(memory_input)
   # Memory is immediately available across all processes

Retrieve Memory
~~~~~~~~~~~~~~~

.. code-block:: python

   # Retrieve from any process/server
   retrieve_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY
   )

   memories = memory_manager.retrieve(retrieve_input)

   if memories:
       print(f"Found {len(memories)} memories")
       for memory in memories:
           print(f"Memory: {memory.working_memory}")
   else:
       print("No memories found")

Update Memory
~~~~~~~~~~~~~

.. code-block:: python

   # Update existing memory (replaces previous value)
   updated_data = [IWorkingMemory(
       working_memory="User prefers SMS (updated)"
   )]

   update_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=updated_data,
       metadata={"updated_at": "2024-01-02"}
   )

   memory_manager.update(update_input)

Delete Memory
~~~~~~~~~~~~~

.. code-block:: python

   # Delete conversation memory
   delete_input = IMemoryInput(
       conversation_id="conv_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY
   )

   memory_manager.delete(delete_input)
   # Memory is deleted from Redis immediately

Production Patterns
-------------------

Multi-Process Deployment
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   """
   All processes share the same Redis instance.
   Memory is synchronized automatically.
   """

   # Process 1 (Web Server 1)
   async def process_1():
       memory_manager = RedisWorkingMemoryManager()
       agent = WorkingMemoryAgent(llm_client, memory_manager)

       await agent.process(IAgentInput(
           message="I'm Alice",
           metadata={"conversation_id": "conv_123"}
       ))

   # Process 2 (Web Server 2) - can access same memory
   async def process_2():
       memory_manager = RedisWorkingMemoryManager()
       agent = WorkingMemoryAgent(llm_client, memory_manager)

       response = await agent.process(IAgentInput(
           message="What's my name?",
           metadata={"conversation_id": "conv_123"}
       ))
       # Agent retrieves memory from Redis: "Alice"

Connection Pooling
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from redis import ConnectionPool, Redis
   from arshai.memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager

   # Create connection pool for better performance
   pool = ConnectionPool.from_url(
       "redis://localhost:6379/1",
       max_connections=50,
       decode_responses=False
   )

   # Custom Redis client with pool
   redis_client = Redis(connection_pool=pool)

   # Use with memory manager (note: you may need to extend the class)
   class PooledRedisMemoryManager(RedisWorkingMemoryManager):
       def __init__(self, redis_client, **kwargs):
           self.redis_client = redis_client
           self.prefix = "memory"
           self.ttl = kwargs.get('ttl', 60 * 60 * 12)

   memory_manager = PooledRedisMemoryManager(redis_client)

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   import redis

   class RobustMemoryManager:
       """Wrapper with fallback and retry logic."""

       def __init__(self, primary_url: str, max_retries: int = 3):
           self.memory_manager = RedisWorkingMemoryManager(storage_url=primary_url)
           self.max_retries = max_retries

       def store_with_retry(self, memory_input: IMemoryInput) -> str:
           """Store with automatic retry on failure."""
           for attempt in range(self.max_retries):
               try:
                   return self.memory_manager.store(memory_input)
               except redis.ConnectionError as e:
                   if attempt == self.max_retries - 1:
                       raise
                   # Wait before retry
                   import time
                   time.sleep(2 ** attempt)  # Exponential backoff

           raise Exception("Max retries exceeded")

       def retrieve_safe(self, memory_input: IMemoryInput) -> list:
           """Retrieve with fallback to empty list on error."""
           try:
               return self.memory_manager.retrieve(memory_input)
           except redis.ConnectionError:
               # Log error and return empty
               print("Redis connection error, returning empty memories")
               return []

Monitoring and Debugging
-------------------------

Check Redis Contents
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Connect to Redis
   redis-cli

   # List all memory keys
   KEYS memory:*

   # Get specific memory
   GET memory:WORKING_MEMORY:conv_123

   # Check TTL
   TTL memory:WORKING_MEMORY:conv_123

   # Count all memory entries
   KEYS memory:* | wc -l

Monitor Memory Usage
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import redis

   def get_memory_stats(redis_url: str) -> dict:
       """Get Redis memory statistics."""
       client = redis.from_url(redis_url)

       info = client.info('memory')
       stats = {
           "used_memory": info.get("used_memory_human"),
           "used_memory_peak": info.get("used_memory_peak_human"),
           "total_keys": client.dbsize()
       }

       return stats

   # Usage
   stats = get_memory_stats("redis://localhost:6379/1")
   print(f"Memory usage: {stats['used_memory']}")
   print(f"Total keys: {stats['total_keys']}")

Cleanup Utilities
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def cleanup_old_conversations(redis_url: str, pattern: str = "memory:*"):
       """Clean up all memories matching pattern."""
       client = redis.from_url(redis_url)

       # Get all matching keys
       keys = client.keys(pattern)

       if keys:
           client.delete(*keys)
           print(f"Deleted {len(keys)} memory entries")
       else:
           print("No memories found to delete")

   # Cleanup all memories
   cleanup_old_conversations("redis://localhost:6379/1")

   # Cleanup specific conversation
   cleanup_old_conversations("redis://localhost:6379/1", "memory:*:conv_123")

Performance Optimization
------------------------

Batch Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_store_memories(
       memory_manager: RedisWorkingMemoryManager,
       conversation_memories: dict
   ):
       """Store multiple conversation memories efficiently."""

       for conversation_id, memory_text in conversation_memories.items():
           memory_input = IMemoryInput(
               conversation_id=conversation_id,
               memory_type=ConversationMemoryType.WORKING_MEMORY,
               data=[IWorkingMemory(working_memory=memory_text)]
           )
           memory_manager.store(memory_input)

   # Usage
   batch_store_memories(memory_manager, {
       "conv_1": "Alice prefers email",
       "conv_2": "Bob likes Python",
       "conv_3": "Carol uses dark mode"
   })

Connection Reuse
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Reuse memory manager instance across requests
   class MemoryService:
       """Singleton memory service for connection reuse."""

       _instance = None
       _memory_manager = None

       @classmethod
       def get_manager(cls):
           if cls._memory_manager is None:
               cls._memory_manager = RedisWorkingMemoryManager()
           return cls._memory_manager

   # Usage across application
   manager = MemoryService.get_manager()

Testing
-------

Unit Testing with Redis Mock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch

   @pytest.fixture
   def mock_redis():
       """Mock Redis client for testing."""
       mock = Mock()
       mock.get.return_value = None
       mock.setex.return_value = True
       mock.delete.return_value = 1
       return mock

   def test_store_memory(mock_redis):
       """Test memory storage."""
       with patch('redis.from_url', return_value=mock_redis):
           manager = RedisWorkingMemoryManager()

           memory_input = IMemoryInput(
               conversation_id="test_conv",
               memory_type=ConversationMemoryType.WORKING_MEMORY,
               data=[IWorkingMemory(working_memory="test")]
           )

           key = manager.store(memory_input)

           # Verify Redis was called
           assert mock_redis.setex.called

Integration Testing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest

   @pytest.mark.integration
   @pytest.mark.asyncio
   async def test_redis_persistence():
       """Test actual Redis persistence."""
       # Requires real Redis instance
       manager = RedisWorkingMemoryManager("redis://localhost:6379/15")  # Use test DB

       try:
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

       finally:
           # Cleanup
           manager.delete(retrieve_input)

Migration from In-Memory
------------------------

.. code-block:: python

   import os

   def create_memory_manager():
       """Factory function for environment-specific memory manager."""

       environment = os.getenv("ENVIRONMENT", "development")

       if environment == "production":
           # Production: Use Redis
           from arshai.memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager
           return RedisWorkingMemoryManager(
               storage_url=os.getenv("REDIS_URL"),
               ttl=86400  # 24 hours
           )
       else:
           # Development: Use in-memory
           from arshai.memory.working_memory.in_memory_manager import InMemoryManager
           return InMemoryManager(ttl=3600)

   # Usage
   memory_manager = create_memory_manager()
   agent = WorkingMemoryAgent(llm_client, memory_manager)
   # Same code works in both environments!

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Connection Refused**

.. code-block:: python

   # Error: redis.exceptions.ConnectionError: Error connecting to Redis
   # Solution: Verify Redis is running
   redis-cli ping  # Should return PONG

**Authentication Failed**

.. code-block:: python

   # Error: NOAUTH Authentication required
   # Solution: Include password in URL
   REDIS_URL="redis://:your_password@localhost:6379/1"

**Wrong Database**

.. code-block:: python

   # Using wrong Redis database number
   # Solution: Specify correct DB in URL
   REDIS_URL="redis://localhost:6379/1"  # Use DB 1

**Memory Full**

.. code-block:: bash

   # Error: OOM command not allowed when used memory > 'maxmemory'
   # Solution: Increase Redis maxmemory or cleanup old data
   redis-cli CONFIG SET maxmemory 256mb

Best Practices
--------------

1. **Use Environment Variables**
   Always configure Redis URL via environment variables.

2. **Set Appropriate TTL**
   Balance memory usage with user experience needs.

3. **Monitor Redis Health**
   Track memory usage, connection count, and performance.

4. **Use Separate Databases**
   Use different Redis DBs for different environments (dev/staging/prod).

5. **Implement Retry Logic**
   Handle transient Redis connection errors gracefully.

6. **Regular Cleanup**
   Implement periodic cleanup of old/unused memories.

7. **Backup Strategy**
   Configure Redis persistence (RDB/AOF) for data durability.

Security
--------

.. code-block:: bash

   # Use SSL/TLS in production
   REDIS_URL="rediss://username:password@redis-host:6380/1"

   # Restrict Redis access with firewall rules
   # Only allow application servers to connect

   # Use strong passwords
   # Set in Redis config: requirepass your_strong_password

   # Enable Redis AUTH
   # Configure ACL for fine-grained access control

Next Steps
----------

- **Development Memory**: See :doc:`in-memory` for local development
- **Custom Implementation**: Implement ``IMemoryManager`` for your database
- **Agent Integration**: See :doc:`../../framework/agents/examples/03-memory-patterns`

Remember: This is **one way** to implement distributed memory in Arshai. The framework provides the interface - you choose the storage backend that fits your infrastructure.
