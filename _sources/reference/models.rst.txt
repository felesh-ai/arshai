Models
======

Complete reference for all Pydantic models and DTOs (Data Transfer Objects) used throughout the Arshai framework.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Arshai uses Pydantic models for all data structures to provide:

- **Automatic Validation**: Type checking and constraint enforcement
- **JSON Serialization**: Easy conversion to/from JSON
- **IDE Support**: Full autocomplete and type hints
- **Immutability**: Predictable state management
- **Documentation**: Self-documenting field descriptions

All models inherit from ``IDTO`` (``pydantic.BaseModel``) and follow consistent patterns.

.. _idto-model:

Base Models
-----------

IDTO
~~~~

**Location:** ``arshai/core/interfaces/idto.py``

Base class for all Data Transfer Objects with Pydantic validation.

**Definition:**

.. code-block:: python

   from pydantic import BaseModel

   class IDTO(BaseModel):
       """Base class for all DTOs with Pydantic validation"""

       class Config:
           arbitrary_types_allowed = True
           extra = "forbid"  # Prevent extra fields

**Features:**

- **Strict Field Validation**: Only defined fields are allowed (``extra = "forbid"``)
- **Arbitrary Types**: Can include custom types (``arbitrary_types_allowed = True``)
- **JSON Serialization**: Automatic ``model_dump()`` and ``model_dump_json()``
- **Immutability**: Fields are immutable by default (Pydantic v2)

**Usage:**

.. code-block:: python

   from arshai.core.interfaces import IDTO
   from pydantic import Field

   class MyDTO(IDTO):
       name: str = Field(description="User name")
       age: int = Field(ge=0, description="User age")

   # Create instance
   data = MyDTO(name="Alice", age=30)

   # Validation error on invalid data
   try:
       MyDTO(name="Bob", age=-5)  # Raises ValidationError
   except ValidationError as e:
       print(e)

LLM Models
----------

.. _illmconfig-model:

ILLMConfig
~~~~~~~~~~

**Location:** ``arshai/core/interfaces/illm.py``

Configuration for LLM providers.

**Fields:**

.. code-block:: python

   class ILLMConfig(IDTO):
       model: str                            # Required: Model identifier
       temperature: float = 0.0              # Sampling temperature (0.0-2.0)
       max_tokens: Optional[int] = None      # Maximum tokens to generate
       top_p: Optional[float] = None         # Nucleus sampling parameter
       frequency_penalty: Optional[float] = None  # Repetition penalty (-2.0 to 2.0)
       presence_penalty: Optional[float] = None   # Presence penalty (-2.0 to 2.0)

**Field Descriptions:**

``model: str`` (required)
   Model identifier specific to the LLM provider.

   Examples:
      - OpenAI: ``"gpt-4"``, ``"gpt-3.5-turbo"``
      - Google: ``"gemini-1.5-pro"``, ``"gemini-1.5-flash"``

``temperature: float`` (default: 0.0)
   Controls randomness in generation. Range: 0.0 (deterministic) to 2.0 (very random).

``max_tokens: Optional[int]``
   Maximum number of tokens to generate. ``None`` means provider default.

``top_p: Optional[float]``
   Nucleus sampling: only consider tokens with cumulative probability mass of top_p.

``frequency_penalty: Optional[float]``
   Penalize frequent tokens to reduce repetition. Range: -2.0 to 2.0.

``presence_penalty: Optional[float]``
   Penalize tokens based on presence in prior text. Range: -2.0 to 2.0.

**Example:**

.. code-block:: python

   from arshai.core.interfaces import ILLMConfig

   # Simple configuration
   config = ILLMConfig(model="gpt-4")

   # Advanced configuration
   config = ILLMConfig(
       model="gpt-4",
       temperature=0.7,
       max_tokens=1000,
       top_p=0.9,
       frequency_penalty=0.5
   )

.. _illminput-model:

ILLMInput
~~~~~~~~~

**Location:** ``arshai/core/interfaces/illm.py``

Input for LLM chat and streaming operations with support for function calling and structured output.

**Fields:**

.. code-block:: python

   class ILLMInput(IDTO):
       system_prompt: str                          # Required: System prompt
       user_message: str                           # Required: User message
       regular_functions: Dict[str, Callable] = {} # Regular callable functions
       background_tasks: Dict[str, Callable] = {}  # Fire-and-forget functions
       structure_type: Type[T] = None              # Output structure type
       max_turns: int = 10                         # Max function calling turns

**Field Descriptions:**

``system_prompt: str`` (required)
   The system prompt defining the assistant's behavior, personality, and context.

``user_message: str`` (required)
   The user's message or query to process.

``regular_functions: Dict[str, Callable]`` (default: {})
   Dictionary mapping function names to callable functions for tool integration.

   - Functions execute synchronously in the conversation
   - Results are returned to the LLM for reasoning
   - Supports multi-turn function calling

``background_tasks: Dict[str, Callable]`` (default: {})
   Dictionary mapping function names to background task callables.

   - Tasks execute in fire-and-forget mode using ``asyncio.create_task()``
   - Results are NOT returned to the LLM conversation
   - Useful for logging, notifications, analytics

``structure_type: Type[T]`` (default: None)
   Pydantic model class for structured output.

   - When provided, LLM returns validated instance of this type
   - Ensures type-safe structured responses
   - Uses provider-specific structured output mechanisms

``max_turns: int`` (default: 10)
   Maximum number of function calling turns before forcing final response.

**Validation:**

.. code-block:: python

   @model_validator(mode='before')
   @classmethod
   def validate_input(cls, data):
       if not data.get('system_prompt'):
           raise ValueError("system_prompt is required")
       if not data.get('user_message'):
           raise ValueError("user_message is required")
       return data

**Example Usage:**

**Simple Chat:**

.. code-block:: python

   from arshai.core.interfaces import ILLMInput

   llm_input = ILLMInput(
       system_prompt="You are a helpful assistant",
       user_message="What is Python?"
   )

   result = await llm_client.chat(llm_input)

**With Function Calling:**

.. code-block:: python

   def get_weather(location: str) -> dict:
       """Get current weather for a location"""
       return {"temp": 72, "condition": "sunny"}

   def search_web(query: str) -> list:
       """Search the web for information"""
       return ["result1", "result2"]

   llm_input = ILLMInput(
       system_prompt="You are a helpful assistant with access to tools",
       user_message="What's the weather in San Francisco?",
       regular_functions={
           "get_weather": get_weather,
           "search_web": search_web
       }
   )

   result = await llm_client.chat(llm_input)

**With Background Tasks:**

.. code-block:: python

   def log_interaction(action: str, metadata: dict = None):
       """Log user interaction for analytics"""
       print(f"Logged: {action} with metadata: {metadata}")

   def send_notification(event: str, user_id: str):
       """Send notification to admin system"""
       print(f"Notification: {event} for user {user_id}")

   llm_input = ILLMInput(
       system_prompt="You are a helpful assistant",
       user_message="Hello!",
       background_tasks={
           "log_interaction": log_interaction,
           "send_notification": send_notification
       }
   )

**With Structured Output:**

.. code-block:: python

   from pydantic import BaseModel

   class SentimentAnalysis(BaseModel):
       sentiment: str
       confidence: float
       reasoning: str

   llm_input = ILLMInput(
       system_prompt="Analyze the sentiment of user messages",
       user_message="I love this product!",
       structure_type=SentimentAnalysis
   )

   result = await llm_client.chat(llm_input)
   # result contains validated SentimentAnalysis instance

Agent Models
------------

.. _iagentinput-model:

IAgentInput
~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/iagent.py``

Input for agent processing operations.

**Fields:**

.. code-block:: python

   class IAgentInput(IDTO):
       message: str                          # Required: User message
       metadata: Optional[Dict[str, Any]] = None  # Optional metadata

**Field Descriptions:**

``message: str`` (required)
   The user's message to process.

``metadata: Optional[Dict[str, Any]]`` (default: None)
   Optional metadata for agent-specific context.

   Common metadata keys:
      - ``conversation_id``: Unique conversation identifier
      - ``user_id``: User identifier
      - ``stream``: Whether to stream response
      - ``session_data``: Session-specific data

**Example:**

.. code-block:: python

   from arshai.core.interfaces import IAgentInput

   # Simple input
   agent_input = IAgentInput(message="Hello!")

   # With metadata
   agent_input = IAgentInput(
       message="What's my order status?",
       metadata={
           "conversation_id": "conv_123",
           "user_id": "user_456",
           "session_data": {"previous_topic": "orders"}
       }
   )

   result = await agent.process(agent_input)

Memory Models
-------------

.. _conversationmemorytype-enum:

ConversationMemoryType
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/imemorymanager.py``

Enumeration of memory types supported by the framework.

**Definition:**

.. code-block:: python

   from enum import StrEnum

   class ConversationMemoryType(StrEnum):
       SHORT_TERM_MEMORY = "SHORT_TERM_MEMORY"
       LONG_TERM_MEMORY = "LONG_TERM_MEMORY"
       WORKING_MEMORY = "WORKING_MEMORY"

**Values:**

``SHORT_TERM_MEMORY``
   Recent conversations and temporary information.

   - Typically limited time window (e.g., last 24 hours)
   - Automatically expires after period
   - Used for contextual awareness in conversations

``LONG_TERM_MEMORY``
   Persistent knowledge and important information.

   - Stored indefinitely
   - User preferences, facts, important decisions
   - Requires explicit deletion

``WORKING_MEMORY``
   Current conversation state and context.

   - Active context for ongoing conversation
   - Updated with each interaction
   - Summarizes key information from conversation

**Example:**

.. code-block:: python

   from arshai.core.interfaces import ConversationMemoryType, IMemoryInput

   memory_input = IMemoryInput(
       conversation_id="user_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=[working_memory]
   )

.. _imemoryinput-model:

IMemoryInput
~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/imemorymanager.py``

Input for memory management operations (store, retrieve, update, delete).

**Fields:**

.. code-block:: python

   class IMemoryInput(IDTO):
       conversation_id: str                                       # Required
       memory_type: ConversationMemoryType = SHORT_TERM_MEMORY   # Memory type
       data: Optional[List[IWorkingMemory]] = None               # Data for store/update
       query: Optional[str] = None                               # Search query
       memory_id: Optional[str] = None                           # Specific memory ID
       limit: Optional[int] = 5                                  # Max results
       filters: Optional[Dict[str, Any]] = None                  # Additional filters
       metadata: Optional[Dict[str, Any]] = None                 # Entry metadata

**Field Descriptions:**

``conversation_id: str`` (required)
   Unique identifier for the conversation.

``memory_type: ConversationMemoryType`` (default: SHORT_TERM_MEMORY)
   Type of memory to operate on.

``data: Optional[List[IWorkingMemory]]``
   Memory data for store/update operations.

``query: Optional[str]``
   Search query for retrieve operations.

``memory_id: Optional[str]``
   Specific memory ID for update/delete operations.

``limit: Optional[int]`` (default: 5)
   Maximum number of items to retrieve.

``filters: Optional[Dict[str, Any]]``
   Additional filters to apply (implementation-specific).

``metadata: Optional[Dict[str, Any]]``
   Additional metadata for the memory entry.

**Example Usage:**

**Store Memory:**

.. code-block:: python

   from arshai.core.interfaces import IMemoryInput, IWorkingMemory

   memory_input = IMemoryInput(
       conversation_id="user_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=[IWorkingMemory(working_memory="User prefers concise answers")]
   )

   memory_id = memory_manager.store(memory_input)

**Retrieve Memory:**

.. code-block:: python

   memory_input = IMemoryInput(
       conversation_id="user_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       limit=10
   )

   memories = memory_manager.retrieve(memory_input)

**Update Memory:**

.. code-block:: python

   memory_input = IMemoryInput(
       conversation_id="user_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       memory_id="mem_456",
       data=[IWorkingMemory(working_memory="Updated context")]
   )

   memory_manager.update(memory_input)

**Delete Memory:**

.. code-block:: python

   memory_input = IMemoryInput(
       conversation_id="user_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       memory_id="mem_456"
   )

   memory_manager.delete(memory_input)

.. _imemoryitem-model:

IMemoryItem
~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/imemorymanager.py``

Represents a single item in conversation memory (for short-term/long-term memory).

**Fields:**

.. code-block:: python

   class IMemoryItem(IDTO):
       role: str                                  # Required: Message role
       content: str                               # Required: Content
       metadata: Optional[Dict[str, Any]] = {}    # Optional metadata

**Field Descriptions:**

``role: str`` (required)
   Role of the message sender.

   Common values:
      - ``"user"`` - User message
      - ``"assistant"`` - Assistant response
      - ``"system"`` - System message

``content: str`` (required)
   Content of the memory item.

``metadata: Optional[Dict[str, Any]]`` (default: {})
   Additional metadata for the memory item (timestamps, tags, etc.).

**Example:**

.. code-block:: python

   from arshai.core.interfaces import IMemoryItem

   memory_item = IMemoryItem(
       role="user",
       content="What's the weather today?",
       metadata={"timestamp": "2025-10-07T10:00:00Z"}
   )

.. _iworkingmemory-model:

IWorkingMemory
~~~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/imemorymanager.py``

Maintains the assistant's working memory during conversations as a structured string.

**Fields:**

.. code-block:: python

   class IWorkingMemory(IDTO):
       working_memory: str  # Structured string containing all working memory components

**Field Descriptions:**

``working_memory: str`` (required)
   A structured string containing all working memory components.

   Should be clearly delineated with sections like:
      - USER CONTEXT: User profile and preferences
      - CONVERSATION FLOW: Current conversation state
      - CURRENT FOCUS: Active topics and goals
      - INTERACTION TONE: Communication style

**Class Methods:**

``initialize_memory() -> IWorkingMemory``
   Create a new working memory state with initial values.

   **Returns:**
      IWorkingMemory: Fresh working memory instance

``to_dict() -> Dict``
   Convert the working memory state to dictionary format for storage.

   **Returns:**
      Dictionary containing working_memory field

``from_dict(data: Dict) -> IWorkingMemory``
   Create working memory state from stored dictionary format.

   **Args:**
      - ``data``: Dictionary with working_memory field

   **Returns:**
      IWorkingMemory: Reconstructed instance

**Example:**

.. code-block:: python

   from arshai.core.interfaces import IWorkingMemory

   # Initialize new memory
   memory = IWorkingMemory.initialize_memory()
   print(memory.working_memory)
   # Output:
   # ### USER CONTEXT:
   # New user with no established profile yet...
   #
   # ### CONVERSATION FLOW:
   # Conversation just initiated...

   # Create custom memory
   custom_memory = IWorkingMemory(
       working_memory="""
       ### USER CONTEXT:
       User is interested in AI and machine learning. Prefers technical details.

       ### CONVERSATION FLOW:
       Currently discussing RAG systems and vector databases.

       ### CURRENT FOCUS:
       Helping user implement document retrieval system.

       ### INTERACTION TONE:
       Technical, detailed, with code examples.
       """
   )

   # Store and retrieve
   memory_dict = custom_memory.to_dict()
   restored_memory = IWorkingMemory.from_dict(memory_dict)

Component Models
----------------

.. _embeddingconfig-model:

EmbeddingConfig
~~~~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/iembedding.py``

Configuration for embedding services.

**Fields:**

.. code-block:: python

   class EmbeddingConfig(IDTO):
       model_name: Optional[str] = None           # Embedding model name
       batch_size: int = 16                       # Batch size for processing
       additional_params: Dict[str, Any] = {}     # Model-specific parameters

**Field Descriptions:**

``model_name: Optional[str]`` (default: None)
   Name of the embedding model to use.

   Examples:
      - OpenAI: ``"text-embedding-3-small"``, ``"text-embedding-3-large"``
      - VoyageAI: ``"voyage-2"``, ``"voyage-code-2"``

``batch_size: int`` (default: 16)
   Number of documents to process in each batch for efficiency.

``additional_params: Dict[str, Any]`` (default: {})
   Additional model-specific parameters (dimensions, encoding format, etc.).

**Example:**

.. code-block:: python

   from arshai.core.interfaces import EmbeddingConfig

   config = EmbeddingConfig(
       model_name="text-embedding-3-small",
       batch_size=32,
       additional_params={"dimensions": 1536}
   )

.. _ivectordbconfig-model:

IVectorDBConfig
~~~~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/ivector_db_client.py``

Configuration for vector database connections.

**Fields:**

.. code-block:: python

   class IVectorDBConfig(IDTO):
       host: str                                  # Required: Database host
       port: str                                  # Required: Database port
       db_name: str                               # Required: Database name
       batch_size: int = 50                       # Batch size for bulk operations
       additional_params: Optional[Dict] = {}     # Database-specific parameters

**Example:**

.. code-block:: python

   from arshai.core.interfaces import IVectorDBConfig

   config = IVectorDBConfig(
       host="localhost",
       port="19530",
       db_name="my_database",
       batch_size=100,
       additional_params={"token": "auth_token"}
   )

.. _icollectionconfig-model:

ICollectionConfig
~~~~~~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/ivector_db_client.py``

Configuration for vector database collections (schema, fields, search capabilities).

**Fields:**

.. code-block:: python

   class ICollectionConfig(IDTO):
       collection_name: str              # Required: Collection name
       dense_dim: int                    # Required: Dense vector dimension
       text_field: str                   # Required: Text content field name
       pk_field: str = "doc_id"          # Primary key field name
       dense_field: str = "dense_vector" # Dense vector field name
       sparse_field: Optional[str] = "sparse_vector"  # Sparse vector field name
       metadata_field: str = "metadata"  # Metadata field name
       schema_model: Optional[Type[IDTO]] = None  # Optional schema validation
       is_hybrid: bool = False           # Enable hybrid search

**Field Descriptions:**

``collection_name: str`` (required)
   Name of the collection.

``dense_dim: int`` (required)
   Dimension of dense vector embeddings (e.g., 1536 for OpenAI text-embedding-3-small).

``text_field: str`` (required)
   Field name for text content.

``pk_field: str`` (default: "doc_id")
   Primary key field name.

``dense_field: str`` (default: "dense_vector")
   Field name for dense vector data.

``sparse_field: Optional[str]`` (default: "sparse_vector")
   Field name for sparse vector data (optional, for hybrid search).

``metadata_field: str`` (default: "metadata")
   Field name for metadata storage.

``schema_model: Optional[Type[IDTO]]`` (default: None)
   Optional Pydantic model for schema validation.

``is_hybrid: bool`` (default: False)
   Whether to enable hybrid search capabilities (dense + sparse vectors).

**Example:**

.. code-block:: python

   from arshai.core.interfaces import ICollectionConfig
   from pydantic import BaseModel

   # Simple collection
   config = ICollectionConfig(
       collection_name="documents",
       dense_dim=1536,
       text_field="content"
   )

   # Hybrid search collection with schema
   class DocumentSchema(IDTO):
       title: str
       content: str
       category: str

   config = ICollectionConfig(
       collection_name="documents",
       dense_dim=1536,
       text_field="content",
       is_hybrid=True,
       schema_model=DocumentSchema
   )

Model Validation
----------------

**Automatic Validation:**

All models inherit Pydantic validation:

.. code-block:: python

   from arshai.core.interfaces import ILLMConfig
   from pydantic import ValidationError

   # Valid configuration
   config = ILLMConfig(model="gpt-4", temperature=0.7)

   # Validation error - temperature out of range
   try:
       config = ILLMConfig(model="gpt-4", temperature=3.0)
   except ValidationError as e:
       print(e)  # Shows validation error details

**Custom Validators:**

Models can include custom validation logic:

.. code-block:: python

   from pydantic import field_validator, model_validator

   class CustomConfig(IDTO):
       value: int

       @field_validator('value')
       @classmethod
       def validate_value(cls, v):
           if v < 0:
               raise ValueError("Value must be non-negative")
           return v

       @model_validator(mode='after')
       def validate_complete_model(self):
           # Cross-field validation
           return self

Model Serialization
-------------------

**To Dictionary:**

.. code-block:: python

   config = ILLMConfig(model="gpt-4", temperature=0.7)

   # Convert to dictionary
   config_dict = config.model_dump()
   # {"model": "gpt-4", "temperature": 0.7, ...}

**To JSON:**

.. code-block:: python

   # Convert to JSON string
   config_json = config.model_dump_json()
   # '{"model":"gpt-4","temperature":0.7,...}'

**From Dictionary:**

.. code-block:: python

   config_dict = {"model": "gpt-4", "temperature": 0.7}
   config = ILLMConfig(**config_dict)

**From JSON:**

.. code-block:: python

   import json

   config_json = '{"model":"gpt-4","temperature":0.7}'
   config_dict = json.loads(config_json)
   config = ILLMConfig(**config_dict)

Complete Model List
-------------------

**Core Models:**

- :ref:`IDTO <idto-model>` - Base DTO class

**LLM Models:**

- :ref:`ILLMConfig <illmconfig-model>` - LLM configuration
- :ref:`ILLMInput <illminput-model>` - LLM input

**Agent Models:**

- :ref:`IAgentInput <iagentinput-model>` - Agent input

**Memory Models:**

- :ref:`ConversationMemoryType <conversationmemorytype-enum>` - Memory type enum
- :ref:`IMemoryInput <imemoryinput-model>` - Memory operation input
- :ref:`IMemoryItem <imemoryitem-model>` - Memory item structure
- :ref:`IWorkingMemory <iworkingmemory-model>` - Working memory state

**Component Models:**

- :ref:`EmbeddingConfig <embeddingconfig-model>` - Embedding configuration
- :ref:`IVectorDBConfig <ivectordbconfig-model>` - Vector DB configuration
- :ref:`ICollectionConfig <icollectionconfig-model>` - Collection configuration

**Workflow Models:**

- ``IWorkflowState`` - Workflow state (see source)
- ``IWorkflowConfig`` - Workflow configuration (see source)
- Node input/output models (see source)

Next Steps
----------

- **Use Models**: See :doc:`../tutorials/index` for practical examples
- **Implement Interfaces**: See :doc:`interfaces` for protocol specifications
- **Extend Base Classes**: See :doc:`base-classes` for foundation classes
- **Build Applications**: See :doc:`../framework/index` for development guides
