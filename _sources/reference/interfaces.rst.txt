Interfaces
==========

Complete reference for all Arshai framework interfaces. All interfaces use Python protocols for flexible, duck-typed implementations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Core Interfaces
---------------

These are the essential protocols that define the framework's primary abstractions.

.. _illm-interface:

ILLM
~~~~

**Location:** ``arshai/core/interfaces/illm.py``

The LLM client interface defines the contract for language model providers. All LLM implementations must satisfy this protocol.

**Protocol Definition:**

.. code-block:: python

   class ILLM(Protocol):
       """Protocol class for LLM providers"""

       def __init__(self, config: ILLMConfig) -> None: ...

       async def chat(self, input: ILLMInput) -> Dict[str, Any]:
           """Main chat interface supporting all functionality"""
           ...

       async def stream(self, input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]:
           """Main streaming interface supporting all functionality"""
           ...

       def _initialize_client(self) -> Any:
           """Initialize the LLM provider client"""
           ...

       def _convert_callables_to_provider_format(
           self,
           functions: Dict[str, Callable]
       ) -> Any:
           """Convert python callables to provider-specific function declarations"""
           ...

**Methods:**

``async chat(input: ILLMInput) -> Dict[str, Any]``
   Main chat interface supporting all functionality including:

   - Regular function calling
   - Background tasks (fire-and-forget execution)
   - Structured output
   - Multi-turn conversations

   **Args:**
      - ``input`` (ILLMInput): Complete input specification

   **Returns:**
      Dictionary containing:
         - ``llm_response`` (str): The generated response
         - ``usage`` (dict): Token usage information
         - ``function_calls`` (list, optional): Function calls made during generation

``async stream(input: ILLMInput) -> AsyncGenerator[Dict[str, Any], None]``
   Streaming version of chat interface. Yields progressive results.

   **Args:**
      - ``input`` (ILLMInput): Complete input specification

   **Yields:**
      Dictionaries containing partial results with same structure as chat()

**Implementation Notes:**

- All LLM providers must support both ``chat()`` and ``stream()`` methods
- Function calling should be handled internally by the provider
- Background tasks execute in fire-and-forget mode using ``asyncio.create_task()``
- Structured output uses provider-specific mechanisms (function calling or native support)

**Example Implementation:**

.. code-block:: python

   from arshai.core.interfaces import ILLM, ILLMInput, ILLMConfig

   class MyLLMClient:
       def __init__(self, config: ILLMConfig):
           self.config = config
           self.client = self._initialize_client()
           self._background_tasks = set()

       async def chat(self, input: ILLMInput) -> Dict[str, Any]:
           # Implementation with function calling support
           response = await self._generate_with_tools(input)
           return {
               "llm_response": response.text,
               "usage": {"total_tokens": response.usage.total_tokens}
           }

       async def stream(self, input: ILLMInput):
           async for chunk in self._stream_with_tools(input):
               yield {"text": chunk.text, "usage": chunk.usage}

       def _initialize_client(self):
           # Provider-specific initialization
           return SomeProviderClient(api_key=self.config.api_key)

**See Also:**
   - :ref:`ILLMInput <illminput-model>` - Input structure
   - :ref:`ILLMConfig <illmconfig-model>` - Configuration model
   - :doc:`../framework/llm-clients/index` - Implementation guide

.. _iagent-interface:

IAgent
~~~~~~

**Location:** ``arshai/core/interfaces/iagent.py``

The agent interface defines the contract for all agents. It provides maximum flexibility for response format and implementation.

**Protocol Definition:**

.. code-block:: python

   class IAgent(Protocol):
       """Agent interface - all agents must implement this protocol"""

       async def process(self, input: IAgentInput) -> Any:
           """
           Process the input and return a response.

           Args:
               input: The input containing message and optional metadata

           Returns:
               Any: Developer-defined response format
           """
           ...

**Methods:**

``async process(input: IAgentInput) -> Any``
   Process user input and return a response. This is the only required method.

   **Args:**
      - ``input`` (IAgentInput): Input containing message and optional metadata

   **Returns:**
      Any: Developer-defined response format. Can be:
         - Simple string
         - Dictionary with structured data
         - Streaming response
         - Custom DTO
         - Tuple (response, metadata)

   **Design Philosophy:**
      The return type is intentionally ``Any`` to give developers full authority over:

      - Response format (streaming, structured, simple string)
      - Data structure (custom DTOs, tuples, dicts)
      - Error handling patterns
      - Additional metadata inclusion

**Implementation Patterns:**

**Simple Response:**

.. code-block:: python

   class SimpleAgent:
       async def process(self, input: IAgentInput) -> str:
           return f"Processed: {input.message}"

**Structured Response:**

.. code-block:: python

   class StructuredAgent:
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           return {
               "response": "...",
               "confidence": 0.95,
               "sources": [...]
           }

**Streaming Response:**

.. code-block:: python

   class StreamingAgent:
       async def process(self, input: IAgentInput) -> AsyncGenerator[str, None]:
           async for chunk in self._generate_streaming(input):
               yield chunk

**With Tools and Memory:**

.. code-block:: python

   from arshai.agents.base import BaseAgent

   class SmartAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt, tools=None):
           super().__init__(llm_client, system_prompt)
           self.tools = tools or []

       async def process(self, input: IAgentInput) -> dict:
           # Agents internally use tools, memory, or any capabilities
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions={tool.name: tool.execute for tool in self.tools}
           )
           result = await self.llm_client.chat(llm_input)
           return {"response": result["llm_response"]}

**See Also:**
   - :ref:`IAgentInput <iagentinput-model>` - Input structure
   - :ref:`BaseAgent <baseagent-class>` - Foundation class
   - :doc:`../framework/agents/index` - Agent development guide

.. _imemorymanager-interface:

IMemoryManager
~~~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/imemorymanager.py``

Universal interface for memory management systems providing unified approach to handling different memory types.

**Protocol Definition:**

.. code-block:: python

   class IMemoryManager(Protocol):
       """Universal interface for memory management systems"""

       def store(self, input: IMemoryInput) -> str:
           """Store any type of memory data"""
           ...

       def retrieve(self, input: IMemoryInput) -> List[IWorkingMemory]:
           """Retrieve any type of memory data"""
           ...

       def update(self, input: IMemoryInput) -> None:
           """Update any type of memory data"""
           ...

       def delete(self, input: IMemoryInput) -> None:
           """Delete memory data"""
           ...

**Methods:**

``store(input: IMemoryInput) -> str``
   Universal method to store any type of memory data.

   **Args:**
      - ``input`` (IMemoryInput): Memory data and metadata for storage

   **Returns:**
      str: Unique identifier for the stored memory

``retrieve(input: IMemoryInput) -> List[IWorkingMemory]``
   Universal method to retrieve any type of memory data.

   **Args:**
      - ``input`` (IMemoryInput): Query and retrieval parameters

   **Returns:**
      List[IWorkingMemory]: Matching memory entries

``update(input: IMemoryInput) -> None``
   Universal method to update any type of memory data.

   **Args:**
      - ``input`` (IMemoryInput): Memory ID and update data

``delete(input: IMemoryInput) -> None``
   Universal method to delete memory data.

   **Args:**
      - ``input`` (IMemoryInput): Memory ID or deletion criteria

**Memory Types:**

The interface supports three memory types via ``ConversationMemoryType`` enum:

- ``SHORT_TERM_MEMORY``: Recent conversations and temporary information
- ``LONG_TERM_MEMORY``: Persistent knowledge and important information
- ``WORKING_MEMORY``: Current conversation state and context

**Example Usage:**

.. code-block:: python

   from arshai.core.interfaces import IMemoryManager, IMemoryInput, IWorkingMemory
   from arshai.core.interfaces import ConversationMemoryType

   # Store working memory
   memory_input = IMemoryInput(
       conversation_id="user_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY,
       data=[IWorkingMemory(working_memory="User prefers concise answers")]
   )
   memory_id = memory_manager.store(memory_input)

   # Retrieve working memory
   retrieve_input = IMemoryInput(
       conversation_id="user_123",
       memory_type=ConversationMemoryType.WORKING_MEMORY
   )
   memories = memory_manager.retrieve(retrieve_input)

**Implementations:**

- ``InMemoryManager``: For development/testing (see :doc:`../implementations/memory/in-memory`)
- ``RedisWorkingMemoryManager``: For production (see :doc:`../implementations/memory/redis-memory`)

**See Also:**
   - :ref:`IMemoryInput <imemoryinput-model>` - Memory operation input
   - :ref:`IWorkingMemory <iworkingmemory-model>` - Working memory structure
   - :doc:`../framework/memory/index` - Memory systems guide

Component Interfaces
--------------------

Specialized protocols for specific capabilities.

.. _iembedding-interface:

IEmbedding
~~~~~~~~~~

**Location:** ``arshai/core/interfaces/iembedding.py``

Interface for embedding services that convert text to vector representations.

**Protocol Definition:**

.. code-block:: python

   class IEmbedding(Protocol):
       """Interface for embedding services"""

       @property
       def dimension(self) -> int:
           """Get the dimension of embeddings produced"""
           ...

       def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
           """Generate embeddings for multiple documents"""
           ...

       def embed_document(self, text: str) -> Dict[str, Any]:
           """Generate embeddings for a single document"""
           ...

**Methods:**

``dimension: int`` (property)
   Get the dimension of embeddings produced by this service.

   **Returns:**
      int: Dimension of the embedding vectors (e.g., 1536 for OpenAI text-embedding-3-small)

``embed_documents(texts: List[str]) -> Dict[str, Any]``
   Generate embeddings for multiple documents efficiently (batched).

   **Args:**
      - ``texts`` (List[str]): List of text documents to embed

   **Returns:**
      Dictionary containing embeddings with keys:
         - ``dense`` (List[List[float]]): Dense vector embeddings
         - ``sparse`` (List[Dict], optional): Sparse vector embeddings (if supported)

``embed_document(text: str) -> Dict[str, Any]``
   Generate embeddings for a single document.

   **Args:**
      - ``text`` (str): Text document to embed

   **Returns:**
      Dictionary containing embeddings with keys:
         - ``dense`` (List[float]): Dense vector embedding
         - ``sparse`` (Dict, optional): Sparse vector embedding (if supported)

**Example Implementation:**

.. code-block:: python

   from arshai.core.interfaces import IEmbedding
   from openai import OpenAI

   class OpenAIEmbedding:
       def __init__(self, model_name: str = "text-embedding-3-small"):
           self.model_name = model_name
           self.client = OpenAI()
           self._dimension = 1536  # For text-embedding-3-small

       @property
       def dimension(self) -> int:
           return self._dimension

       def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
           response = self.client.embeddings.create(
               model=self.model_name,
               input=texts
           )
           embeddings = [item.embedding for item in response.data]
           return {"dense": embeddings}

       def embed_document(self, text: str) -> Dict[str, Any]:
           result = self.embed_documents([text])
           return {"dense": result["dense"][0]}

**Implementations:**

- ``OpenAIEmbedding``: OpenAI embedding models
- ``VoyageAIEmbedding``: VoyageAI models with sparse vector support
- ``MGTEEmbedding``: Multilingual models

**See Also:**
   - :ref:`EmbeddingConfig <embeddingconfig-model>` - Configuration model
   - :doc:`../implementations/components/embeddings` - Implementation guide

.. _ivectordbclient-interface:

IVectorDBClient
~~~~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/ivector_db_client.py``

Interface for vector database clients combining general database operations with vector-specific operations.

**Protocol Definition:**

.. code-block:: python

   class IVectorDBClient(Protocol):
       """Interface for vector database clients"""

       def __init__(self, config: Any) -> None: ...
       def connect(self) -> None: ...
       def disconnect(self) -> None: ...
       def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]: ...
       def insert(self, data: Dict[str, Any]) -> bool: ...
       def update(self, query_params: Dict[str, Any], data: Dict[str, Any]) -> bool: ...
       def delete(self, query_params: Dict[str, Any]) -> bool: ...

       # Vector-specific operations
       def get_or_create_collection(self, config: ICollectionConfig) -> Any: ...
       def get_collection_stats(self, config: ICollectionConfig) -> Dict[str, Any]: ...
       def insert_entity(self, config: ICollectionConfig, entity: Dict, embeddings: Dict) -> None: ...
       def insert_entities(self, config: ICollectionConfig, data: List[Dict], embeddings: Dict) -> None: ...
       def search_by_vector(self, config: ICollectionConfig, query_vectors: List[List[float]], **kwargs) -> List[Dict]: ...
       def hybrid_search(self, config: ICollectionConfig, dense_vectors: List[List[float]], sparse_vectors: List[Dict], **kwargs) -> List[Dict]: ...
       def delete_entity(self, config: ICollectionConfig, filter_expr: str) -> Any: ...

**Key Methods:**

``connect() -> None``
   Establish connection to the vector database.

``disconnect() -> None``
   Close the vector database connection.

``get_or_create_collection(config: ICollectionConfig) -> Any``
   Get an existing collection or create a new one if it doesn't exist.

   **Args:**
      - ``config`` (ICollectionConfig): Collection configuration including schema

``search_by_vector(config: ICollectionConfig, query_vectors: List[List[float]], **kwargs) -> List[Dict]``
   Search for similar vectors in the collection.

   **Args:**
      - ``config``: Collection configuration
      - ``query_vectors``: Query vectors to search for
      - ``limit`` (int): Maximum results (default: 10)
      - ``expr`` (str, optional): Filter expression
      - ``output_fields`` (List[str], optional): Fields to return

   **Returns:**
      List of search results with distances and metadata

``hybrid_search(config: ICollectionConfig, dense_vectors: List, sparse_vectors: List, **kwargs) -> List[Dict]``
   Perform hybrid search using both dense and sparse vectors.

   **Args:**
      - ``dense_vectors``: Dense vectors for search
      - ``sparse_vectors``: Sparse vectors for search
      - ``weights``: Weights for dense and sparse [dense_weight, sparse_weight]
      - ``limit`` (int): Maximum results

**Example Usage:**

.. code-block:: python

   from arshai.vector_db.milvus import MilvusClient
   from arshai.core.interfaces import ICollectionConfig

   # Initialize client
   client = MilvusClient(config)
   client.connect()

   # Create collection
   collection_config = ICollectionConfig(
       collection_name="documents",
       dense_dim=1536,
       text_field="content",
       is_hybrid=True
   )
   collection = client.get_or_create_collection(collection_config)

   # Search
   results = client.search_by_vector(
       config=collection_config,
       query_vectors=[query_embedding],
       limit=5,
       expr='metadata["category"] == "technical"'
   )

**See Also:**
   - :ref:`IVectorDBConfig <ivectordbconfig-model>` - Database configuration
   - :ref:`ICollectionConfig <icollectionconfig-model>` - Collection configuration
   - :doc:`../implementations/components/vector-databases` - Implementation guide

Orchestration Interfaces
-------------------------

Protocols for workflow and system coordination.

.. _iworkflowrunner-interface:

IWorkflowRunner
~~~~~~~~~~~~~~~

**Location:** ``arshai/core/interfaces/iworkflowrunner.py``

Interface for workflow runners that execute workflows and manage state.

**Protocol Definition:**

.. code-block:: python

   class IWorkflowRunner(Protocol):
       """Interface for workflow runners"""

       def __init__(
           self,
           workflow_config: IWorkflowConfig,
           debug_mode: bool = False,
           **kwargs: Any
       ): ...

       async def execute_workflow(
           self,
           user_id: str,
           input_data: Dict[str, Any],
           callbacks: Optional[Dict[str, Any]] = None
       ) -> Dict[str, Any]: ...

**Methods:**

``async execute_workflow(user_id: str, input_data: Dict, callbacks: Optional[Dict] = None) -> Dict``
   Execute a workflow with the given input.

   **Process:**
      1. Initializes workflow state if not provided
      2. Prepares input data with state and callbacks
      3. Executes the workflow
      4. Processes and returns the results

   **Args:**
      - ``user_id`` (str): User initiating the workflow
      - ``input_data`` (Dict): Input data for the workflow
      - ``callbacks`` (Dict, optional): Callback functions for workflow execution

   **Returns:**
      Dict with workflow execution results including state

**Example Usage:**

.. code-block:: python

   from arshai.workflows.runner import WorkflowRunner
   from arshai.workflows.config import WorkflowConfig

   # Create workflow configuration
   workflow_config = WorkflowConfig(
       name="customer_support",
       nodes=[triage_node, support_node, response_node]
   )

   # Initialize runner
   runner = WorkflowRunner(workflow_config, debug_mode=True)

   # Execute workflow
   result = await runner.execute_workflow(
       user_id="user_123",
       input_data={"message": "I need help with billing"},
       callbacks={"notify_admin": notify_function}
   )

**See Also:**
   - :doc:`../implementations/orchestration/workflow-system` - Workflow system guide
   - :doc:`../framework/building-systems/index` - System building guide

Infrastructure Interfaces
--------------------------

Supporting protocols for notifications and utilities.

.. _idto-interface:

IDTO
~~~~

**Location:** ``arshai/core/interfaces/idto.py``

Base class for all Data Transfer Objects using Pydantic.

**Definition:**

.. code-block:: python

   from pydantic import BaseModel

   class IDTO(BaseModel):
       """Base class for all DTOs with Pydantic validation"""

       class Config:
           arbitrary_types_allowed = True
           extra = "forbid"  # Prevent extra fields

**Usage:**

All framework DTOs inherit from IDTO to get:

- Automatic field validation
- Type checking
- JSON serialization/deserialization
- Immutability (by default)

**Example:**

.. code-block:: python

   from arshai.core.interfaces import IDTO
   from pydantic import Field

   class MyCustomDTO(IDTO):
       name: str = Field(description="User name")
       age: int = Field(ge=0, le=150, description="User age")
       email: Optional[str] = Field(default=None, description="Email address")

**See Also:**
   - :doc:`models` - All DTO models

Complete Interface List
-----------------------

**Core Interfaces:**

- :ref:`ILLM <illm-interface>` - LLM client interface
- :ref:`IAgent <iagent-interface>` - Agent interface
- :ref:`IMemoryManager <imemorymanager-interface>` - Memory management

**Component Interfaces:**

- :ref:`IEmbedding <iembedding-interface>` - Embedding generation
- :ref:`IVectorDBClient <ivectordbclient-interface>` - Vector database operations
- ``IDocumentProcessor`` - Document processing (see source)
- ``IReranker`` - Result reranking (see source)
- ``IWebSearch`` - Web search integration (see source)
- ``INotificationService`` - Notification delivery (see source)

**Orchestration Interfaces:**

- ``IWorkflowOrchestrator`` - Workflow orchestration (see source)
- :ref:`IWorkflowRunner <iworkflowrunner-interface>` - Workflow execution
- ``IWorkflowNode`` - Workflow nodes (see source)
- ``IWorkflowState`` - Workflow state (see source)
- ``IWorkflowConfig`` - Workflow configuration (see source)

**Infrastructure Interfaces:**

- :ref:`IDTO <idto-interface>` - Data transfer object base
- ``IStreamDTO`` - Streaming DTO base (see source)

Interface Source Files
----------------------

All interfaces are located in ``arshai/core/interfaces/``:

.. code-block:: text

   arshai/core/interfaces/
   ├── __init__.py
   ├── iagent.py            # Agent interface
   ├── idocument.py         # Document processing
   ├── idto.py              # DTO base classes
   ├── iembedding.py        # Embedding interface
   ├── illm.py              # LLM client interface
   ├── imemorymanager.py    # Memory management
   ├── inotification.py     # Notifications
   ├── ireranker.py         # Reranking
   ├── ivector_db_client.py # Vector database
   ├── iwebsearch.py        # Web search
   ├── iworkflow.py         # Workflow definitions
   └── iworkflowrunner.py   # Workflow execution

Next Steps
----------

- **Implement an Interface**: See :doc:`../tutorials/index` for complete examples
- **Explore Base Classes**: See :doc:`base-classes` for foundation implementations
- **Review Models**: See :doc:`models` for all DTO structures
- **Build Agents**: See :doc:`../framework/agents/index` for agent development
