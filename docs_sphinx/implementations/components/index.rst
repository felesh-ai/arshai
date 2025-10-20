Component Reference Implementations
===================================

This section documents the reference component implementations provided with Arshai. These demonstrate how to implement various framework interfaces for embeddings, vector databases, and other system components.

.. toctree::
   :maxdepth: 2
   :caption: Component Implementations
   
   embeddings
   vector-databases

.. note::
   **Reference Implementation Philosophy**
   
   These component implementations are **not part of the core framework**. They represent working examples of how to implement framework interfaces for different providers and use cases. You can:
   
   - Use them directly if they support your required providers
   - Modify them for your specific integration needs
   - Learn implementation patterns to build your own components
   - Combine multiple implementations for different scenarios

Available Reference Implementations
-----------------------------------

**Embedding Implementations** (:doc:`embeddings`)
   Working implementations for different embedding providers: OpenAI, VoyageAI, and MGTE. Demonstrate the ``IEmbedding`` interface implementation patterns.

**Vector Database Implementations** (:doc:`vector-databases`)
   Production-ready Milvus vector database client showing how to implement the ``IVectorDBClient`` interface.

Component Integration Patterns
-------------------------------

**Interface Implementation**
   How reference components properly implement framework interfaces to ensure compatibility and consistency.

**Provider Abstraction**
   Patterns for abstracting different service providers behind common interfaces.

**Configuration Management**
   How components handle configuration, credentials, and environment-specific settings.

**Error Handling**
   Robust error handling patterns that gracefully handle provider-specific failures.

**Async Operations**
   How components implement asynchronous operations for better performance and scalability.

Framework Interface Compliance
-------------------------------

All reference component implementations follow their respective framework interfaces:

**Embedding Interface (IEmbedding)**

.. code-block:: python

   from arshai.core.interfaces.iembedding import IEmbedding, EmbeddingConfig
   
   class IEmbedding:
       """Interface for embedding implementations."""
       
       @property
       def dimension(self) -> int:
           """Get the dimension of embeddings produced by this service."""
           pass
       
       def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
           """Generate embeddings for multiple documents."""
           pass
       
       def embed_document(self, text: str) -> Dict[str, Any]:
           """Generate embeddings for a single document."""
           pass
       
       async def aembed_documents(self, texts: List[str]) -> Dict[str, Any]:
           """Asynchronously generate embeddings for multiple documents."""
           pass

**Vector Database Interface (IVectorDBClient)**

.. code-block:: python

   from arshai.core.interfaces.ivector_db_client import IVectorDBClient, ICollectionConfig
   
   class IVectorDBClient:
       """Interface for vector database implementations."""
       
       def connect(self):
           """Connect to the vector database."""
           pass
       
       def get_or_create_collection(self, config: ICollectionConfig):
           """Get existing collection or create new one."""
           pass
       
       def insert_entities(self, config: ICollectionConfig, data: list, documents_embedding):
           """Insert documents with embeddings into collection."""
           pass
       
       def search_by_vector(self, config: ICollectionConfig, query_vectors, **kwargs):
           """Search documents using vector similarity."""
           pass
       
       def hybrid_search(self, config: ICollectionConfig, dense_vectors=None, sparse_vectors=None, **kwargs):
           """Perform hybrid search using multiple vector types."""
           pass

Basic Usage Patterns
---------------------

**Embedding Usage**

.. code-block:: python

   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig
   
   # Create embedding configuration
   config = EmbeddingConfig(
       model_name="text-embedding-3-small",
       batch_size=32
   )
   
   # Create embedding service
   embedding_service = OpenAIEmbedding(config)
   
   # Generate embeddings
   texts = ["Hello world", "How are you?", "Machine learning is fascinating"]
   embeddings = embedding_service.embed_documents(texts)
   
   print(f"Generated {len(embeddings['dense'])} embeddings")
   print(f"Embedding dimension: {embedding_service.dimension}")

**Vector Database Usage**

.. code-block:: python

   from arshai.vector_db.milvus_client import MilvusClient
   from arshai.core.interfaces.ivector_db_client import ICollectionConfig
   import os
   
   # Set environment variables
   os.environ["MILVUS_HOST"] = "localhost"
   os.environ["MILVUS_PORT"] = "19530"
   os.environ["MILVUS_DB_NAME"] = "default"
   
   # Create vector database client
   vector_client = MilvusClient()
   
   # Configure collection
   collection_config = ICollectionConfig(
       collection_name="my_documents",
       dense_dim=1536,  # For OpenAI embeddings
       is_hybrid=False  # Dense vectors only
   )
   
   # Prepare documents and embeddings
   documents = [
       {"content": "Document 1 content", "metadata": {"source": "file1.txt"}},
       {"content": "Document 2 content", "metadata": {"source": "file2.txt"}}
   ]
   
   # Get embeddings
   texts = [doc["content"] for doc in documents]
   embeddings = embedding_service.embed_documents(texts)
   
   # Insert into vector database
   vector_client.insert_entities(
       config=collection_config,
       data=documents,
       documents_embedding=embeddings
   )
   
   # Search similar documents
   query_text = "Tell me about document content"
   query_embedding = embedding_service.embed_document(query_text)
   
   results = vector_client.search_by_vector(
       config=collection_config,
       query_vectors=[query_embedding["dense"]],
       limit=5
   )

**Combined RAG System**

.. code-block:: python

   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.vector_db.milvus_client import MilvusClient
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLMInput
   
   class RAGAgent(BaseAgent):
       """Agent that uses embeddings and vector search for RAG."""
       
       def __init__(self, llm_client, embedding_service, vector_client, collection_config):
           super().__init__(llm_client, "You are a helpful assistant with access to documents")
           self.embedding_service = embedding_service
           self.vector_client = vector_client
           self.collection_config = collection_config
       
       async def process(self, input: IAgentInput) -> str:
           # Get query embedding
           query_embedding = self.embedding_service.embed_document(input.message)
           
           # Search for relevant documents
           search_results = self.vector_client.search_by_vector(
               config=self.collection_config,
               query_vectors=[query_embedding["dense"]],
               limit=3
           )
           
           # Extract relevant content
           context_docs = []
           for result in search_results[0]:  # First query results
               content = result.entity.get("content", "")
               context_docs.append(content)
           
           context = "\n\n".join(context_docs)
           
           # Generate response with context
           llm_input = ILLMInput(
               system_prompt=f"{self.system_prompt}\n\nRelevant context:\n{context}",
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

Component Extension Patterns
-----------------------------

**Custom Embedding Provider**

.. code-block:: python

   from arshai.core.interfaces.iembedding import IEmbedding, EmbeddingConfig
   from typing import List, Dict, Any
   import requests
   
   class CustomEmbeddingProvider(IEmbedding):
       """Custom embedding provider implementation."""
       
       def __init__(self, config: EmbeddingConfig):
           self.api_endpoint = config.model_name  # Using model_name for endpoint
           self.batch_size = config.batch_size
           self._dimension = 768  # Custom provider dimension
       
       @property
       def dimension(self) -> int:
           return self._dimension
       
       def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
           """Generate embeddings using custom API."""
           embeddings = []
           
           for i in range(0, len(texts), self.batch_size):
               batch = texts[i:i + self.batch_size]
               
               # Call custom embedding API
               response = requests.post(
                   self.api_endpoint,
                   json={"texts": batch},
                   headers={"Authorization": f"Bearer {self.api_key}"}
               )
               
               batch_embeddings = response.json()["embeddings"]
               embeddings.extend(batch_embeddings)
           
           return {"dense": embeddings}
       
       def embed_document(self, text: str) -> Dict[str, Any]:
           embeddings = self.embed_documents([text])
           return {"dense": embeddings["dense"][0]}
       
       async def aembed_documents(self, texts: List[str]) -> Dict[str, Any]:
           # Implement async version
           import asyncio
           import aiohttp
           
           async with aiohttp.ClientSession() as session:
               tasks = []
               for i in range(0, len(texts), self.batch_size):
                   batch = texts[i:i + self.batch_size]
                   task = self._async_embed_batch(session, batch)
                   tasks.append(task)
               
               batch_results = await asyncio.gather(*tasks)
               
               embeddings = []
               for batch_embeddings in batch_results:
                   embeddings.extend(batch_embeddings)
               
               return {"dense": embeddings}
       
       async def _async_embed_batch(self, session, texts):
           async with session.post(
               self.api_endpoint,
               json={"texts": texts},
               headers={"Authorization": f"Bearer {self.api_key}"}
           ) as response:
               result = await response.json()
               return result["embeddings"]

**Custom Vector Database Client**

.. code-block:: python

   from arshai.core.interfaces.ivector_db_client import IVectorDBClient, ICollectionConfig
   from typing import List, Dict, Any
   import sqlite3
   import numpy as np
   import json
   
   class SQLiteVectorClient(IVectorDBClient):
       """Simple SQLite-based vector database for development."""
       
       def __init__(self, db_path: str = "vectors.db"):
           self.db_path = db_path
           self.connection = None
       
       def connect(self):
           """Connect to SQLite database."""
           self.connection = sqlite3.connect(self.db_path)
           self.connection.execute("PRAGMA journal_mode=WAL")
       
       def get_or_create_collection(self, config: ICollectionConfig):
           """Create table if it doesn't exist."""
           if not self.connection:
               self.connect()
           
           cursor = self.connection.cursor()
           cursor.execute(f"""
               CREATE TABLE IF NOT EXISTS {config.collection_name} (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   content TEXT NOT NULL,
                   metadata TEXT,
                   vector BLOB,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               )
           """)
           self.connection.commit()
           
           return config.collection_name
       
       def insert_entities(self, config: ICollectionConfig, data: List[Dict], documents_embedding: Dict[str, Any]):
           """Insert documents with embeddings."""
           cursor = self.connection.cursor()
           
           for i, doc in enumerate(data):
               vector_blob = np.array(documents_embedding["dense"][i]).tobytes()
               
               cursor.execute(f"""
                   INSERT INTO {config.collection_name} 
                   (content, metadata, vector) VALUES (?, ?, ?)
               """, (
                   doc["content"],
                   json.dumps(doc.get("metadata", {})),
                   vector_blob
               ))
           
           self.connection.commit()
       
       def search_by_vector(self, config: ICollectionConfig, query_vectors: List[List[float]], limit: int = 5, **kwargs):
           """Simple cosine similarity search."""
           query_vector = np.array(query_vectors[0])
           
           cursor = self.connection.cursor()
           cursor.execute(f"""
               SELECT id, content, metadata, vector FROM {config.collection_name}
           """)
           
           results = []
           for row in cursor.fetchall():
               doc_id, content, metadata, vector_blob = row
               doc_vector = np.frombuffer(vector_blob, dtype=np.float32)
               
               # Calculate cosine similarity
               similarity = np.dot(query_vector, doc_vector) / (
                   np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
               )
               
               results.append({
                   "id": doc_id,
                   "content": content,
                   "metadata": json.loads(metadata),
                   "similarity": similarity
               })
           
           # Sort by similarity and return top results
           results.sort(key=lambda x: x["similarity"], reverse=True)
           return [results[:limit]]

**Configuration-Driven Component Factory**

.. code-block:: python

   from typing import Dict, Type, Any
   from arshai.core.interfaces.iembedding import IEmbedding, EmbeddingConfig
   
   class ComponentFactory:
       """Factory for creating components based on configuration."""
       
       def __init__(self):
           self.embedding_providers: Dict[str, Type[IEmbedding]] = {}
           self.vector_db_providers: Dict[str, Type[IVectorDBClient]] = {}
       
       def register_embedding_provider(self, name: str, provider_class: Type[IEmbedding]):
           """Register an embedding provider."""
           self.embedding_providers[name] = provider_class
       
       def register_vector_db_provider(self, name: str, provider_class: Type[IVectorDBClient]):
           """Register a vector database provider."""
           self.vector_db_providers[name] = provider_class
       
       def create_embedding_service(self, provider: str, config: EmbeddingConfig) -> IEmbedding:
           """Create embedding service based on provider name."""
           if provider not in self.embedding_providers:
               raise ValueError(f"Unknown embedding provider: {provider}")
           
           provider_class = self.embedding_providers[provider]
           return provider_class(config)
       
       def create_vector_db_client(self, provider: str, **kwargs) -> IVectorDBClient:
           """Create vector database client based on provider name."""
           if provider not in self.vector_db_providers:
               raise ValueError(f"Unknown vector database provider: {provider}")
           
           provider_class = self.vector_db_providers[provider]
           return provider_class(**kwargs)
   
   # Usage
   factory = ComponentFactory()
   
   # Register providers
   factory.register_embedding_provider("openai", OpenAIEmbedding)
   factory.register_embedding_provider("custom", CustomEmbeddingProvider)
   factory.register_vector_db_provider("milvus", MilvusClient)
   factory.register_vector_db_provider("sqlite", SQLiteVectorClient)
   
   # Create components from configuration
   embedding_config = EmbeddingConfig(model_name="text-embedding-3-small")
   embedding_service = factory.create_embedding_service("openai", embedding_config)
   vector_client = factory.create_vector_db_client("milvus")

Testing Component Implementations
----------------------------------

**Unit Testing Embeddings**

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch
   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig
   
   @pytest.fixture
   def embedding_config():
       return EmbeddingConfig(
           model_name="text-embedding-3-small",
           batch_size=2
       )
   
   @pytest.fixture
   def mock_openai_response():
       return Mock(data=[
           Mock(embedding=[0.1, 0.2, 0.3]),
           Mock(embedding=[0.4, 0.5, 0.6])
       ])
   
   @patch('arshai.embeddings.openai_embeddings.OpenAI')
   def test_embed_documents(mock_openai_class, embedding_config, mock_openai_response):
       # Setup mock
       mock_client = Mock()
       mock_openai_class.return_value = mock_client
       mock_client.embeddings.create.return_value = mock_openai_response
       
       # Create embedding service
       with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
           embedding_service = OpenAIEmbedding(embedding_config)
       
       # Test embedding generation
       texts = ["Hello world", "How are you?"]
       result = embedding_service.embed_documents(texts)
       
       # Verify results
       assert "dense" in result
       assert len(result["dense"]) == 2
       assert result["dense"][0] == [0.1, 0.2, 0.3]
       assert result["dense"][1] == [0.4, 0.5, 0.6]
       
       # Verify API call
       mock_client.embeddings.create.assert_called_once_with(
           model="text-embedding-3-small",
           input=texts,
           encoding_format="float"
       )

**Integration Testing Vector Database**

.. code-block:: python

   import pytest
   import os
   from arshai.vector_db.milvus_client import MilvusClient
   from arshai.core.interfaces.ivector_db_client import ICollectionConfig
   
   @pytest.mark.integration
   def test_milvus_integration():
       # Set up test environment
       os.environ["MILVUS_HOST"] = "localhost"
       os.environ["MILVUS_PORT"] = "19530"
       os.environ["MILVUS_DB_NAME"] = "test_db"
       
       # Create client and config
       client = MilvusClient()
       config = ICollectionConfig(
           collection_name="test_collection",
           dense_dim=3,
           is_hybrid=False
       )
       
       try:
           # Test collection creation
           collection = client.get_or_create_collection(config)
           assert collection is not None
           
           # Test document insertion
           documents = [
               {"content": "Test document 1", "metadata": {"type": "test"}},
               {"content": "Test document 2", "metadata": {"type": "test"}}
           ]
           
           embeddings = {
               "dense": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
           }
           
           client.insert_entities(config, documents, embeddings)
           
           # Test search
           query_vector = [0.1, 0.2, 0.3]
           results = client.search_by_vector(
               config=config,
               query_vectors=[query_vector],
               limit=2
           )
           
           assert len(results) > 0
           assert len(results[0]) > 0
           
       finally:
           # Cleanup - delete test collection
           try:
               client.delete_entity(config, "metadata['type'] == 'test'")
           except:
               pass

**Performance Testing**

.. code-block:: python

   import time
   import asyncio
   
   def test_embedding_performance():
       embedding_service = OpenAIEmbedding(EmbeddingConfig(
           model_name="text-embedding-3-small",
           batch_size=100
       ))
       
       # Generate test data
       texts = [f"Test document {i}" for i in range(1000)]
       
       # Test synchronous performance
       start_time = time.time()
       embeddings = embedding_service.embed_documents(texts)
       sync_time = time.time() - start_time
       
       print(f"Synchronous embedding of 1000 texts: {sync_time:.2f} seconds")
       assert len(embeddings["dense"]) == 1000
       
   @pytest.mark.asyncio
   async def test_async_embedding_performance():
       embedding_service = OpenAIEmbedding(EmbeddingConfig(
           model_name="text-embedding-3-small",
           batch_size=100
       ))
       
       texts = [f"Test document {i}" for i in range(1000)]
       
       # Test asynchronous performance
       start_time = time.time()
       embeddings = await embedding_service.aembed_documents(texts)
       async_time = time.time() - start_time
       
       print(f"Asynchronous embedding of 1000 texts: {async_time:.2f} seconds")
       assert len(embeddings["dense"]) == 1000

Best Practices for Component Implementation
-------------------------------------------

**Interface Compliance**
   - Implement all required interface methods completely
   - Follow the exact method signatures defined in interfaces
   - Return data in the expected format and structure
   - Handle errors gracefully without breaking interface contracts

**Configuration Management**
   - Use environment variables for sensitive configuration like API keys
   - Provide sensible defaults for optional configuration
   - Validate configuration at initialization time
   - Support both programmatic and environment-based configuration

**Error Handling**
   - Handle provider-specific errors and translate to meaningful messages
   - Implement appropriate retry logic for transient failures
   - Log errors with sufficient context for debugging
   - Fail gracefully without exposing sensitive information

**Performance Optimization**
   - Implement batching for operations that support it
   - Use connection pooling for database/API connections
   - Implement appropriate caching strategies
   - Support asynchronous operations where beneficial

**Testing and Reliability**
   - Write comprehensive unit tests with mocked dependencies
   - Include integration tests with real services
   - Test error conditions and edge cases
   - Monitor performance characteristics

The reference component implementations provide solid foundations for integrating different providers and services with the Arshai framework. Use them as starting points for your own integrations or as complete solutions if they meet your needs.