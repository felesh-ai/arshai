Vector Database - Milvus Client
================================

The Milvus client is a reference implementation for storing and searching vector embeddings. It demonstrates how to implement the ``IVectorDBClient`` interface for Milvus, a popular open-source vector database.

.. important::

   **This is a Reference Implementation**

   The Milvus client shows **one way** to implement vector storage in Arshai. You can:

   - Use it as-is for Milvus deployments
   - Extend it for your specific needs
   - Build your own implementation for other vector databases (Pinecone, Weaviate, Qdrant, etc.)
   - Implement ``IVectorDBClient`` for any vector storage backend

   The framework provides the interface - you choose the implementation.

Overview
--------

The Milvus client provides:

- **Collection Management**: Create and configure collections
- **Vector Storage**: Store embeddings with metadata
- **Similarity Search**: Find similar vectors
- **Hybrid Search**: Combine dense and sparse vectors
- **Batch Operations**: Efficient bulk insert and search
- **Schema Generation**: Automatic schema from Pydantic models

**Use Cases:**

- Semantic search applications
- Retrieval-Augmented Generation (RAG)
- Recommendation systems
- Image and document similarity
- Question answering systems

Installation
------------

Install with Milvus support:

.. code-block:: bash

   # Install Arshai with Milvus
   pip install arshai[milvus]

   # Or install separately
   pip install arshai pymilvus

Milvus Setup
------------

Local Development
~~~~~~~~~~~~~~~~~

Using Docker Compose:

.. code-block:: bash

   # Download docker-compose file
   wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

   # Start Milvus
   docker-compose up -d

   # Verify it's running
   docker-compose ps

Using Milvus Lite (Python-only):

.. code-block:: bash

   # Install Milvus Lite
   pip install milvus

   # No separate server needed - runs in process

Environment Configuration:

.. code-block:: bash

   # Set Milvus connection
   export MILVUS_HOST="localhost"
   export MILVUS_PORT="19530"
   export MILVUS_DB_NAME="default"
   export MILVUS_BATCH_SIZE="50"

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

**Managed Milvus (Zilliz Cloud)**

.. code-block:: bash

   export MILVUS_HOST="your-cluster.api.gcp-us-west1.zillizcloud.com"
   export MILVUS_PORT="19530"
   export MILVUS_API_KEY="your-api-key"

**Self-Hosted Kubernetes**

.. code-block:: bash

   # Deploy with Helm
   helm repo add milvus https://zilliztech.github.io/milvus-helm/
   helm install my-release milvus/milvus

Basic Usage
-----------

Simple Example
~~~~~~~~~~~~~~

.. code-block:: python

   import os
   from arshai.vector_db.milvus_client import MilvusClient
   from arshai.core.interfaces.ivector_db_client import ICollectionConfig
   from pydantic import BaseModel

   # Define your data model
   class Document(BaseModel):
       id: int
       content: str
       embedding: list  # Vector field

   # Create Milvus client
   os.environ.update({
       "MILVUS_HOST": "localhost",
       "MILVUS_PORT": "19530",
       "MILVUS_DB_NAME": "default"
   })

   client = MilvusClient()

   # Create collection configuration
   config = ICollectionConfig(
       collection_name="my_documents",
       dimension=1536,  # Match your embedding model
       model_class=Document
   )

   # Create or get collection
   collection = client.get_or_create_collection(config)

   print(f"Collection '{collection.name}' ready with {collection.num_entities} documents")

Complete RAG Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os
   from arshai.vector_db.milvus_client import MilvusClient
   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig
   from arshai.core.interfaces.ivector_db_client import ICollectionConfig
   from pydantic import BaseModel
   from typing import List

   # 1. Define data model
   class DocumentChunk(BaseModel):
       id: int
       text: str
       source: str
       embedding: List[float]

   # 2. Initialize components
   embedder = OpenAIEmbedding(
       EmbeddingConfig(model_name="text-embedding-3-small")
   )

   client = MilvusClient()

   # 3. Create collection
   config = ICollectionConfig(
       collection_name="knowledge_base",
       dimension=embedder.dimension,
       model_class=DocumentChunk
   )

   collection = client.get_or_create_collection(config)

   # 4. Insert documents
   documents = [
       "Artificial intelligence is transforming technology",
       "Machine learning powers modern AI systems",
       "Deep learning uses neural networks for pattern recognition"
   ]

   # Generate embeddings
   embedding_result = embedder.embed_documents(documents)

   # Prepare data
   data = [
       DocumentChunk(
           id=i,
           text=doc,
           source="example.txt",
           embedding=emb
       )
       for i, (doc, emb) in enumerate(zip(documents, embedding_result['embeddings']))
   ]

   # Insert into Milvus
   client.insert(config, [d.dict() for d in data])

   # 5. Search
   query = "What is AI?"
   query_result = embedder.embed_query(query)
   query_embedding = query_result['embedding']

   # Search similar documents
   search_results = client.search(
       config=config,
       query_vectors=[query_embedding],
       limit=3
   )

   print("\nSearch Results:")
   for result in search_results[0]:
       print(f"Score: {result['distance']:.3f}")
       print(f"Text: {result['text']}")
       print()

Collection Management
---------------------

Creating Collections
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pydantic import BaseModel
   from typing import List

   # Define schema
   class Article(BaseModel):
       article_id: int
       title: str
       category: str
       embedding: List[float]
       sparse_embedding: dict  # For hybrid search

   # Configure collection
   config = ICollectionConfig(
       collection_name="articles",
       dimension=1536,  # Dense vector dimension
       model_class=Article,
       metric_type="COSINE",  # or "L2", "IP"
       index_params={
           "index_type": "IVF_FLAT",
           "metric_type": "COSINE",
           "params": {"nlist": 128}
       }
   )

   # Create collection
   collection = client.get_or_create_collection(config)

Listing Collections
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get all collection names
   collections = client.list_collections()
   print("Available collections:", collections)

   # Check if collection exists
   exists = client.has_collection("articles")
   print(f"Collection 'articles' exists: {exists}")

Dropping Collections
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Delete a collection
   client.drop_collection("old_collection")

Data Operations
---------------

Insert Data
~~~~~~~~~~~

.. code-block:: python

   # Single insert
   data = [{
       "article_id": 1,
       "title": "Introduction to AI",
       "category": "technology",
       "embedding": [0.1, 0.2, ...]  # 1536-dim vector
   }]

   client.insert(config, data)

   # Batch insert
   batch_data = [
       {"article_id": i, "title": f"Article {i}", "embedding": [...]}
       for i in range(100)
   ]

   client.insert(config, batch_data)

Query Data
~~~~~~~~~~

.. code-block:: python

   # Query by filter
   results = client.query(
       config=config,
       filter_expr="article_id in [1, 2, 3]",
       output_fields=["article_id", "title", "category"]
   )

   for result in results:
       print(f"{result['article_id']}: {result['title']}")

   # Query with limit
   results = client.query(
       config=config,
       filter_expr="category == 'technology'",
       limit=10
   )

Delete Data
~~~~~~~~~~~

.. code-block:: python

   # Delete by ID
   client.delete(
       config=config,
       filter_expr="article_id in [1, 2, 3]"
   )

   # Delete by condition
   client.delete(
       config=config,
       filter_expr="category == 'outdated'"
   )

Search Operations
-----------------

Similarity Search
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare query vector
   query_text = "What is machine learning?"
   query_result = embedder.embed_query(query_text)
   query_vector = query_result['embedding']

   # Search
   results = client.search(
       config=config,
       query_vectors=[query_vector],
       limit=5,
       output_fields=["article_id", "title", "category"]
   )

   # Process results
   for result in results[0]:
       print(f"Distance: {result['distance']:.3f}")
       print(f"Title: {result['title']}")
       print(f"Category: {result['category']}")
       print()

Search with Filters
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Search within specific category
   results = client.search(
       config=config,
       query_vectors=[query_vector],
       limit=5,
       filter_expr="category == 'technology'",
       output_fields=["title", "category"]
   )

   # Complex filter
   results = client.search(
       config=config,
       query_vectors=[query_vector],
       limit=10,
       filter_expr="category in ['tech', 'science'] and article_id > 100"
   )

Batch Search
~~~~~~~~~~~~

.. code-block:: python

   # Search multiple queries at once
   queries = [
       "What is AI?",
       "How does ML work?",
       "Explain neural networks"
   ]

   # Generate embeddings
   query_embeddings = [
       embedder.embed_query(q)['embedding']
       for q in queries
   ]

   # Batch search
   results = client.search(
       config=config,
       query_vectors=query_embeddings,
       limit=3
   )

   # Process results for each query
   for i, query_results in enumerate(results):
       print(f"\nResults for: {queries[i]}")
       for result in query_results:
           print(f"  - {result['title']} (score: {result['distance']:.3f})")

Hybrid Search
-------------

Combine dense and sparse vectors:

.. code-block:: python

   from pydantic import BaseModel
   from typing import List, Dict

   class HybridDocument(BaseModel):
       id: int
       text: str
       dense_vector: List[float]  # Semantic embeddings
       sparse_vector: Dict[int, float]  # BM25 or keyword-based

   # Configure for hybrid search
   config = ICollectionConfig(
       collection_name="hybrid_docs",
       dimension=1536,
       model_class=HybridDocument,
       # Additional configuration for sparse vectors
   )

   # Hybrid search (implementation-specific)
   # Combine dense and sparse search results
   dense_results = client.search(config, [dense_query_vector], limit=10)
   # Rerank or combine with sparse results

Advanced Patterns
-----------------

Semantic Search with Reranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def semantic_search_with_reranking(
       client,
       config,
       embedder,
       query: str,
       initial_k: int = 100,
       final_k: int = 10
   ):
       """Search with reranking for better results."""

       # 1. Initial vector search (broad)
       query_embedding = embedder.embed_query(query)['embedding']

       initial_results = client.search(
           config=config,
           query_vectors=[query_embedding],
           limit=initial_k,
           output_fields=["id", "text"]
       )

       # 2. Rerank using cross-encoder (example)
       # In practice, use a reranking model
       scored_results = []
       for result in initial_results[0]:
           # Calculate relevance score
           # (simplified - use actual reranking model)
           score = result['distance']
           scored_results.append({
               "text": result['text'],
               "score": score
           })

       # 3. Sort and return top-k
       scored_results.sort(key=lambda x: x['score'], reverse=True)
       return scored_results[:final_k]

Metadata Filtering
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datetime import datetime, timedelta

   class TimestampedDocument(BaseModel):
       id: int
       text: str
       created_at: str
       category: str
       embedding: List[float]

   # Search recent documents only
   week_ago = (datetime.now() - timedelta(days=7)).isoformat()

   results = client.search(
       config=config,
       query_vectors=[query_vector],
       filter_expr=f"created_at > '{week_ago}' and category == 'news'",
       limit=10
   )

Pagination
~~~~~~~~~~

.. code-block:: python

   def paginated_search(client, config, query_vector, page_size=10):
       """Implement pagination for search results."""

       offset = 0
       all_results = []

       while True:
           # Search with offset
           results = client.search(
               config=config,
               query_vectors=[query_vector],
               limit=page_size,
               offset=offset
           )

           if not results[0]:
               break

           all_results.extend(results[0])
           offset += page_size

           # Stop if we've got enough or no more results
           if len(results[0]) < page_size:
               break

       return all_results

Production Patterns
-------------------

Connection Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MilvusService:
       """Singleton Milvus service for connection reuse."""

       _instance = None
       _client = None

       @classmethod
       def get_client(cls):
           """Get or create Milvus client instance."""
           if cls._client is None:
               cls._client = MilvusClient()
           return cls._client

   # Usage across application
   client = MilvusService.get_client()

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   from pymilvus import MilvusException

   def robust_insert(client, config, data, max_retries=3):
       """Insert with retry logic."""

       for attempt in range(max_retries):
           try:
               client.insert(config, data)
               return True

           except MilvusException as e:
               print(f"Attempt {attempt + 1} failed: {e}")
               if attempt == max_retries - 1:
                   raise

               import time
               time.sleep(2 ** attempt)

       return False

Monitoring
~~~~~~~~~~

.. code-block:: python

   def get_collection_stats(client, collection_name: str):
       """Get collection statistics."""

       from pymilvus import Collection, connections

       connections.connect(
           host=os.getenv("MILVUS_HOST"),
           port=os.getenv("MILVUS_PORT")
       )

       collection = Collection(collection_name)
       collection.load()

       stats = {
           "name": collection.name,
           "num_entities": collection.num_entities,
           "schema": collection.schema,
           "loaded": True
       }

       return stats

   # Usage
   stats = get_collection_stats(client, "knowledge_base")
   print(f"Collection has {stats['num_entities']} documents")

Performance Optimization
------------------------

Batch Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_insert_large_dataset(client, config, documents, batch_size=1000):
       """Efficiently insert large datasets."""

       for i in range(0, len(documents), batch_size):
           batch = documents[i:i + batch_size]
           client.insert(config, batch)
           print(f"Inserted {min(i + batch_size, len(documents))}/{len(documents)}")

Index Optimization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Choose appropriate index type
   index_configs = {
       # Fast, less accurate
       "fast": {
           "index_type": "IVF_FLAT",
           "params": {"nlist": 128}
       },
       # Balanced
       "balanced": {
           "index_type": "IVF_SQ8",
           "params": {"nlist": 1024}
       },
       # Accurate, slower
       "accurate": {
           "index_type": "HNSW",
           "params": {"M": 16, "efConstruction": 200}
       }
   }

   # Use in configuration
   config = ICollectionConfig(
       collection_name="optimized_collection",
       dimension=1536,
       model_class=Document,
       index_params=index_configs["balanced"]
   )

Testing
-------

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch

   @pytest.fixture
   def mock_milvus():
       """Mock Milvus client for testing."""
       mock = Mock()
       mock.get_or_create_collection.return_value = Mock(name="test_collection")
       mock.insert.return_value = None
       mock.search.return_value = [[{"id": 1, "distance": 0.9}]]
       return mock

   def test_vector_storage(mock_milvus):
       """Test vector storage operations."""
       with patch('arshai.vector_db.milvus_client.MilvusClient', return_value=mock_milvus):
           client = MilvusClient()

           # Test insert
           data = [{"id": 1, "embedding": [0.1] * 1536}]
           client.insert(config, data)

           assert mock_milvus.insert.called

Building Custom Vector DB Client
---------------------------------

Implement ``IVectorDBClient`` for other databases:

.. code-block:: python

   from arshai.core.interfaces.ivector_db_client import IVectorDBClient, ICollectionConfig
   from typing import List, Dict, Any

   class CustomVectorDB(IVectorDBClient):
       """Custom vector database implementation."""

       def get_or_create_collection(self, config: ICollectionConfig):
           """Create or retrieve collection."""
           # Your implementation
           pass

       def insert(self, config: ICollectionConfig, data: List[Dict[str, Any]]):
           """Insert vectors."""
           # Your implementation
           pass

       def search(
           self,
           config: ICollectionConfig,
           query_vectors: List[List[float]],
           limit: int = 10,
           **kwargs
       ) -> List[List[Dict[str, Any]]]:
           """Search similar vectors."""
           # Your implementation
           pass

       def query(self, config: ICollectionConfig, **kwargs) -> List[Dict[str, Any]]:
           """Query by filter."""
           # Your implementation
           pass

       def delete(self, config: ICollectionConfig, filter_expr: str):
           """Delete vectors."""
           # Your implementation
           pass

Best Practices
--------------

1. **Choose Appropriate Index**
   Balance speed vs. accuracy based on your use case.

2. **Batch Operations**
   Use batch insert/search for better performance.

3. **Proper Schema Design**
   Define clear schemas with appropriate field types.

4. **Monitor Collection Size**
   Track growth and plan for scaling.

5. **Use Metadata Filters**
   Combine vector similarity with metadata filtering.

6. **Regular Maintenance**
   Compact collections and rebuild indexes periodically.

7. **Test with Real Data**
   Validate performance with production-like data volumes.

Next Steps
----------

- **Embeddings**: See :doc:`embeddings` for generating vectors
- **RAG Systems**: Build complete retrieval-augmented generation systems
- **Production Deployment**: Scale your vector database for production

Remember: This is **one way** to implement vector storage in Arshai. The framework provides the ``IVectorDBClient`` interface - you can implement it for Pinecone, Weaviate, Qdrant, PostgreSQL with pgvector, or any other vector database.
