Embedding Implementations
=========================

Arshai provides reference implementations for generating text embeddings using various providers. These implementations follow the ``IEmbedding`` interface, allowing you to swap providers easily.

.. important::

   **These are Reference Implementations**

   The embedding implementations are examples showing how to:

   - Implement the ``IEmbedding`` interface
   - Integrate with different embedding providers
   - Handle batch processing and error cases

   You can use these as-is, extend them, or build your own implementations.

Overview
--------

**Available Implementations:**

1. **OpenAI Embeddings** - High-quality general-purpose embeddings
2. **VoyageAI Embeddings** - Specialized embeddings for various domains
3. **MGTE Embeddings** - Multi-granularity text embeddings
4. **Cloudflare AI Gateway Embeddings** - Multi-provider embeddings via Cloudflare (BYOK mode)

All implementations provide:

- Batch text embedding
- Query embedding (single text)
- Configurable dimensions (where supported)
- Async support
- Error handling

Installation
------------

Install with embedding support:

.. code-block:: bash

   # Core package (includes interfaces)
   pip install arshai

   # For OpenAI embeddings
   pip install arshai openai

   # For VoyageAI embeddings
   pip install arshai voyageai

   # For MGTE embeddings
   pip install arshai sentence-transformers

OpenAI Embeddings
-----------------

High-quality embeddings from OpenAI's API.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   import os
   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig

   # Set API key
   os.environ["OPENAI_API_KEY"] = "your-api-key"

   # Create configuration
   config = EmbeddingConfig(
       model_name="text-embedding-3-small",  # or text-embedding-3-large, text-embedding-ada-002
       batch_size=100
   )

   # Create embedding instance
   embedder = OpenAIEmbedding(config)

   print(f"Embedding dimension: {embedder.dimension}")
   # Output: 1536

Available Models
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Model
     - Dimension
     - Best For
   * - ``text-embedding-3-small``
     - 1536
     - Fast, cost-effective embeddings
   * - ``text-embedding-3-large``
     - 3072
     - Highest quality embeddings
   * - ``text-embedding-ada-002``
     - 1536
     - Legacy model (still supported)

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig

   # Initialize
   config = EmbeddingConfig(model_name="text-embedding-3-small")
   embedder = OpenAIEmbedding(config)

   # Embed documents
   documents = [
       "Artificial intelligence is transforming technology",
       "Machine learning powers modern AI systems",
       "Deep learning uses neural networks"
   ]

   result = embedder.embed_documents(documents)

   print(f"Generated {len(result['embeddings'])} embeddings")
   print(f"Embedding dimension: {len(result['embeddings'][0])}")
   print(f"Tokens used: {result['total_tokens']}")

   # Embed query
   query_result = embedder.embed_query("What is AI?")
   print(f"Query embedding dimension: {len(query_result['embedding'])}")

Async Usage
~~~~~~~~~~~

.. code-block:: python

   import asyncio

   async def embed_async():
       config = EmbeddingConfig(model_name="text-embedding-3-small")
       embedder = OpenAIEmbedding(config)

       documents = ["Document 1", "Document 2", "Document 3"]

       # Async embedding
       result = await embedder.embed_documents_async(documents)
       print(f"Embedded {len(result['embeddings'])} documents asynchronously")

   asyncio.run(embed_async())

VoyageAI Embeddings
-------------------

Specialized embeddings for different domains and use cases.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   import os
   from arshai.embeddings.voyageai_embedding import VoyageAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig

   # Set API key
   os.environ["VOYAGE_API_KEY"] = "your-api-key"

   # Create configuration
   config = EmbeddingConfig(
       model_name="voyage-3-large",
       batch_size=100
   )

   # Create embedding instance
   embedder = VoyageAIEmbedding(config)

Available Models
~~~~~~~~~~~~~~~~

**Flexible Dimension Models:**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Model
     - Default Dimension
     - Allowed Dimensions
   * - ``voyage-3-large``
     - 1024
     - 256, 512, 1024, 2048
   * - ``voyage-3.5``
     - 1024
     - 256, 512, 1024, 2048
   * - ``voyage-3.5-lite``
     - 1024
     - 256, 512, 1024, 2048
   * - ``voyage-code-3``
     - 1024
     - 256, 512, 1024, 2048

**Domain-Specific Models:**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Model
     - Dimension
     - Specialization
   * - ``voyage-finance-2``
     - 1024
     - Financial documents
   * - ``voyage-law-2``
     - 1024
     - Legal documents
   * - ``voyage-code-2``
     - 1536
     - Code and programming
   * - ``voyage-multilingual-2``
     - 1024
     - Multilingual text

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from arshai.embeddings.voyageai_embedding import VoyageAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig

   # General purpose
   config = EmbeddingConfig(model_name="voyage-3-large")
   embedder = VoyageAIEmbedding(config)

   # Embed documents
   documents = ["AI is revolutionary", "ML powers innovation"]
   result = embedder.embed_documents(documents)

   # Domain-specific (legal)
   legal_config = EmbeddingConfig(model_name="voyage-law-2")
   legal_embedder = VoyageAIEmbedding(legal_config)

   legal_docs = [
       "The defendant pleaded guilty to charges",
       "Court ruled in favor of the plaintiff"
   ]
   legal_result = legal_embedder.embed_documents(legal_docs)

Custom Dimensions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use custom dimension (for supported models)
   config = EmbeddingConfig(
       model_name="voyage-3-large",
       dimension=512  # Choose from [256, 512, 1024, 2048]
   )
   embedder = VoyageAIEmbedding(config)

   result = embedder.embed_documents(["Sample text"])
   print(f"Embedding dimension: {len(result['embeddings'][0])}")
   # Output: 512

MGTE Embeddings
---------------

Multi-granularity text embeddings using sentence transformers.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from arshai.embeddings.mgte_embeddings import MGTEEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig

   # Create configuration
   config = EmbeddingConfig(
       model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
       batch_size=32
   )

   # Create embedding instance (downloads model on first use)
   embedder = MGTEEmbedding(config)

Available Models
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Default model
   config = EmbeddingConfig(model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct")

   # Other Sentence Transformer models
   config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from arshai.embeddings.mgte_embeddings import MGTEEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig

   # Initialize (model cached after first download)
   config = EmbeddingConfig(
       model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
   )
   embedder = MGTEEmbedding(config)

   # Embed documents
   documents = [
       "Natural language processing enables machines to understand text",
       "Embeddings convert text into numerical vectors"
   ]

   result = embedder.embed_documents(documents)
   print(f"Dimension: {embedder.dimension}")

Cloudflare AI Gateway Embeddings
--------------------------------

Multi-provider embeddings through Cloudflare AI Gateway using BYOK (Bring Your Own Key) mode.
Provider API keys are stored securely in Cloudflare, and you only need the gateway token.

.. note::
   **BYOK Mode**: Provider API keys are configured in Cloudflare AI Gateway dashboard.
   Only the gateway token is needed in your application code.

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   from arshai.embeddings import CloudflareGatewayEmbedding, CloudflareGatewayEmbeddingConfig

   # Configure (BYOK mode)
   config = CloudflareGatewayEmbeddingConfig(
       account_id="your-cloudflare-account-id",
       gateway_id="your-gateway-id",
       gateway_token="your-gateway-token",  # Or set CLOUDFLARE_GATEWAY_TOKEN env var
       provider="openrouter",               # Provider name
       model_name="openai/text-embedding-3-small",  # Model name
   )

   # Create embedding instance
   embedder = CloudflareGatewayEmbedding(config)
   print(f"Dimension: {embedder.dimension}")

Supported Providers
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Provider
     - Example Models
     - Dimension
   * - ``openrouter``
     - openai/text-embedding-3-small
     - 1536
   * - ``openai``
     - text-embedding-3-small, text-embedding-3-large
     - 1536 / 3072
   * - ``cohere``
     - embed-english-v3.0, embed-multilingual-v3.0
     - 1024
   * - ``mistral``
     - mistral-embed
     - 1024
   * - ``workers-ai``
     - bge-base-en-v1.5, bge-large-en-v1.5
     - 768 / 1024

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from arshai.embeddings import CloudflareGatewayEmbedding, CloudflareGatewayEmbeddingConfig

   # Initialize
   config = CloudflareGatewayEmbeddingConfig(
       account_id="xxx",
       gateway_id="my-gateway",
       gateway_token="xxx",
       provider="openrouter",
       model_name="openai/text-embedding-3-small",
   )
   embedder = CloudflareGatewayEmbedding(config)

   # Embed documents
   documents = [
       "Artificial intelligence is transforming technology",
       "Machine learning powers modern AI systems",
   ]

   result = embedder.embed_documents(documents)
   vectors = result["dense"]

   print(f"Embedded {len(vectors)} documents")
   print(f"Vector dimension: {len(vectors[0])}")

   # Embed single document
   single_result = embedder.embed_document("What is AI?")
   print(f"Single embedding dimension: {len(single_result['dense'])}")

Async Usage
~~~~~~~~~~~

.. code-block:: python

   import asyncio

   async def embed_async():
       config = CloudflareGatewayEmbeddingConfig(
           account_id="xxx",
           gateway_id="my-gateway",
           gateway_token="xxx",
           provider="openrouter",
           model_name="openai/text-embedding-3-small",
       )
       embedder = CloudflareGatewayEmbedding(config)

       documents = ["Document 1", "Document 2", "Document 3"]

       # Async embedding
       result = await embedder.aembed_documents(documents)
       print(f"Embedded {len(result['dense'])} documents asynchronously")

   asyncio.run(embed_async())

Health Check
~~~~~~~~~~~~

.. code-block:: python

   # Check if embedding service is healthy
   health = embedder.health_check()

   print(f"Initialized: {health['initialized']}")
   print(f"Provider: {health['provider']}")
   print(f"Model: {health['model']}")
   print(f"Healthy: {health['healthy']}")
   if health.get('latency_ms'):
       print(f"Latency: {health['latency_ms']:.2f}ms")

Benefits of Cloudflare Gateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Centralized Key Management** - Provider keys stored in Cloudflare, not in code
2. **Multi-Provider Access** - Switch providers by changing config
3. **Unified Caching** - Built-in caching reduces costs
4. **Analytics** - Monitor usage across all providers
5. **Rate Limiting** - Unified rate limit management

Advanced Usage
--------------

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_embed_large_dataset(embedder, documents: list, batch_size: int = 100):
       """Embed large dataset in batches."""

       all_embeddings = []

       for i in range(0, len(documents), batch_size):
           batch = documents[i:i + batch_size]
           result = embedder.embed_documents(batch)
           all_embeddings.extend(result['embeddings'])

           print(f"Processed {len(all_embeddings)}/{len(documents)} documents")

       return all_embeddings

   # Usage
   large_dataset = ["Document " + str(i) for i in range(1000)]
   embeddings = batch_embed_large_dataset(embedder, large_dataset)

Similarity Search
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   def cosine_similarity(vec1, vec2):
       """Calculate cosine similarity between two vectors."""
       return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

   def find_similar_documents(query: str, documents: list, embedder, top_k: int = 3):
       """Find most similar documents to query."""

       # Embed query
       query_result = embedder.embed_query(query)
       query_embedding = query_result['embedding']

       # Embed documents
       doc_result = embedder.embed_documents(documents)
       doc_embeddings = doc_result['embeddings']

       # Calculate similarities
       similarities = [
           cosine_similarity(query_embedding, doc_emb)
           for doc_emb in doc_embeddings
       ]

       # Get top-k
       top_indices = np.argsort(similarities)[-top_k:][::-1]

       results = [
           {"document": documents[i], "similarity": similarities[i]}
           for i in top_indices
       ]

       return results

   # Usage
   query = "What is artificial intelligence?"
   documents = [
       "AI is the simulation of human intelligence",
       "Machine learning is a subset of AI",
       "Python is a programming language"
   ]

   similar_docs = find_similar_documents(query, documents, embedder)
   for doc in similar_docs:
       print(f"Similarity: {doc['similarity']:.3f} - {doc['document']}")

Caching Embeddings
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pickle
   from pathlib import Path

   class CachedEmbedder:
       """Wrapper that caches embeddings to disk."""

       def __init__(self, embedder, cache_dir: str = ".embedding_cache"):
           self.embedder = embedder
           self.cache_dir = Path(cache_dir)
           self.cache_dir.mkdir(exist_ok=True)

       def _get_cache_key(self, text: str) -> str:
           """Generate cache key from text."""
           import hashlib
           return hashlib.md5(text.encode()).hexdigest()

       def embed_with_cache(self, text: str):
           """Embed with disk caching."""
           cache_key = self._get_cache_key(text)
           cache_file = self.cache_dir / f"{cache_key}.pkl"

           # Check cache
           if cache_file.exists():
               with open(cache_file, 'rb') as f:
                   return pickle.load(f)

           # Generate embedding
           result = self.embedder.embed_query(text)
           embedding = result['embedding']

           # Save to cache
           with open(cache_file, 'wb') as f:
               pickle.dump(embedding, f)

           return embedding

   # Usage
   cached_embedder = CachedEmbedder(embedder)
   embedding1 = cached_embedder.embed_with_cache("Sample text")  # Generates
   embedding2 = cached_embedder.embed_with_cache("Sample text")  # From cache

Error Handling
--------------

.. code-block:: python

   from openai import OpenAIError

   def safe_embed(embedder, documents: list, max_retries: int = 3):
       """Embed with retry logic."""

       for attempt in range(max_retries):
           try:
               result = embedder.embed_documents(documents)
               return result

           except OpenAIError as e:
               print(f"Attempt {attempt + 1} failed: {e}")
               if attempt == max_retries - 1:
                   raise

               import time
               time.sleep(2 ** attempt)  # Exponential backoff

   # Usage
   try:
       result = safe_embed(embedder, documents)
   except OpenAIError as e:
       print(f"Failed after retries: {e}")

Choosing an Embedding Provider
-------------------------------

**Use OpenAI when:**

- You need high-quality general-purpose embeddings
- You're already using OpenAI for LLMs
- You want reliable, well-tested embeddings
- Cost is not the primary concern

**Use VoyageAI when:**

- You have domain-specific content (finance, legal, code)
- You need flexible embedding dimensions
- You want specialized models for your use case
- You need multilingual support

**Use MGTE when:**

- You want to run embeddings locally
- You need offline operation
- You want to avoid API costs
- You have GPU resources available
- Privacy is a concern

**Use Cloudflare Gateway when:**

- You want centralized API key management
- You need multi-provider access from one interface
- You want unified caching and analytics
- You're already using Cloudflare AI Gateway for LLMs
- You need to switch providers without code changes

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Provider
     - Speed
     - Quality
     - Cost
   * - OpenAI
     - Fast
     - High
     - API calls
   * - VoyageAI
     - Fast
     - Specialized
     - API calls
   * - MGTE
     - Medium
     - Good
     - Free (local)
   * - Cloudflare Gateway
     - Fast
     - Depends on provider
     - API calls + caching savings

Building Custom Embeddings
---------------------------

Implement ``IEmbedding`` interface:

.. code-block:: python

   from arshai.core.interfaces.iembedding import IEmbedding, EmbeddingConfig
   from typing import List, Dict, Any

   class CustomEmbedding(IEmbedding):
       """Custom embedding implementation."""

       def __init__(self, config: EmbeddingConfig):
           self.model_name = config.model_name
           self.batch_size = config.batch_size
           self._dimension = 768  # Your model's dimension

       @property
       def dimension(self) -> int:
           return self._dimension

       def embed_documents(self, texts: List[str]) -> Dict[str, Any]:
           """Embed multiple documents."""
           # Your implementation here
           embeddings = [self._embed_single(text) for text in texts]

           return {
               "embeddings": embeddings,
               "total_tokens": len(texts) * 100  # Approximate
           }

       def embed_query(self, text: str) -> Dict[str, Any]:
           """Embed single query."""
           embedding = self._embed_single(text)

           return {
               "embedding": embedding,
               "total_tokens": 100  # Approximate
           }

       def _embed_single(self, text: str) -> List[float]:
           """Your embedding logic."""
           # Implement your embedding generation
           pass

       async def embed_documents_async(self, texts: List[str]) -> Dict[str, Any]:
           """Async version."""
           return self.embed_documents(texts)

       async def embed_query_async(self, text: str) -> Dict[str, Any]:
           """Async version."""
           return self.embed_query(text)

Best Practices
--------------

1. **Consistent Models**
   Use the same embedding model for documents and queries.

2. **Batch Processing**
   Process multiple documents at once for better performance.

3. **Cache Results**
   Cache embeddings for frequently accessed documents.

4. **Error Handling**
   Implement retry logic for API-based embeddings.

5. **Monitor Costs**
   Track API usage for cost management.

6. **Choose Appropriate Dimensions**
   Higher dimensions = better quality but more storage/compute.

7. **Test Different Providers**
   Benchmark providers for your specific use case.

Integration with Vector Databases
----------------------------------

See :doc:`vector-databases` for using embeddings with vector stores.

.. code-block:: python

   # Quick example
   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.vector_db.milvus_client import MilvusClient

   # Create embedder
   embedder = OpenAIEmbedding(EmbeddingConfig(model_name="text-embedding-3-small"))

   # Generate embeddings
   documents = ["Doc 1", "Doc 2", "Doc 3"]
   result = embedder.embed_documents(documents)

   # Store in vector database
   # (See vector-databases documentation for details)

Next Steps
----------

- **Vector Storage**: See :doc:`vector-databases` for storing and searching embeddings
- **RAG Systems**: Build retrieval-augmented generation systems
- **Semantic Search**: Implement semantic search for your application

Remember: These are **reference implementations**. The framework provides the ``IEmbedding`` interface - you can implement it for any embedding provider or custom model.
