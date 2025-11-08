Building a RAG System
======================

This tutorial guides you through building a complete Retrieval-Augmented Generation (RAG) system using Arshai. You'll create a document-based question-answering system that combines vector search with LLM reasoning.

**What You'll Build:**

- Document ingestion and chunking pipeline
- Vector embeddings and storage system
- Semantic search with ranking
- RAG-enabled conversational agent
- Complete Q&A application

**What You'll Learn:**

- Document processing strategies
- Embedding generation and storage
- Vector similarity search
- RAG agent implementation
- Production patterns

**Prerequisites:**

- Python 3.9+
- Arshai with extras: ``pip install arshai[openai,milvus]``
- OpenAI API key
- Milvus running (Docker or cloud)

**Time to Complete:** 60-90 minutes

Project Setup
-------------

Install Dependencies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create project
   mkdir rag-system
   cd rag-system

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install arshai[openai,milvus] python-dotenv

   # Create project structure
   mkdir -p documents data
   touch .env
   touch rag_system.py
   touch document_processor.py
   touch rag_agent.py

Configure Environment
~~~~~~~~~~~~~~~~~~~~~

Create ``.env`` file:

.. code-block:: bash

   # .env
   OPENAI_API_KEY=your-openai-key
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
   MILVUS_DB_NAME=rag_db

Start Milvus (Docker):

.. code-block:: bash

   # Download docker-compose
   wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

   # Start Milvus
   docker-compose up -d

   # Verify
   docker-compose ps

Step 1: Document Processing
----------------------------

Create document processor for chunking:

.. code-block:: python

   # document_processor.py
   from typing import List, Dict, Any
   from pathlib import Path
   import re

   class DocumentChunker:
       """Process and chunk documents for RAG."""

       def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
           """
           Initialize chunker.

           Args:
               chunk_size: Target size for each chunk in characters
               chunk_overlap: Overlap between chunks in characters
           """
           self.chunk_size = chunk_size
           self.chunk_overlap = chunk_overlap

       def load_document(self, file_path: str) -> str:
           """Load document from file."""
           path = Path(file_path)

           if not path.exists():
               raise FileNotFoundError(f"File not found: {file_path}")

           # Handle different file types
           if path.suffix == '.txt':
               with open(path, 'r', encoding='utf-8') as f:
                   return f.read()
           elif path.suffix == '.md':
               with open(path, 'r', encoding='utf-8') as f:
                   return f.read()
           else:
               raise ValueError(f"Unsupported file type: {path.suffix}")

       def chunk_text(self, text: str) -> List[Dict[str, Any]]:
           """
           Chunk text into smaller pieces with metadata.

           Args:
               text: Text to chunk

           Returns:
               List of chunks with metadata
           """
           # Clean text
           text = self._clean_text(text)

           # Split into sentences
           sentences = self._split_into_sentences(text)

           # Create chunks
           chunks = []
           current_chunk = []
           current_length = 0

           for sentence in sentences:
               sentence_length = len(sentence)

               # If adding this sentence exceeds chunk size
               if current_length + sentence_length > self.chunk_size and current_chunk:
                   # Save current chunk
                   chunk_text = ' '.join(current_chunk)
                   chunks.append({
                       'text': chunk_text,
                       'length': len(chunk_text),
                       'sentence_count': len(current_chunk)
                   })

                   # Start new chunk with overlap
                   overlap_sentences = self._get_overlap_sentences(
                       current_chunk,
                       self.chunk_overlap
                   )
                   current_chunk = overlap_sentences
                   current_length = sum(len(s) for s in current_chunk)

               # Add sentence to current chunk
               current_chunk.append(sentence)
               current_length += sentence_length

           # Add final chunk
           if current_chunk:
               chunk_text = ' '.join(current_chunk)
               chunks.append({
                   'text': chunk_text,
                   'length': len(chunk_text),
                   'sentence_count': len(current_chunk)
               })

           return chunks

       def process_document(
           self,
           file_path: str,
           metadata: Dict[str, Any] = None
       ) -> List[Dict[str, Any]]:
           """
           Process document into chunks with metadata.

           Args:
               file_path: Path to document
               metadata: Additional metadata for chunks

           Returns:
               List of processed chunks
           """
           # Load document
           text = self.load_document(file_path)

           # Chunk text
           chunks = self.chunk_text(text)

           # Add metadata
           file_name = Path(file_path).name
           base_metadata = metadata or {}

           processed_chunks = []
           for i, chunk in enumerate(chunks):
               chunk_data = {
                   'id': f"{file_name}_{i}",
                   'text': chunk['text'],
                   'source': file_name,
                   'chunk_index': i,
                   'total_chunks': len(chunks),
                   **base_metadata
               }
               processed_chunks.append(chunk_data)

           return processed_chunks

       def _clean_text(self, text: str) -> str:
           """Clean and normalize text."""
           # Remove extra whitespace
           text = re.sub(r'\s+', ' ', text)
           # Remove special characters (optional)
           text = text.strip()
           return text

       def _split_into_sentences(self, text: str) -> List[str]:
           """Split text into sentences."""
           # Simple sentence splitting (can be improved)
           sentences = re.split(r'(?<=[.!?])\s+', text)
           return [s.strip() for s in sentences if s.strip()]

       def _get_overlap_sentences(
           self,
           sentences: List[str],
           target_overlap: int
       ) -> List[str]:
           """Get sentences for overlap."""
           overlap_sentences = []
           overlap_length = 0

           # Take sentences from the end
           for sentence in reversed(sentences):
               if overlap_length >= target_overlap:
                   break
               overlap_sentences.insert(0, sentence)
               overlap_length += len(sentence)

           return overlap_sentences

Step 2: Build RAG Agent
------------------------

Create the RAG agent with retrieval:

.. code-block:: python

   # rag_agent.py
   import asyncio
   from typing import List, Dict, Any
   from dotenv import load_dotenv

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLMInput
   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig
   from arshai.vector_db.milvus_client import MilvusClient
   from arshai.core.interfaces.ivector_db_client import ICollectionConfig
   from pydantic import BaseModel

   load_dotenv()

   class DocumentChunk(BaseModel):
       """Schema for document chunks in vector DB."""
       id: str
       text: str
       source: str
       chunk_index: int
       embedding: List[float]

   class RAGAgent(BaseAgent):
       """Retrieval-Augmented Generation agent."""

       def __init__(
           self,
           llm_client,
           embedding_service,
           vector_client,
           collection_name: str = "knowledge_base",
           top_k: int = 3
       ):
           """
           Initialize RAG agent.

           Args:
               llm_client: LLM client for generation
               embedding_service: Service for generating embeddings
               vector_client: Vector database client
               collection_name: Name of vector collection
               top_k: Number of documents to retrieve
           """
           super().__init__(
               llm_client,
               system_prompt="""You are a helpful AI assistant with access to a knowledge base.
               Use the provided context to answer questions accurately.
               If the context doesn't contain the answer, say so clearly.
               Always cite the source when using information from the context."""
           )
           self.embedding_service = embedding_service
           self.vector_client = vector_client
           self.top_k = top_k

           # Configure collection
           self.collection_config = ICollectionConfig(
               collection_name=collection_name,
               dimension=embedding_service.dimension,
               model_class=DocumentChunk
           )

           # Ensure collection exists
           self.vector_client.get_or_create_collection(self.collection_config)

       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           """Process query with RAG."""

           # Step 1: Retrieve relevant documents
           retrieved_docs = await self._retrieve_documents(input.message)

           # Step 2: Build context from retrieved documents
           context = self._build_context(retrieved_docs)

           # Step 3: Generate response with context
           response = await self._generate_response(input.message, context)

           return {
               'response': response,
               'sources': [doc['source'] for doc in retrieved_docs],
               'retrieved_chunks': len(retrieved_docs),
               'context': context
           }

       async def _retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
           """Retrieve relevant documents from vector store."""

           # Generate query embedding
           query_result = self.embedding_service.embed_query(query)
           query_embedding = query_result['embedding']

           # Search vector database
           search_results = self.vector_client.search(
               config=self.collection_config,
               query_vectors=[query_embedding],
               limit=self.top_k,
               output_fields=["id", "text", "source", "chunk_index"]
           )

           # Extract and format results
           retrieved_docs = []
           if search_results and len(search_results) > 0:
               for result in search_results[0]:
                   retrieved_docs.append({
                       'text': result.get('text', ''),
                       'source': result.get('source', 'unknown'),
                       'chunk_index': result.get('chunk_index', 0),
                       'distance': result.get('distance', 0.0)
                   })

           return retrieved_docs

       def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
           """Build context string from retrieved documents."""
           if not retrieved_docs:
               return "No relevant information found in the knowledge base."

           context_parts = []
           for i, doc in enumerate(retrieved_docs, 1):
               context_parts.append(
                   f"[Source {i}: {doc['source']}]\n{doc['text']}"
               )

           return "\n\n".join(context_parts)

       async def _generate_response(self, query: str, context: str) -> str:
           """Generate response using LLM with context."""

           # Build enhanced prompt
           enhanced_prompt = f"""Context from knowledge base:
   {context}

   Question: {query}

   Please provide a detailed answer based on the context above. If the context doesn't contain enough information, say so clearly."""

           # Call LLM
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=enhanced_prompt
           )

           result = await self.llm_client.chat(llm_input)
           return result.get('llm_response', 'Unable to generate response')

       def add_documents(self, chunks: List[Dict[str, Any]]):
           """Add document chunks to vector store."""

           # Extract texts for embedding
           texts = [chunk['text'] for chunk in chunks]

           # Generate embeddings
           embedding_result = self.embedding_service.embed_documents(texts)
           embeddings = embedding_result['embeddings']

           # Prepare data for insertion
           insert_data = []
           for i, chunk in enumerate(chunks):
               insert_data.append({
                   'id': chunk.get('id', f'chunk_{i}'),
                   'text': chunk['text'],
                   'source': chunk.get('source', 'unknown'),
                   'chunk_index': chunk.get('chunk_index', i),
                   'embedding': embeddings[i]
               })

           # Insert into vector database
           self.vector_client.insert(
               config=self.collection_config,
               data=insert_data
           )

           print(f"✓ Added {len(chunks)} document chunks to knowledge base")

Step 3: Build Complete RAG System
----------------------------------

Create the main application:

.. code-block:: python

   # rag_system.py
   import asyncio
   import os
   from pathlib import Path
   from typing import List
   from dotenv import load_dotenv

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig
   from arshai.embeddings.openai_embeddings import OpenAIEmbedding
   from arshai.core.interfaces.iembedding import EmbeddingConfig
   from arshai.vector_db.milvus_client import MilvusClient

   from document_processor import DocumentChunker
   from rag_agent import RAGAgent

   load_dotenv()

   class RAGSystem:
       """Complete RAG system with document ingestion and Q&A."""

       def __init__(self):
           """Initialize RAG system components."""

           # Create LLM client
           self.llm_client = OpenAIClient(
               ILLMConfig(
                   model="gpt-3.5-turbo",
                   temperature=0.7,
                   max_tokens=500
               )
           )

           # Create embedding service
           self.embedding_service = OpenAIEmbedding(
               EmbeddingConfig(
                   model_name="text-embedding-3-small",
                   batch_size=100
               )
           )

           # Create vector database client
           self.vector_client = MilvusClient()

           # Create document processor
           self.document_processor = DocumentChunker(
               chunk_size=500,
               chunk_overlap=50
           )

           # Create RAG agent
           self.rag_agent = RAGAgent(
               llm_client=self.llm_client,
               embedding_service=self.embedding_service,
               vector_client=self.vector_client,
               collection_name="knowledge_base",
               top_k=3
           )

           print("✓ RAG system initialized")

       def ingest_documents(self, document_paths: List[str]):
           """
           Ingest documents into the knowledge base.

           Args:
               document_paths: List of paths to documents
           """
           all_chunks = []

           for doc_path in document_paths:
               try:
                   print(f"Processing: {doc_path}")

                   # Process document
                   chunks = self.document_processor.process_document(
                       doc_path,
                       metadata={'type': 'documentation'}
                   )

                   all_chunks.extend(chunks)
                   print(f"  ✓ Created {len(chunks)} chunks")

               except Exception as e:
                   print(f"  ✗ Error processing {doc_path}: {e}")

           if all_chunks:
               # Add to vector store
               print(f"\nAdding {len(all_chunks)} chunks to knowledge base...")
               self.rag_agent.add_documents(all_chunks)
               print("✓ Document ingestion complete")
           else:
               print("No chunks to add")

       async def query(self, question: str) -> dict:
           """
           Query the RAG system.

           Args:
               question: User question

           Returns:
               Response with answer and sources
           """
           from arshai.core.interfaces.iagent import IAgentInput

           result = await self.rag_agent.process(
               IAgentInput(message=question)
           )

           return result

       async def interactive_session(self):
           """Run interactive Q&A session."""
           print("\n" + "=" * 60)
           print("RAG System - Interactive Q&A")
           print("=" * 60)
           print("\nCommands:")
           print("  /quit   - Exit")
           print("  /stats  - Show system statistics")
           print("\nAsk questions about your documents!\n")

           while True:
               try:
                   # Get user question
                   question = input("\n❓ Question: ").strip()

                   if not question:
                       continue

                   # Handle commands
                   if question.lower() == '/quit':
                       print("\n👋 Goodbye!")
                       break

                   if question.lower() == '/stats':
                       self._show_statistics()
                       continue

                   # Process question
                   print("🔍 Searching knowledge base...", end="", flush=True)
                   result = await self.query(question)
                   print("\r" + " " * 40 + "\r", end="")

                   # Display answer
                   print(f"\n💡 Answer:\n{result['response']}\n")

                   # Display sources
                   if result['sources']:
                       print(f"📚 Sources: {', '.join(set(result['sources']))}")
                       print(f"📊 Retrieved {result['retrieved_chunks']} relevant chunks")

               except KeyboardInterrupt:
                   print("\n\n👋 Goodbye!")
                   break
               except Exception as e:
                   print(f"\n❌ Error: {e}")

       def _show_statistics(self):
           """Show system statistics."""
           print("\n📊 System Statistics:")
           print(f"  Embedding Model: {self.embedding_service.model_name}")
           print(f"  Embedding Dimension: {self.embedding_service.dimension}")
           print(f"  LLM Model: {self.llm_client.config.model}")
           print(f"  Top-K Retrieval: {self.rag_agent.top_k}")

   async def main():
       """Main application entry point."""

       # Create RAG system
       rag_system = RAGSystem()

       # Ingest documents
       documents_dir = Path("documents")
       if documents_dir.exists():
           document_files = list(documents_dir.glob("*.txt")) + \
                           list(documents_dir.glob("*.md"))

           if document_files:
               print(f"\nFound {len(document_files)} documents to ingest\n")
               rag_system.ingest_documents([str(f) for f in document_files])
           else:
               print("\n⚠️ No documents found in 'documents/' directory")
               print("Add .txt or .md files to get started\n")
       else:
           print("\n⚠️ 'documents/' directory not found")
           documents_dir.mkdir()
           print("Created 'documents/' directory - add your files there\n")

       # Start interactive session
       await rag_system.interactive_session()

   if __name__ == "__main__":
       asyncio.run(main())

Step 4: Test the System
------------------------

Create sample documents:

.. code-block:: bash

   # Create sample document
   cat > documents/arshai_intro.txt << 'EOF'
   Arshai Framework

   Arshai is a framework for building agentic AI systems. It provides three main layers:

   Layer 1 - LLM Clients: Standardized interfaces for different language model providers including OpenAI, Google Gemini, Azure OpenAI, and OpenRouter.

   Layer 2 - Agents: Building blocks for creating conversational agents with BaseAgent as the foundation. Agents can have memory, use tools, and handle complex interactions.

   Layer 3 - Systems: Patterns for composing multiple agents into complete agentic systems through orchestration and workflow management.

   The framework follows these core principles:
   - Direct Control: Developers explicitly create and configure all components
   - Building Blocks: Framework provides foundations, developers build solutions
   - Progressive Complexity: Start simple and scale to sophisticated systems as needed
   EOF

Run the system:

.. code-block:: bash

   python rag_system.py

Test queries:

.. code-block:: text

   RAG System - Interactive Q&A
   ============================================================

   Commands:
     /quit   - Exit
     /stats  - Show system statistics

   Ask questions about your documents!

   ❓ Question: What is Arshai?

   💡 Answer:
   Arshai is a framework for building agentic AI systems. It provides a three-layer architecture consisting of LLM Clients (Layer 1), Agents (Layer 2), and Systems (Layer 3). The framework emphasizes direct control, provides building blocks for developers, and supports progressive complexity.

   📚 Sources: arshai_intro.txt
   📊 Retrieved 3 relevant chunks

   ❓ Question: What are the three layers?

   💡 Answer:
   The three layers in Arshai are:
   1. Layer 1 - LLM Clients: Standardized interfaces for language model providers
   2. Layer 2 - Agents: Building blocks for conversational agents
   3. Layer 3 - Systems: Patterns for composing agents into complete systems

   📚 Sources: arshai_intro.txt
   📊 Retrieved 3 relevant chunks

Step 5: Add Advanced Features
------------------------------

Hybrid Search with Reranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class AdvancedRAGAgent(RAGAgent):
       """RAG agent with reranking."""

       async def _retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
           """Retrieve with reranking."""

           # Step 1: Initial retrieval (more candidates)
           query_result = self.embedding_service.embed_query(query)
           query_embedding = query_result['embedding']

           search_results = self.vector_client.search(
               config=self.collection_config,
               query_vectors=[query_embedding],
               limit=self.top_k * 3,  # Get 3x candidates
               output_fields=["id", "text", "source", "chunk_index"]
           )

           # Step 2: Rerank using cross-encoder (simplified)
           candidates = []
           if search_results and len(search_results) > 0:
               for result in search_results[0]:
                   candidates.append({
                       'text': result.get('text', ''),
                       'source': result.get('source', 'unknown'),
                       'chunk_index': result.get('chunk_index', 0),
                       'distance': result.get('distance', 0.0),
                       'score': self._calculate_relevance_score(
                           query,
                           result.get('text', '')
                       )
                   })

           # Sort by reranked score and return top-k
           candidates.sort(key=lambda x: x['score'], reverse=True)
           return candidates[:self.top_k]

       def _calculate_relevance_score(self, query: str, text: str) -> float:
           """Simple relevance scoring (can be improved with cross-encoder)."""
           # Simple keyword matching score
           query_terms = set(query.lower().split())
           text_terms = set(text.lower().split())

           if not query_terms:
               return 0.0

           overlap = len(query_terms & text_terms)
           score = overlap / len(query_terms)

           return score

Conversation Memory with RAG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arshai.memory.working_memory.in_memory_manager import InMemoryManager

   class ConversationalRAGAgent(RAGAgent):
       """RAG agent with conversation memory."""

       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.memory_manager = InMemoryManager()
           self.conversation_id = "rag_session"

       async def process(self, input):
           """Process with conversation history."""

           # Get conversation history
           history = self._get_conversation_history()

           # Enhance query with history context
           enhanced_query = self._build_query_with_history(
               input.message,
               history
           )

           # Standard RAG processing
           result = await super().process(
               IAgentInput(message=enhanced_query)
           )

           # Store conversation turn
           self._store_conversation_turn(
               input.message,
               result['response']
           )

           return result

       def _get_conversation_history(self) -> str:
           """Get recent conversation history."""
           from arshai.core.interfaces.imemorymanager import IMemoryInput
           from arshai.memory.memory_types import ConversationMemoryType

           try:
               memory_input = IMemoryInput(
                   conversation_id=self.conversation_id,
                   memory_type=ConversationMemoryType.WORKING_MEMORY
               )
               memories = self.memory_manager.retrieve(memory_input)
               if memories:
                   return memories[0].working_memory
           except:
               pass

           return ""

       def _build_query_with_history(self, query: str, history: str) -> str:
           """Build query enhanced with conversation history."""
           if history:
               # Only use last 3 turns
               history_lines = history.split('\n')[-6:]  # 3 turns * 2 lines
               recent_history = '\n'.join(history_lines)
               return f"Previous conversation:\n{recent_history}\n\nCurrent question: {query}"
           return query

       def _store_conversation_turn(self, question: str, answer: str):
           """Store conversation turn."""
           from arshai.core.interfaces.imemorymanager import IMemoryInput, IWorkingMemory
           from arshai.memory.memory_types import ConversationMemoryType

           try:
               history = self._get_conversation_history()
               new_turn = f"\nQ: {question}\nA: {answer}"
               updated_history = history + new_turn

               memory_input = IMemoryInput(
                   conversation_id=self.conversation_id,
                   memory_type=ConversationMemoryType.WORKING_MEMORY,
                   data=[IWorkingMemory(working_memory=updated_history)]
               )
               self.memory_manager.store(memory_input)
           except Exception as e:
               print(f"Error storing conversation: {e}")

Document Metadata Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class FilteredRAGAgent(RAGAgent):
       """RAG with metadata filtering."""

       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           """Process with optional metadata filters."""

           # Extract filters from input metadata
           filters = input.metadata or {}
           document_type = filters.get('document_type')
           date_range = filters.get('date_range')

           # Build filter expression
           filter_expr = self._build_filter_expression(
               document_type=document_type,
               date_range=date_range
           )

           # Retrieve with filters
           retrieved_docs = await self._retrieve_documents(
               input.message,
               filter_expr=filter_expr
           )

           # Rest of processing...
           context = self._build_context(retrieved_docs)
           response = await self._generate_response(input.message, context)

           return {
               'response': response,
               'sources': [doc['source'] for doc in retrieved_docs],
               'filters_applied': filter_expr
           }

       def _build_filter_expression(self, **kwargs) -> str:
           """Build Milvus filter expression."""
           filters = []

           if kwargs.get('document_type'):
               filters.append(f"type == '{kwargs['document_type']}'")

           if kwargs.get('date_range'):
               start, end = kwargs['date_range']
               filters.append(f"date >= '{start}' and date <= '{end}'")

           return " and ".join(filters) if filters else ""

Production Deployment
---------------------

Environment-Based Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os

   class ProductionRAGSystem(RAGSystem):
       """Production-ready RAG system."""

       def __init__(self):
           """Initialize with environment-based config."""

           # Determine environment
           env = os.getenv('ENVIRONMENT', 'development')

           # Environment-specific configuration
           if env == 'production':
               llm_config = ILLMConfig(
                   model="gpt-4",  # Better model for production
                   temperature=0.3,  # Lower temperature for consistency
                   max_tokens=1000
               )

               # Use Redis for production memory
               from arshai.memory.working_memory.redis_memory_manager import RedisWorkingMemoryManager
               memory_manager = RedisWorkingMemoryManager()

               chunk_size = 1000  # Larger chunks
               top_k = 5  # More context
           else:
               llm_config = ILLMConfig(
                   model="gpt-3.5-turbo",
                   temperature=0.7,
                   max_tokens=500
               )

               memory_manager = InMemoryManager()
               chunk_size = 500
               top_k = 3

           # Initialize components with env-specific config
           # ... rest of initialization

Error Handling and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   from datetime import datetime

   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('rag_system.log'),
           logging.StreamHandler()
       ]
   )

   class MonitoredRAGAgent(RAGAgent):
       """RAG agent with monitoring."""

       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.logger = logging.getLogger('RAGAgent')
           self.query_count = 0
           self.error_count = 0

       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           """Process with monitoring."""
           self.query_count += 1
           start_time = datetime.now()

           try:
               self.logger.info(f"Processing query: {input.message[:100]}")

               result = await super().process(input)

               # Log success
               duration = (datetime.now() - start_time).total_seconds()
               self.logger.info(
                   f"Query successful - Duration: {duration:.2f}s, "
                   f"Sources: {len(result['sources'])}"
               )

               return result

           except Exception as e:
               self.error_count += 1
               self.logger.error(f"Query failed: {e}", exc_info=True)

               # Return error response
               return {
                   'response': 'I encountered an error processing your question. Please try again.',
                   'error': str(e),
                   'sources': []
               }

Next Steps
----------

**Enhance the RAG system:**

- Add support for more document types (PDF, DOCX, HTML)
- Implement citation tracking and verification
- Add query expansion and rephrasing
- Implement answer validation

**Scale for production:**

- Deploy with Redis memory: :doc:`../implementations/memory/redis-memory`
- Use production vector database (Zilliz Cloud, Pinecone)
- Add caching layer for frequent queries
- Implement rate limiting and authentication

**Learn more:**

- :doc:`custom-system` - Build custom orchestration
- :doc:`../implementations/components/embeddings` - Explore embedding options
- :doc:`../implementations/components/vector-databases` - Vector DB patterns

Congratulations! You've built a complete RAG system with Arshai! 🎉
