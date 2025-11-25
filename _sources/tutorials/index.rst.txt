Tutorials
=========

Complete end-to-end tutorials showing how to build real AI applications with Arshai. Each tutorial builds a fully functional application from scratch, demonstrating core concepts and best practices.

.. toctree::
   :maxdepth: 2
   :caption: Available Tutorials

   simple-chatbot
   rag-system
   custom-system

Overview
--------

These tutorials take you from concept to working application, with complete code examples and explanations. Each tutorial is self-contained and can be completed independently.

**Learning Path:**

1. Start with :doc:`simple-chatbot` to learn the basics
2. Move to :doc:`rag-system` to add document retrieval
3. Complete with :doc:`custom-system` for advanced orchestration

Available Tutorials
-------------------

Simple Chatbot
~~~~~~~~~~~~~~

**:doc:`simple-chatbot`**

Build a conversational chatbot with memory and command handling.

**What You'll Learn:**

- Direct instantiation of framework components
- Memory integration for conversations
- CLI interface development
- Basic agent patterns
- Error handling and user experience

**Duration:** 30-45 minutes

**Prerequisites:** Python basics, API key

**Builds:** Complete CLI chatbot with memory, commands, and optional web interface

RAG System
~~~~~~~~~~

**:doc:`rag-system`**

Create a complete Retrieval-Augmented Generation system.

**What You'll Learn:**

- Document processing and chunking
- Embedding generation and storage
- Vector similarity search
- RAG agent implementation
- Production deployment patterns

**Duration:** 60-90 minutes

**Prerequisites:** Completion of simple-chatbot tutorial, Milvus setup

**Builds:** Full document Q&A system with vector search and LLM reasoning

Custom Agentic System
~~~~~~~~~~~~~~~~~~~~~~

**:doc:`custom-system`**

Build a multi-agent customer support system from scratch.

**What You'll Learn:**

- Multi-agent system design
- Custom orchestration without workflows
- Agent specialization and routing
- State management across agents
- Advanced coordination patterns

**Duration:** 90-120 minutes

**Prerequisites:** Understanding of async Python, previous tutorials

**Builds:** Production-ready multi-agent support system with intelligent routing

Tutorial Structure
------------------

Each tutorial follows a consistent structure:

**1. Introduction**
   What you'll build, what you'll learn, time estimate

**2. Setup**
   Project structure, dependencies, environment configuration

**3. Step-by-Step Implementation**
   Incremental development with explanations

**4. Testing and Verification**
   How to test your implementation

**5. Enhancements**
   Optional advanced features

**6. Production Considerations**
   Deployment, scaling, monitoring

**7. Next Steps**
   Where to go from here

Prerequisites
-------------

General Requirements
~~~~~~~~~~~~~~~~~~~~

All tutorials require:

- Python 3.9 or higher
- Virtual environment (recommended)
- Text editor or IDE
- Terminal/command prompt

API Keys
~~~~~~~~

You'll need at least one LLM provider API key:

- **OpenAI** (recommended for tutorials)
- Google Gemini
- Azure OpenAI
- OpenRouter

See :doc:`../getting-started/installation` for setup details.

Installation
~~~~~~~~~~~~

Basic installation for all tutorials:

.. code-block:: bash

   pip install arshai[openai]

Additional dependencies are specified in each tutorial.

Choosing Your Path
------------------

**I'm New to Arshai**

Start with :doc:`simple-chatbot`. This tutorial introduces core concepts and builds confidence with hands-on coding.

**I Want to Build RAG Applications**

Go directly to :doc:`rag-system` if you're comfortable with Python and want to build document-based Q&A systems.

**I Need Multi-Agent Systems**

Jump to :doc:`custom-system` if you have experience with Arshai and need to build complex agentic systems.

**I Want to Learn Everything**

Follow the tutorials in order:
1. :doc:`simple-chatbot` - Foundation
2. :doc:`rag-system` - Document integration
3. :doc:`custom-system` - Advanced orchestration

Tutorial Features
-----------------

**Complete Code**

Every tutorial includes full, working code - not just snippets. You can copy, run, and modify the complete applications.

**Incremental Development**

Build applications step-by-step, testing at each stage. This helps you understand how components work together.

**Production-Ready Patterns**

Learn patterns used in real production systems, including error handling, logging, and monitoring.

**Extension Points**

Each tutorial includes optional enhancements and suggestions for taking the application further.

**Clear Explanations**

Code is explained with inline comments and accompanying documentation that describes the why, not just the what.

Getting Help
------------

**Stuck on a Tutorial?**

- Re-read the prerequisites section
- Check the troubleshooting tips in each tutorial
- Review the :doc:`../getting-started/index` section
- Consult the :doc:`../framework/index` for detailed concepts

**Found an Issue?**

- Report it on `GitHub Issues <https://github.com/felesh-ai/arshai/issues>`_
- Include which tutorial and step you're on
- Provide error messages and context

**Want to Share Your Work?**

We'd love to see what you build! Share your projects with the community.

After the Tutorials
-------------------

Once you've completed the tutorials, you're ready to:

**Build Your Own Applications**

Apply the patterns you've learned to your specific use cases.

**Explore Advanced Topics**

- :doc:`../implementations/orchestration/building-your-own` - More orchestration patterns
- :doc:`../implementations/memory/index` - Advanced memory strategies
- :doc:`../implementations/components/index` - Component integrations

**Contribute**

Help improve Arshai by:

- Sharing your implementations
- Contributing to documentation
- Reporting issues and suggesting features
- Helping other developers

Best Practices from Tutorials
------------------------------

**Direct Instantiation**

Create components explicitly rather than relying on magic configuration:

.. code-block:: python

   llm_client = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))
   agent = MyAgent(llm_client, system_prompt)

**Error Handling**

Always handle errors gracefully:

.. code-block:: python

   try:
       result = await agent.process(input)
   except Exception as e:
       logger.error(f"Error: {e}")
       return fallback_response

**State Management**

Keep state explicit and manageable:

.. code-block:: python

   @dataclass
   class State:
       user_id: str
       conversation_history: List[Turn]
       metadata: Dict[str, Any]

**Testing**

Write tests for your components:

.. code-block:: python

   async def test_agent():
       mock_llm = AsyncMock()
       agent = MyAgent(mock_llm, "test prompt")
       result = await agent.process(test_input)
       assert result is not None

**Documentation**

Document your code:

.. code-block:: python

   class MyAgent(BaseAgent):
       """
       Custom agent that does X.

       Args:
           llm_client: LLM client for processing
           system_prompt: Agent's system prompt
       """

Common Patterns Learned
-----------------------

**Memory Integration**

.. code-block:: python

   memory_manager = InMemoryManager()
   agent = WorkingMemoryAgent(llm_client, memory_manager)

**Document Processing**

.. code-block:: python

   chunker = DocumentChunker(chunk_size=500)
   chunks = chunker.process_document(file_path)

**Vector Search**

.. code-block:: python

   embedding = embedder.embed_query(query)
   results = vector_db.search(query_vector=embedding)

**Multi-Agent Coordination**

.. code-block:: python

   coordinator = Coordinator()
   result = await coordinator.process_request(user_input)

Start Your Journey
------------------

Ready to build? Choose your first tutorial:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Tutorial
     - Best For
     - Duration
   * - :doc:`simple-chatbot`
     - Beginners, learning basics
     - 30-45 min
   * - :doc:`rag-system`
     - Document Q&A systems
     - 60-90 min
   * - :doc:`custom-system`
     - Multi-agent systems
     - 90-120 min

Happy building! 🚀

----

*These tutorials are living documents. If you have suggestions for improvements or additional tutorials, please open an issue on GitHub.*
