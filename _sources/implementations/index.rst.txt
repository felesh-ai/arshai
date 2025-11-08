Reference Implementations
=========================

This section documents reference implementations provided with the Arshai framework. These are working examples that demonstrate how to use the framework's building blocks to create real-world components.

.. important::
   **Framework vs Reference Implementations**
   
   These implementations are **not part of the core framework**. They represent "our experience" with using the framework - working examples that show one approach to solving common problems. You are encouraged to:
   
   - Use them as-is if they fit your needs
   - Modify them for your specific requirements  
   - Build completely different implementations
   - Ignore them entirely and create your own approach

.. toctree::
   :maxdepth: 2
   :caption: Reference Implementation Categories
   
   agents/index
   orchestration/index
   memory/index
   components/index

Philosophy
----------

**Reference, Not Prescription**
   These implementations show how we've used the framework in our projects. They're not the "right way" or "only way" - just working examples of what's possible.

**Learning Resources**
   Use these implementations to understand patterns, see real code, and learn techniques. Then adapt them or build your own.

**Starting Points**
   Many developers find it helpful to start with a reference implementation and modify it rather than building from scratch.

**Production Examples**
   All reference implementations are production-quality code that we've used in real projects, complete with error handling and testing.

What You'll Find
-----------------

**Agent Implementations**
   Working agents that demonstrate memory integration, specialized behaviors, and real-world patterns.

**Orchestration Examples**
   Systems that coordinate multiple agents using different patterns and coordination mechanisms.

**Memory Implementations**
   Complete memory management solutions for different storage backends and use cases.

**Component Implementations**
   Working examples of embeddings, vector databases, and other framework components.

Key Principles
--------------

**Complete Examples**
   Each implementation is fully functional with real code, not just snippets or pseudo-code.

**Clear Documentation**
   Every implementation explains why it was built this way, what problems it solves, and how to adapt it.

**Framework-Focused**
   Examples emphasize how they use the framework's building blocks rather than external dependencies.

**Honest Limitations**
   Documentation clearly states what each implementation can and cannot do, and when you might need something different.

Using Reference Implementations
--------------------------------

**Direct Usage**
   Import and use reference implementations directly in your projects if they meet your needs.

.. code-block:: python

   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   from arshai.memory.in_memory_manager import InMemoryManager
   
   # Use reference implementations directly
   memory_manager = InMemoryManager()
   agent = WorkingMemoryAgent(llm_client, memory_manager)

**Adaptation Pattern**
   Copy reference implementation code and modify for your specific needs.

.. code-block:: python

   # Start with reference implementation
   from arshai.agents.hub.working_memory import WorkingMemoryAgent
   
   class MyCustomMemoryAgent(WorkingMemoryAgent):
       """Customized version of WorkingMemoryAgent"""
       
       async def process(self, input: IAgentInput) -> str:
           # Add your custom logic
           custom_preprocessing(input)
           
           # Call parent implementation
           result = await super().process(input)
           
           # Add your custom post-processing
           return custom_postprocessing(result)

**Learning Pattern**
   Study reference implementations to understand techniques, then build your own from scratch.

.. code-block:: python

   # Learn from WorkingMemoryAgent, then build your own
   class MyAgentApproach(BaseAgent):
       """My own approach inspired by reference implementations"""
       
       def __init__(self, llm_client, my_storage_system):
           super().__init__(llm_client, "My system prompt")
           self.storage = my_storage_system
       
       async def process(self, input: IAgentInput) -> str:
           # Your own implementation using lessons learned
           pass

Implementation Categories
-------------------------

**Agents** (:doc:`agents/index`)
   Reference agent implementations that demonstrate real-world patterns:
   
   - ``WorkingMemoryAgent``: Memory-enabled conversational agent
   - Future agent examples as they're developed

**Orchestration** (:doc:`orchestration/index`)
   Reference system implementations that coordinate multiple agents:
   
   - Workflow-based orchestration systems
   - Custom coordination patterns
   - Multi-agent system examples

**Memory** (:doc:`memory/index`)
   Reference memory management implementations:
   
   - ``InMemoryManager``: Simple in-memory storage
   - ``RedisMemoryManager``: Redis-backed persistent storage
   - Custom memory patterns

**Components** (:doc:`components/index`)
   Reference implementations for framework components:
   
   - Embedding implementations (OpenAI, VoyageAI, MGTE)
   - Vector database clients (Milvus)
   - Tool implementations and patterns

Next Steps
----------

1. **Browse Categories**: Explore the implementation categories that interest you
2. **Study Code**: Look at the actual implementation code to understand patterns
3. **Try Examples**: Use reference implementations in your own projects
4. **Adapt and Build**: Modify examples or create your own implementations
5. **Contribute**: Share your own implementations with the community

Remember: These are reference implementations showing what's possible with the framework. The framework itself provides the building blocks - these examples show how we've assembled them into working solutions.