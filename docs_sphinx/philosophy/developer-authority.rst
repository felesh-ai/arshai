Developer Authority
==================

Developer authority is the core principle that drives Arshai's design. It's the belief that developers should have complete control over their application's components, lifecycle, and behavior.

The Philosophy
--------------

**You're the Architect**: Arshai provides interfaces and building blocks, but you decide how to use them, when to use them, and whether to use them at all.

**No Hidden Magic**: Every component is created explicitly by you. No factories, no global state, no configuration files that hide behavior.

**Interface Respect**: Any component you build that implements our interfaces works seamlessly with the framework.

Traditional vs Arshai Approach
-------------------------------

**Traditional Framework Problems**:

.. code-block:: python

   # Traditional approach - framework controls everything
   framework = AIFramework()
   framework.load_config("config.yaml")
   agent = framework.create_agent("chatbot")  # What's happening inside?
   response = agent.chat("Hello")  # How does this work?

Problems with this approach:
- 🔒 Hidden complexity - You don't know what's being created
- ⛔ Limited control - Framework decides component lifecycle  
- 🎭 Magic behavior - Implicit configuration and dependencies
- 🧩 Poor testability - Hard to mock framework-controlled components

**Arshai Approach**:

.. code-block:: python

   # Arshai approach - you control everything
   llm_config = ILLMConfig(model="gpt-4", temperature=0.7)
   llm_client = OpenAIClient(llm_config)  # You create it

   agent = ChatbotAgent(
       llm_client=llm_client,  # You inject dependencies
       system_prompt="Be helpful",  # You configure it
   )

   response = await agent.process(input)  # You understand the flow

Benefits:
- ✅ Full visibility - See exactly what's created and when
- ✅ Complete control - You manage component lifecycle
- ✅ Explicit behavior - All dependencies are visible
- ✅ Easy testing - Simple to mock and test

Core Principles
---------------

Explicit Over Implicit
^^^^^^^^^^^^^^^^^^^^^^^

**❌ Implicit (Hidden)**:

.. code-block:: python

   # Where do these come from? What do they do?
   agent = factory.create("assistant")
   agent.configure()  # What's being configured?

**✅ Explicit (Visible)**:

.. code-block:: python

   # Everything is clear and visible
   llm_client = OpenAIClient(config)
   memory = RedisMemory(url="redis://localhost")
   agent = AssistantAgent(llm_client, memory, tools=[search, calculate])

Direct Control Over Abstraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**❌ Abstracted Away**:

.. code-block:: python

   # Framework hides the details
   settings = Settings()
   agent = settings.create_agent("type", config)
   # How do I customize this? When is it created?

**✅ Direct Control**:

.. code-block:: python

   # You control creation and configuration
   class MyCustomAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, custom_param: str):
           super().__init__(llm_client, "My prompt")
           self.custom_param = custom_param
       
       async def process(self, input: IAgentInput):
           # Your logic, your control
           return self.my_custom_logic(input)

   # Create when YOU want
   agent = MyCustomAgent(llm_client, "my_value")

Composition Over Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**❌ Configuration-Driven**:

.. code-block:: yaml

   # config.yaml - behavior hidden in configuration
   agent:
     type: assistant
     model: gpt-4
     temperature: 0.7
     memory: redis

**✅ Code-Driven Composition**:

.. code-block:: python

   # Behavior defined in code, not configuration
   def create_assistant(env: str = "prod"):
       if env == "prod":
           llm_client = OpenAIClient(ILLMConfig(model="gpt-4"))
           memory = RedisMemory(url=os.getenv("REDIS_URL"))
       else:
           llm_client = MockLLMClient()
           memory = InMemoryStorage()
       
       return AssistantAgent(
           llm_client=llm_client,
           memory=memory,
           tools=[SearchTool(), CalculateTool()]
       )

Dependency Injection Over Service Location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**❌ Service Location (Anti-pattern)**:

.. code-block:: python

   class BadAgent:
       def __init__(self):
           # Agent finds its own dependencies
           self.llm = ServiceLocator.get("llm")
           self.memory = ServiceLocator.get("memory")
           # Hidden dependencies!

**✅ Dependency Injection**:

.. code-block:: python

   class GoodAgent:
       def __init__(self, llm: ILLM, memory: IMemoryManager):
           # Dependencies are injected
           self.llm = llm
           self.memory = memory
           # Clear, testable, controllable

Real-World Benefits
-------------------

Testing Made Simple
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Easy to test with explicit dependencies
   def test_agent():
       # Create test doubles
       mock_llm = Mock(spec=ILLM)
       mock_llm.chat.return_value = {"llm_response": "Test response"}
       
       mock_memory = Mock(spec=IMemoryManager)
       
       # Inject mocks
       agent = MyAgent(mock_llm, mock_memory)
       
       # Test behavior
       result = await agent.process(test_input)
       assert result == expected
       mock_llm.chat.assert_called_once()

Debugging Transparency
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # You can see exactly what's happening
   agent = DebugAgent(
       llm_client=llm_client,
       system_prompt="Assistant prompt"
   )

   # Add logging where YOU want it
   if debug_mode:
       agent = LoggingWrapper(agent)
       
   # Control the flow
   result = await agent.process(input)
   print(f"Agent used: {agent.__class__.__name__}")

Performance Control
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # You control performance characteristics
   class OptimizedSystem:
       def __init__(self):
           # You decide on connection pooling
           self.llm_pool = [
               OpenAIClient(config) for _ in range(3)
           ]
           
           # You control caching
           self.cache = Redis(max_connections=10)
       
       async def process(self, requests: List[str]):
           # You control the execution strategy
           tasks = [
               self.process_one(req, self.llm_pool[i % 3])
               for i, req in enumerate(requests)
           ]
           return await asyncio.gather(*tasks)

Common Patterns
---------------

Progressive Enhancement
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Start simple
   basic_agent = SimpleAgent(llm_client)

   # Add capabilities as needed
   if needs_memory:
       agent_with_memory = MemoryWrapper(basic_agent, memory_manager)
   else:
       agent_with_memory = basic_agent

   if needs_tools:
       final_agent = ToolsWrapper(agent_with_memory, tools)
   else:
       final_agent = agent_with_memory

Custom Pipelines
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Build your own processing pipeline
   class CustomPipeline:
       def __init__(self, components: List[Component]):
           self.components = components  # You control the order
       
       async def process(self, input: Any) -> Any:
           result = input
           for component in self.components:
               result = await component.process(result)
               # You can add logging, caching, etc.
           return result

   # Compose your pipeline
   pipeline = CustomPipeline([
       InputValidator(),
       PreProcessor(), 
       MainAgent(llm_client),
       PostProcessor(),
       OutputFormatter()
   ])

What This Means for You
-----------------------

When using Arshai:

1. **You decide** when components are created
2. **You control** how they're configured  
3. **You manage** their lifecycle
4. **You understand** the flow
5. **You own** the architecture

The framework provides powerful building blocks, but you're the architect. You decide how to use them, when to use them, and whether to use them at all.

**The Bottom Line**: "The framework should empower you, not constrain you."