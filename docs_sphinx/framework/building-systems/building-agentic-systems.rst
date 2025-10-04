Building Agentic Systems
========================

This section explores the capabilities and building blocks that Arshai provides for creating sophisticated agentic systems. The framework's architecture enables you to build various coordination mechanisms and multi-agent systems without prescribing any specific approach.

Core Framework Capabilities
----------------------------

**Function Calling as Coordination Mechanism**
   The framework's LLM function calling system enables agents to coordinate naturally through tool invocations. Agents can call other agents as tools, creating dynamic coordination patterns.

**Background Tasks for System Events**
   Background tasks allow agents to trigger system-wide events, logging, notifications, and coordination without blocking the conversation flow. This enables fire-and-forget coordination patterns.

**Direct Agent Communication**
   Agents can communicate directly by calling each other's process methods, enabling peer-to-peer coordination and complex interaction patterns.

**Shared Context and Memory**
   The memory system and metadata propagation allow agents to share context and maintain system-wide state across multiple interactions.

**Flexible Response Types**
   Agents can return any data type - strings, structured objects, streams, or custom types - enabling rich information exchange between system components.

Framework Building Blocks for Agentic Systems
----------------------------------------------

**BaseAgent Foundation**
   Every agent extends BaseAgent, providing a consistent interface while allowing complete customization of behavior, making agents composable system components.

**Tool Integration System**
   Any Python callable can be a tool, including other agents. This enables natural composition where agents become tools for other agents, creating hierarchical systems.

**Memory and Context Management**
   WorkingMemoryAgent and memory managers provide shared context across agents, enabling persistent conversations and coordinated decision-making.

**Asynchronous Execution**
   Full async support enables parallel agent execution, concurrent tool calling, and non-blocking system coordination.

**Metadata Propagation**
   Input metadata flows through the system, allowing agents to share context, user information, session data, and coordination signals.

Coordination Mechanisms the Framework Enables
----------------------------------------------

**LLM-Driven Coordination**
   The function calling system allows LLMs to intelligently decide which agents to invoke, how to combine their results, and when to escalate or delegate tasks.

**Event-Driven Architecture**
   Background tasks enable agents to trigger system events, coordinate with external systems, and maintain system-wide awareness without blocking conversations.

**Sequential Processing**
   Agents can process tasks sequentially where each agent's output becomes the next agent's input, enabled by flexible response types and standardized interfaces.

**Hierarchical Decision Making**
   Master agents can coordinate specialist agents, with the framework handling the communication, context sharing, and result aggregation automatically.

**Parallel Processing**
   Multiple agents can process different aspects of a problem simultaneously, with results coordinated through the framework's async capabilities.

**Dynamic Agent Selection**
   Agents can dynamically decide which other agents to invoke based on context, enabled by the tool integration system and function calling.

Real-World System Capabilities
-------------------------------

**Multi-Expertise Systems**
   Combine specialized agents (research, analysis, writing, fact-checking) where each focuses on their domain expertise while the framework handles coordination.

**Scalable Processing Systems**
   Build systems that can process large volumes of requests by distributing work across multiple agent instances and coordinating results.

**Context-Aware Conversations**
   Maintain conversation state across multiple agents, allowing users to interact with a system that remembers context even as different agents handle different parts of the conversation.

**Adaptive System Behavior**
   Create systems that adapt their agent usage based on request complexity, user preferences, or system load, enabled by the framework's flexible composition.

**Cross-Agent Learning**
   Share insights and context between agents through the memory system, enabling system-wide learning and improvement over time.

System Coordination Examples
-----------------------------

**Function-Based Agent Coordination**

.. code-block:: python

   class SystemCoordinator(BaseAgent):
       def __init__(self, llm_client, specialist_agents):
           super().__init__(llm_client, "Coordinate specialists")
           # Convert agents to callable tools
           self.agent_tools = {
               name: agent.process for name, agent in specialist_agents.items()
           }
       
       async def process(self, input: IAgentInput) -> str:
           # LLM decides which agents to use via function calling
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=self.agent_tools
           )
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Background Task Coordination**

.. code-block:: python

   def notify_system_event(event_type: str, agent_id: str, details: str = ""):
       """System-wide event notification (background task)"""
       # Triggers coordination without blocking conversation
       system_coordinator.handle_event(event_type, agent_id, details)
   
   def update_shared_context(key: str, value: str, scope: str = "global"):
       """Update system-wide context (background task)"""
       # Maintains shared state across agents
       context_manager.update(key, value, scope)
   
   class CoordinatedAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           background_tasks = {
               "notify_system": notify_system_event,
               "update_context": update_shared_context
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               background_tasks=background_tasks
           )
           
           # LLM can trigger system coordination automatically
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Memory-Coordinated Systems**

.. code-block:: python

   class MemoryCoordinatedSystem:
       def __init__(self, memory_manager, agents):
           self.memory = memory_manager
           self.agents = agents
       
       async def process_with_coordination(self, request: str, user_id: str):
           # Shared memory enables coordination
           context = await self.memory.get_working_memory(user_id)
           
           # Multiple agents can access shared context
           for agent_name, agent in self.agents.items():
               if self.should_involve_agent(agent_name, request, context):
                   agent_input = IAgentInput(
                       message=request,
                       metadata={"user_id": user_id, "shared_context": context}
                   )
                   result = await agent.process(agent_input)
                   
                   # Update shared context with agent results
                   await self.memory.update_working_memory(
                       user_id, f"{agent_name}_result", result
                   )

**Dynamic System Adaptation**

.. code-block:: python

   class AdaptiveSystem:
       def __init__(self, llm_client, agent_registry):
           self.coordinator = self.create_coordinator(llm_client)
           self.agent_registry = agent_registry
       
       async def process_request(self, request: str, context: dict):
           # System adapts agent selection based on context
           available_agents = self.select_agents_for_context(context)
           
           # Create tools from selected agents
           agent_tools = {
               name: agent.process 
               for name, agent in available_agents.items()
           }
           
           # Coordinator uses available agents dynamically
           coordinator_input = IAgentInput(
               message=request,
               metadata={"available_capabilities": list(agent_tools.keys())}
           )
           
           # Update coordinator's tools dynamically
           self.coordinator.update_tools(agent_tools)
           return await self.coordinator.process(coordinator_input)

**Parallel Agent Execution**

.. code-block:: python

   class ParallelProcessingSystem:
       def __init__(self, agents):
           self.agents = agents
       
       async def process_parallel(self, request: str, user_context: dict):
           # Execute multiple agents concurrently
           tasks = []
           for agent_name, agent in self.agents.items():
               agent_input = IAgentInput(
                   message=request,
                   metadata={"context": user_context, "agent_role": agent_name}
               )
               tasks.append(agent.process(agent_input))
           
           # Gather results from all agents
           results = await asyncio.gather(*tasks)
           
           # Combine results using another agent
           combiner_input = IAgentInput(
               message=f"Combine these agent results: {results}",
               metadata={"original_request": request}
           )
           return await self.result_combiner.process(combiner_input)

Framework Advantages for Agentic Systems
-----------------------------------------

**Natural Composition**
   Agents are just Python objects with async process methods, making them naturally composable into larger systems without framework overhead.

**Flexible Communication**
   Multiple communication patterns (function calling, direct invocation, background tasks, shared memory) allow you to choose the right approach for each use case.

**Intelligent Coordination**
   LLM-driven function calling enables systems that adapt their coordination based on context and requirements rather than fixed rules.

**Scalable Architecture**
   Stateless agent design and async execution enable systems that scale horizontally by adding more agent instances.

**Observable Systems**
   Background tasks and metadata propagation provide natural hooks for monitoring, logging, and system observability.

**Testable Components**
   Each agent is independently testable, making complex systems easier to validate and debug.

Building Your Agentic System
-----------------------------

**Start with Purpose**
   Define what your system needs to accomplish and identify the distinct capabilities required.

**Design Agent Responsibilities**
   Create focused agents that each handle one aspect well, rather than monolithic agents that do everything.

**Choose Coordination Mechanisms**
   Select coordination mechanisms based on your needs:
   
   - Function calling for intelligent delegation
   - Background tasks for system events
   - Shared memory for persistent context
   - Direct communication for simple interactions

**Implement Incrementally**
   Start with simple agent interactions and add coordination complexity as needed.

**Test System Behavior**
   Test not just individual agents but their interactions and coordination patterns.

**Monitor and Adapt**
   Use background tasks and metadata to monitor system behavior and adapt coordination over time.

Common System Architectures the Framework Enables
--------------------------------------------------

**Coordinator-Based Systems**
   Master agents coordinate specialists using function calling and intelligent delegation.

**Sequential Processing Systems**
   Step-by-step processing enabled by flexible input/output types and standardized interfaces.

**Peer-to-Peer Systems**
   Agents communicate directly using method calls and shared context.

**Hierarchical Systems**
   Multi-level coordination using nested function calling and context propagation.

**Event-Driven Systems**
   System-wide coordination using background tasks and event notification.

**Adaptive Systems**
   Dynamic behavior using metadata-driven agent selection and configuration.

Key Framework Principles
-------------------------

**You Control the Architecture**
   The framework provides capabilities; you decide how to use them for your specific agentic system.

**Composition Over Prescription**
   Build complex systems by composing simple agents rather than following rigid architectural patterns.

**Intelligence-Driven Coordination**
   Leverage LLM function calling for adaptive, context-aware coordination rather than fixed rules.

**Progressive Complexity**
   Start with simple agent interactions and add sophisticated coordination as your system evolves.

**Observable and Maintainable**
   Built-in hooks for monitoring, logging, and debugging complex multi-agent interactions.

The framework doesn't prescribe specific agentic architectures but provides the foundational capabilities that make any architecture possible. Your system's design emerges from your specific needs and the intelligent coordination enabled by the framework's LLM-driven function calling and background task systems.