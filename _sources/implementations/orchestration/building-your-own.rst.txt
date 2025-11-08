Building Your Own Orchestration
================================

The workflow system is just one approach to orchestrating agents in Arshai. This guide shows you how to build custom orchestration patterns from scratch, leveraging the framework's three-layer architecture and direct instantiation philosophy.

Philosophy
----------

**You're in Control**

Arshai provides building blocks (LLM clients, agents), not prescriptive patterns. The framework empowers you to:

- Create custom orchestration logic that fits your exact needs
- Choose your own coordination patterns
- Build lightweight or sophisticated systems as required
- Avoid framework lock-in by using simple Python

**When to Build Custom Orchestration**

Build your own orchestration when:

- The workflow system is too rigid for your needs
- You want LLM-driven dynamic coordination
- Your use case requires unique patterns
- You prefer complete control over execution flow
- You want minimal dependencies

Custom Orchestration Patterns
------------------------------

Pattern 1: Simple Sequential Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest orchestration: pass output from one agent to the next.

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from typing import List, Any

   class SimplePipeline:
       """Sequential agent pipeline - each agent processes the previous output."""

       def __init__(self, agents: List[BaseAgent]):
           """Initialize with a list of agents to execute in order."""
           self.agents = agents

       async def execute(self, initial_input: str) -> str:
           """Execute agents sequentially, passing output to next agent."""
           current_input = initial_input

           for agent in self.agents:
               # Process with current agent
               result = await agent.process(IAgentInput(message=current_input))

               # Output becomes input for next agent
               current_input = result if isinstance(result, str) else str(result)

           return current_input

   # Usage
   async def main():
       from arshai.llms.openai import OpenAIClient
       from arshai.core.interfaces.illm import ILLMConfig

       llm_client = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))

       # Create specialized agents
       research_agent = ResearchAgent(llm_client, "Research information")
       analysis_agent = AnalysisAgent(llm_client, "Analyze findings")
       report_agent = ReportAgent(llm_client, "Generate report")

       # Create pipeline
       pipeline = SimplePipeline([research_agent, analysis_agent, report_agent])

       # Execute
       result = await pipeline.execute("What are the latest AI trends?")
       print(result)

Pattern 2: Parallel Processing with Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run multiple agents concurrently and combine results.

.. code-block:: python

   import asyncio
   from typing import List, Dict, Any

   class ParallelOrchestrator:
       """Execute multiple agents in parallel and aggregate results."""

       def __init__(self, agents: Dict[str, BaseAgent]):
           """Initialize with named agents for parallel execution."""
           self.agents = agents

       async def execute(self, input_message: str) -> Dict[str, Any]:
           """Execute all agents in parallel and return aggregated results."""

           # Create tasks for all agents
           tasks = {
               name: agent.process(IAgentInput(message=input_message))
               for name, agent in self.agents.items()
           }

           # Wait for all agents to complete
           results = await asyncio.gather(*tasks.values(), return_exceptions=True)

           # Aggregate results
           aggregated = {}
           for (name, _), result in zip(tasks.items(), results):
               if isinstance(result, Exception):
                   aggregated[name] = {"error": str(result)}
               else:
                   aggregated[name] = result

           return aggregated

   # Usage
   async def parallel_example():
       llm_client = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))

       orchestrator = ParallelOrchestrator({
           "sentiment": SentimentAgent(llm_client, "Analyze sentiment"),
           "summary": SummaryAgent(llm_client, "Summarize text"),
           "keywords": KeywordAgent(llm_client, "Extract keywords")
       })

       results = await orchestrator.execute("This product is amazing!")
       print(results)
       # {
       #     "sentiment": "positive",
       #     "summary": "User praises product quality",
       #     "keywords": ["product", "amazing"]
       # }

Pattern 3: LLM-Driven Dynamic Routing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let the LLM decide which agent to use based on the request.

.. code-block:: python

   from typing import Dict, Optional
   from arshai.core.interfaces.illm import ILLMInput

   class DynamicRouter:
       """LLM-driven agent routing system."""

       def __init__(self, llm_client, agents: Dict[str, BaseAgent]):
           """Initialize with LLM client and available agents."""
           self.llm_client = llm_client
           self.agents = agents

       async def execute(self, user_message: str, context: Optional[str] = None) -> Any:
           """Route message to appropriate agent based on LLM analysis."""

           # Step 1: Analyze request and determine routing
           routing_prompt = f"""
           Available agents:
           {self._format_agent_descriptions()}

           User message: {user_message}

           Which agent should handle this request? Reply with only the agent name.
           """

           llm_input = ILLMInput(
               system_prompt="You are a routing expert. Choose the most appropriate agent.",
               user_message=routing_prompt
           )

           result = await self.llm_client.chat(llm_input)
           chosen_agent_name = result['llm_response'].strip().lower()

           # Step 2: Execute with chosen agent
           if chosen_agent_name in self.agents:
               agent = self.agents[chosen_agent_name]
               return await agent.process(IAgentInput(message=user_message))
           else:
               # Fallback to default agent
               default_agent = list(self.agents.values())[0]
               return await default_agent.process(IAgentInput(message=user_message))

       def _format_agent_descriptions(self) -> str:
           """Format agent descriptions for the routing prompt."""
           descriptions = []
           for name, agent in self.agents.items():
               descriptions.append(f"- {name}: {agent.system_prompt[:100]}")
           return "\n".join(descriptions)

   # Usage
   async def dynamic_routing_example():
       llm_client = OpenAIClient(ILLMConfig(model="gpt-4"))

       router = DynamicRouter(llm_client, {
           "technical": TechnicalAgent(llm_client, "Handle technical questions"),
           "sales": SalesAgent(llm_client, "Handle sales inquiries"),
           "support": SupportAgent(llm_client, "Handle support requests")
       })

       result = await router.execute("How do I install this software?")

Pattern 4: Stateful Orchestration with Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build orchestration with conversation memory across interactions.

.. code-block:: python

   from dataclasses import dataclass, field
   from typing import Dict, Any, List
   from datetime import datetime

   @dataclass
   class OrchestrationState:
       """State container for stateful orchestration."""
       conversation_id: str
       user_id: str
       interaction_count: int = 0
       history: List[Dict[str, Any]] = field(default_factory=list)
       context: Dict[str, Any] = field(default_factory=dict)
       created_at: datetime = field(default_factory=datetime.utcnow)

   class StatefulOrchestrator:
       """Orchestrator with conversation state management."""

       def __init__(self, agent: BaseAgent, memory_manager):
           """Initialize with agent and memory storage."""
           self.agent = agent
           self.memory_manager = memory_manager
           self.states: Dict[str, OrchestrationState] = {}

       async def execute(self, user_id: str, conversation_id: str, message: str) -> str:
           """Execute with state management across interactions."""

           # Get or create state
           state = self._get_or_create_state(user_id, conversation_id)

           # Update state
           state.interaction_count += 1
           state.history.append({
               "message": message,
               "timestamp": datetime.utcnow().isoformat()
           })

           # Build context from history
           context = self._build_context(state)

           # Process with agent
           agent_input = IAgentInput(
               message=message,
               metadata={
                   "conversation_id": conversation_id,
                   "context": context
               }
           )

           result = await self.agent.process(agent_input)

           # Update state with response
           state.history.append({
               "response": result,
               "timestamp": datetime.utcnow().isoformat()
           })

           # Persist state
           await self._persist_state(state)

           return result

       def _get_or_create_state(self, user_id: str, conversation_id: str) -> OrchestrationState:
           """Get existing state or create new one."""
           key = f"{user_id}:{conversation_id}"
           if key not in self.states:
               self.states[key] = OrchestrationState(
                   conversation_id=conversation_id,
                   user_id=user_id
               )
           return self.states[key]

       def _build_context(self, state: OrchestrationState) -> str:
           """Build context string from conversation history."""
           last_n = 5  # Include last 5 interactions
           recent_history = state.history[-last_n*2:] if len(state.history) > last_n*2 else state.history

           context_parts = []
           for entry in recent_history:
               if "message" in entry:
                   context_parts.append(f"User: {entry['message']}")
               elif "response" in entry:
                   context_parts.append(f"Assistant: {entry['response']}")

           return "\n".join(context_parts)

       async def _persist_state(self, state: OrchestrationState):
           """Persist state to memory manager."""
           # Implement persistence logic
           pass

Pattern 5: Hierarchical Agent System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Master agent coordinates specialist agents.

.. code-block:: python

   class HierarchicalOrchestrator:
       """Master agent that coordinates specialist agents."""

       def __init__(self, master_llm, specialists: Dict[str, BaseAgent]):
           """Initialize with master LLM and specialist agents."""
           self.master_llm = master_llm
           self.specialists = specialists

       async def execute(self, user_message: str) -> str:
           """Master agent coordinates specialists to solve complex tasks."""

           # Create tools for master agent (each specialist becomes a tool)
           specialist_tools = self._create_specialist_tools()

           # Master agent decides which specialists to use and how
           master_input = ILLMInput(
               system_prompt="""You are a master coordinator.
               You have access to specialist agents as tools.
               Analyze the request and use specialists as needed to provide comprehensive answers.""",
               user_message=user_message,
               regular_functions=specialist_tools
           )

           result = await self.master_llm.chat(master_input)
           return result.get('llm_response', '')

       def _create_specialist_tools(self) -> Dict[str, callable]:
           """Convert specialist agents into callable tools for the master."""
           tools = {}

           for name, agent in self.specialists.items():
               # Create a closure that captures the agent
               async def call_specialist(query: str, agent=agent) -> str:
                   """Call specialist agent with query."""
                   result = await agent.process(IAgentInput(message=query))
                   return str(result)

               # Add as tool
               call_specialist.__name__ = f"use_{name}_specialist"
               call_specialist.__doc__ = f"Use {name} specialist for: {agent.system_prompt[:100]}"
               tools[f"use_{name}_specialist"] = call_specialist

           return tools

   # Usage
   async def hierarchical_example():
       master_llm = OpenAIClient(ILLMConfig(model="gpt-4"))
       specialist_llm = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))

       orchestrator = HierarchicalOrchestrator(
           master_llm=master_llm,
           specialists={
               "research": ResearchAgent(specialist_llm, "Research specialist"),
               "math": MathAgent(specialist_llm, "Math specialist"),
               "writing": WritingAgent(specialist_llm, "Writing specialist")
           }
       )

       result = await orchestrator.execute(
           "Research AI trends, calculate market size, and write a summary"
       )

Advanced Patterns
-----------------

Pattern 6: Event-Driven Orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use events to trigger agent actions.

.. code-block:: python

   from typing import Callable, Dict, List
   from dataclasses import dataclass

   @dataclass
   class Event:
       """Event in the system."""
       type: str
       data: Dict[str, Any]
       timestamp: datetime = field(default_factory=datetime.utcnow)

   class EventDrivenOrchestrator:
       """Orchestrator that responds to events."""

       def __init__(self):
           """Initialize event handlers."""
           self.handlers: Dict[str, List[Callable]] = {}
           self.event_history: List[Event] = []

       def register_handler(self, event_type: str, handler: Callable):
           """Register handler for specific event type."""
           if event_type not in self.handlers:
               self.handlers[event_type] = []
           self.handlers[event_type].append(handler)

       async def emit_event(self, event: Event):
           """Emit event and trigger handlers."""
           self.event_history.append(event)

           if event.type in self.handlers:
               for handler in self.handlers[event.type]:
                   await handler(event)

       async def execute(self, initial_event: Event) -> List[Event]:
           """Execute orchestration starting with initial event."""
           await self.emit_event(initial_event)
           return self.event_history

   # Usage
   async def event_driven_example():
       orchestrator = EventDrivenOrchestrator()

       # Define event handlers
       async def on_user_message(event: Event):
           message = event.data.get("message")
           # Process with agent
           result = await some_agent.process(IAgentInput(message=message))
           # Emit new event
           await orchestrator.emit_event(Event(
               type="agent_response",
               data={"response": result}
           ))

       async def on_agent_response(event: Event):
           response = event.data.get("response")
           # Log or further process
           print(f"Agent responded: {response}")

       # Register handlers
       orchestrator.register_handler("user_message", on_user_message)
       orchestrator.register_handler("agent_response", on_agent_response)

       # Trigger orchestration
       await orchestrator.execute(Event(
           type="user_message",
           data={"message": "Hello!"}
       ))

Pattern 7: Retry and Fallback Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robust orchestration with retries and fallbacks.

.. code-block:: python

   from typing import Optional, List
   import asyncio

   class RobustOrchestrator:
       """Orchestrator with retry and fallback logic."""

       def __init__(self, primary_agent: BaseAgent, fallback_agents: Optional[List[BaseAgent]] = None):
           """Initialize with primary and fallback agents."""
           self.primary_agent = primary_agent
           self.fallback_agents = fallback_agents or []

       async def execute(
           self,
           message: str,
           max_retries: int = 3,
           retry_delay: float = 1.0
       ) -> Dict[str, Any]:
           """Execute with retry logic and fallback."""

           # Try primary agent with retries
           for attempt in range(max_retries):
               try:
                   result = await self._execute_with_timeout(
                       self.primary_agent,
                       message,
                       timeout=10.0
                   )

                   return {
                       "result": result,
                       "agent": "primary",
                       "attempts": attempt + 1
                   }

               except Exception as e:
                   if attempt < max_retries - 1:
                       await asyncio.sleep(retry_delay)
                       continue
                   else:
                       # All retries exhausted, try fallbacks
                       return await self._try_fallbacks(message, e)

       async def _execute_with_timeout(
           self,
           agent: BaseAgent,
           message: str,
           timeout: float
       ) -> Any:
           """Execute agent with timeout."""
           return await asyncio.wait_for(
               agent.process(IAgentInput(message=message)),
               timeout=timeout
           )

       async def _try_fallbacks(self, message: str, original_error: Exception) -> Dict[str, Any]:
           """Try fallback agents in order."""
           for i, fallback_agent in enumerate(self.fallback_agents):
               try:
                   result = await self._execute_with_timeout(
                       fallback_agent,
                       message,
                       timeout=10.0
                   )

                   return {
                       "result": result,
                       "agent": f"fallback_{i}",
                       "original_error": str(original_error)
                   }

               except Exception:
                   continue

           # All agents failed
           return {
               "result": None,
               "error": "All agents failed",
               "original_error": str(original_error)
           }

Design Principles
-----------------

When building custom orchestration:

1. **Start Simple**
   Begin with the simplest pattern that works, add complexity only when needed.

2. **Explicit Over Implicit**
   Make orchestration logic visible and understandable.

3. **Composability**
   Build small, focused orchestration units that can be composed.

4. **Error Handling**
   Handle errors gracefully at every level.

5. **Testability**
   Design for easy testing with mocks and stubs.

6. **Performance**
   Use parallel execution where appropriate.

7. **Observability**
   Include logging and monitoring from the start.

Testing Custom Orchestration
-----------------------------

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock

   @pytest.mark.asyncio
   async def test_simple_pipeline():
       # Create mock agents
       agent1 = AsyncMock()
       agent1.process.return_value = "intermediate_result"

       agent2 = AsyncMock()
       agent2.process.return_value = "final_result"

       # Create pipeline
       pipeline = SimplePipeline([agent1, agent2])

       # Execute
       result = await pipeline.execute("input")

       # Verify
       assert result == "final_result"
       agent1.process.assert_called_once()
       agent2.process.assert_called_once()

   @pytest.mark.asyncio
   async def test_parallel_orchestrator():
       # Create mock agents
       agent1 = AsyncMock()
       agent1.process.return_value = "result1"

       agent2 = AsyncMock()
       agent2.process.return_value = "result2"

       # Create orchestrator
       orchestrator = ParallelOrchestrator({
           "agent1": agent1,
           "agent2": agent2
       })

       # Execute
       results = await orchestrator.execute("input")

       # Verify
       assert results["agent1"] == "result1"
       assert results["agent2"] == "result2"

When to Use Each Pattern
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pattern
     - Best For
   * - Sequential Pipeline
     - Simple multi-step processing, data transformation chains
   * - Parallel Processing
     - Independent analyses, multiple perspectives on same input
   * - Dynamic Routing
     - Variable request types, intelligent task decomposition
   * - Stateful Orchestration
     - Conversations, multi-turn interactions, context preservation
   * - Hierarchical System
     - Complex tasks requiring coordination, master-specialist patterns
   * - Event-Driven
     - Reactive systems, decoupled components, audit trails
   * - Retry/Fallback
     - Production reliability, handling external service failures

Next Steps
----------

- **Review Workflow System**: See :doc:`workflow-system` for a complete reference implementation
- **Framework Patterns**: Study :doc:`../../framework/building-systems/index` for composition patterns
- **Example Code**: Check the `examples/ directory <https://github.com/felesh-ai/arshai/tree/main/examples>`_

Remember: The best orchestration pattern is the one that solves your specific problem most simply and effectively. Don't over-engineer - start with what you need and evolve from there.
