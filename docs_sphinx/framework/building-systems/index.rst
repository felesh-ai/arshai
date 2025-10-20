Building Systems (Layer 3)
===========================

Layer 3 provides the framework for composing agents into complete agentic systems. This is where individual agents become components in larger, coordinated solutions.

.. toctree::
   :maxdepth: 2
   :caption: System Building Patterns

   composition-patterns
   building-agentic-systems

Core Philosophy
---------------

**Agentic Systems, Not Just Agents**
   The framework's vision is to enable building complete agentic systems where agents work together to solve complex problems.

**Composition Over Monoliths**
   Build sophisticated systems by composing simple, focused agents rather than creating complex monolithic agents.

**Flexible Orchestration**
   Choose the coordination pattern that fits your problem - orchestrator, pipeline, mesh, or custom.

**Direct Control**
   You explicitly define how agents interact and coordinate - no hidden orchestration magic.

System Building Concepts
------------------------

**Agent Composition**
   Combining multiple specialized agents to handle complex tasks that no single agent could solve effectively.

**Orchestration Patterns**
   Different ways to coordinate agent interactions:
   
   - **Orchestrator**: Master agent coordinates specialists
   - **Pipeline**: Sequential processing through agents
   - **Mesh**: Peer-to-peer agent communication
   - **Factory**: Dynamic agent creation as needed

**Communication Mechanisms**
   How agents share information and coordinate:
   
   - **Function Calling**: LLM-based coordination through tools
   - **Message Passing**: Direct agent-to-agent communication
   - **Shared Context**: Common metadata and memory

**State Management**
   Handling system-wide state across multiple agents:
   
   - **Conversation Context**: Shared memory across agents
   - **Workflow State**: System-level state tracking
   - **Session Management**: User-specific context

Basic Composition Example
-------------------------

Here's a simple multi-agent system:

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   
   class ResearchAgent(BaseAgent):
       """Agent that researches information."""
       async def process(self, input: IAgentInput) -> dict:
           # Research logic
           return {"findings": "research results"}
   
   class AnalysisAgent(BaseAgent):
       """Agent that analyzes data."""
       async def process(self, input: IAgentInput) -> dict:
           # Analysis logic
           return {"analysis": "analysis results"}
   
   class ReportAgent(BaseAgent):
       """Agent that generates reports."""
       async def process(self, input: IAgentInput) -> str:
           # Report generation
           return "Final report"
   
   # Compose agents into a system
   class ResearchSystem:
       def __init__(self, llm_client):
           self.research_agent = ResearchAgent(llm_client, "Research prompt")
           self.analysis_agent = AnalysisAgent(llm_client, "Analysis prompt")
           self.report_agent = ReportAgent(llm_client, "Report prompt")
       
       async def process_request(self, query: str) -> str:
           # Step 1: Research
           research_result = await self.research_agent.process(
               IAgentInput(message=query)
           )
           
           # Step 2: Analyze findings
           analysis_result = await self.analysis_agent.process(
               IAgentInput(message=str(research_result))
           )
           
           # Step 3: Generate report
           report = await self.report_agent.process(
               IAgentInput(message=str(analysis_result))
           )
           
           return report

Orchestration Patterns
----------------------

**1. Orchestrator Pattern**

A master agent coordinates multiple specialized agents:

.. code-block:: python

   class OrchestratorAgent(BaseAgent):
       def __init__(self, llm_client, specialized_agents):
           self.agents = specialized_agents
           super().__init__(llm_client, "Orchestration prompt")
       
       async def process(self, input: IAgentInput) -> Any:
           # LLM decides which agents to use
           # Coordinates their work
           # Returns combined result
           pass

**2. Pipeline Pattern**

Sequential processing through multiple agents:

.. code-block:: python

   class PipelineSystem:
       def __init__(self, agents: List[BaseAgent]):
           self.pipeline = agents
       
       async def process(self, input: Any) -> Any:
           current_input = input
           for agent in self.pipeline:
               current_input = await agent.process(
                   IAgentInput(message=str(current_input))
               )
           return current_input

**3. Mesh Pattern**

Agents communicate peer-to-peer:

.. code-block:: python

   class MeshSystem:
       def __init__(self, agents: Dict[str, BaseAgent]):
           self.mesh = agents
           # Each agent can call others
       
       async def process(self, request: str) -> Any:
           # Agents coordinate dynamically
           # No central controller
           pass

**4. Factory Pattern**

Dynamic agent creation based on needs:

.. code-block:: python

   class AgentFactory:
       def __init__(self, llm_client):
           self.llm_client = llm_client
           self.agent_cache = {}
       
       def create_specialist(self, specialty: str) -> BaseAgent:
           if specialty not in self.agent_cache:
               # Create specialized agent
               agent = self.create_agent_for(specialty)
               self.agent_cache[specialty] = agent
           return self.agent_cache[specialty]

When to Use Each Pattern
------------------------

**Use Orchestrator when:**
- You need intelligent task decomposition
- The coordination logic is complex
- Different requests need different agent combinations
- You want the LLM to decide coordination

**Use Pipeline when:**
- Processing is sequential and predictable
- Each stage transforms the previous output
- You need clear processing stages
- Order of operations matters

**Use Mesh when:**
- Agents need to collaborate dynamically
- No single controller makes sense
- Peer-to-peer communication is natural
- Complex cross-referencing is needed

**Use Factory when:**
- Agent types aren't known at startup
- You need specialized agents on-demand
- Resource optimization is important
- Scaling requires dynamic agent creation

System Design Principles
------------------------

**1. Separation of Concerns**
   Each agent should have a single, well-defined responsibility.

**2. Loose Coupling**
   Agents should communicate through well-defined interfaces, not internal details.

**3. High Cohesion**
   Related functionality should be grouped within the same agent.

**4. Scalability**
   Design systems that can handle increased load by adding agents.

**5. Fault Tolerance**
   Individual agent failures shouldn't crash the entire system.

**6. Observability**
   Include logging and monitoring to understand system behavior.

Real-World System Example
-------------------------

A customer service system using multiple patterns:

.. code-block:: python

   class CustomerServiceSystem:
       """Complete customer service agentic system."""
       
       def __init__(self, llm_client, database, knowledge_base):
           # Core agents
           self.triage_agent = TriageAgent(llm_client)
           self.support_agent = SupportAgent(llm_client, knowledge_base)
           self.escalation_agent = EscalationAgent(llm_client)
           
           # Orchestrator for complex requests
           self.orchestrator = OrchestratorAgent(llm_client, {
               'triage': self.triage_agent,
               'support': self.support_agent,
               'escalation': self.escalation_agent
           })
           
           # Pipeline for standard flow
           self.standard_pipeline = PipelineSystem([
               self.triage_agent,
               self.support_agent
           ])
           
           self.database = database
       
       async def handle_request(self, customer_request: str, customer_id: str):
           """Route request through appropriate pattern."""
           
           # Determine complexity
           if self.is_complex_request(customer_request):
               # Use orchestrator for complex cases
               result = await self.orchestrator.process(
                   IAgentInput(
                       message=customer_request,
                       metadata={"customer_id": customer_id}
                   )
               )
           else:
               # Use pipeline for standard cases
               result = await self.standard_pipeline.process(
                   customer_request
               )
           
           # Log to database
           await self.database.log_interaction(customer_id, result)
           
           return result

Performance Considerations
--------------------------

**Parallel Execution**
   Run independent agents concurrently:
   
   .. code-block:: python
   
      results = await asyncio.gather(
          agent1.process(input1),
          agent2.process(input2),
          agent3.process(input3)
      )

**Caching**
   Cache agent results to avoid redundant processing.

**Resource Management**
   Use semaphores to limit concurrent agent executions.

**Load Balancing**
   Distribute work across multiple agent instances.

Testing Multi-Agent Systems
---------------------------

**Unit Testing**
   Test each agent independently with mocks.

**Integration Testing**
   Test agent interactions with real components.

**System Testing**
   Test the complete system end-to-end.

**Performance Testing**
   Measure throughput, latency, and resource usage.

Next Steps
----------

- **Study Patterns**: Deep dive into :doc:`composition-patterns`
- **Learn System Building**: Explore :doc:`building-agentic-systems`
- **See Examples**: Check :doc:`../../implementations/orchestration/index` for reference implementations
- **Build Your System**: Apply these patterns to your use case

Key Takeaway
------------

Layer 3 is where agents become systems. The framework provides the patterns and mechanisms - you design the solution. Whether you need a simple pipeline or a complex orchestrated system, the building blocks are here for you to compose as needed.

Remember: **Agents are components. Systems are solutions.**