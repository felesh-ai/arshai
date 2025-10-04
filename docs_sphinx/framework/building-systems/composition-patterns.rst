Example 5: Agent Composition
============================

This example demonstrates advanced agent composition patterns for building multi-agent systems that solve complex problems through coordination and collaboration.

**File**: ``examples/agents/05_agent_composition.py`` (504 lines)

**Focus**: Multi-agent systems and orchestration

**Best For**: Understanding how agents work together in complex workflows

Overview
--------

This example showcases four major composition patterns:

- **Orchestrator Pattern**: Master agent coordinates specialized agents
- **Pipeline Pattern**: Sequential processing through multiple agents
- **Mesh Pattern**: Interconnected agents that communicate dynamically
- **Factory Pattern**: Dynamic agent creation for specialized tasks

These patterns demonstrate how Arshai's agent architecture enables sophisticated multi-agent coordination using LLM function calling as the coordination mechanism.

Key Concepts Demonstrated
-------------------------

**Agent Orchestration**
   How a master agent coordinates multiple specialized agents to complete complex tasks.

**Sequential Processing**
   Pipeline pattern where each agent processes the output of the previous agent.

**Dynamic Collaboration**
   Mesh pattern enabling agents to communicate and cross-reference information.

**Dynamic Agent Creation**
   Factory pattern for creating specialized agents on-demand.

**Function-Based Coordination**
   Using LLM function calls to enable agent-to-agent communication.

Code Walkthrough
----------------

**1. Specialized Agent Definitions**

The example starts by creating domain-specific agents:

.. code-block:: python

   class DataAnalysisAgent(BaseAgent):
       """Specialized agent for data analysis tasks."""
       
       def __init__(self, llm_client: ILLM, **kwargs):
           system_prompt = """You are a data analysis expert.
           Analyze data patterns, provide insights, and make recommendations."""
           super().__init__(llm_client, system_prompt, **kwargs)
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Analyze this data: {input.message}"
           )
           
           result = await self.llm_client.chat(llm_input)
           
           return {
               "analysis_type": "data_pattern_analysis",
               "insights": result.get('llm_response', ''),
               "confidence": 0.85,
               "recommendations": ["Consider trend analysis", "Look for outliers"],
               "data_quality": "good"
           }

   class ReportGenerationAgent(BaseAgent):
       """Specialized agent for generating reports."""
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Process report generation
           return {
               "report_type": "analytical_report",
               "content": result.get('llm_response', ''),
               "sections": ["Executive Summary", "Analysis", "Recommendations"],
               "word_count": len(result.get('llm_response', '').split()),
               "status": "completed"
           }

**Key Points:**
- Each agent has a focused domain of expertise
- Structured return formats enable composition
- Clear interfaces facilitate coordination

**2. Orchestrator Pattern**

Master agent that coordinates multiple specialized agents:

.. code-block:: python

   class OrchestratorAgent(BaseAgent):
       """Master agent that orchestrates multiple specialized agents."""
       
       def __init__(self, llm_client: ILLM, specialized_agents: Dict[str, IAgent], **kwargs):
           system_prompt = """You are an intelligent orchestrator.
           
           Available agents:
           - data_analyst: Analyzes data patterns and provides insights
           - report_generator: Creates structured reports
           - knowledge_search: Searches knowledge base for information
           - memory_manager: Manages conversation context
           
           Decide which agents to use and coordinate their work."""
           
           super().__init__(llm_client, system_prompt, **kwargs)
           self.agents = specialized_agents
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Define agent functions for LLM to call
           async def analyze_data(data_description: str) -> str:
               """Use the data analysis agent to analyze data."""
               result = await self.agents['data_analyst'].process(
                   IAgentInput(message=data_description)
               )
               return json.dumps(result, indent=2)
           
           async def generate_report(content: str) -> str:
               """Use the report generator agent to create reports."""
               result = await self.agents['report_generator'].process(
                   IAgentInput(message=content)
               )
               return result['content']
           
           async def search_knowledge(query: str) -> str:
               """Use the knowledge base agent to find information."""
               result = await self.agents['knowledge_search'].process(
                   IAgentInput(message=query)
               )
               return str(result)
           
           # Background task for memory management
           async def update_memory(interaction: str) -> None:
               """Update conversation memory in background."""
               if 'memory_manager' in self.agents:
                   await self.agents['memory_manager'].process(IAgentInput(
                       message=interaction,
                       metadata=input.metadata
                   ))
           
           # Create LLM input with agent functions
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions={
                   "analyze_data": analyze_data,
                   "generate_report": generate_report,
                   "search_knowledge": search_knowledge
               },
               background_tasks={
                   "update_memory": update_memory
               }
           )
           
           result = await self.llm_client.chat(llm_input)
           return {
               "orchestrator_response": result.get('llm_response', ''),
               "agents_available": list(self.agents.keys()),
               "coordination_metadata": {
                   "complexity": "multi_agent_orchestration"
               }
           }

**Key Points:**
- LLM decides which agents to use based on the request
- Agents wrapped as functions for LLM tool calling
- Background tasks enable memory coordination
- Orchestrator manages complexity and coordination

**3. Pipeline Pattern**

Sequential processing through multiple agents:

.. code-block:: python

   class PipelineAgent(BaseAgent):
       """Agent that implements a processing pipeline pattern."""
       
       def __init__(self, llm_client: ILLM, pipeline_agents: List[IAgent], **kwargs):
           system_prompt = "You coordinate a processing pipeline of specialized agents."
           super().__init__(llm_client, system_prompt, **kwargs)
           self.pipeline = pipeline_agents
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           """Process input through a pipeline of agents."""
           
           pipeline_results = []
           current_input = input.message
           
           # Process through each agent in the pipeline
           for i, agent in enumerate(self.pipeline):
               print(f"Stage {i+1}: Processing with {agent.__class__.__name__}")
               
               # Create input for this stage
               stage_input = IAgentInput(
                   message=current_input,
                   metadata=input.metadata
               )
               
               # Process with current agent
               stage_result = await agent.process(stage_input)
               
               # Store result and prepare for next stage
               stage_info = {
                   "stage": i + 1,
                   "agent": agent.__class__.__name__,
                   "input": current_input[:100] + "...",
                   "output": str(stage_result)[:200] + "..."
               }
               pipeline_results.append(stage_info)
               
               # Extract content for next stage
               if isinstance(stage_result, dict):
                   if 'content' in stage_result:
                       current_input = stage_result['content']
                   elif 'insights' in stage_result:
                       current_input = stage_result['insights']
                   else:
                       current_input = json.dumps(stage_result)
               else:
                   current_input = str(stage_result)
           
           return {
               "final_result": current_input,
               "pipeline_stages": len(self.pipeline),
               "stage_results": pipeline_results,
               "processing_complete": True
           }

**Key Points:**
- Sequential processing through predefined agent sequence
- Output of each stage becomes input to the next
- Complete audit trail of processing stages
- Handles different output formats automatically

**4. Mesh Pattern**

Interconnected agents that can communicate dynamically:

.. code-block:: python

   class MeshCoordinatorAgent(BaseAgent):
       """Agent that coordinates a mesh of interconnected agents."""
       
       def __init__(self, llm_client: ILLM, agent_mesh: Dict[str, IAgent], **kwargs):
           system_prompt = """You coordinate a mesh of interconnected agents.
           Agents can communicate with each other to solve complex problems collaboratively."""
           super().__init__(llm_client, system_prompt, **kwargs)
           self.mesh = agent_mesh
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           """Coordinate agents in a mesh pattern."""
           
           # Agents can call each other
           async def get_analysis(data: str) -> str:
               """Get analysis from data analyst."""
               result = await self.mesh['analyst'].process(IAgentInput(message=data))
               return json.dumps(result)
           
           async def search_info(query: str) -> str:
               """Search knowledge base."""
               result = await self.mesh['searcher'].process(IAgentInput(message=query))
               return str(result)
           
           async def cross_reference(analysis: str, knowledge: str) -> str:
               """Cross-reference analysis with knowledge base."""
               combined = f"Analysis: {analysis}\nKnowledge: {knowledge}"
               result = await self.mesh['reporter'].process(IAgentInput(message=combined))
               return result['content']
           
           # All agents available to each other
           mesh_functions = {
               "get_analysis": get_analysis,
               "search_info": search_info,
               "cross_reference": cross_reference
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt + f"\nMesh functions: {list(mesh_functions.keys())}",
               user_message=input.message,
               regular_functions=mesh_functions
           )
           
           result = await self.llm_client.chat(llm_input)
           return {
               "mesh_response": result.get('llm_response', ''),
               "mesh_agents": list(self.mesh.keys()),
               "interconnected": True
           }

**Key Points:**
- Agents can dynamically call each other
- Cross-referencing and collaboration capabilities
- LLM decides communication patterns
- Flexible, adaptive coordination

**5. Factory Pattern**

Dynamic agent creation for specialized tasks:

.. code-block:: python

   class AgentFactory:
       """Factory for creating agents dynamically."""
       
       def __init__(self, llm_client: ILLM):
           self.llm_client = llm_client
           self.agent_cache = {}
       
       def create_specialist_agent(self, specialty: str) -> IAgent:
           """Create a specialist agent for a specific domain."""
           
           if specialty in self.agent_cache:
               return self.agent_cache[specialty]
           
           class SpecialistAgent(BaseAgent):
               def __init__(self, llm_client, specialty):
                   prompt = f"You are a {specialty} specialist. Provide expert advice and analysis."
                   super().__init__(llm_client, prompt)
                   self.specialty = specialty
               
               async def process(self, input: IAgentInput) -> str:
                   llm_input = ILLMInput(
                       system_prompt=self.system_prompt,
                       user_message=f"As a {self.specialty} expert, help with: {input.message}"
                   )
                   result = await self.llm_client.chat(llm_input)
                   return result.get('llm_response', '')
           
           agent = SpecialistAgent(self.llm_client, specialty)
           self.agent_cache[specialty] = agent
           return agent

**Key Points:**
- On-demand agent creation for specialized domains
- Caching for performance optimization
- Flexible specialization based on requirements
- Scalable architecture for growing needs

Running the Example
--------------------

**Prerequisites:**

.. code-block:: bash

   export OPENROUTER_API_KEY=your_key_here

**Run the example:**

.. code-block:: bash

   cd examples/agents
   python 05_agent_composition.py

**Expected Output:**

The example demonstrates four composition patterns:

1. **Orchestrator Pattern** - Complex task coordination
2. **Pipeline Pattern** - Sequential processing workflows  
3. **Mesh Pattern** - Dynamic agent collaboration
4. **Factory Pattern** - Specialized agent creation

Composition Patterns Deep Dive
-------------------------------

**When to Use Each Pattern**

**Orchestrator Pattern:**
- Complex tasks requiring multiple specialized capabilities
- When you need intelligent task decomposition
- Scenarios where the LLM should decide agent coordination
- Master-slave coordination requirements

**Pipeline Pattern:**
- Sequential processing workflows
- Data transformation pipelines
- Document processing chains
- When each stage builds on the previous

**Mesh Pattern:**
- Dynamic collaboration requirements
- Cross-referencing and validation needs
- When agents need to communicate peer-to-peer
- Complex problem-solving requiring multiple perspectives

**Factory Pattern:**
- Dynamic specialization requirements
- When agent types aren't known at startup
- Microservice-style architectures
- Performance optimization through caching

**Combining Patterns**

Patterns can be combined for sophisticated architectures:

.. code-block:: python

   class HybridSystem:
       """System combining multiple composition patterns."""
       
       def __init__(self, llm_client):
           self.llm_client = llm_client
           self.factory = AgentFactory(llm_client)
           
           # Core agents
           self.core_agents = {
               'orchestrator': None,  # Created on demand
               'pipeline': None,      # Created on demand
               'mesh': None          # Created on demand
           }
       
       async def process_complex_request(self, request: str, pattern: str = "auto"):
           """Process request using appropriate composition pattern."""
           
           if pattern == "auto":
               pattern = self.determine_pattern(request)
           
           if pattern == "orchestrator":
               return await self.use_orchestrator_pattern(request)
           elif pattern == "pipeline":
               return await self.use_pipeline_pattern(request)
           elif pattern == "mesh":
               return await self.use_mesh_pattern(request)
           else:
               return await self.use_hybrid_pattern(request)
       
       def determine_pattern(self, request: str) -> str:
           """Determine best pattern based on request analysis."""
           # Analyze request characteristics
           if "step by step" in request.lower():
               return "pipeline"
           elif "cross-reference" in request.lower():
               return "mesh"
           elif "complex" in request.lower():
               return "orchestrator"
           else:
               return "hybrid"

Real-World Implementation Examples
----------------------------------

**Document Processing System:**

.. code-block:: python

   class DocumentProcessingSystem:
       """Real-world document processing using composition patterns."""
       
       def __init__(self, llm_client):
           self.llm_client = llm_client
           
           # Specialized document agents
           self.agents = {
               'extractor': DocumentExtractionAgent(llm_client),
               'classifier': DocumentClassificationAgent(llm_client),
               'analyzer': ContentAnalysisAgent(llm_client),
               'summarizer': SummarizationAgent(llm_client),
               'validator': ValidationAgent(llm_client)
           }
           
           # Create orchestrator for coordination
           self.orchestrator = DocumentOrchestratorAgent(llm_client, self.agents)
       
       async def process_document(self, document_path: str, processing_type: str = "full"):
           """Process document using appropriate agent composition."""
           
           if processing_type == "full":
               # Use orchestrator for complex processing
               return await self.orchestrator.process(IAgentInput(
                   message=f"Process document: {document_path}",
                   metadata={"processing_type": "comprehensive"}
               ))
           
           elif processing_type == "pipeline":
               # Use pipeline for standard workflow
               pipeline = PipelineAgent(self.llm_client, [
                   self.agents['extractor'],
                   self.agents['classifier'],
                   self.agents['analyzer'],
                   self.agents['summarizer']
               ])
               return await pipeline.process(IAgentInput(message=document_path))
           
           elif processing_type == "validation":
               # Use mesh for cross-validation
               mesh_agents = {
                   'analyzer': self.agents['analyzer'],
                   'validator': self.agents['validator'],
                   'classifier': self.agents['classifier']
               }
               mesh = MeshCoordinatorAgent(self.llm_client, mesh_agents)
               return await mesh.process(IAgentInput(
                   message=f"Validate document analysis: {document_path}"
               ))

**Customer Service System:**

.. code-block:: python

   class CustomerServiceSystem:
       """Customer service system using agent composition."""
       
       def __init__(self, llm_client, crm_client, knowledge_base):
           self.llm_client = llm_client
           
           # Create specialized customer service agents
           self.agents = {
               'triage': TriageAgent(llm_client),
               'technical': TechnicalSupportAgent(llm_client),
               'billing': BillingAgent(llm_client, crm_client),
               'escalation': EscalationAgent(llm_client),
               'knowledge': KnowledgeSearchAgent(llm_client, knowledge_base)
           }
           
           # Create orchestrator for request routing
           self.orchestrator = CustomerServiceOrchestrator(llm_client, self.agents)
       
       async def handle_customer_request(self, request: str, customer_id: str):
           """Handle customer request using agent composition."""
           
           input_data = IAgentInput(
               message=request,
               metadata={
                   "customer_id": customer_id,
                   "timestamp": datetime.now().isoformat(),
                   "channel": "agent_system"
               }
           )
           
           # Use orchestrator to coordinate response
           result = await self.orchestrator.process(input_data)
           
           # If escalation needed, use mesh pattern for collaboration
           if result.get("requires_escalation"):
               escalation_mesh = {
                   'technical': self.agents['technical'],
                   'escalation': self.agents['escalation'],
                   'knowledge': self.agents['knowledge']
               }
               
               mesh_coordinator = MeshCoordinatorAgent(self.llm_client, escalation_mesh)
               escalation_result = await mesh_coordinator.process(input_data)
               
               return {
                   "primary_response": result,
                   "escalation_response": escalation_result,
                   "coordination_pattern": "orchestrator_plus_mesh"
               }
           
           return {
               "response": result,
               "coordination_pattern": "orchestrator"
           }

Testing Composition Systems
---------------------------

**Unit Testing Individual Patterns:**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock

   @pytest.mark.asyncio
   async def test_orchestrator_pattern():
       """Test orchestrator coordination."""
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Analysis completed using data agent",
           "usage": {"total_tokens": 100}
       }
       
       # Mock specialized agents
       mock_data_agent = AsyncMock()
       mock_data_agent.process.return_value = {"insights": "test insights"}
       
       mock_report_agent = AsyncMock()
       mock_report_agent.process.return_value = {"content": "test report"}
       
       agents = {
           'data_analyst': mock_data_agent,
           'report_generator': mock_report_agent
       }
       
       orchestrator = OrchestratorAgent(mock_llm, agents)
       result = await orchestrator.process(IAgentInput(
           message="Analyze data and create report"
       ))
       
       # Verify orchestrator response
       assert "orchestrator_response" in result
       assert result["agents_available"] == ['data_analyst', 'report_generator']
       
       # Verify LLM had access to agent functions
       call_args = mock_llm.chat.call_args[0][0]
       assert "regular_functions" in call_args.__dict__
       assert "analyze_data" in call_args.regular_functions
       assert "generate_report" in call_args.regular_functions

   @pytest.mark.asyncio
   async def test_pipeline_pattern():
       """Test pipeline sequential processing."""
       # Create mock agents
       mock_agent1 = AsyncMock()
       mock_agent1.process.return_value = {"content": "stage1 output"}
       mock_agent1.__class__.__name__ = "MockAgent1"
       
       mock_agent2 = AsyncMock()
       mock_agent2.process.return_value = "final output"
       mock_agent2.__class__.__name__ = "MockAgent2"
       
       mock_llm = AsyncMock()
       pipeline = PipelineAgent(mock_llm, [mock_agent1, mock_agent2])
       
       result = await pipeline.process(IAgentInput(message="test input"))
       
       # Verify pipeline execution
       assert result["processing_complete"] is True
       assert result["pipeline_stages"] == 2
       assert result["final_result"] == "final output"
       assert len(result["stage_results"]) == 2
       
       # Verify agent calls
       mock_agent1.process.assert_called_once()
       mock_agent2.process.assert_called_once()

**Integration Testing:**

.. code-block:: python

   @pytest.mark.integration
   @pytest.mark.asyncio
   async def test_real_composition_system():
       """Test composition with real LLM client."""
       config = ILLMConfig(model="gpt-4o-mini", temperature=0.1)
       llm_client = OpenAIClient(config)
       
       # Create real agents
       agents = {
           'analyst': DataAnalysisAgent(llm_client),
           'reporter': ReportGenerationAgent(llm_client)
       }
       
       orchestrator = OrchestratorAgent(llm_client, agents)
       
       result = await orchestrator.process(IAgentInput(
           message="Analyze sales data: Q1=$100K, Q2=$120K, Q3=$135K and create a report"
       ))
       
       # Verify realistic output
       assert "orchestrator_response" in result
       assert len(result["orchestrator_response"]) > 0
       assert "agents_available" in result

Performance Considerations
--------------------------

**Optimizing Composition Systems:**

1. **Caching Strategy:**
   ```python
   class CachedCompositionSystem:
       def __init__(self):
           self.agent_cache = {}
           self.result_cache = {}
       
       async def get_cached_result(self, request_hash: str):
           return self.result_cache.get(request_hash)
       
       async def cache_result(self, request_hash: str, result: Any):
           self.result_cache[request_hash] = result
   ```

2. **Parallel Execution:**
   ```python
   async def parallel_agent_execution(self, tasks: List[Tuple[IAgent, IAgentInput]]):
       """Execute multiple agents in parallel when possible."""
       agent_tasks = [
           agent.process(input_data) 
           for agent, input_data in tasks
       ]
       return await asyncio.gather(*agent_tasks)
   ```

3. **Resource Management:**
   ```python
   class ResourceManagedComposition:
       def __init__(self, max_concurrent_agents: int = 5):
           self.semaphore = asyncio.Semaphore(max_concurrent_agents)
       
       async def execute_with_limits(self, agent: IAgent, input_data: IAgentInput):
           async with self.semaphore:
               return await agent.process(input_data)
   ```

Key Takeaways
-------------

**Composition Enables Complexity**
   Complex problems solved through coordinated simple agents.

**LLM Function Calling as Coordination**
   Natural coordination mechanism using function calls between agents.

**Pattern Selection Matters**
   Choose composition patterns based on problem characteristics and requirements.

**Flexibility Through Interfaces**
   IAgent interface enables any agent to participate in composition patterns.

**Scalable Architecture**
   Composition patterns scale from simple coordination to complex multi-agent systems.

**Testing Is Critical**
   Complex systems require comprehensive testing at multiple levels.

Next Steps
----------

After mastering agent composition:

1. **Implement Testing**: :doc:`06-testing-agents` - Test complex multi-agent systems
2. **Build Production Systems**: Apply composition patterns to real-world problems
3. **Optimize Performance**: Implement caching and parallel execution strategies
4. **Monitor Systems**: Add observability to composition systems

**Related Documentation:**
- :doc:`../agent-patterns` - Additional patterns for agent design
- :doc:`../creating-agents` - Building agents for composition
- :doc:`../tools-and-callables` - Understanding the coordination mechanism