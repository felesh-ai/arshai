Orchestration Reference Implementations
======================================

This section documents the reference orchestration implementations provided with Arshai. These demonstrate how to coordinate multiple agents and components into complete agentic systems using the framework's building blocks.

.. toctree::
   :maxdepth: 2
   :caption: Orchestration Implementations
   
   workflow-system
   building-your-own

.. note::
   **Reference Implementation Philosophy**
   
   These orchestration implementations are **not part of the core framework**. They represent working examples of how to build coordination systems using the framework's capabilities. You can:
   
   - Use them as complete solutions if they fit your needs
   - Adapt them for your specific coordination requirements
   - Learn coordination patterns to build your own systems
   - Combine multiple approaches for complex systems

Available Reference Implementations
-----------------------------------

**Workflow System** (:doc:`workflow-system`)
   A complete workflow-based orchestration system with state management, node-based execution, and error handling. Demonstrates stateful multi-step processes.

**Building Your Own** (:doc:`building-your-own`)
   Patterns and examples for creating custom orchestration systems using the framework's building blocks.

Core Orchestration Capabilities
--------------------------------

**State Management**
   How orchestration systems maintain and coordinate state across multiple processing steps and agents.

**Node-Based Processing**
   Breaking complex workflows into discrete, testable nodes that can be composed into larger systems.

**Error Handling and Recovery**
   Robust error handling patterns that allow systems to recover from partial failures.

**Dynamic Routing**
   How orchestration systems can adapt their execution paths based on runtime conditions and data.

**Agent Coordination**
   Patterns for coordinating multiple agents within orchestrated workflows.

Key Orchestration Patterns
---------------------------

**Workflow Pattern**
   Sequential or conditional execution through predefined nodes with state management.

**Event-Driven Pattern**
   Orchestration based on events and triggers rather than fixed sequences.

**Pipeline Pattern**
   Simple linear processing through multiple stages with data transformation.

**Decision Tree Pattern**
   Complex branching logic based on data analysis and conditions.

**Parallel Processing Pattern**
   Coordinating multiple concurrent operations with result aggregation.

When to Use Orchestration
--------------------------

**Multi-Step Processes**
   When you need to coordinate multiple agents or steps in a specific sequence.

**State Management**
   When you need to maintain context and state across multiple interactions.

**Error Recovery**
   When partial failures should not stop the entire process.

**Complex Decision Logic**
   When routing and decision logic is too complex for simple agent composition.

**Monitoring and Observability**
   When you need detailed tracking of process execution and state changes.

Framework Integration
---------------------

**Agent Integration**
   How orchestration systems use agents as workflow nodes and processing components.

**Memory Integration**
   How orchestrated workflows maintain state using the framework's memory management capabilities.

**Tool Integration**
   How workflow nodes can use tools and external services through the framework's tool system.

**LLM Integration**
   How orchestration systems can leverage LLM capabilities for dynamic decision making.

**Background Tasks**
   How orchestration systems coordinate background processing and side effects.

Basic Orchestration Example
----------------------------

Here's a simple orchestration system using the framework:

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   
   class SimpleOrchestrator:
       """Basic orchestration pattern using direct agent coordination."""
       
       def __init__(self, agents: Dict[str, BaseAgent]):
           self.agents = agents
       
       async def process_request(self, request: str, context: dict) -> dict:
           results = {}
           
           # Step 1: Analyze request
           analysis_result = await self.agents["analyzer"].process(
               IAgentInput(message=request, metadata=context)
           )
           results["analysis"] = analysis_result
           
           # Step 2: Route based on analysis
           if "complex" in str(analysis_result).lower():
               # Complex path: use specialist
               specialist_result = await self.agents["specialist"].process(
                   IAgentInput(
                       message=f"Handle complex request: {request}",
                       metadata={**context, "analysis": analysis_result}
                   )
               )
               results["response"] = specialist_result
           else:
               # Simple path: use general agent
               general_result = await self.agents["general"].process(
                   IAgentInput(
                       message=request,
                       metadata={**context, "analysis": analysis_result}
                   )
               )
               results["response"] = general_result
           
           # Step 3: Post-processing
           final_result = await self.agents["formatter"].process(
               IAgentInput(
                   message=str(results["response"]),
                   metadata={**context, "processing_results": results}
               )
           )
           results["final_response"] = final_result
           
           return results

**Function-Based Orchestration**

.. code-block:: python

   def create_analysis_tool(analyzer_agent):
       """Convert agent to tool for orchestration."""
       async def analyze_request(request: str, priority: str = "normal") -> dict:
           result = await analyzer_agent.process(IAgentInput(
               message=request,
               metadata={"priority": priority}
           ))
           return {"analysis": str(result), "priority": priority}
       return analyze_request
   
   def create_routing_tool(specialist_agents):
       """Create routing tool for dynamic agent selection."""
       async def route_to_specialist(analysis: dict, request: str) -> str:
           # Determine which specialist to use
           specialist_type = determine_specialist_type(analysis)
           
           if specialist_type in specialist_agents:
               result = await specialist_agents[specialist_type].process(
                   IAgentInput(
                       message=request,
                       metadata={"analysis": analysis}
                   )
               )
               return str(result)
           else:
               return "No suitable specialist found"
       return route_to_specialist
   
   class LLMOrchestrator(BaseAgent):
       """LLM-driven orchestration using function calling."""
       
       def __init__(self, llm_client, analyzer_agent, specialist_agents):
           super().__init__(llm_client, "You coordinate analysis and specialist processing")
           
           # Create orchestration tools
           self.tools = {
               "analyze_request": create_analysis_tool(analyzer_agent),
               "route_to_specialist": create_routing_tool(specialist_agents)
           }
       
       async def process(self, input: IAgentInput) -> str:
           # LLM decides how to orchestrate the request
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=self.tools
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Async Parallel Orchestration**

.. code-block:: python

   import asyncio
   from typing import List, Dict, Any
   
   class ParallelOrchestrator:
       """Orchestrate multiple agents in parallel for faster processing."""
       
       def __init__(self, parallel_agents: Dict[str, BaseAgent]):
           self.parallel_agents = parallel_agents
       
       async def process_parallel(self, request: str, context: dict) -> dict:
           # Prepare tasks for parallel execution
           tasks = {}
           
           for agent_name, agent in self.parallel_agents.items():
               task = agent.process(IAgentInput(
                   message=request,
                   metadata={**context, "agent_role": agent_name}
               ))
               tasks[agent_name] = task
           
           # Execute all agents in parallel
           results = await asyncio.gather(
               *tasks.values(),
               return_exceptions=True
           )
           
           # Combine results
           combined_results = {}
           for (agent_name, task), result in zip(tasks.items(), results):
               if isinstance(result, Exception):
                   combined_results[agent_name] = f"Error: {str(result)}"
               else:
                   combined_results[agent_name] = str(result)
           
           return combined_results

Testing Orchestration Systems
------------------------------

**Unit Testing Orchestration Logic**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock
   
   @pytest.mark.asyncio
   async def test_simple_orchestrator():
       # Mock agents
       mock_agents = {
           "analyzer": AsyncMock(),
           "specialist": AsyncMock(),
           "general": AsyncMock(),
           "formatter": AsyncMock()
       }
       
       # Configure mock responses
       mock_agents["analyzer"].process.return_value = "complex request"
       mock_agents["specialist"].process.return_value = "specialist response"
       mock_agents["formatter"].process.return_value = "formatted response"
       
       # Test orchestrator
       orchestrator = SimpleOrchestrator(mock_agents)
       result = await orchestrator.process_request("test request", {"user_id": "123"})
       
       assert result["final_response"] == "formatted response"
       mock_agents["specialist"].process.assert_called_once()
       mock_agents["general"].process.assert_not_called()

**Integration Testing**

.. code-block:: python

   @pytest.mark.asyncio
   async def test_orchestration_integration():
       # Create real agents for integration testing
       llm_client = OpenAIClient(test_config)
       
       agents = {
           "analyzer": AnalysisAgent(llm_client, "Analyze requests"),
           "specialist": SpecialistAgent(llm_client, "Handle complex requests"),
           "general": GeneralAgent(llm_client, "Handle general requests"),
           "formatter": FormatterAgent(llm_client, "Format responses")
       }
       
       orchestrator = SimpleOrchestrator(agents)
       
       # Test with real request
       result = await orchestrator.process_request(
           "I need help with a complex technical integration",
           {"user_id": "test_user"}
       )
       
       assert "final_response" in result
       assert len(result["final_response"]) > 0

**Performance Testing**

.. code-block:: python

   import time
   import asyncio
   
   @pytest.mark.asyncio
   async def test_orchestration_performance():
       orchestrator = ParallelOrchestrator(test_agents)
       
       # Measure execution time
       start_time = time.time()
       
       # Run multiple requests
       tasks = []
       for i in range(10):
           task = orchestrator.process_parallel(
               f"Test request {i}",
               {"request_id": i}
           )
           tasks.append(task)
       
       results = await asyncio.gather(*tasks)
       execution_time = time.time() - start_time
       
       # Verify performance
       assert execution_time < 10.0  # Should complete within 10 seconds
       assert len(results) == 10
       assert all("Error" not in str(result) for result in results)

Best Practices for Orchestration
---------------------------------

**Design Principles**
   - Keep orchestration logic separate from business logic
   - Make orchestration steps independently testable
   - Design for failure recovery and graceful degradation
   - Use clear, descriptive names for orchestration steps

**State Management**
   - Minimize shared state between orchestration steps
   - Use immutable data structures where possible
   - Clearly define state transitions and ownership
   - Implement state validation and consistency checks

**Error Handling**
   - Plan for partial failures in multi-step processes
   - Implement appropriate retry logic for transient failures
   - Provide meaningful error context for debugging
   - Consider compensation patterns for rollback scenarios

**Performance**
   - Use parallel execution where steps are independent
   - Implement appropriate timeouts for orchestration steps
   - Monitor and optimize critical orchestration paths
   - Consider caching for expensive orchestration operations

**Monitoring**
   - Instrument orchestration steps for observability
   - Track orchestration performance and failure rates
   - Implement health checks for orchestration systems
   - Log sufficient context for debugging orchestration issues

Common Orchestration Challenges
--------------------------------

**State Consistency**
   Managing consistent state across multiple asynchronous operations.

**Error Recovery**
   Deciding how to handle partial failures in complex orchestration flows.

**Performance**
   Balancing orchestration flexibility with execution performance.

**Testing Complexity**
   Testing orchestration logic with multiple interacting components.

**Debugging**
   Tracing execution flow through complex orchestration systems.

The reference implementations in this section demonstrate proven patterns for building robust orchestration systems using Arshai's capabilities. Use them as starting points for your own orchestration needs.