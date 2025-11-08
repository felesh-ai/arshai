Workflow System
===============

The Arshai workflow system provides a reference implementation for orchestrating multiple agents and nodes into a coordinated multi-step process. This is **one way** to build agentic systems in Arshai - not **the way**.

.. important::

   **This is a Reference Implementation**

   The workflow system is provided as an example of how to orchestrate agents in Arshai. You are encouraged to:

   - Use it as-is if it fits your needs
   - Adapt it for your specific use case
   - Build your own custom orchestration from scratch

   The framework provides the building blocks - you create the solution that works for you.

Overview
--------

The workflow system consists of three main components:

1. **WorkflowRunner** - Manages workflow execution and state
2. **WorkflowOrchestrator** - Coordinates nodes and routing logic
3. **WorkflowConfig** - Defines workflow structure and dependencies

**Key Features:**

- Multi-step agent coordination
- State management across workflow steps
- Dynamic routing based on input
- Node-based execution graph
- Error handling and recovery
- Callback support for custom logic

Core Concepts
-------------

Workflow Architecture
~~~~~~~~~~~~~~~~~~~~~

The workflow system follows a node-based execution model:

.. code-block:: text

   [Input] → [WorkflowRunner] → [WorkflowOrchestrator]
                                       ↓
                         [Node 1] → [Node 2] → [Node 3]
                            ↓          ↓          ↓
                         [State] ← [State] ← [State]
                                       ↓
                                   [Output]

**Workflow State**

Carries information across the entire workflow:

- User context (ID, interaction count, etc.)
- Current step and processing path
- Agent-specific data
- Working memories
- Errors and notifications

**Nodes**

Individual processing units that:

- Receive workflow state as input
- Perform specific operations (agent processing, data transformation, etc.)
- Return updated state
- Determine next node via routing logic

Getting Started
---------------

Basic Workflow Example
~~~~~~~~~~~~~~~~~~~~~~

Here's a simple workflow with three nodes:

.. code-block:: python

   from arshai.workflows.workflow_config import WorkflowConfig
   from arshai.workflows.workflow_runner import BaseWorkflowRunner
   from arshai.workflows.node import Node
   from arshai.core.interfaces.iworkflow import IWorkflowState
   from typing import Dict, Any

   # Step 1: Create workflow nodes
   class GreetingNode(Node):
       """Node that greets the user."""

       async def run(self, state: IWorkflowState, input_data: Dict[str, Any]) -> IWorkflowState:
           # Add greeting to state
           state.agent_data["greeting"] = f"Hello, {state.user_context.user_id}!"
           return state

   class ProcessingNode(Node):
       """Node that processes user input."""

       def __init__(self, agent):
           self.agent = agent

       async def run(self, state: IWorkflowState, input_data: Dict[str, Any]) -> IWorkflowState:
           # Process message with agent
           message = input_data.get("message", "")
           response = await self.agent.process(message)
           state.agent_data["response"] = response
           return state

   class ResponseNode(Node):
       """Node that formats final response."""

       async def run(self, state: IWorkflowState, input_data: Dict[str, Any]) -> IWorkflowState:
           # Combine greeting and response
           greeting = state.agent_data.get("greeting", "")
           response = state.agent_data.get("response", "")
           state.agent_data["final_response"] = f"{greeting}\n{response}"
           return state

   # Step 2: Create workflow configuration
   class SimpleWorkflowConfig(WorkflowConfig):
       def __init__(self, agent):
           super().__init__()
           self.agent = agent

       def _configure_workflow(self, workflow):
           # Create nodes
           greeting = GreetingNode(name="greeting")
           processing = ProcessingNode(name="processing", agent=self.agent)
           response = ResponseNode(name="response")

           # Add nodes to workflow
           workflow.add_node("greeting", greeting)
           workflow.add_node("processing", processing)
           workflow.add_node("response", response)

           # Define edges (node connections)
           workflow.add_edge("greeting", "processing")
           workflow.add_edge("processing", "response")

           # Set entry point
           workflow.set_entry_point("greeting")

       def _route_input(self, input_data: Dict[str, Any]) -> str:
           # Route all input to greeting node
           return "greeting"

   # Step 3: Use the workflow
   async def main():
       from arshai.llms.openai import OpenAIClient
       from arshai.core.interfaces.illm import ILLMConfig
       from arshai.agents.base import BaseAgent
       from arshai.core.interfaces.iagent import IAgentInput
       from arshai.core.interfaces.illm import ILLMInput

       # Create agent
       llm_config = ILLMConfig(model="gpt-3.5-turbo")
       llm_client = OpenAIClient(llm_config)

       class SimpleAgent(BaseAgent):
           async def process(self, message: str) -> str:
               llm_input = ILLMInput(
                   system_prompt=self.system_prompt,
                   user_message=message
               )
               result = await self.llm_client.chat(llm_input)
               return result.get('llm_response', '')

       agent = SimpleAgent(llm_client, "You are a helpful assistant")

       # Create workflow
       config = SimpleWorkflowConfig(agent)
       runner = BaseWorkflowRunner(config)

       # Execute workflow
       result = await runner.execute_workflow(
           user_id="user_123",
           input_data={"message": "Tell me about AI"}
       )

       print(result['state'].agent_data['final_response'])

Advanced Patterns
-----------------

Conditional Routing
~~~~~~~~~~~~~~~~~~~

Route to different nodes based on input conditions:

.. code-block:: python

   class ConditionalWorkflowConfig(WorkflowConfig):
       def __init__(self, simple_agent, complex_agent):
           super().__init__()
           self.simple_agent = simple_agent
           self.complex_agent = complex_agent

       def _configure_workflow(self, workflow):
           # Create nodes for different complexity levels
           simple_node = SimpleProcessingNode(self.simple_agent)
           complex_node = ComplexProcessingNode(self.complex_agent)
           analysis_node = AnalysisNode()

           workflow.add_node("analyze", analysis_node)
           workflow.add_node("simple", simple_node)
           workflow.add_node("complex", complex_node)

           # Conditional edges based on analysis result
           workflow.add_conditional_edges(
               "analyze",
               lambda state: "simple" if state.agent_data.get("complexity") == "low" else "complex"
           )

           workflow.set_entry_point("analyze")

       def _route_input(self, input_data: Dict[str, Any]) -> str:
           return "analyze"

Agent Integration
~~~~~~~~~~~~~~~~~

Integrate multiple agents in a workflow:

.. code-block:: python

   class MultiAgentWorkflow(WorkflowConfig):
       def __init__(self, triage_agent, support_agent, escalation_agent):
           super().__init__()
           self.triage = triage_agent
           self.support = support_agent
           self.escalation = escalation_agent

       def _configure_workflow(self, workflow):
           # Create nodes wrapping agents
           triage_node = AgentNode("triage", self.triage)
           support_node = AgentNode("support", self.support)
           escalation_node = AgentNode("escalation", self.escalation)

           workflow.add_node("triage", triage_node)
           workflow.add_node("support", support_node)
           workflow.add_node("escalation", escalation_node)

           # Route based on triage decision
           workflow.add_conditional_edges(
               "triage",
               lambda state: state.agent_data.get("next_agent", "support")
           )

           workflow.add_edge("support", "END")
           workflow.add_edge("escalation", "END")

           workflow.set_entry_point("triage")

       def _route_input(self, input_data: Dict[str, Any]) -> str:
           return "triage"

Memory-Enabled Workflows
~~~~~~~~~~~~~~~~~~~~~~~~

Workflows with conversation memory:

.. code-block:: python

   from arshai.agents.working_memory import WorkingMemoryAgent
   from arshai.memory.working_memory.in_memory_manager import InMemoryManager

   class MemoryWorkflowConfig(WorkflowConfig):
       def __init__(self, llm_client, memory_manager):
           super().__init__()
           self.llm_client = llm_client
           self.memory_manager = memory_manager

       def _configure_workflow(self, workflow):
           # Create memory-enabled agent
           agent = WorkingMemoryAgent(
               llm_client=self.llm_client,
               memory_manager=self.memory_manager
           )

           # Create node with memory agent
           chat_node = AgentNode("chat", agent)
           workflow.add_node("chat", chat_node)
           workflow.set_entry_point("chat")

       def _route_input(self, input_data: Dict[str, Any]) -> str:
           return "chat"

   # Usage
   async def use_memory_workflow():
       llm_client = OpenAIClient(ILLMConfig(model="gpt-3.5-turbo"))
       memory_manager = InMemoryManager()

       config = MemoryWorkflowConfig(llm_client, memory_manager)
       runner = BaseWorkflowRunner(config)

       # First interaction
       result1 = await runner.execute_workflow(
           user_id="user_123",
           input_data={
               "message": "My name is Alice",
               "conversation_id": "conv_123"
           }
       )

       # Second interaction (agent remembers)
       result2 = await runner.execute_workflow(
           user_id="user_123",
           input_data={
               "message": "What's my name?",
               "conversation_id": "conv_123"
           }
       )

State Management
----------------

Workflow State Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from arshai.core.interfaces.iworkflow import IWorkflowState, IUserContext

   # Workflow state contains:
   state = IWorkflowState(
       user_context=IUserContext(
           user_id="user_123",
           last_active=datetime.utcnow(),
           interaction_count=0
       ),
       current_step="processing",
       step_count=3,
       processing_path="simple",
       agent_data={
           "response": "...",
           "metadata": {...}
       },
       working_memories={
           "conversation_id": "..."
       },
       errors=[]
   )

Accessing State in Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomNode(Node):
       async def run(self, state: IWorkflowState, input_data: Dict[str, Any]) -> IWorkflowState:
           # Access user context
           user_id = state.user_context.user_id

           # Read from agent_data
           previous_response = state.agent_data.get("previous_response", "")

           # Update agent_data
           state.agent_data["new_data"] = "some value"

           # Track errors
           if some_error:
               state.errors.append("Error description")

           # Increment step count
           state.step_count += 1

           return state

Error Handling
--------------

Handling Errors in Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ErrorHandlingNode(Node):
       async def run(self, state: IWorkflowState, input_data: Dict[str, Any]) -> IWorkflowState:
           try:
               # Attempt processing
               result = await self.process(input_data)
               state.agent_data["result"] = result
           except Exception as e:
               # Log error
               state.errors.append(f"Processing error: {str(e)}")

               # Set error state
               state.agent_data["has_error"] = True
               state.agent_data["error_message"] = str(e)

           return state

   # Configure error handling in workflow
   class RobustWorkflowConfig(WorkflowConfig):
       def _configure_workflow(self, workflow):
           processing_node = ErrorHandlingNode("processing")
           error_node = ErrorRecoveryNode("error_recovery")
           success_node = SuccessNode("success")

           workflow.add_node("processing", processing_node)
           workflow.add_node("error_recovery", error_node)
           workflow.add_node("success", success_node)

           # Conditional routing based on errors
           workflow.add_conditional_edges(
               "processing",
               lambda state: "error_recovery" if state.agent_data.get("has_error") else "success"
           )

           workflow.set_entry_point("processing")

Callbacks
---------

Using Callbacks for Custom Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def execute_with_callbacks():
       def on_node_start(node_name: str):
           print(f"Starting node: {node_name}")

       def on_node_complete(node_name: str, state: IWorkflowState):
           print(f"Completed node: {node_name}")
           print(f"Current step: {state.current_step}")

       callbacks = {
           "on_node_start": on_node_start,
           "on_node_complete": on_node_complete
       }

       result = await runner.execute_workflow(
           user_id="user_123",
           input_data={"message": "Hello"},
           callbacks=callbacks
       )

Best Practices
--------------

1. **Single Responsibility Nodes**
   Each node should have one clear purpose.

2. **Immutable State**
   Create new state objects rather than modifying existing ones when possible.

3. **Error Handling**
   Always handle errors in nodes to prevent workflow crashes.

4. **State Validation**
   Validate state before and after node execution.

5. **Testing**
   Test nodes independently before integrating into workflows.

6. **Documentation**
   Document node purpose, inputs, and outputs clearly.

Example: Complete Customer Service Workflow
-------------------------------------------

.. code-block:: python

   class CustomerServiceWorkflow(WorkflowConfig):
       def __init__(self, llm_client, memory_manager, knowledge_base):
           super().__init__()
           self.llm_client = llm_client
           self.memory_manager = memory_manager
           self.knowledge_base = knowledge_base

       def _configure_workflow(self, workflow):
           # Create specialized agents
           triage_agent = TriageAgent(self.llm_client)
           support_agent = SupportAgent(self.llm_client, self.knowledge_base)
           escalation_agent = EscalationAgent(self.llm_client)

           # Create nodes
           triage_node = AgentNode("triage", triage_agent)
           support_node = AgentNode("support", support_agent)
           escalation_node = AgentNode("escalation", escalation_agent)
           memory_node = MemoryNode("memory", self.memory_manager)

           # Add to workflow
           workflow.add_node("triage", triage_node)
           workflow.add_node("support", support_node)
           workflow.add_node("escalation", escalation_node)
           workflow.add_node("memory", memory_node)

           # Define routing
           workflow.add_conditional_edges(
               "triage",
               lambda state: state.agent_data.get("route", "support")
           )

           workflow.add_edge("support", "memory")
           workflow.add_edge("escalation", "memory")
           workflow.add_edge("memory", "END")

           workflow.set_entry_point("triage")

       def _route_input(self, input_data: Dict[str, Any]) -> str:
           return "triage"

When to Use Workflows
---------------------

**Use the workflow system when:**

- You have multi-step processes with clear stages
- You need state management across steps
- You want declarative orchestration
- You have conditional routing requirements

**Consider alternatives when:**

- You have simple sequential processing (use pipelines)
- You need dynamic, LLM-driven orchestration (build custom)
- You have very complex routing (build custom)
- The workflow structure is too rigid for your needs

Next Steps
----------

- **Build Your Own**: See :doc:`building-your-own` for custom orchestration patterns
- **Examples**: Check the `examples/ directory <https://github.com/felesh-ai/arshai/tree/main/examples>`_ for complete workflow examples
- **Framework Patterns**: Review :doc:`../../framework/building-systems/index` for system building concepts

Remember: This is **one way** to orchestrate agents in Arshai. The framework empowers you to build the solution that fits your specific needs.
