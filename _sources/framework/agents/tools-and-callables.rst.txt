Tools and Callables
===================

In Arshai, tools are Python callables that extend agents with external capabilities. This design reflects our vision of agents as intelligent components that can seamlessly integrate into complex agentic systems through natural function-based interfaces.

The Arshai Vision for Tools
----------------------------

**Tools as Natural Extensions**
   Tools should feel like natural extensions of an agent's capabilities, not foreign objects that require special handling.

**Function-Based Integration**
   Python functions provide the most natural interface for tool integration - familiar, type-safe, and immediately understandable.

**Dual-Purpose Design**
   Tools serve two distinct purposes in agentic systems: information gathering (regular functions) and system coordination (background tasks).

**Component Composition**
   Agents equipped with tools become composable components that can participate in larger agentic architectures.

Core Tool Philosophy
--------------------

**Functions as Capabilities**

In Arshai, every function represents a specific capability that an agent can possess:

.. code-block:: python

   def search_knowledge_base(query: str, limit: int = 5) -> List[Dict]:
       """Search internal knowledge base for relevant information."""
       # This function gives the agent search capability
       return search_results
   
   def validate_user_input(input_text: str, rules: List[str]) -> Dict[str, bool]:
       """Validate user input against business rules."""
       # This function gives the agent validation capability
       return validation_results

**Background Tasks as System Coordination**

Background tasks enable agents to participate in system-wide coordination without blocking the main conversation flow:

.. code-block:: python

   def update_user_context(user_id: str, interaction_data: Dict):
       """Update user context in system-wide storage."""
       # This enables the agent to contribute to system state
       pass
   
   def trigger_workflow(workflow_id: str, parameters: Dict):
       """Trigger a system workflow based on conversation."""
       # This allows the agent to orchestrate system actions
       pass

Types of Tools in Agentic Systems
----------------------------------

**Regular Functions: Information and Computation**

These tools provide agents with the ability to gather information, perform computations, and access external systems. Results flow back to the agent for processing and inclusion in responses.

**Knowledge Access Tools**:

.. code-block:: python

   def search_documents(query: str, document_type: str = "all") -> List[Dict]:
       """Search document repository for relevant information."""
       # Gives agent access to organizational knowledge
       return document_results
   
   def get_user_preferences(user_id: str) -> Dict[str, Any]:
       """Retrieve user preferences and settings."""
       # Enables personalized agent behavior
       return user_data

**Computation Tools**:

.. code-block:: python

   def calculate_metrics(data: List[float], metric_type: str) -> Dict[str, float]:
       """Calculate various metrics from data."""
       # Provides agent with analytical capabilities
       return computed_metrics
   
   def validate_business_rules(entity: Dict, rules: List[str]) -> Dict[str, bool]:
       """Validate entity against business rules."""
       # Enables agent to enforce business logic
       return validation_results

**External Integration Tools**:

.. code-block:: python

   def fetch_market_data(symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
       """Fetch real-time market data for analysis."""
       # Connects agent to external data sources
       return market_data
   
   def send_email_notification(recipient: str, subject: str, body: str) -> bool:
       """Send email notification to specified recipient."""
       # Enables agent to communicate externally
       return success_status

**Background Tasks: System Orchestration**

Background tasks enable agents to participate in system-wide orchestration, trigger workflows, and maintain system state without interrupting the user experience.

**State Management Tasks**:

.. code-block:: python

   def update_conversation_memory(conversation_id: str, key_points: List[str]):
       """Update conversation memory with key insights."""
       # Maintains system-wide conversation state
       pass
   
   def record_user_intent(user_id: str, intent: str, confidence: float):
       """Record detected user intent for system learning."""
       # Contributes to system intelligence
       pass

**Workflow Coordination Tasks**:

.. code-block:: python

   def initiate_approval_workflow(request_data: Dict, approver_role: str):
       """Start approval workflow for user request."""
       # Triggers system processes
       pass
   
   def schedule_followup_task(task_data: Dict, delay_hours: int):
       """Schedule follow-up task in system task queue."""
       # Coordinates future system actions
       pass

**Analytics and Monitoring Tasks**:

.. code-block:: python

   def log_agent_interaction(agent_type: str, user_query: str, outcome: str):
       """Log interaction for system analytics."""
       # Contributes to system observability
       pass
   
   def update_performance_metrics(agent_id: str, response_time: float, quality_score: float):
       """Update agent performance metrics."""
       # Maintains system performance awareness
       pass

Agent Tool Integration Patterns
-------------------------------

**Capability-Driven Agent Design**

Agents are designed around the capabilities they need to fulfill their role in the system:

.. code-block:: python

   class CustomerSupportAgent(BaseAgent):
       """Agent specialized in customer support with relevant capabilities."""
       
       async def process(self, input: IAgentInput) -> str:
           # Information gathering capabilities
           def lookup_customer_account(customer_id: str) -> Dict[str, Any]:
               """Access customer account information."""
               return account_data
           
           def search_support_articles(query: str) -> List[Dict]:
               """Search knowledge base for solutions."""
               return articles
           
           def check_system_status(service_name: str) -> Dict[str, str]:
               """Check current system status."""
               return status_info
           
           # System coordination capabilities
           def escalate_to_human(ticket_data: Dict, priority: str = "normal"):
               """Escalate issue to human support."""
               pass
           
           def log_support_interaction(interaction_summary: str, resolution: str):
               """Log support interaction for training."""
               pass
           
           # Configure agent with appropriate tools
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions={
                   "lookup_customer_account": lookup_customer_account,
                   "search_support_articles": search_support_articles,
                   "check_system_status": check_system_status
               },
               background_tasks={
                   "escalate_to_human": escalate_to_human,
                   "log_support_interaction": log_support_interaction
               }
           )
           
           result = await self.llm_client.chat(llm_input)
           return result.get("llm_response", "")

**Dynamic Tool Selection**

Agents can dynamically select tools based on context and requirements:

.. code-block:: python

   class AdaptiveAgent(BaseAgent):
       """Agent that adapts tools based on user context."""
       
       def __init__(self, llm_client, system_prompt, tool_registry):
           super().__init__(llm_client, system_prompt)
           self.tool_registry = tool_registry
       
       async def process(self, input: IAgentInput) -> str:
           # Analyze context to select appropriate tools
           user_role = input.metadata.get("user_role", "guest")
           task_complexity = self._assess_complexity(input.message)
           
           # Select tools based on context
           selected_tools = self._select_tools(user_role, task_complexity)
           selected_background_tasks = self._select_background_tasks(user_role)
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=selected_tools,
               background_tasks=selected_background_tasks
           )
           
           result = await self.llm_client.chat(llm_input)
           return result.get("llm_response", "")
       
       def _select_tools(self, user_role: str, complexity: str) -> Dict:
           """Select appropriate tools based on context."""
           if user_role == "admin" and complexity == "high":
               return {
                   "advanced_query": self.tool_registry.advanced_query,
                   "system_analysis": self.tool_registry.system_analysis,
                   "generate_report": self.tool_registry.generate_report
               }
           elif user_role == "user":
               return {
                   "basic_search": self.tool_registry.basic_search,
                   "get_help": self.tool_registry.get_help
               }
           return {}

Tools as System Interfaces
---------------------------

**Inter-Agent Communication Through Tools**

Tools enable agents to communicate and coordinate with each other through the system:

.. code-block:: python

   # Agent A can trigger Agent B through background tasks
   def request_analysis_from_specialist(data: Dict, analysis_type: str):
       """Request specialized analysis from analysis agent."""
       # This creates a task for the analysis agent
       pass
   
   # Agent B can provide results through regular functions  
   def get_analysis_results(request_id: str) -> Dict[str, Any]:
       """Retrieve analysis results from specialist agent."""
       # This allows retrieval of analysis results
       return analysis_data

**System State Coordination**

Background tasks enable agents to maintain shared system state:

.. code-block:: python

   def update_shared_context(context_key: str, context_value: Any):
       """Update shared context accessible by all agents."""
       # Maintains system-wide state
       pass
   
   def signal_system_event(event_type: str, event_data: Dict):
       """Signal system-wide event to interested agents."""
       # Coordinates system-wide notifications
       pass

**Resource Management**

Tools enable agents to coordinate resource usage:

.. code-block:: python

   def reserve_computational_resource(resource_type: str, duration_minutes: int) -> str:
       """Reserve computational resource for intensive operations."""
       # Returns reservation ID
       return reservation_id
   
   def release_computational_resource(reservation_id: str):
       """Release previously reserved computational resource."""
       # Background task for resource cleanup
       pass

Advanced Tool Patterns
-----------------------

**Stateful Tool Factories**

Create tools that maintain state across interactions:

.. code-block:: python

   def create_session_tools(session_id: str):
       """Create tools bound to a specific session."""
       session_data = {}
       
       def store_session_data(key: str, value: Any) -> bool:
           """Store data in session context."""
           session_data[key] = value
           return True
       
       def retrieve_session_data(key: str) -> Any:
           """Retrieve data from session context."""
           return session_data.get(key)
       
       def clear_session():
           """Clear session data."""
           session_data.clear()
       
       return {
           "store_session_data": store_session_data,
           "retrieve_session_data": retrieve_session_data
       }, {
           "clear_session": clear_session
       }

**Hierarchical Tool Organization**

Organize tools in hierarchies for complex agents:

.. code-block:: python

   class EnterpriseAgentTools:
       """Hierarchically organized tools for enterprise agents."""
       
       def __init__(self, user_permissions: List[str]):
           self.permissions = user_permissions
       
       def get_customer_tools(self) -> Dict:
           """Get customer-related tools based on permissions."""
           tools = {}
           
           if "customer_read" in self.permissions:
               tools["get_customer_info"] = self._get_customer_info
               tools["search_customers"] = self._search_customers
           
           if "customer_write" in self.permissions:
               tools["update_customer"] = self._update_customer
           
           return tools
       
       def get_background_tasks(self) -> Dict:
           """Get background tasks based on permissions."""
           tasks = {}
           
           if "audit_log" in self.permissions:
               tasks["log_customer_access"] = self._log_customer_access
           
           if "workflow_trigger" in self.permissions:
               tasks["trigger_customer_workflow"] = self._trigger_workflow
           
           return tasks

Benefits for Agentic Systems
-----------------------------

**Composable Intelligence**
   Agents with well-defined tools become composable building blocks for larger intelligent systems.

**Clear Separation of Concerns**
   Regular functions handle information flow, background tasks handle system coordination.

**Natural Scalability**
   Function-based tools scale naturally with system complexity and can be distributed across services.

**Emergent Capabilities**
   Complex behaviors emerge from simple tool combinations rather than monolithic implementations.

**System Evolution**
   New capabilities can be added by introducing new tools without changing agent core logic.

**Debugging and Observability**
   Function calls provide clear audit trails of agent behavior and system interactions.

Testing Tool-Enabled Agents
----------------------------

**Isolated Tool Testing**:

.. code-block:: python

   def test_search_knowledge_base():
       """Test individual tool functionality."""
       results = search_knowledge_base("user authentication", limit=3)
       assert len(results) <= 3
       assert all("title" in result for result in results)

**Agent Integration Testing**:

.. code-block:: python

   @pytest.mark.asyncio
   async def test_customer_support_agent():
       """Test agent with mocked tools."""
       # Mock external dependencies
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {"llm_response": "I found your account details."}
       
       # Test agent with tools
       agent = CustomerSupportAgent(mock_llm, "You provide customer support")
       response = await agent.process(IAgentInput(
           message="What's my account status?",
           metadata={"customer_id": "12345"}
       ))
       
       assert "account" in response.lower()

**System Integration Testing**:

.. code-block:: python

   @pytest.mark.asyncio
   async def test_multi_agent_coordination():
       """Test agents coordinating through background tasks."""
       # Test that Agent A triggers workflows that Agent B can respond to
       support_agent = CustomerSupportAgent(llm_client, "Customer support")
       specialist_agent = SpecialistAgent(llm_client, "Technical specialist")
       
       # Simulate escalation workflow
       result = await support_agent.process(IAgentInput(
           message="Complex technical issue requiring specialist help"
       ))
       
       # Verify background task was triggered
       assert workflow_was_triggered("technical_escalation")

This tool philosophy enables agents to become intelligent, capable components that can seamlessly participate in sophisticated agentic systems while maintaining clarity, testability, and composability.