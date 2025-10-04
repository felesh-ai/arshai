Example 4: Tool Integration
===========================

This example demonstrates comprehensive tool integration patterns with agents, covering both regular functions and background tasks for external capabilities.

**File**: ``examples/agents/04_tool_integration.py`` (529 lines)

**Focus**: External function integration and background tasks

**Best For**: Understanding tool patterns and system coordination

Overview
--------

This example showcases:

- **Regular Functions**: Tools that return results to the conversation
- **Background Tasks**: Fire-and-forget operations for system coordination
- **Multiple Tool Categories**: Organizing tools by domain (math, file ops, network)
- **Dynamic Tool Selection**: Adapting available tools based on request analysis
- **Complex Agent Architectures**: Multi-tool agents for sophisticated operations

The example demonstrates how Arshai's vision of tools as natural extensions enables agents to become proper components in agentic systems.

Key Concepts Demonstrated
-------------------------

**Regular Functions vs Background Tasks**
   Understanding when tools should return results versus when they should run independently.

**Tool Organization**
   Grouping related tools by domain and functionality.

**Dynamic Tool Selection**
   Adapting available tools based on input analysis and context.

**System Coordination**
   Using background tasks for monitoring, logging, and metrics collection.

**Tool Composition**
   Building complex capabilities from simple, composable functions.

Code Walkthrough
----------------

**1. Calculator Agent - Mathematical Tools**

Demonstrates basic tool integration with mathematical functions:

.. code-block:: python

   class CalculatorAgent(BaseAgent):
       """Agent with mathematical calculation capabilities."""
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Define calculation tools
           def add(a: float, b: float) -> float:
               """Add two numbers."""
               return a + b
           
           def divide(a: float, b: float) -> float:
               """Divide two numbers."""
               if b == 0:
                   return float('inf')  # Handle division by zero
               return a / b
           
           def factorial(n: int) -> int:
               """Calculate factorial of n."""
               if n < 0:
                   return -1  # Invalid input
               return math.factorial(min(n, 170))  # Prevent overflow
           
           # Prepare tools for LLM
           tools = {
               "add": add,
               "divide": divide,
               "factorial": factorial
               # ... more tools
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=tools  # Tools available to LLM
           )
           
           result = await self.llm_client.chat(llm_input)
           return {
               "response": result.get('llm_response', ''),
               "tools_available": list(tools.keys()),
               "usage": result.get('usage', {})
           }

**Key Points:**
- Tools are simple Python functions with clear docstrings
- Error handling built into tool implementations
- Tools return results that become part of the conversation
- Metadata includes available tools for debugging

**2. Research Agent - Tools + Background Tasks**

Shows the combination of regular functions and background tasks:

.. code-block:: python

   class ResearchAgent(BaseAgent):
       """Agent with research capabilities and background monitoring."""
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Regular tools (return results to conversation)
           def search_web(query: str) -> str:
               """Search the web for information."""
               mock_results = [
                   f"Article: '{query}' - Comprehensive overview",
                   f"Research paper: 'Analysis of {query}' - Academic Journal",
                   f"News: Recent developments in {query} - Tech News"
               ]
               return " | ".join(mock_results)
           
           def analyze_data(data_description: str) -> Dict[str, Any]:
               """Analyze provided data description."""
               return {
                   "data_type": data_description,
                   "sample_size": random.randint(100, 1000),
                   "key_findings": ["Correlation found", "Outliers detected"],
                   "confidence": random.randint(80, 95)
               }
           
           # Background tasks (fire-and-forget)
           async def log_research_activity(query: str, user_id: str = "unknown"):
               """Log research activity for analytics."""
               log_entry = {
                   "query": query,
                   "user_id": user_id,
                   "timestamp": "2024-current-time"
               }
               self.research_log.append(log_entry)
               print(f"📊 [BACKGROUND] Logged: {query}")
           
           async def generate_usage_report(activity: str):
               """Generate usage statistics."""
               print(f"📈 [BACKGROUND] Generating report for: {activity}")
           
           # Separate regular functions and background tasks
           regular_functions = {
               "search_web": search_web,
               "analyze_data": analyze_data
           }
           
           background_tasks = {
               "log_research_activity": lambda q: log_research_activity(q, user_id),
               "generate_usage_report": generate_usage_report
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=regular_functions,    # Results return to conversation
               background_tasks=background_tasks       # Fire-and-forget execution
           )

**Key Points:**
- Regular functions provide data that enhances the conversation
- Background tasks handle system coordination independently
- User context passed to background tasks for personalization
- Clear separation between conversation and coordination tools

**3. Multi-Tool Agent - Organized Tool Categories**

Demonstrates organizing tools by domain for complex operations:

.. code-block:: python

   class MultiToolAgent(BaseAgent):
       """Agent with multiple tool categories."""
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # File operations
           file_tools = {
               "read_file": lambda filename: f"Content of {filename}",
               "write_file": lambda filename, content: True,
               "list_files": lambda directory=".": ["file1.txt", "file2.py"]
           }
           
           # Network operations
           network_tools = {
               "fetch_url": lambda url: {"status": 200, "content": "data"},
               "send_notification": lambda recipient, message: True
           }
           
           # Data processing
           data_tools = {
               "process_json": lambda data: {"processed": True, "records": 50},
               "validate_data": lambda type, data: {"valid": True, "errors": []}
           }
           
           # System monitoring
           system_tools = {
               "check_system_status": lambda: {
                   "cpu_usage": "45%", "memory_usage": "60%", "status": "healthy"
               }
           }
           
           # Combine all tools
           all_tools = {**file_tools, **network_tools, **data_tools, **system_tools}
           
           # Enhanced system prompt with tool organization
           enhanced_prompt = f"""{self.system_prompt}
           
           Tool Categories Available:
           - File Operations: {list(file_tools.keys())}
           - Network Operations: {list(network_tools.keys())}
           - Data Processing: {list(data_tools.keys())}
           - System Monitoring: {list(system_tools.keys())}
           """
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message,
               regular_functions=all_tools
           )

**Key Points:**
- Tools organized by functional domain
- Enhanced system prompt explains available capabilities
- Modular organization makes tools easier to manage
- Supports complex multi-step operations

**4. Dynamic Tool Selection - Adaptive Patterns**

Shows how agents can adapt their tools based on request analysis:

.. code-block:: python

   class AdaptiveAgent(BaseAgent):
       """Agent that adapts tools based on request analysis."""
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           message = input.message.lower()
           tools = {}
           
           # Math tools for mathematical queries
           if any(word in message for word in ['calculate', 'math', 'number']):
               def calculate(expression: str) -> float:
                   """Safe calculation."""
                   try:
                       return eval(expression.replace('^', '**'))
                   except:
                       return 0
               tools['calculate'] = calculate
           
           # Text tools for text processing
           if any(word in message for word in ['text', 'word', 'analyze']):
               def count_words(text: str) -> int:
                   return len(text.split())
               
               def analyze_text(text: str) -> Dict[str, Any]:
                   return {
                       "characters": len(text),
                       "words": len(text.split()),
                       "sentences": text.count('.') + text.count('!')
                   }
               
               tools.update({
                   'count_words': count_words,
                   'analyze_text': analyze_text
               })
           
           # Create LLM input with selected tools only
           llm_input = ILLMInput(
               system_prompt=f"{self.system_prompt}\nTools: {list(tools.keys())}",
               user_message=input.message,
               regular_functions=tools
           )

**Key Points:**
- Tools selected based on keyword analysis
- Only relevant tools loaded for each request
- Improves performance by reducing tool complexity
- Demonstrates intelligent tool orchestration

Running the Example
--------------------

**Prerequisites:**

.. code-block:: bash

   export OPENROUTER_API_KEY=your_key_here

**Run the example:**

.. code-block:: bash

   cd examples/agents
   python 04_tool_integration.py

**Expected Output:**

The example demonstrates four scenarios:

1. **Calculator Agent** - Mathematical tool usage
2. **Research Agent** - Tools + background task coordination
3. **Multi-Tool Agent** - Complex organized tool categories
4. **Dynamic Tool Selection** - Adaptive tool loading

Tool Integration Patterns
--------------------------

**Pattern 1: Domain-Specific Tools**

Organize tools by functional domain:

.. code-block:: python

   # Mathematical domain
   math_tools = {
       "add": lambda a, b: a + b,
       "multiply": lambda a, b: a * b,
       "sqrt": lambda x: math.sqrt(x)
   }
   
   # Text processing domain
   text_tools = {
       "count_words": lambda text: len(text.split()),
       "extract_keywords": lambda text: text.split()[:5],
       "sentiment": lambda text: "positive"  # Mock
   }
   
   # Network domain
   network_tools = {
       "fetch_data": lambda url: {"status": 200},
       "send_email": lambda to, subject, body: True,
       "check_connectivity": lambda host: True
   }

**Pattern 2: Regular Functions + Background Tasks**

Combine conversation tools with system coordination:

.. code-block:: python

   # Tools that enhance conversation
   regular_functions = {
       "search_knowledge": search_knowledge_base,
       "analyze_sentiment": analyze_text_sentiment,
       "get_weather": fetch_weather_data
   }
   
   # System coordination tasks
   background_tasks = {
       "log_interaction": log_user_interaction,
       "update_metrics": update_usage_metrics,
       "send_admin_alert": send_alert_if_needed
   }
   
   llm_input = ILLMInput(
       system_prompt=system_prompt,
       user_message=user_message,
       regular_functions=regular_functions,    # Conversation enhancement
       background_tasks=background_tasks       # System coordination
   )

**Pattern 3: Context-Aware Tool Selection**

Tools adapt based on user context and request type:

.. code-block:: python

   def select_tools_for_user(user_role: str, request_type: str) -> dict:
       """Select appropriate tools based on user context."""
       tools = {}
       
       # Base tools for all users
       tools.update(get_base_tools())
       
       # Role-specific tools
       if user_role == "admin":
           tools.update(get_admin_tools())
       elif user_role == "analyst":
           tools.update(get_analysis_tools())
       
       # Request-specific tools
       if request_type == "data_analysis":
           tools.update(get_data_tools())
       elif request_type == "system_maintenance":
           tools.update(get_system_tools())
       
       return tools
   
   # Usage in agent
   user_role = input.metadata.get("user_role", "user")
   request_type = classify_request(input.message)
   tools = select_tools_for_user(user_role, request_type)

**Pattern 4: Tool Composition**

Build complex capabilities from simple tools:

.. code-block:: python

   # Basic tools
   def read_file(filename: str) -> str:
       """Read file content."""
       # Implementation
   
   def parse_json(content: str) -> dict:
       """Parse JSON content."""
       # Implementation
   
   def validate_schema(data: dict, schema: dict) -> bool:
       """Validate data against schema."""
       # Implementation
   
   # Composed capability
   def process_config_file(filename: str, schema: dict) -> dict:
       """Process and validate configuration file."""
       content = read_file(filename)
       data = parse_json(content)
       is_valid = validate_schema(data, schema)
       return {"data": data, "valid": is_valid}
   
   # Agent can use either basic tools or composed tools
   tools = {
       # Basic tools for flexibility
       "read_file": read_file,
       "parse_json": parse_json,
       "validate_schema": validate_schema,
       
       # Composed tool for common workflows
       "process_config_file": process_config_file
   }

Real-World Implementation Examples
----------------------------------

**Customer Support Agent with CRM Integration:**

.. code-block:: python

   class CustomerSupportAgent(BaseAgent):
       def __init__(self, llm_client, crm_client, ticket_system):
           super().__init__(llm_client, "You are a customer support agent...")
           self.crm = crm_client
           self.tickets = ticket_system
       
       async def process(self, input: IAgentInput) -> str:
           customer_id = input.metadata.get("customer_id")
           
           # CRM tools
           def get_customer_info(customer_id: str) -> dict:
               return self.crm.get_customer(customer_id)
           
           def get_order_history(customer_id: str) -> list:
               return self.crm.get_orders(customer_id)
           
           def create_support_ticket(issue: str, priority: str = "medium") -> str:
               return self.tickets.create(customer_id, issue, priority)
           
           # Background tasks
           async def log_interaction(interaction_type: str, resolution: str = ""):
               await self.crm.log_interaction(customer_id, interaction_type, resolution)
           
           async def update_customer_satisfaction(rating: int):
               await self.crm.update_satisfaction(customer_id, rating)
           
           tools = {
               "get_customer_info": get_customer_info,
               "get_order_history": get_order_history,
               "create_support_ticket": create_support_ticket
           }
           
           background_tasks = {
               "log_interaction": log_interaction,
               "update_customer_satisfaction": update_customer_satisfaction
           }
           
           llm_input = ILLMInput(
               system_prompt=f"{self.system_prompt}\nCustomer ID: {customer_id}",
               user_message=input.message,
               regular_functions=tools,
               background_tasks=background_tasks
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Data Analysis Agent with Database Access:**

.. code-block:: python

   class DataAnalysisAgent(BaseAgent):
       def __init__(self, llm_client, database, visualization_service):
           super().__init__(llm_client, "You are a data analyst...")
           self.db = database
           self.viz = visualization_service
       
       async def process(self, input: IAgentInput) -> dict:
           # Database tools
           def query_data(sql: str) -> list:
               """Execute SQL query safely."""
               # Add SQL injection protection
               return self.db.execute_safe_query(sql)
           
           def get_table_schema(table_name: str) -> dict:
               """Get table structure."""
               return self.db.describe_table(table_name)
           
           def aggregate_data(table: str, groupby: str, metric: str) -> dict:
               """Perform data aggregation."""
               return self.db.aggregate(table, groupby, metric)
           
           # Visualization tools
           def create_chart(data: list, chart_type: str) -> str:
               """Create visualization."""
               chart_url = self.viz.create_chart(data, chart_type)
               return f"Chart created: {chart_url}"
           
           # Background tasks
           async def cache_results(query: str, results: list):
               """Cache query results for performance."""
               await self.db.cache_set(f"query:{hash(query)}", results)
           
           async def log_analysis(query_type: str, tables_used: list):
               """Log analysis for auditing."""
               await self.db.log_analysis(query_type, tables_used)
           
           tools = {
               "query_data": query_data,
               "get_table_schema": get_table_schema,
               "aggregate_data": aggregate_data,
               "create_chart": create_chart
           }
           
           background_tasks = {
               "cache_results": cache_results,
               "log_analysis": log_analysis
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=tools,
               background_tasks=background_tasks
           )
           
           result = await self.llm_client.chat(llm_input)
           return {
               "analysis": result["llm_response"],
               "usage": result["usage"]
           }

Testing Tool Integration
------------------------

**Unit Testing Tools:**

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock, MagicMock

   @pytest.mark.asyncio
   async def test_calculator_agent_tools():
       """Test calculator agent tool integration."""
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "The result is 42",
           "usage": {"total_tokens": 25}
       }
       
       agent = CalculatorAgent(mock_llm, "Math assistant")
       result = await agent.process(IAgentInput(message="What is 6 * 7?"))
       
       # Verify tools were provided
       call_args = mock_llm.chat.call_args[0][0]
       assert "regular_functions" in call_args.__dict__
       assert "add" in call_args.regular_functions
       assert "multiply" in call_args.regular_functions
       
       # Verify response structure
       assert "response" in result
       assert "tools_available" in result
       assert result["response"] == "The result is 42"

   @pytest.mark.asyncio
   async def test_background_tasks_execution():
       """Test background task integration."""
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Research completed",
           "usage": {"total_tokens": 30}
       }
       
       agent = ResearchAgent(mock_llm)
       result = await agent.process(IAgentInput(
           message="Research AI trends",
           metadata={"user_id": "test_user"}
       ))
       
       # Verify background tasks were provided
       call_args = mock_llm.chat.call_args[0][0]
       assert "background_tasks" in call_args.__dict__
       assert len(call_args.background_tasks) > 0

**Integration Testing:**

.. code-block:: python

   @pytest.mark.integration
   @pytest.mark.asyncio
   async def test_real_tool_execution():
       """Test tools execute correctly with real LLM."""
       config = ILLMConfig(model="gpt-4o-mini", temperature=0.1)
       llm_client = OpenAIClient(config)
       
       agent = CalculatorAgent(llm_client, "You are a math assistant")
       result = await agent.process(IAgentInput(message="Calculate 15 + 25"))
       
       # Verify response contains calculation
       assert "40" in result["response"] or "forty" in result["response"].lower()
       
       # Verify tool information
       assert "tools_available" in result
       assert len(result["tools_available"]) > 0

Best Practices for Tool Integration
-----------------------------------

**1. Clear Tool Boundaries**
   - Regular functions for data that enhances conversation
   - Background tasks for system coordination and side effects
   - Keep tools focused and single-purpose

**2. Error Handling in Tools**
   - Handle edge cases within tool implementations
   - Return sensible defaults for invalid inputs
   - Log errors for debugging without breaking conversation flow

**3. Tool Documentation**
   - Provide clear docstrings for all tools
   - Include parameter types and return value descriptions
   - Document expected behavior and edge cases

**4. Performance Considerations**
   - Keep tools lightweight and fast
   - Use async tools for I/O operations
   - Cache expensive computations when appropriate

**5. Security**
   - Validate all tool inputs
   - Implement proper authentication for external services
   - Avoid exposing sensitive operations as tools

**6. Testing Strategy**
   - Unit test tools independently
   - Mock external dependencies in agent tests
   - Include integration tests with real LLM calls

Key Takeaways
-------------

**Tool Philosophy**
   Tools are natural extensions that make agents proper components in agentic systems.

**Dual Tool Types**
   Regular functions enhance conversations; background tasks coordinate systems.

**Dynamic Selection**
   Agents can adapt their capabilities based on context and request analysis.

**Organization Matters**
   Group tools by domain for better management and clearer system prompts.

**Composition Enables Complexity**
   Build sophisticated capabilities from simple, composable tools.

**Background Tasks Enable Coordination**
   Fire-and-forget operations allow agents to participate in larger system workflows.

Next Steps
----------

After mastering tool integration:

1. **Explore Agent Composition**: :doc:`05-agent-composition` - Multi-agent systems
2. **Implement Testing**: :doc:`06-testing-agents` - Comprehensive testing strategies  
3. **Build Real Tools**: Create domain-specific tools for your use case
4. **System Integration**: Use tool-enabled agents in production workflows

**Related Documentation:**
- :doc:`../tools-and-callables` - Deep dive into tool philosophy
- :doc:`../agent-patterns` - Advanced patterns using tools
- :doc:`../creating-agents` - Step-by-step agent development