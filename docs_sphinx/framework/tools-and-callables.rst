Tools and Callables Architecture
==================================

In Arshai's architecture, tools are Python callables (functions), not classes with interfaces. This design choice provides maximum flexibility and aligns with the framework's philosophy of developer authority.

Philosophy: Functions Over Interfaces
--------------------------------------

Traditional frameworks often require tools to implement specific interfaces or inherit from base classes. Arshai takes a different approach:

**Traditional Approach (Constrained)**:

.. code-block:: python

   # Traditional - forced interface
   class SearchTool(BaseTool):
       def __init__(self, config):
           super().__init__(config)
       
       def execute(self, input: ToolInput) -> ToolOutput:
           # Forced to use framework types
           pass

**Arshai Approach (Flexible)**:

.. code-block:: python

   # Arshai - any function works
   def search_web(query: str, max_results: int = 5) -> List[Dict]:
       """Search the web and return results."""
       # Use any types you want
       # Return any structure you need
       return results

Core Principles
---------------

**1. Any Callable is a Tool**

.. code-block:: python

   # Functions
   def calculate(expression: str) -> float:
       return eval(expression)
   
   # Methods
   class MathUtils:
       def add(self, a: float, b: float) -> float:
           return a + b
   
   # Lambdas
   multiply = lambda x, y: x * y
   
   # All can be used as tools
   tools = [calculate, MathUtils().add, multiply]

**2. Type Safety Through Function Signatures**

.. code-block:: python

   # Function signature provides all type information
   def search_documents(
       query: str,
       limit: int = 10,
       include_metadata: bool = True
   ) -> List[Dict[str, Any]]:
       """Search through documents."""
       # LLM understands the parameters and return type
       pass

**3. Flexible Return Types**

.. code-block:: python

   # Return strings
   def get_weather(city: str) -> str:
       return f"Weather in {city}: Sunny, 72°F"
   
   # Return structured data
   def get_stock_price(symbol: str) -> Dict[str, Any]:
       return {
           "symbol": symbol,
           "price": 150.25,
           "change": +2.5
       }
   
   # Return lists
   def list_files(directory: str) -> List[str]:
       return ["file1.txt", "file2.py", "file3.md"]

**4. Easy Integration with Existing Code**

.. code-block:: python

   # Use existing functions without modification
   import os
   import json
   
   # Standard library functions work as tools
   tools = [
       os.listdir,      # List directory contents
       json.loads,      # Parse JSON
       len,             # Get length
       str.upper        # Convert to uppercase
   ]

Using Tools with LLM Clients
-----------------------------

Tools integrate seamlessly with the LLM layer:

.. code-block:: python

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
   
   # Define your tools
   def search_web(query: str, max_results: int = 5) -> List[Dict]:
       """Search the web and return results."""
       # Implementation here
       return results
   
   def calculate(expression: str) -> float:
       """Evaluate a mathematical expression."""
       return eval(expression)
   
   def get_current_time() -> str:
       """Get the current time."""
       from datetime import datetime
       return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
   # Use with LLM
   llm_client = OpenAIClient(ILLMConfig(model="gpt-4"))
   
   llm_input = ILLMInput(
       system_prompt="You are a helpful assistant with access to tools",
       user_message="What time is it and what's 25 * 4?",
       tools_list=[search_web, calculate, get_current_time]
   )
   
   response = await llm_client.chat(llm_input)

Advanced Tool Patterns
-----------------------

**Stateful Tools Using Closures**:

.. code-block:: python

   def create_counter():
       count = 0
       
       def increment() -> int:
           nonlocal count
           count += 1
           return count
       
       return increment
   
   # Each agent gets its own counter
   counter_tool = create_counter()

**Configuration Through Partial Application**:

.. code-block:: python

   from functools import partial
   
   def api_call(endpoint: str, api_key: str, params: Dict) -> Dict:
       """Make API call with authentication."""
       # Implementation
       pass
   
   # Pre-configure API key
   weather_api = partial(api_call, api_key="your-key-here")
   
   # Now it's a simpler tool
   def get_weather(location: str) -> Dict:
       return weather_api("weather", {"location": location})

**Async Tools**:

.. code-block:: python

   import asyncio
   import aiohttp
   
   async def fetch_url(url: str) -> str:
       """Fetch content from URL asynchronously."""
       async with aiohttp.ClientSession() as session:
           async with session.get(url) as response:
               return await response.text()
   
   # Async tools work seamlessly
   tools = [fetch_url]

Tool Documentation and Discovery
--------------------------------

**Self-Documenting Tools**:

.. code-block:: python

   def search_database(
       table: str,
       filters: Dict[str, Any],
       limit: int = 100
   ) -> List[Dict[str, Any]]:
       """
       Search database table with filters.
       
       Args:
           table: Name of the database table
           filters: Dictionary of column:value filters
           limit: Maximum number of results to return
       
       Returns:
           List of matching records as dictionaries
           
       Examples:
           search_database("users", {"age": 25}, limit=10)
           search_database("products", {"category": "electronics"})
       """
       # Implementation
       pass

**Tool Introspection**:

.. code-block:: python

   import inspect
   
   def describe_tool(func):
       """Get tool information for LLM."""
       sig = inspect.signature(func)
       return {
           "name": func.__name__,
           "description": func.__doc__ or "No description",
           "parameters": {
               name: {
                   "type": param.annotation.__name__ if param.annotation != param.empty else "Any",
                   "default": param.default if param.default != param.empty else None
               }
               for name, param in sig.parameters.items()
           },
           "return_type": sig.return_annotation.__name__ if sig.return_annotation != sig.empty else "Any"
       }

Benefits of This Approach
-------------------------

**1. No Vendor Lock-in**
   - Any function can be a tool
   - No framework-specific base classes
   - Easy migration to/from other frameworks

**2. Excellent IDE Support**
   - Full type hints and autocomplete
   - Jump to definition works
   - Refactoring tools work correctly

**3. Easy Testing**
   - Test functions directly
   - No mocking of framework components
   - Standard unit testing patterns apply

**4. Reusability**
   - Tools work outside the framework
   - Share tools between projects
   - Use in non-AI contexts

**5. Performance**
   - No abstraction overhead
   - Direct function calls
   - Optimal for high-frequency operations

Integration with Framework Layers
----------------------------------

Tools fit naturally into Arshai's three-layer architecture:

**Layer 1 (LLM Clients)**:
   - Accept tools as list of callables
   - Generate function schemas automatically
   - Handle tool execution and results

**Layer 2 (Agents)**:
   - Choose which tools to provide to LLM
   - Can wrap or modify tool behavior
   - Filter or validate tool results

**Layer 3 (Systems)**:
   - Orchestrate tools across multiple agents
   - Manage tool state and lifecycle
   - Implement tool access controls

This design ensures tools are flexible, powerful, and aligned with Arshai's philosophy of developer control.