Example 01: Basic Agent Usage
==============================

This example demonstrates the simplest way to create and use an agent in Arshai. Perfect for getting started quickly and understanding the core concepts.

**File**: ``examples/agents/01_basic_usage.py``

**Prerequisites**: Set ``OPENROUTER_API_KEY`` environment variable

Overview
--------

This example shows:

- Creating a simple agent that extends ``BaseAgent``
- Implementing the required ``process()`` method
- Basic LLM integration with ``ILLMInput``
- Using agent input with metadata
- Simple string responses

Key Concepts Demonstrated
-------------------------

**Basic Agent Structure**

The example creates a ``SimpleAgent`` that demonstrates the minimal agent implementation:

.. code-block:: python

   class SimpleAgent(BaseAgent):
       """A basic agent that processes messages and returns responses."""
       
       async def process(self, input: IAgentInput) -> str:
           # Create LLM input
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           # Get response from LLM
           result = await self.llm_client.chat(llm_input)
           
           # Return the response
           return result.get('llm_response', 'No response generated')

**Direct Instantiation Pattern**

The framework's core principle of direct instantiation is demonstrated:

.. code-block:: python

   # Step 1: Configure LLM client
   llm_config = ILLMConfig(
       model="openai/gpt-4o-mini",
       temperature=0.7,
       max_tokens=150
   )
   
   # Step 2: Initialize LLM client
   llm_client = OpenRouterClient(llm_config)
   
   # Step 3: Create agent with explicit dependencies
   agent = SimpleAgent(
       llm_client=llm_client,
       system_prompt="You are a helpful AI assistant. Be concise and friendly."
   )

**Input Structure and Metadata**

Shows how to use ``IAgentInput`` with optional metadata:

.. code-block:: python

   # Simple input
   agent_input = IAgentInput(message="Hello! How are you today?")
   
   # Input with metadata for context
   input_with_metadata = IAgentInput(
       message="Tell me about Python",
       metadata={
           "user_id": "user_123",
           "session_id": "session_456", 
           "max_length": 100
       }
   )

What You'll Learn
-----------------

**1. Agent Creation Basics**
   - Extending ``BaseAgent``
   - Implementing the ``process()`` method
   - Working with ``IAgentInput`` and ``ILLMInput``

**2. Framework Patterns**
   - Direct instantiation over factory patterns
   - Explicit dependency injection
   - Clear separation between LLM client and agent logic

**3. Practical Usage**
   - How to configure and initialize components
   - Processing user input through agents
   - Using metadata to pass context

**4. Development Workflow**
   - Environment setup with API keys
   - Testing agents with various inputs
   - Observing agent responses and behavior

Code Walkthrough
----------------

**1. Environment Setup**

.. code-block:: python

   # Check for required API key
   if not os.environ.get("OPENROUTER_API_KEY"):
       print("⚠️  Please set OPENROUTER_API_KEY environment variable")
       return

**2. LLM Configuration**

.. code-block:: python

   # Configure with specific model and parameters
   llm_config = ILLMConfig(
       model="openai/gpt-4o-mini",    # Using OpenRouter with GPT-4o Mini
       temperature=0.7,              # Balanced creativity
       max_tokens=150                # Reasonable response length
   )

**3. Agent Testing**

.. code-block:: python

   test_messages = [
       "Hello! How are you today?",
       "What is the capital of France?", 
       "Can you explain what an AI agent is in simple terms?"
   ]
   
   for message in test_messages:
       agent_input = IAgentInput(message=message)
       response = await agent.process(agent_input)
       print(f"🤖 Agent: {response}")

**4. Metadata Usage**

.. code-block:: python

   # Demonstrate passing context without changing the API
   input_with_metadata = IAgentInput(
       message="Tell me about Python",
       metadata={
           "user_id": "user_123",
           "session_id": "session_456",
           "max_length": 100
       }
   )

Running the Example
-------------------

**Setup**:

.. code-block:: bash

   export OPENROUTER_API_KEY=your_key_here
   cd examples/agents
   python 01_basic_usage.py

**Expected Output**:

.. code-block:: text

   ============================================================
   EXAMPLE 1: Basic Agent Usage
   ============================================================
   
   🔄 Initializing OpenRouter client...
   🤖 Creating simple agent...
   
   ----------------------------------------
   
   👤 User: Hello! How are you today?
   🤖 Agent: Hello! I'm doing well, thank you for asking...
   
   👤 User: What is the capital of France?
   🤖 Agent: The capital of France is Paris...
   
   👤 User: Can you explain what an AI agent is in simple terms?
   🤖 Agent: An AI agent is a computer program that can understand...
   
   ============================================================
   BONUS: Using Metadata
   ============================================================
   
   📋 Input with metadata: {'user_id': 'user_123', 'session_id': 'session_456', 'max_length': 100}
   🤖 Agent: Python is a versatile programming language...
   
   ✨ Metadata can be used to pass context without changing the API!

Key Takeaways
-------------

**1. Simplicity**
   - Creating agents is straightforward - just extend ``BaseAgent`` and implement ``process()``
   - The framework doesn't hide complexity behind abstractions

**2. Explicit Control**
   - You create and configure all components yourself
   - No hidden factories or magic configuration
   - Clear dependency relationships

**3. Flexible Input**
   - ``IAgentInput`` provides both message and metadata
   - Metadata allows context passing without API changes
   - Agents can ignore or use metadata as needed

**4. Return Type Freedom**
   - This example returns strings, but agents can return any type
   - The framework doesn't constrain your response format

**5. Testing Friendly**
   - Simple structure makes agents easy to test
   - Clear input/output relationships
   - No complex setup required

Next Steps
----------

After understanding basic usage:

1. **Example 02**: Learn about custom agents with different return types
2. **Example 03**: Explore memory patterns for stateful conversations  
3. **Example 04**: Add tool integration for external capabilities
4. **Example 05**: Compose multiple agents into complex systems

**Key Files to Explore**:

- ``arshai/agents/base.py`` - The BaseAgent foundation
- ``arshai/core/interfaces/iagent.py`` - IAgent interface definition
- ``arshai/core/interfaces/illm.py`` - LLM integration interfaces

This example provides the foundation for all agent development in Arshai. Once you understand this pattern, you can build increasingly sophisticated agents while maintaining the same clear, direct approach.