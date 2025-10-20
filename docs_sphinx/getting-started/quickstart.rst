Quickstart Guide
================

Get up and running with Arshai agents in 5 minutes. This guide shows you the minimal code needed to create and use your first agent.

Prerequisites
-------------

1. Install Arshai:

.. code-block:: bash

   pip install arshai

2. Set up your API key:

.. code-block:: bash

   export OPENROUTER_API_KEY=your_key_here

Your First Agent in 30 Seconds
-------------------------------

Here's the simplest possible agent:

.. code-block:: python

   import asyncio
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLMInput, ILLMConfig
   from arshai.llms.openrouter import OpenRouterClient

   # Step 1: Create your custom agent
   class MyFirstAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)
           return result.get('llm_response', 'No response')

   # Step 2: Configure and use
   async def main():
       # Configure LLM
       config = ILLMConfig(model="openai/gpt-4o-mini", temperature=0.7)
       llm_client = OpenRouterClient(config)
       
       # Create agent
       agent = MyFirstAgent(
           llm_client=llm_client,
           system_prompt="You are a helpful AI assistant."
       )
       
       # Use agent
       response = await agent.process(IAgentInput(message="Hello!"))
       print(response)

   # Run
   asyncio.run(main())

Interactive Example
-------------------

Want to chat with your agent? Here's an interactive version:

.. code-block:: python

   import os
   import asyncio
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLMInput, ILLMConfig
   from arshai.llms.openrouter import OpenRouterClient

   class ChatAgent(BaseAgent):
       """A simple chat agent."""
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)
           return result.get('llm_response', 'No response')

   async def main():
       # Check API key
       if not os.environ.get("OPENROUTER_API_KEY"):
           print("⚠️  Please set OPENROUTER_API_KEY environment variable")
           return
       
       # Setup
       config = ILLMConfig(
           model="openai/gpt-4o-mini",
           temperature=0.7,
           max_tokens=150
       )
       llm_client = OpenRouterClient(config)
       
       # Create agent
       agent = ChatAgent(
           llm_client=llm_client,
           system_prompt="You are a helpful AI assistant. Be concise and friendly."
       )
       
       # Interactive chat
       print("🤖 Agent is ready! Type 'quit' to exit.\n")
       
       while True:
           user_message = input("You: ")
           if user_message.lower() == 'quit':
               break
           
           response = await agent.process(IAgentInput(message=user_message))
           print(f"Agent: {response}\n")
       
       print("Goodbye! 👋")

   if __name__ == "__main__":
       asyncio.run(main())

Key Concepts
------------

**1. Extend BaseAgent**
   Every agent extends ``BaseAgent`` and implements the ``process`` method.

**2. Process Method**
   The ``process`` method takes ``IAgentInput`` and returns any type you need (string, dict, object, etc.).

**3. Direct Control**
   You explicitly create and configure your agent - no hidden magic.

**4. LLM Integration**
   Agents wrap LLM clients, adding your custom logic around AI capabilities.

What's Next?
------------

**10 Minutes:**
   Read :doc:`comprehensive-guide` for a complete overview of agent capabilities.

**30 Minutes:**
   Explore :doc:`../framework/agents/creating-agents` to build custom agents.

**1 Hour:**
   Learn :doc:`../framework/building-systems/index` to compose agents into systems.

Common Patterns
---------------

**Agent with Configuration:**

.. code-block:: python

   class ConfigurableAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt, max_tokens=150, temperature=0.7):
           super().__init__(llm_client, system_prompt)
           self.max_tokens = max_tokens
           self.temperature = temperature
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               max_tokens=self.max_tokens,
               temperature=self.temperature
           )
           result = await self.llm_client.chat(llm_input)
           return result.get('llm_response', '')

**Agent with Error Handling:**

.. code-block:: python

   class SafeAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           try:
               if not input.message:
                   return "Please provide a message"
               
               llm_input = ILLMInput(
                   system_prompt=self.system_prompt,
                   user_message=input.message
               )
               result = await self.llm_client.chat(llm_input)
               return result.get('llm_response', 'No response')
               
           except Exception as e:
               return f"Error: {str(e)}"

**Agent with Metadata:**

.. code-block:: python

   class ContextAwareAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           # Access metadata
           user_id = input.metadata.get("user_id", "anonymous") if input.metadata else "anonymous"
           
           # Include context in prompt
           enhanced_prompt = f"{self.system_prompt}\nUser ID: {user_id}"
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)
           return result.get('llm_response', '')

Tips for Success
----------------

1. **Start Simple**: Begin with basic agents before adding complexity
2. **Use Type Hints**: Helps with IDE support and debugging
3. **Handle Errors**: Always include error handling in production
4. **Test Early**: Write tests for your agents from the start
5. **Keep Agents Focused**: Each agent should have one clear purpose

Troubleshooting
---------------

**"API Key not found"**
   Ensure you've exported your API key:
   
   .. code-block:: bash
   
      export OPENROUTER_API_KEY=your_key_here

**"Module not found"**
   Install Arshai and its dependencies:
   
   .. code-block:: bash
   
      pip install arshai

**"Connection error"**
   Check your internet connection and API key validity.

Next Steps
----------

Now that you have a working agent:

- **Learn More**: Read the :doc:`comprehensive-guide` for deeper understanding
- **Explore Examples**: Check :doc:`../framework/agents/examples/index` for patterns
- **Build Systems**: Learn to compose agents in :doc:`../framework/building-systems/index`
- **Add Tools**: Enable agents with tools in :doc:`../framework/agents/tools-and-callables`

Remember: Arshai provides the building blocks - you create the solution!