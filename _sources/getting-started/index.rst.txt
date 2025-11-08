Getting Started
===============

Welcome to Arshai! This section will get you up and running with the framework in minutes.

.. toctree::
   :maxdepth: 2
   :caption: Quick Start Guides
   
   installation
   quickstart
   comprehensive-guide
   first-agent

What is Arshai?
---------------

Arshai is a framework for building agentic AI systems. It provides:

- **Layer 1**: Standardized LLM client interfaces
- **Layer 2**: Agent building blocks with BaseAgent
- **Layer 3**: Patterns for composing agents into systems

The framework follows these principles:

- **Direct Control**: You explicitly create and configure components
- **Building Blocks**: Framework provides foundations, you build solutions
- **Progressive Complexity**: Start simple, scale to sophisticated systems

Choose Your Path
----------------

**5 Minutes - Quickstart**
   Jump straight into code with :doc:`quickstart`. Create your first agent and start chatting.

**30 Minutes - Comprehensive Guide**
   Read :doc:`comprehensive-guide` for a complete overview of agent capabilities and patterns.

**1 Hour - Build Your First Agent**
   Follow :doc:`first-agent` to build a production-ready agent step-by-step.

Quick Example
-------------

Here's the simplest possible agent:

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLMInput

   class MyAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

Prerequisites
-------------

**Python Version**
   Python 3.8 or higher is required.

**API Keys**
   You'll need an API key from one of:
   
   - OpenAI
   - Azure OpenAI
   - Google Gemini
   - OpenRouter

**Installation**
   Install Arshai via pip:
   
   .. code-block:: bash
   
      pip install arshai

What You'll Learn
-----------------

**Foundation Concepts**
   - Creating agents by extending BaseAgent
   - Processing inputs and returning responses
   - Configuring LLM clients

**Core Patterns**
   - Adding tools to agents
   - Managing conversation memory
   - Handling errors gracefully

**System Building**
   - Composing multiple agents
   - Orchestration patterns
   - Building complete applications

Framework Philosophy
--------------------

Arshai believes in:

1. **Developer Authority**: You control every aspect
2. **Explicit Over Implicit**: No hidden magic
3. **Composition Over Monoliths**: Build from simple pieces
4. **Progressive Enhancement**: Start simple, add complexity as needed

Next Steps
----------

1. **Install Arshai**: Follow :doc:`installation`
2. **Try the Quickstart**: Get hands-on in :doc:`quickstart`
3. **Read the Guide**: Understand concepts in :doc:`comprehensive-guide`
4. **Build an Agent**: Create something real in :doc:`first-agent`

After getting started, explore:

- :doc:`../framework/agents/index` - Deep dive into agents
- :doc:`../framework/building-systems/index` - Compose agents into systems
- :doc:`../framework/llm-clients/index` - Understand LLM integration

Welcome to the Arshai community!