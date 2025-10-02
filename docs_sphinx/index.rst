.. Arshai documentation master file

==============================
Arshai Documentation
==============================

.. image:: https://img.shields.io/pypi/v/arshai.svg
   :target: https://pypi.org/project/arshai/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/arshai.svg
   :target: https://pypi.org/project/arshai/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/felesh-ai/arshai.svg
   :target: https://github.com/felesh-ai/arshai/blob/main/LICENSE
   :alt: License

**Arshai** is an AI framework built on a clean three-layer architecture that provides direct control over components through explicit instantiation and dependency injection. It empowers developers to build AI systems without hidden abstractions or forced patterns.

Core Features
=============

**Three-Layer Architecture**
   Clean separation with Layer 1 (LLM Clients), Layer 2 (Agents), and Layer 3 (Agentic Systems) providing progressive developer authority

**Interface-Driven Design**
   Protocol-based interfaces that define clear contracts, enabling you to implement and integrate any component that respects the structure

**Direct Instantiation Philosophy**
   You create, configure, and control all components explicitly - no hidden factories, no magic configuration, no framework lock-in

**Developer Authority**
   Complete control over component lifecycle, dependencies, and behavior through explicit composition and dependency injection

Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install arshai

Three-Layer Example
-------------------

Arshai's power comes from its three-layer architecture where you build exactly what you need:

.. code-block:: python

   import os
   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput

   # Layer 1: LLM Client - You create and configure
   os.environ["OPENAI_API_KEY"] = "your-api-key"
   llm_config = ILLMConfig(model="gpt-4", temperature=0.7)
   llm_client = OpenAIClient(llm_config)

   # Layer 2: Agent - You implement your logic
   class MyCustomAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> str:
           # Your business logic here
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Process this: {input.message}"
           )
           result = await self.llm_client.chat(llm_input)
           return result['llm_response']

   # Layer 3: System - You orchestrate components
   class MyAISystem:
       def __init__(self):
           # You control component creation
           self.agent = MyCustomAgent(
               llm_client=llm_client,
               system_prompt="You are a helpful assistant"
           )
       
       async def handle_request(self, message: str) -> str:
           # You control the flow
           return await self.agent.process(IAgentInput(message=message))

   # Usage - You're in complete control
   system = MyAISystem()
   response = await system.handle_request("Hello!")

The Framework Philosophy
------------------------

**You're the Architect**: Arshai provides interfaces and building blocks, but you decide how to use them, when to use them, and whether to use them at all.

**No Hidden Magic**: Every component is created explicitly by you. No factories, no global state, no configuration files that hide behavior.

**Interface Respect**: Any component you build that implements our interfaces works seamlessly with the framework.

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/installation
   getting-started/quickstart
   getting-started/first-agent

.. toctree::
   :maxdepth: 2
   :caption: Architecture & Philosophy

   architecture/architecture
   architecture/developer-authority
   architecture/design-decisions
   architecture/tools-and-callables

.. toctree::
   :maxdepth: 2
   :caption: Three-Layer Architecture

   architecture/layer1-llm-clients
   architecture/layer2-agents
   architecture/layer3-systems
   
.. toctree::
   :maxdepth: 2
   :caption: Building Components

   guides/implementing-llm-clients
   guides/creating-agents
   guides/building-systems
   guides/custom-interfaces

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Examples & Patterns

   examples/basic-patterns
   examples/advanced-systems
   examples/custom-components

.. toctree::
   :maxdepth: 2
   :caption: Deployment

   deployment/index

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing/documentation

Links
=====

* **PyPI**: https://pypi.org/project/arshai/
* **GitHub**: https://github.com/felesh-ai/arshai
* **Documentation**: https://felesh-ai.github.io/arshai/
* **Issues**: https://github.com/felesh-ai/arshai/issues

License
=======

This project is licensed under the MIT License - see the `LICENSE <https://github.com/felesh-ai/arshai/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`