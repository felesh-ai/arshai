Framework Core
==============

The Arshai framework consists of three fundamental layers that provide building blocks for AI applications. Unlike traditional frameworks that prescribe specific solutions, Arshai provides the essential infrastructure while giving you complete control over how to build your systems.

.. toctree::
   :maxdepth: 2
   :caption: Framework Layers

   llm-clients/index
   agents/index
   building-systems/index

What IS the Framework
---------------------

The core framework consists of:

**Layer 1: LLM Clients**
   Standardized interfaces to language model providers (OpenAI, Azure, Google Gemini, OpenRouter). These provide unified access to different LLMs with consistent interfaces and functionality.

**Layer 2: Agent Foundation**  
   The BaseAgent class and IAgent interface that define how to build purpose-driven components. This layer provides the foundation for creating agents that wrap LLM clients with custom logic.

**Layer 3: System Building Blocks**
   Interfaces and patterns for building complex multi-agent systems. This layer provides the contracts and patterns for orchestrating agents and components.

What is NOT the Framework
--------------------------

Everything else in the package represents **reference implementations** - examples showing how we've used the framework in our projects. These include:

- Workflow orchestration systems
- Memory management implementations  
- Tool integrations
- Embedding providers
- Vector database clients

These implementations are provided as working examples and starting points, but they are not prescriptive. You're encouraged to:

- Use them as-is if they fit your needs
- Modify them for your requirements  
- Build completely different implementations
- Ignore them entirely and create your own

Core Design Principles
----------------------

**Direct Instantiation**
   You create and configure all components explicitly. No hidden factories, no magic configuration files, no framework taking control.

**Interface-Driven**
   Components implement well-defined interfaces. You can replace any component with your own implementation as long as it follows the interface contract.

**Progressive Authority**
   As you move up the layers (1→2→3), you gain more control and authority over the system behavior.

**Stateless Design**
   Agents and core components don't maintain internal state, making them easier to test, debug, and scale.

**Minimal Core**
   The framework includes only what's essential. Additional functionality is provided through reference implementations, not core framework features.

Getting Started with the Framework
-----------------------------------

To understand the framework:

1. **Start with Layer 1** - Learn how LLM clients provide standardized access to language models
2. **Move to Layer 2** - Understand how BaseAgent lets you build custom logic around LLMs  
3. **Explore Layer 3** - See how interfaces enable building complex systems

The framework documentation focuses on these core building blocks and how to extend them for your specific needs.