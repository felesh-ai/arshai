Introduction
============

What is Arshai?
---------------

Arshai is an **AI framework that provides building blocks, not solutions**. It empowers developers to create AI applications with complete control over component lifecycle, configuration, and behavior.

Unlike traditional frameworks that prescribe how to build AI applications, Arshai provides:

- **Fundamental building blocks** - Core interfaces and base classes
- **Reference implementations** - Examples showing one way to use the framework
- **Complete control** - You decide what to build and how to build it

The Framework vs. The Implementations
--------------------------------------

It's crucial to understand what IS and what ISN'T the framework:

**The Core Framework**

- **Layer 1**: LLM client implementations (OpenAI, Azure, Gemini, OpenRouter)
- **Layer 2**: BaseAgent class and IAgent interface  
- **Layer 3**: Interfaces and patterns for building systems

**Reference Implementations**

Everything else in the package - workflows, memory managers, tools - are **reference implementations**. They show how we've used the framework in our projects, but they're not prescriptive. You're encouraged to:

- Use them as-is if they fit your needs
- Modify them for your requirements
- Build completely different implementations
- Ignore them entirely and create your own

Why Arshai Exists
-----------------

The Problem
^^^^^^^^^^^

Most AI frameworks suffer from:

1. **Over-abstraction**: Hiding important details behind layers of magic
2. **Rigid patterns**: Forcing specific architectural patterns
3. **Kitchen-sink approach**: Including everything, needed or not
4. **Framework lock-in**: Making it hard to migrate or customize

The Solution
^^^^^^^^^^^^

Arshai takes a fundamentally different approach:

1. **Minimal core**: Just the essential building blocks
2. **Direct instantiation**: You create and control components
3. **Interface-driven**: Clean contracts, multiple implementations
4. **Example-rich**: Learn from implementations, don't be constrained by them

Who is Arshai For?
-------------------

Arshai is for developers who:

- Want **control** over their AI application architecture
- Prefer **explicit** over implicit behavior
- Value **simplicity** and understanding over magic
- Need **flexibility** to build custom solutions
- Appreciate **clean interfaces** and testable code

Who Might Prefer Other Frameworks?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Arshai might not be the best choice if you:

- Want a complete, out-of-the-box solution
- Prefer convention over configuration
- Need extensive pre-built components
- Want the framework to make architectural decisions for you

Core Philosophy
---------------

**Empower, Don't Prescribe**
   We provide tools and patterns, you decide how to use them.

**Transparent, Not Magic**
   You should understand what every line of code does.

**Minimal, Not Minimal-Viable**
   We include only what's essential, but that essential is production-ready.

**Examples, Not Rules**
   Our implementations show possibilities, not requirements.

What You'll Build
-----------------

With Arshai, you can build:

- **Custom chatbots** with your own logic and flow
- **RAG systems** with your choice of components
- **Multi-agent orchestrations** using your patterns
- **AI APIs** with your architecture
- **Anything else** that needs LLM integration

The key is: **you build it your way**, using our building blocks.

Getting Started
---------------

Ready to explore the framework? Continue to :doc:`three-layer-architecture` to understand the structure, or jump to :doc:`/getting-started/index` to start building.