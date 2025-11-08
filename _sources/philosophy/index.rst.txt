Philosophy
==========

The Arshai framework is built on a fundamental philosophy: **developers should have complete control over their AI applications**. This section explains the core principles, architectural decisions, and design philosophy that guide the framework.

.. toctree::
   :maxdepth: 2
   :caption: Philosophy & Principles

   introduction
   three-layer-architecture
   developer-authority
   design-decisions

Core Principles
---------------

**You're in Control**
   Every component is created explicitly by you. No hidden factories, no magic configuration, no framework taking over.

**Building Blocks, Not Solutions**  
   Arshai provides the foundation - interfaces and base classes. You build the solution that fits your needs.

**Progressive Complexity**
   Start simple with Layer 1 (LLM clients), add logic with Layer 2 (agents), orchestrate with Layer 3 (systems).

**Direct Instantiation**
   You create components when you need them, configure them how you want, and control their lifecycle.

**Interface-Driven Design**
   Clean interfaces define contracts. Implement them your way, extend as needed, compose freely.

Why This Matters
----------------

Traditional AI frameworks often:

- Hide complexity behind abstractions
- Force specific patterns and workflows  
- Create vendor lock-in through proprietary abstractions
- Make testing and debugging difficult

Arshai takes a different approach:

- **Transparent**: You see and control everything
- **Flexible**: Use what you need, ignore what you don't
- **Testable**: Easy to mock and test components
- **Portable**: No vendor lock-in, just clean interfaces

Next Steps
----------

Start with :doc:`introduction` to understand what Arshai is and why it exists, then explore the :doc:`three-layer-architecture` that provides the framework's structure.