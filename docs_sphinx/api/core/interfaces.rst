==========
Interfaces
==========

The interfaces module defines the contracts that all components must implement. These interfaces enable you to create custom components that integrate seamlessly with Arshai's three-layer architecture.

Core Design Philosophy
======================

Arshai uses Python protocols to define interfaces, providing:

- **Duck typing with type safety** - No forced inheritance
- **Multiple protocol implementation** - Components can implement multiple interfaces
- **Easy testing** - Simple to create mocks and test doubles
- **Clear contracts** - Explicit method signatures and behavior

Layer 1: LLM Client Interfaces
===============================

.. currentmodule:: arshai.core.interfaces

LLM Provider Interface
----------------------

.. autoclass:: ILLM
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ILLMConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ILLMInput
   :members:
   :undoc-members:
   :show-inheritance:

Layer 2: Agent Interfaces
==========================

Agent Core Interface
--------------------

.. autoclass:: IAgent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IAgentInput
   :members:
   :undoc-members:
   :show-inheritance:

Layer 3: System Interfaces
===========================

Workflow Interfaces
-------------------

.. autoclass:: IWorkflowState
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IUserContext
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IWorkflowOrchestrator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IWorkflowConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: INode
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IWorkflowRunner
   :members:
   :undoc-members:
   :show-inheritance:

Memory Management Interface
---------------------------

.. autoclass:: IMemoryManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IWorkingMemory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ConversationMemoryType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IMemoryInput
   :members:
   :undoc-members:
   :show-inheritance:

Supporting Interfaces
======================

Document Processing
-------------------

.. autoclass:: Document
   :members:
   :undoc-members:
   :show-inheritance:

Data Transfer Objects
---------------------

.. autoclass:: IDTO
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IStreamDTO
   :members:
   :undoc-members:
   :show-inheritance:

Vector Database & Embeddings
-----------------------------

.. autoclass:: IEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IVectorDBClient
   :members:
   :undoc-members:
   :show-inheritance:


Creating Custom Interfaces
===========================

When creating your own interfaces, follow these patterns:

.. code-block:: python

   from typing import Protocol
   from arshai.core.interfaces import IDTO
   
   # Define your interface
   class IMyCustomComponent(Protocol):
       async def process(self, input: IDTO) -> IDTO:
           """Process input and return result."""
           ...
   
   # Implement the interface
   class MyCustomComponent:
       async def process(self, input: IDTO) -> IDTO:
           # Your implementation
           return result
   
   # Use with type safety
   def use_component(component: IMyCustomComponent):
       return component.process(input_data)