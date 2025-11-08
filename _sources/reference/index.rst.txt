API Reference
=============

Complete technical API reference for Arshai framework interfaces, base classes, and models.

This section provides detailed specifications for all framework components, including method signatures, parameters, return types, and usage contracts.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   interfaces
   base-classes
   models

Overview
--------

The Arshai framework follows **interface-driven design** principles. All major components implement well-defined protocols that specify behavior contracts without implementation details.

**Key Principles:**

- **Protocol-Based Interfaces**: All interfaces use Python protocols for flexible, duck-typed implementations
- **DTO Pattern**: Structured data transfer using Pydantic models with validation
- **Provider Pattern**: Multiple implementations for external services (LLMs, memory, vector DBs)
- **Developer Authority**: Interfaces provide flexibility while maintaining clear contracts

Interface Categories
--------------------

**Core Interfaces**

Essential protocols that define the framework's primary abstractions:

- :ref:`ILLM <illm-interface>` - LLM client interface
- :ref:`IAgent <iagent-interface>` - Agent interface
- :ref:`IMemoryManager <imemorymanager-interface>` - Memory management interface

**Component Interfaces**

Specialized protocols for specific capabilities:

- :ref:`IEmbedding <iembedding-interface>` - Embedding generation
- :ref:`IVectorDBClient <ivectordbclient-interface>` - Vector database operations
- :ref:`IDocumentProcessor <idocumentprocessor-interface>` - Document processing
- :ref:`IReranker <ireranker-interface>` - Result reranking
- :ref:`IWebSearch <iwebsearch-interface>` - Web search integration

**Orchestration Interfaces**

Protocols for workflow and system coordination:

- :ref:`IWorkflowOrchestrator <iworkfloworchestrator-interface>` - Workflow orchestration
- :ref:`IWorkflowRunner <iworkflowrunner-interface>` - Workflow execution
- :ref:`IWorkflowNode <iworkflownode-interface>` - Workflow nodes

**Infrastructure Interfaces**

Supporting protocols for notifications and utilities:

- :ref:`INotificationService <inotificationservice-interface>` - Notification delivery
- :ref:`IDTO <idto-interface>` - Data transfer object base

Base Classes
------------

Foundation classes that provide reusable implementations:

- :ref:`BaseAgent <baseagent-class>` - Agent foundation with LLM integration
- :ref:`BaseLLMClient <basellmclient-class>` - LLM client foundation with function calling
- :ref:`WorkingMemoryAgent <workingmemoryagent-class>` - Agent with working memory

See :doc:`base-classes` for complete documentation.

Data Models
-----------

Pydantic models and DTOs used throughout the framework:

**LLM Models**

- :ref:`ILLMInput <illminput-model>` - LLM input structure
- :ref:`ILLMConfig <illmconfig-model>` - LLM configuration

**Agent Models**

- :ref:`IAgentInput <iagentinput-model>` - Agent input structure
- :ref:`IAgentConfig <iagentconfig-model>` - Agent configuration

**Memory Models**

- :ref:`IMemoryInput <imemoryinput-model>` - Memory operation input
- :ref:`IMemoryItem <imemoryitem-model>` - Memory item structure
- :ref:`IWorkingMemory <iworkingmemory-model>` - Working memory state
- :ref:`ConversationMemoryType <conversationmemorytype-enum>` - Memory type enum

**Workflow Models**

- Workflow state and configuration models
- Node input/output models

See :doc:`models` for complete documentation.

Usage Patterns
--------------

**Implementing an Interface**

All Arshai interfaces are protocols, so you implement them through duck typing:

.. code-block:: python

   from arshai.core.interfaces import IAgent, IAgentInput

   class MyCustomAgent:
       """Custom agent implementation"""

       async def process(self, input: IAgentInput) -> dict:
           """Process input and return response"""
           return {"response": f"Processed: {input.message}"}

   # MyCustomAgent automatically satisfies IAgent protocol

**Using DTOs**

All data structures are Pydantic models with validation:

.. code-block:: python

   from arshai.core.interfaces import IAgentInput

   # Valid input
   agent_input = IAgentInput(
       message="Hello",
       metadata={"conversation_id": "123"}
   )

   # Validation error - missing required field
   try:
       bad_input = IAgentInput()  # Raises ValidationError
   except ValidationError as e:
       print(f"Validation failed: {e}")

**Extending Base Classes**

Inherit from base classes to get default implementations:

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces import IAgentInput, ILLM

   class SmartAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str):
           super().__init__(llm_client, system_prompt)
           self.interaction_count = 0

       async def process(self, input: IAgentInput) -> dict:
           self.interaction_count += 1
           # Use inherited self.llm_client
           response = await self.llm_client.chat(...)
           return {"response": response, "count": self.interaction_count}

Type Safety
-----------

**Protocol Runtime Checking**

Use ``isinstance()`` to check protocol implementation at runtime:

.. code-block:: python

   from typing import runtime_checkable
   from arshai.core.interfaces import IAgent

   @runtime_checkable
   class IAgent(Protocol):
       async def process(self, input: IAgentInput) -> Any: ...

   agent = MyCustomAgent()
   if isinstance(agent, IAgent):
       result = await agent.process(input)

**Type Hints**

All interfaces use proper type hints for IDE support:

.. code-block:: python

   from arshai.core.interfaces import ILLM, ILLMInput

   async def chat_with_llm(llm: ILLM, message: str) -> dict:
       """IDE provides autocomplete and type checking"""
       input_data = ILLMInput(
           system_prompt="You are helpful",
           user_message=message
       )
       return await llm.chat(input_data)

Validation
----------

**Pydantic Validation**

All DTOs include automatic validation:

.. code-block:: python

   from arshai.core.interfaces import ILLMInput

   # Automatic validation
   llm_input = ILLMInput(
       system_prompt="",  # Will raise ValidationError
       user_message="Hello"
   )

   # Custom validators
   class ILLMInput(IDTO):
       @model_validator(mode='before')
       @classmethod
       def validate_input(cls, data):
           if not data.get('system_prompt'):
               raise ValueError("system_prompt is required")
           return data

**Field Validation**

Use Pydantic field validators for custom logic:

.. code-block:: python

   from pydantic import Field, field_validator

   class CustomInput(IDTO):
       temperature: float = Field(ge=0.0, le=2.0)

       @field_validator('temperature')
       @classmethod
       def validate_temperature(cls, v):
           if v > 1.0:
               print("Warning: High temperature may produce random results")
           return v

Navigation
----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Section
     - Description
   * - :doc:`interfaces`
     - All framework protocols and their specifications
   * - :doc:`base-classes`
     - Foundation classes for extension
   * - :doc:`models`
     - Pydantic models and DTOs

Related Documentation
---------------------

**For Implementation Guides**
   - :doc:`../framework/llm-clients/index` - LLM client implementations
   - :doc:`../framework/agents/index` - Agent development
   - :doc:`../framework/memory/index` - Memory systems

**For Examples**
   - :doc:`../framework/agents/examples/index` - Practical examples
   - :doc:`../tutorials/index` - Complete tutorials

**For Architecture**
   - :doc:`../philosophy/architecture` - System design principles
   - :doc:`../philosophy/direct-instantiation` - Component creation patterns

Contributing to API Documentation
----------------------------------

The API reference is maintained alongside the codebase. To improve documentation:

1. **Inline Docstrings**: Add comprehensive docstrings to interfaces and classes
2. **Type Hints**: Use complete type annotations for all parameters and returns
3. **Examples**: Include usage examples in docstrings
4. **Validation**: Document validation rules and constraints

See :doc:`../extending/index` for contribution guidelines.
