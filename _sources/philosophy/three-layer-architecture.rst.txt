Three-Layer Architecture  
========================

The three-layer architecture is the foundational design pattern of Arshai. It organizes the framework into three distinct layers, each with increasing levels of developer authority and control. 

.. important::
   This architecture describes the **framework's structure**, not a prescribed way to build applications. You decide how to use these layers in your implementation.

Overview
--------

The three layers provide a natural progression of complexity and control:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                    Layer 3: Agentic Systems                    │
   │                  (Maximum Developer Authority)                 │
   │                                                                 │
   │   • Workflows      • Orchestration     • State Management      │
   │   • Memory         • Tool Integration  • Complex Patterns      │
   └───────────────────────────┬─────────────────────────────────────┘
                               │ Composes
   ┌───────────────────────────▼─────────────────────────────────────┐
   │                      Layer 2: Agents                           │
   │                  (Moderate Developer Authority)                │
   │                                                                 │
   │   • BaseAgent      • Custom Logic      • Process Methods       │
   │   • Stateless      • Tool Usage        • Response Formatting   │
   └───────────────────────────┬─────────────────────────────────────┘
                               │ Uses
   ┌───────────────────────────▼─────────────────────────────────────┐
   │                    Layer 1: LLM Clients                        │
   │                  (Minimal Developer Authority)                 │
   │                                                                 │
   │   • OpenAI         • Gemini           • Azure                  │
   │   • Standardized   • Environment Vars  • Streaming             │
   └─────────────────────────────────────────────────────────────────┘

Layer 1: LLM Clients
--------------------

**Purpose**: Provide standardized access to various LLM providers with consistent interfaces.

**Characteristics**:
- Minimal developer authority with standard patterns
- Environment configuration for API keys
- Unified interface through ILLM protocol
- Provider abstraction for easy switching

**Example**:

.. code-block:: python

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput

   # Direct instantiation - you create it
   config = ILLMConfig(model="gpt-4", temperature=0.7)
   llm_client = OpenAIClient(config)

   # Standardized usage
   llm_input = ILLMInput(
       system_prompt="You are a helpful assistant",
       user_message="Hello, how are you?"
   )
   response = await llm_client.chat(llm_input)

Layer 2: Agents
---------------

**Purpose**: Wrap LLM clients with purpose-driven logic and business rules.

**Characteristics**:
- Moderate developer authority with custom logic
- Stateless design for scalability
- Explicit dependency injection
- Flexible return types

**Example**:

.. code-block:: python

   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput

   class AnalysisAgent(BaseAgent):
       """Custom agent for data analysis."""
       
       def __init__(self, llm_client: ILLM, system_prompt: str):
           super().__init__(llm_client, system_prompt)
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Your custom logic here
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Analyze: {input.message}"
           )
           
           result = await self.llm_client.chat(llm_input)
           
           return {
               "analysis": result["llm_response"],
               "confidence": self._calculate_confidence(result)
           }

Layer 3: Agentic Systems
------------------------

**Purpose**: Compose agents and components into complex, multi-step systems.

**Characteristics**:
- Maximum developer authority with complete control
- Component composition and orchestration
- Complex state management
- Workflow definition and execution

**Example**:

.. code-block:: python

   class DataPipelineSystem:
       """Complex system orchestrating multiple agents."""
       
       def __init__(self):
           # You create all components
           llm_config = ILLMConfig(model="gpt-4")
           llm_client = OpenAIClient(llm_config)
           
           # You compose the system
           self.extractor = DataExtractorAgent(llm_client)
           self.analyzer = AnalysisAgent(llm_client)
           self.reporter = ReportAgent(llm_client)
       
       async def process_data(self, data: str) -> Dict[str, Any]:
           # You control execution flow
           extracted = await self.extractor.process(data)
           analyzed = await self.analyzer.process(extracted)
           report = await self.reporter.process(analyzed)
           
           return {
               "report": report,
               "metadata": self._generate_metadata()
           }

Key Benefits
------------

**Clear Separation of Concerns**
   - Layer 1: Provider integration
   - Layer 2: Business logic  
   - Layer 3: System orchestration

**Progressive Complexity**
   - Start simple with Layer 1
   - Add logic with Layer 2
   - Build systems with Layer 3

**Maximum Flexibility**
   - Use only what you need
   - Skip layers if not required
   - Mix and match components

**Independent Testing**
   Each layer can be tested independently with mocks and test doubles.

Design Principles
-----------------

1. **Explicit Dependencies**: All dependencies are injected, nothing is hidden
2. **Progressive Authority**: More control as you move up layers
3. **Stateless Agents**: Agents don't maintain internal state
4. **Interface Compliance**: Components implement well-defined protocols
5. **Direct Instantiation**: You create and configure all components

This architecture ensures you have the building blocks to create any AI system while maintaining complete control over the implementation.