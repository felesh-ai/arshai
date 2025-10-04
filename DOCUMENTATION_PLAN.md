# Arshai Documentation Plan

## Current State Analysis

### Framework Components Analysis

## FUNDAMENTAL FRAMEWORK (Core)

**Layer 1 - LLM Clients** ✅ FUNDAMENTAL
- `base_llm_client.py` - Base class for all clients
- `openai.py` - OpenAI client implementation
- `azure.py` - Azure OpenAI client  
- `google_genai.py` - Google Gemini client
- `openrouter.py` - OpenRouter proxy client
- **These ARE the framework** - provide standardized LLM access

**Layer 2 - Agent Foundation** ✅ FUNDAMENTAL
- `base.py` - BaseAgent abstract class
- `IAgent` interface
- **This IS the framework** - provides agent building blocks

**Layer 3 - System Building Blocks** ✅ FUNDAMENTAL
- Core interfaces for building systems
- Patterns for orchestration
- **Framework provides the foundation, users build systems**

## EXAMPLE IMPLEMENTATIONS (Samples/Hub)

**Example Agents** 📋 SAMPLES
- `working_memory.py` - Example agent with memory integration
- **NOT fundamental** - just shows how to build agents
- **Potential "agents hub"** - examples users can learn from or use

**Example Orchestration** 📋 SAMPLES  
- `workflow_runner.py` - Example workflow execution
- `workflow_orchestrator.py` - Example orchestration logic
- `workflow_config.py` - Example configuration
- `node.py` - Example workflow nodes
- **NOT fundamental** - shows one way to build agentic systems

**Example Memory** 📋 SAMPLES
- `working_memory/in_memory_manager.py` - Example in-memory storage
- `working_memory/redis_memory_manager.py` - Example Redis storage
- **NOT fundamental** - shows how to implement memory

**Example Tools** 📋 SAMPLES
- `knowledge_base_tool.py` - Example RAG tool
- `web_search_tool.py` - Example web search
- `mcp_dynamic_tool.py` - Example MCP integration
- **NOT fundamental** - shows how to build tools as callables

**Example Components** 📋 SAMPLES
- **Embeddings**: OpenAI, VoyageAI, MGTE - implementation examples
- **Vector DB**: Milvus client - one implementation example
- **All are examples** showing how to implement interfaces

## Documentation Structure

### Naming Decision
**Use "philosophy"** instead of "architecture" for the vision section because:
- "Philosophy" better captures the WHY and the mindset
- "Architecture" is more technical/structural
- We have a separate technical architecture section for the three-layer design

### Proposed Structure

```
docs_sphinx/
├── index.rst                          # Landing page - Framework vision + building blocks
│
├── philosophy/                        # WHY - Framework vision & principles
│   ├── index.rst                     # Philosophy overview
│   ├── introduction.rst              # What is Arshai and why it exists
│   ├── three-layer-architecture.rst  # Core architectural pattern
│   ├── developer-authority.rst       # Developer control philosophy
│   └── design-decisions.rst          # Key design choices explained
│
├── getting-started/                   # QUICK START
│   ├── index.rst
│   ├── installation.rst              # pip install arshai
│   ├── quickstart.rst                # 5-minute example using ONLY core framework
│   └── first-agent.rst               # Build first custom agent (extending BaseAgent)
│
├── framework/                         # CORE FRAMEWORK (Fundamental)
│   ├── index.rst                     # What IS the framework vs examples
│   ├── llm-clients/                  # Layer 1 - THE framework for LLM access
│   │   ├── index.rst                 # LLM client foundation
│   │   ├── using-openai.rst
│   │   ├── using-azure.rst
│   │   ├── using-google-gemini.rst
│   │   ├── using-openrouter.rst
│   │   └── extending-llm-clients.rst  # How to add new providers
│   │
│   ├── agents/                       # Layer 2 - THE framework for agents
│   │   ├── index.rst                 # Agent foundation (BaseAgent + IAgent)
│   │   ├── base-agent.rst            # How BaseAgent works
│   │   ├── creating-agents.rst       # Extending BaseAgent
│   │   ├── stateless-design.rst      # Core agent principles
│   │   └── agent-patterns.rst        # Common patterns
│   │
│   └── building-systems/             # Layer 3 - THE framework for systems  
│       ├── index.rst                 # System building concepts
│       ├── interfaces.rst            # Available interfaces for systems
│       ├── composition-patterns.rst   # How to compose components
│       └── orchestration-patterns.rst # Patterns for orchestration
│
├── implementations/                   # REFERENCE IMPLEMENTATIONS (Our Experience)
│   ├── index.rst                     # "These are reference implementations, not framework core"
│   ├── agents/                       # Reference Agent Implementations
│   │   ├── index.rst                 # Available agent implementations
│   │   └── working-memory-agent.rst  # WorkingMemoryAgent reference
│   │
│   ├── orchestration/                # Reference System Implementations
│   │   ├── index.rst                 # Available orchestration patterns
│   │   ├── workflow-system.rst       # Workflow-based orchestration
│   │   └── building-your-own.rst    # Creating custom orchestration
│   │
│   ├── memory/                       # Reference Memory Implementations
│   │   ├── index.rst                 # Available memory managers
│   │   ├── in-memory.rst             # InMemoryManager reference
│   │   └── redis-memory.rst          # RedisMemoryManager reference
│   │
│   └── components/                   # Other Reference Components
│       ├── index.rst                 # Available components
│       ├── embeddings.rst            # Embedding implementations
│       └── vector-databases.rst      # Vector DB implementations
│
├── tutorials/                         # COMPLETE APPLICATIONS
│   ├── index.rst                     # Building complete systems
│   ├── simple-chatbot.rst           # Using only framework core
│   ├── rag-system.rst                # Using example components
│   └── custom-system.rst            # Building your own system
│
├── reference/                         # API REFERENCE (Core only)
│   ├── index.rst                     # Reference overview
│   ├── interfaces.rst                # Core framework interfaces
│   └── base-classes.rst             # BaseAgent, BaseLLMClient
│
└── extending/                         # EXTENDING THE FRAMEWORK
    ├── index.rst                     # How to extend vs how to build examples
    ├── adding-llm-providers.rst      # Extending Layer 1
    ├── agent-patterns.rst            # Extending Layer 2  
    ├── system-patterns.rst           # Extending Layer 3
    └── contributing.rst              # Contributing to core framework
```

## Clarification Notes & Decisions

### 1. Agent Hub Organization
Consider restructuring agents directory:
```
arshai/agents/
├── base.py           # Core framework (FUNDAMENTAL)
├── __init__.py       # Core exports
└── hub/              # Example agents (SAMPLES)
    ├── working_memory.py
    └── other_examples.py
```
This clearly separates framework core from examples.

### 2. Documentation Tone Guidelines
The documentation should emphasize:
- **The framework is intentionally minimal** - provides building blocks, not solutions
- **Examples are "our experience" not "the way"** - show one approach, not the only approach
- **Users are expected to build their own** - framework empowers, doesn't prescribe
- **Examples showcase possibilities** - demonstrate what can be built, not what must be built

### 3. Example Section Disclaimers
All example sections should include clear disclaimers:
> **Note**: These are implementation examples showing how we've used the framework in our projects. 
> They are NOT part of the core framework. You are encouraged to build your own implementations 
> that fit your specific needs. The framework provides the foundation; you create the solution.

### 4. Example Complexity
Examples should follow progressive complexity:
- **Minimal examples** - Show bare minimum to get started
- **Practical examples** - Show real-world patterns
- **Advanced examples** - Show complex orchestrations
- Always emphasize these are "one way" not "the way"

## Documentation Guidelines

### 1. No Hallucination Policy
- ✅ Document ONLY what exists in code
- ✅ Be explicit about limitations
- ✅ Point to extension guides for missing features
- ❌ Don't mention non-existent features
- ❌ Don't promise future features

### 2. Honest Examples
```python
# ✅ GOOD - Shows what actually exists
from arshai.agents.base import BaseAgent
from arshai.llms.openai import OpenAIClient

# ❌ BAD - References non-existent component
from arshai.agents import ConversationAgent  # DOESN'T EXIST!
```

### 3. Clear Limitations
Always be upfront about what the framework provides:
- "Arshai provides building blocks, not complete solutions"
- "Currently supports OpenAI, Azure, Google Gemini, and OpenRouter"
- "Includes one vector database client (Milvus)"

### 4. Focus on Patterns Over Implementations
Since implementations are minimal, emphasize:
- How to extend BaseAgent
- How to implement interfaces
- How to compose components
- Direct instantiation patterns

## Files to Remove

### Completely Remove (outdated/non-existent)
- `api/agents/conversation.rst` - ConversationAgent doesn't exist
- `api/config/` - Settings/ConfigManager removed
- `api/callbacks/` - Deprecated
- `api/clients/` - Old structure
- `deployment/monitoring.rst` - Observability being removed
- `deployment/scaling.rst` - Too specific for current state

### Keep but Regenerate (outdated content)
- Most files in `getting-started/`
- Most files in `api/`
- Workflow documentation

## Implementation Steps

1. **Phase 1: Structure** (Current)
   - Create documentation plan (this file)
   - Create folder structure
   - Remove outdated files

2. **Phase 2: Philosophy**
   - Port philosophy from `docs/00-philosophy/`
   - Adapt tone for Sphinx/RST format
   - Ensure consistency with framework vision

3. **Phase 3: Getting Started**
   - Write installation guide
   - Create realistic quickstart
   - Document actual core concepts

4. **Phase 4: Guides**
   - Document each LLM client
   - Show how to extend BaseAgent
   - Explain workflow system

5. **Phase 5: Reference**
   - Generate interface documentation
   - Document base classes
   - Keep it minimal and accurate

## Key Messages to Emphasize

### Core Philosophy Messages
1. **You're in control** - Direct instantiation, no hidden magic
2. **Three-layer architecture** - Progressive complexity and authority
3. **Building blocks, not solutions** - Framework provides foundation, you build what you need
4. **Interface-driven** - Extend and implement as needed
5. **Minimal but powerful** - Less is more philosophy

### Framework vs Examples Distinction
1. **Core Framework**:
   - Layer 1: LLM Clients (OpenAI, Azure, Gemini, OpenRouter)
   - Layer 2: BaseAgent and IAgent interface
   - Layer 3: Interfaces and patterns for building systems
   
2. **Example Implementations**:
   - Everything else is "how we did it" not "how you must do it"
   - Workflow system is ONE way to orchestrate
   - Memory managers are ONE way to handle state
   - Tools show ONE way to extend functionality

### Documentation Principles
1. **Clear separation** - Always distinguish framework from examples
2. **Empower developers** - Show them how to build, not what to use
3. **Honest about scope** - Framework is minimal by design
4. **Examples as inspiration** - Not prescription, but possibilities