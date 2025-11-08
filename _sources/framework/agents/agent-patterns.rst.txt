Agent Patterns
==============

This guide covers proven patterns for building effective agents in the Arshai framework. These patterns solve common problems and provide reusable solutions for typical agent scenarios.

Core Agent Patterns
--------------------

**1. Simple Response Agent**

The most basic pattern for straightforward text responses:

.. code-block:: python

   class ConversationAgent(BaseAgent):
       """Simple agent for general conversation."""
       
       async def process(self, input: IAgentInput) -> str:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**When to use:** Basic conversational interfaces, simple Q&A systems, straightforward text processing.

**2. Structured Output Agent**

Returns structured data instead of plain text:

.. code-block:: python

   from pydantic import BaseModel, Field
   from typing import List

   class AnalysisResult(BaseModel):
       sentiment: str = Field(description="positive, negative, or neutral")
       confidence: float = Field(description="Confidence score 0-1")
       topics: List[str] = Field(description="Main topics identified")
       summary: str = Field(description="Brief summary")

   class AnalysisAgent(BaseAgent):
       """Agent that returns structured analysis results."""
       
       async def process(self, input: IAgentInput) -> AnalysisResult:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               structure_type=AnalysisResult
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]  # Returns AnalysisResult instance

**When to use:** Data analysis, classification tasks, when you need predictable output format, API responses.

**3. Tool-Enabled Agent**

Integrates external capabilities through Python functions:

.. code-block:: python

   def search_database(query: str, table: str = "products") -> List[dict]:
       """Search database for relevant information."""
       # Your database search implementation
       return search_results

   def calculate_price(base_price: float, discount: float = 0.0, tax_rate: float = 0.08) -> dict:
       """Calculate final price with discount and tax."""
       discounted = base_price * (1 - discount)
       final_price = discounted * (1 + tax_rate)
       return {
           "base_price": base_price,
           "discount_amount": base_price - discounted,
           "tax_amount": final_price - discounted,
           "final_price": final_price
       }

   class ShoppingAgent(BaseAgent):
       """Agent with shopping-related tools."""
       
       async def process(self, input: IAgentInput) -> str:
           tools = {
               "search_database": search_database,
               "calculate_price": calculate_price
           }
           
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               regular_functions=tools
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**When to use:** Agents that need to interact with external systems, perform calculations, access databases, make API calls.

**4. Streaming Agent**

Provides real-time streaming responses:

.. code-block:: python

   from typing import AsyncGenerator

   class StreamingAgent(BaseAgent):
       """Agent that streams responses for real-time interaction."""
       
       async def process(self, input: IAgentInput) -> AsyncGenerator[str, None]:
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message
           )
           
           async for chunk in self.llm_client.stream(llm_input):
               if chunk.get("llm_response"):
                   yield chunk["llm_response"]

**When to use:** Real-time chat interfaces, long-form content generation, user interfaces requiring immediate feedback.

Advanced Agent Patterns
------------------------

**5. Multi-Step Processing Agent**

Breaks complex tasks into sequential steps:

.. code-block:: python

   class DocumentAnalysisAgent(BaseAgent):
       """Agent that performs multi-step document analysis."""
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           document = input.message
           
           # Step 1: Extract key information
           extraction_input = ILLMInput(
               system_prompt="Extract key facts, dates, and entities from the document",
               user_message=document
           )
           extraction_result = await self.llm_client.chat(extraction_input)
           
           # Step 2: Analyze sentiment and tone
           sentiment_input = ILLMInput(
               system_prompt="Analyze the sentiment and tone of the document",
               user_message=document
           )
           sentiment_result = await self.llm_client.chat(sentiment_input)
           
           # Step 3: Generate summary and recommendations
           summary_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"""
               Document: {document}
               
               Key Information: {extraction_result['llm_response']}
               Sentiment Analysis: {sentiment_result['llm_response']}
               
               Generate a comprehensive summary and actionable recommendations.
               """
           )
           summary_result = await self.llm_client.chat(summary_input)
           
           return {
               "extracted_info": extraction_result["llm_response"],
               "sentiment_analysis": sentiment_result["llm_response"],
               "summary_and_recommendations": summary_result["llm_response"],
               "metadata": {
                   "processing_steps": 3,
                   "total_tokens": (
                       extraction_result["usage"]["total_tokens"] +
                       sentiment_result["usage"]["total_tokens"] +
                       summary_result["usage"]["total_tokens"]
                   )
               }
           }

**When to use:** Complex analysis tasks, document processing, multi-faceted evaluations.

**6. Configurable Agent**

Highly adaptable agent that changes behavior based on configuration:

.. code-block:: python

   class FlexibleAgent(BaseAgent):
       """Agent that adapts to different configurations and scenarios."""
       
       def __init__(self, llm_client: ILLM, config: dict):
           system_prompt = config.get("system_prompt", "You are a helpful assistant")
           super().__init__(llm_client, system_prompt)
           
           # Response configuration
           self.response_format = config.get("response_format", "text")
           self.max_tokens = config.get("max_tokens", 500)
           self.temperature = config.get("temperature", 0.7)
           
           # Tool configuration
           self.available_tools = config.get("tools", {})
           self.background_tasks = config.get("background_tasks", {})
           
           # Output configuration
           self.structure_type = config.get("structure_type", None)
           self.include_metadata = config.get("include_metadata", False)
           
           # Preprocessing configuration
           self.enable_validation = config.get("enable_validation", True)
           self.max_input_length = config.get("max_input_length", 5000)
       
       async def process(self, input: IAgentInput) -> Any:
           # Input validation (if enabled)
           if self.enable_validation:
               if not input.message or len(input.message) > self.max_input_length:
                   return {"error": "Invalid input"}
           
           # Build LLM input
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               max_tokens=self.max_tokens,
               temperature=self.temperature,
               regular_functions=self.available_tools,
               background_tasks=self.background_tasks,
               structure_type=self.structure_type
           )
           
           # Handle different response formats
           if self.response_format == "stream":
               return self.llm_client.stream(llm_input)
           
           result = await self.llm_client.chat(llm_input)
           response = result["llm_response"]
           
           # Include metadata if configured
           if self.include_metadata:
               return {
                   "response": response,
                   "usage": result["usage"],
                   "config": {
                       "model": llm_input.model,
                       "temperature": self.temperature,
                       "max_tokens": self.max_tokens
                   }
               }
           
           return response

**When to use:** Multi-tenant applications, A/B testing, environments requiring different agent behaviors.

**7. Error-Resilient Agent**

Implements comprehensive error handling and recovery:

.. code-block:: python

   import logging
   from enum import Enum
   from typing import Union, Dict, Any

   class ErrorSeverity(Enum):
       LOW = "low"
       MEDIUM = "medium"
       HIGH = "high"
       CRITICAL = "critical"

   class AgentError(Exception):
       def __init__(self, message: str, severity: ErrorSeverity, recoverable: bool = True):
           self.message = message
           self.severity = severity
           self.recoverable = recoverable
           super().__init__(message)

   class ResilientAgent(BaseAgent):
       """Agent with comprehensive error handling and recovery mechanisms."""
       
       def __init__(self, llm_client: ILLM, system_prompt: str, 
                    max_retries: int = 3, fallback_response: str = None):
           super().__init__(llm_client, system_prompt)
           self.max_retries = max_retries
           self.fallback_response = fallback_response or "I'm having trouble processing your request. Please try again."
           self.logger = logging.getLogger(__name__)
       
       def validate_input(self, input: IAgentInput) -> None:
           """Validate input and raise AgentError if invalid."""
           if not input.message:
               raise AgentError("Empty message", ErrorSeverity.MEDIUM)
           
           if len(input.message) > 10000:
               raise AgentError("Message too long", ErrorSeverity.LOW)
           
           # Check for potentially harmful content
           harmful_patterns = ["<script>", "javascript:", "data:"]
           if any(pattern in input.message.lower() for pattern in harmful_patterns):
               raise AgentError("Potentially harmful content detected", ErrorSeverity.HIGH, recoverable=False)
       
       async def process_with_retries(self, input: IAgentInput) -> Dict[str, Any]:
           """Process input with retry logic."""
           last_exception = None
           
           for attempt in range(self.max_retries):
               try:
                   llm_input = ILLMInput(
                       system_prompt=self.system_prompt,
                       user_message=input.message,
                       temperature=0.7 + (attempt * 0.1)  # Increase creativity on retries
                   )
                   
                   result = await self.llm_client.chat(llm_input)
                   
                   if not result.get("llm_response"):
                       raise AgentError("Empty response from LLM", ErrorSeverity.MEDIUM)
                   
                   return result
                   
               except Exception as e:
                   last_exception = e
                   self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                   
                   if attempt < self.max_retries - 1:
                       await asyncio.sleep(2 ** attempt)  # Exponential backoff
           
           # All retries failed
           raise AgentError(f"All retries failed. Last error: {last_exception}", ErrorSeverity.HIGH)
       
       async def process(self, input: IAgentInput) -> Union[str, Dict[str, Any]]:
           try:
               # Validate input
               self.validate_input(input)
               
               # Process with retries
               result = await self.process_with_retries(input)
               
               return result["llm_response"]
               
           except AgentError as e:
               self.logger.error(f"Agent error: {e.message} (severity: {e.severity.value})")
               
               if e.severity == ErrorSeverity.CRITICAL or not e.recoverable:
                   return {"error": e.message, "severity": e.severity.value, "recoverable": False}
               
               return self.fallback_response
               
           except Exception as e:
               self.logger.error(f"Unexpected error: {e}")
               return {"error": "Internal error occurred", "recoverable": True}

**When to use:** Production systems, mission-critical applications, environments with unreliable network conditions.

Specialized Agent Patterns
---------------------------

**8. Memory-Aware Agent**

Integrates with memory systems for contextual conversations:

.. code-block:: python

   class ContextualAgent(BaseAgent):
       """Agent that maintains conversation context through memory."""
       
       def __init__(self, llm_client: ILLM, system_prompt: str, memory_manager=None):
           super().__init__(llm_client, system_prompt)
           self.memory_manager = memory_manager
       
       async def retrieve_context(self, conversation_id: str) -> str:
           """Retrieve conversation context from memory."""
           if not self.memory_manager:
               return ""
           
           try:
               memory_records = await self.memory_manager.retrieve({
                   "conversation_id": conversation_id
               })
               
               if memory_records:
                   return memory_records[0].working_memory
               
           except Exception as e:
               self.logger.warning(f"Failed to retrieve memory: {e}")
           
           return ""
       
       async def update_context(self, conversation_id: str, interaction: str):
           """Update conversation context in memory."""
           if not self.memory_manager:
               return
           
           try:
               await self.memory_manager.store({
                   "conversation_id": conversation_id,
                   "working_memory": interaction
               })
           except Exception as e:
               self.logger.warning(f"Failed to store memory: {e}")
       
       async def process(self, input: IAgentInput) -> str:
           conversation_id = input.metadata.get("conversation_id")
           
           # Retrieve context
           context = ""
           if conversation_id:
               context = await self.retrieve_context(conversation_id)
           
           # Build system prompt with context
           enhanced_prompt = self.system_prompt
           if context:
               enhanced_prompt += f"\n\nPrevious conversation context:\n{context}"
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           response = result["llm_response"]
           
           # Update context
           if conversation_id:
               interaction_record = f"User: {input.message}\nAssistant: {response}"
               await self.update_context(conversation_id, interaction_record)
           
           return response

**When to use:** Multi-turn conversations, customer support systems, personalized interactions.

**9. Pipeline Agent**

Processes input through a series of specialized agents:

.. code-block:: python

   class PipelineAgent(BaseAgent):
       """Agent that processes input through a pipeline of specialized agents."""
       
       def __init__(self, llm_client: ILLM, system_prompt: str, pipeline_agents: List[BaseAgent]):
           super().__init__(llm_client, system_prompt)
           self.pipeline_agents = pipeline_agents
       
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           results = {}
           current_input = input
           
           for i, agent in enumerate(self.pipeline_agents):
               stage_name = f"stage_{i+1}_{agent.__class__.__name__}"
               
               try:
                   stage_result = await agent.process(current_input)
                   results[stage_name] = stage_result
                   
                   # Prepare input for next stage
                   if i < len(self.pipeline_agents) - 1:
                       current_input = IAgentInput(
                           message=str(stage_result),
                           metadata=current_input.metadata
                       )
               
               except Exception as e:
                   results[stage_name] = {"error": str(e)}
                   break
           
           return {
               "pipeline_results": results,
               "final_output": results.get(list(results.keys())[-1]),
               "stages_completed": len(results)
           }

**When to use:** Complex processing workflows, document analysis pipelines, multi-step data transformations.

**10. Background Task Agent**

Specializes in fire-and-forget operations:

.. code-block:: python

   def log_user_activity(user_id: str, action: str, details: dict = None):
       """Log user activity for analytics."""
       print(f"Analytics: User {user_id} performed {action}")
       if details:
           print(f"Details: {details}")

   def send_notification(recipient: str, message: str, priority: str = "normal"):
       """Send notification to user or admin."""
       print(f"Notification [{priority}] to {recipient}: {message}")

   def update_user_profile(user_id: str, preferences: dict):
       """Update user preferences based on interaction."""
       print(f"Updated preferences for {user_id}: {preferences}")

   class TaskCoordinatorAgent(BaseAgent):
       """Agent that coordinates background tasks while providing responses."""
       
       async def process(self, input: IAgentInput) -> str:
           user_id = input.metadata.get("user_id", "anonymous")
           session_id = input.metadata.get("session_id")
           
           background_tasks = {
               "log_user_activity": log_user_activity,
               "send_notification": send_notification,
               "update_user_profile": update_user_profile
           }
           
           enhanced_prompt = f"""
           {self.system_prompt}
           
           You can use background tasks to:
           - log_user_activity: Track user interactions
           - send_notification: Send alerts or updates
           - update_user_profile: Modify user preferences
           
           User ID: {user_id}
           Session: {session_id}
           """
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message,
               background_tasks=background_tasks
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**When to use:** System coordination, analytics collection, notification systems, user preference management.

Pattern Selection Guide
-----------------------

**Choose Simple Response Agent when:**
- Building basic conversational interfaces
- Prototyping new agent functionality
- Implementing straightforward text processing
- Creating simple Q&A systems

**Choose Structured Output Agent when:**
- Building APIs that need predictable responses
- Implementing classification or analysis systems
- Creating data extraction tools
- Building systems that process agent output programmatically

**Choose Tool-Enabled Agent when:**
- Agents need to interact with external systems
- Implementing domain-specific functionality
- Building agents that perform calculations or searches
- Creating agents that modify external state

**Choose Streaming Agent when:**
- Building real-time chat interfaces
- Generating long-form content
- Creating interactive writing assistants
- Implementing systems where immediate feedback is important

**Choose Multi-Step Processing when:**
- Implementing complex analysis workflows
- Building document processing systems
- Creating agents that need to break down complex tasks
- Implementing systems with multiple validation steps

**Choose Configurable Agent when:**
- Building multi-tenant systems
- Implementing A/B testing frameworks
- Creating agents for different environments
- Building systems with varying complexity requirements

**Choose Error-Resilient Agent when:**
- Building production systems
- Implementing mission-critical applications
- Creating systems with strict reliability requirements
- Building agents that handle sensitive data

**Choose Memory-Aware Agent when:**
- Building conversational systems
- Implementing customer support agents
- Creating personalized user experiences
- Building agents that learn from interactions

**Choose Pipeline Agent when:**
- Implementing complex processing workflows
- Building document analysis systems
- Creating multi-step validation processes
- Implementing systems with specialized processing stages

**Choose Background Task Agent when:**
- Building systems that need coordination
- Implementing analytics collection
- Creating notification systems
- Building agents that manage user preferences

Combining Patterns
------------------

Patterns can be combined for more sophisticated agents:

.. code-block:: python

   class AdvancedAgent(BaseAgent):
       """Agent combining multiple patterns for robust operation."""
       
       def __init__(self, llm_client: ILLM, config: dict, memory_manager=None):
           super().__init__(llm_client, config["system_prompt"])
           
           # Configurable pattern
           self.max_retries = config.get("max_retries", 3)
           self.tools = config.get("tools", {})
           self.background_tasks = config.get("background_tasks", {})
           
           # Memory-aware pattern
           self.memory_manager = memory_manager
           
           # Error-resilient pattern
           self.fallback_response = config.get("fallback_response", "I encountered an error.")
       
       async def process(self, input: IAgentInput) -> Union[str, Dict[str, Any]]:
           try:
               # Memory-aware: retrieve context
               conversation_id = input.metadata.get("conversation_id")
               context = await self.retrieve_context(conversation_id) if conversation_id else ""
               
               # Enhanced system prompt with context
               enhanced_prompt = self.system_prompt
               if context:
                   enhanced_prompt += f"\n\nContext: {context}"
               
               # Tool-enabled + Background tasks
               llm_input = ILLMInput(
                   system_prompt=enhanced_prompt,
                   user_message=input.message,
                   regular_functions=self.tools,
                   background_tasks=self.background_tasks
               )
               
               # Error-resilient: retry logic
               for attempt in range(self.max_retries):
                   try:
                       result = await self.llm_client.chat(llm_input)
                       response = result["llm_response"]
                       
                       # Memory-aware: update context
                       if conversation_id:
                           await self.update_context(conversation_id, f"User: {input.message}\nAssistant: {response}")
                       
                       return response
                       
                   except Exception as e:
                       if attempt < self.max_retries - 1:
                           await asyncio.sleep(2 ** attempt)
                       else:
                           raise e
           
           except Exception as e:
               return self.fallback_response

**Benefits of Pattern Combination:**
- Increased robustness and reliability
- More sophisticated behavior
- Better user experience
- Easier maintenance and testing

Next Steps
----------

- :doc:`base-agent` - Understand the foundation all patterns build on
- :doc:`creating-agents` - Step-by-step implementation guide
- :doc:`tools-and-callables` - Deep dive into tool integration
- :doc:`examples/index` - See patterns implemented in real examples