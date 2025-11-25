Stateless Agent Design
======================

Stateless design is a fundamental principle in Arshai's agent architecture. This guide explains why agents should be stateless, how to implement stateless patterns, and how to handle state when you need it.

Why Stateless Agents?
---------------------

**Predictable Behavior**
   Stateless agents always produce the same output for the same input, making them easier to test and debug.

**Scalability**
   Multiple instances can handle requests independently without coordination or shared state management.

**Reliability**
   No risk of state corruption, memory leaks, or inconsistent state between requests.

**Testability**
   Easy to unit test because there's no hidden state to set up or clean up.

**Thread Safety**
   Multiple concurrent requests can safely use the same agent instance.

**Deployment Flexibility**
   Agents can be deployed across multiple processes, machines, or containers without state synchronization.

What Makes an Agent Stateless?
-------------------------------

A stateless agent:

- **Doesn't store request data** between ``process()`` calls
- **Doesn't maintain conversation history** internally
- **Doesn't accumulate user data** over time
- **Doesn't modify instance variables** during processing
- **Derives all context** from input parameters

Stateless vs Stateful Examples
-------------------------------

**❌ Stateful Agent (Don't Do This):**

.. code-block:: python

   class StatefulAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str):
           super().__init__(llm_client, system_prompt)
           self.conversation_history = []  # ❌ Stores state
           self.user_preferences = {}      # ❌ Stores state
           self.request_count = 0          # ❌ Stores state
       
       async def process(self, input: IAgentInput) -> str:
           # ❌ Modifying instance state
           self.request_count += 1
           self.conversation_history.append(input.message)
           
           user_id = input.metadata.get("user_id")
           if user_id not in self.user_preferences:
               self.user_preferences[user_id] = {"tone": "casual"}
           
           # ❌ Using stored state in processing
           history_context = "\n".join(self.conversation_history[-5:])
           tone = self.user_preferences[user_id]["tone"]
           
           llm_input = ILLMInput(
               system_prompt=f"{self.system_prompt}\nTone: {tone}\nHistory: {history_context}",
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Problems with the above:**
- Conversation history grows indefinitely (memory leak)
- Multiple users share the same agent instance (data leakage)
- Concurrent requests interfere with each other
- State becomes inconsistent under load
- Impossible to scale horizontally

**✅ Stateless Agent (Correct Approach):**

.. code-block:: python

   class StatelessAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str):
           super().__init__(llm_client, system_prompt)
           # ✅ No state variables
       
       async def process(self, input: IAgentInput) -> str:
           # ✅ All context comes from input
           user_id = input.metadata.get("user_id")
           conversation_id = input.metadata.get("conversation_id")
           user_preferences = input.metadata.get("preferences", {})
           conversation_history = input.metadata.get("history", [])
           
           # ✅ Build context from input data
           tone = user_preferences.get("tone", "professional")
           history_context = "\n".join(conversation_history[-5:]) if conversation_history else ""
           
           # ✅ Create enhanced system prompt
           enhanced_prompt = self.system_prompt
           if tone:
               enhanced_prompt += f"\nTone: {tone}"
           if history_context:
               enhanced_prompt += f"\nRecent conversation:\n{history_context}"
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

Handling State in Stateless Design
-----------------------------------

**1. Pass State in Input Metadata**

.. code-block:: python

   # When calling the agent, include all needed context
   input_data = IAgentInput(
       message="What's my account balance?",
       metadata={
           "user_id": "user123",
           "session_id": "session456",
           "user_preferences": {"currency": "USD", "language": "en"},
           "conversation_history": [
               "User: Hello",
               "Assistant: Hi! How can I help you?",
               "User: I'd like to check my accounts"
           ],
           "account_context": {
               "primary_account": "checking_001",
               "accounts": ["checking_001", "savings_002"]
           }
       }
   )
   
   response = await agent.process(input_data)

**2. External State Management**

.. code-block:: python

   class StatelessAgentWithExternalState(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str, 
                    state_manager=None):
           super().__init__(llm_client, system_prompt)
           self.state_manager = state_manager  # ✅ Dependency injection
       
       async def process(self, input: IAgentInput) -> str:
           user_id = input.metadata.get("user_id")
           conversation_id = input.metadata.get("conversation_id")
           
           # ✅ Retrieve state from external source
           user_context = None
           conversation_history = None
           
           if self.state_manager and user_id:
               user_context = await self.state_manager.get_user_context(user_id)
           
           if self.state_manager and conversation_id:
               conversation_history = await self.state_manager.get_conversation_history(conversation_id)
           
           # ✅ Build context from retrieved state
           enhanced_prompt = self.system_prompt
           if user_context:
               enhanced_prompt += f"\nUser preferences: {user_context.get('preferences', {})}"
           if conversation_history:
               recent_history = conversation_history[-5:]
               enhanced_prompt += f"\nRecent conversation:\n{recent_history}"
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           response = result["llm_response"]
           
           # ✅ Update external state (don't store in agent)
           if self.state_manager and conversation_id:
               await self.state_manager.add_to_conversation(
                   conversation_id, 
                   f"User: {input.message}\nAssistant: {response}"
               )
           
           return response

**3. Immutable Configuration**

Store configuration that never changes during agent lifetime:

.. code-block:: python

   class ConfiguredStatelessAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str, 
                    max_tokens: int = 500, temperature: float = 0.7,
                    available_tools: dict = None):
           super().__init__(llm_client, system_prompt)
           
           # ✅ Immutable configuration (set once, never changed)
           self.max_tokens = max_tokens
           self.temperature = temperature
           self.available_tools = available_tools or {}
       
       async def process(self, input: IAgentInput) -> str:
           # ✅ Configuration is read-only during processing
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=input.message,
               max_tokens=self.max_tokens,
               temperature=self.temperature,
               regular_functions=self.available_tools
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

State Management Patterns
--------------------------

**Pattern 1: Session-Based State**

Use external session storage for user-specific state:

.. code-block:: python

   class SessionAwareAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str, session_store=None):
           super().__init__(llm_client, system_prompt)
           self.session_store = session_store
       
       async def get_session_context(self, session_id: str) -> dict:
           """Retrieve session context from external store."""
           if not self.session_store or not session_id:
               return {}
           
           try:
               return await self.session_store.get(session_id) or {}
           except Exception:
               return {}
       
       async def update_session_context(self, session_id: str, updates: dict):
           """Update session context in external store."""
           if not self.session_store or not session_id:
               return
           
           try:
               current_context = await self.get_session_context(session_id)
               current_context.update(updates)
               await self.session_store.set(session_id, current_context)
           except Exception:
               pass  # Handle gracefully
       
       async def process(self, input: IAgentInput) -> str:
           session_id = input.metadata.get("session_id")
           
           # Get current session state
           session_context = await self.get_session_context(session_id)
           
           # Build enhanced context
           enhanced_prompt = self.system_prompt
           if session_context.get("user_preferences"):
               enhanced_prompt += f"\nUser preferences: {session_context['user_preferences']}"
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           response = result["llm_response"]
           
           # Update session state
           if session_id:
               updates = {
                   "last_interaction": {
                       "message": input.message,
                       "response": response,
                       "timestamp": datetime.utcnow().isoformat()
                   }
               }
               await self.update_session_context(session_id, updates)
           
           return response

**Pattern 2: Request-Scoped State**

Use the request metadata to pass all necessary state:

.. code-block:: python

   class RequestScopedAgent(BaseAgent):
       """Agent that derives all context from request metadata."""
       
       def extract_context(self, metadata: dict) -> dict:
           """Extract and validate context from request metadata."""
           return {
               "user_id": metadata.get("user_id"),
               "conversation_id": metadata.get("conversation_id"),
               "user_preferences": metadata.get("preferences", {}),
               "conversation_history": metadata.get("history", []),
               "domain_context": metadata.get("domain_context", {}),
               "request_context": metadata.get("request_context", {})
           }
       
       def build_enhanced_prompt(self, context: dict) -> str:
           """Build system prompt with all context."""
           prompt = self.system_prompt
           
           if context["user_preferences"]:
               prompt += f"\n\nUser Preferences:\n{context['user_preferences']}"
           
           if context["domain_context"]:
               prompt += f"\n\nDomain Context:\n{context['domain_context']}"
           
           if context["conversation_history"]:
               recent_history = context["conversation_history"][-3:]
               prompt += f"\n\nRecent Conversation:\n{chr(10).join(recent_history)}"
           
           return prompt
       
       async def process(self, input: IAgentInput) -> str:
           # Extract all context from request
           context = self.extract_context(input.metadata or {})
           
           # Build enhanced prompt
           enhanced_prompt = self.build_enhanced_prompt(context)
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           return result["llm_response"]

**Pattern 3: Database-Backed State**

Use database queries for persistent state:

.. code-block:: python

   class DatabaseBackedAgent(BaseAgent):
       def __init__(self, llm_client: ILLM, system_prompt: str, db_connection=None):
           super().__init__(llm_client, system_prompt)
           self.db = db_connection
       
       async def get_user_profile(self, user_id: str) -> dict:
           """Retrieve user profile from database."""
           if not self.db or not user_id:
               return {}
           
           try:
               async with self.db.cursor() as cursor:
                   await cursor.execute(
                       "SELECT preferences, context FROM user_profiles WHERE user_id = %s",
                       (user_id,)
                   )
                   result = await cursor.fetchone()
                   return result or {}
           except Exception:
               return {}
       
       async def get_conversation_summary(self, conversation_id: str) -> str:
           """Get conversation summary from database."""
           if not self.db or not conversation_id:
               return ""
           
           try:
               async with self.db.cursor() as cursor:
                   await cursor.execute(
                       "SELECT summary FROM conversation_summaries WHERE conversation_id = %s",
                       (conversation_id,)
                   )
                   result = await cursor.fetchone()
                   return result.get("summary", "") if result else ""
           except Exception:
               return ""
       
       async def update_conversation_summary(self, conversation_id: str, 
                                           interaction: str):
           """Update conversation summary in database."""
           if not self.db or not conversation_id:
               return
           
           try:
               # This would typically use a smarter summarization strategy
               async with self.db.cursor() as cursor:
                   await cursor.execute(
                       """
                       INSERT INTO conversation_summaries (conversation_id, summary, updated_at)
                       VALUES (%s, %s, NOW())
                       ON DUPLICATE KEY UPDATE 
                       summary = CONCAT(summary, '\n', VALUES(summary)),
                       updated_at = NOW()
                       """,
                       (conversation_id, interaction)
                   )
           except Exception:
               pass  # Handle gracefully
       
       async def process(self, input: IAgentInput) -> str:
           user_id = input.metadata.get("user_id")
           conversation_id = input.metadata.get("conversation_id")
           
           # Retrieve state from database
           user_profile = await self.get_user_profile(user_id)
           conversation_summary = await self.get_conversation_summary(conversation_id)
           
           # Build enhanced prompt
           enhanced_prompt = self.system_prompt
           if user_profile:
               enhanced_prompt += f"\nUser Profile: {user_profile}"
           if conversation_summary:
               enhanced_prompt += f"\nConversation Context: {conversation_summary}"
           
           llm_input = ILLMInput(
               system_prompt=enhanced_prompt,
               user_message=input.message
           )
           
           result = await self.llm_client.chat(llm_input)
           response = result["llm_response"]
           
           # Update database state
           if conversation_id:
               interaction = f"User: {input.message}\nAssistant: {response}"
               await self.update_conversation_summary(conversation_id, interaction)
           
           return response

Testing Stateless Agents
-------------------------

Stateless agents are much easier to test:

.. code-block:: python

   import pytest
   from unittest.mock import AsyncMock

   @pytest.mark.asyncio
   async def test_stateless_agent_deterministic():
       """Test that agent produces same output for same input."""
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Hello! How can I help you?",
           "usage": {"total_tokens": 20}
       }
       
       agent = StatelessAgent(mock_llm, "You are a helpful assistant")
       
       input_data = IAgentInput(
           message="Hello",
           metadata={"user_id": "test123"}
       )
       
       # Multiple calls should be identical
       response1 = await agent.process(input_data)
       response2 = await agent.process(input_data)
       
       assert response1 == response2
       assert mock_llm.chat.call_count == 2

   @pytest.mark.asyncio
   async def test_stateless_agent_no_interference():
       """Test that multiple requests don't interfere."""
       mock_llm = AsyncMock()
       mock_llm.chat.return_value = {
           "llm_response": "Response",
           "usage": {"total_tokens": 10}
       }
       
       agent = StatelessAgent(mock_llm, "You are a helpful assistant")
       
       # Simulate concurrent requests
       tasks = [
           agent.process(IAgentInput(message=f"Message {i}", metadata={"user_id": f"user{i}"}))
           for i in range(10)
       ]
       
       results = await asyncio.gather(*tasks)
       
       # All should succeed without interference
       assert len(results) == 10
       assert all(result == "Response" for result in results)

   @pytest.mark.asyncio
   async def test_stateless_agent_with_context():
       """Test agent with different context produces different outputs."""
       mock_llm = AsyncMock()
       mock_llm.chat.side_effect = [
           {"llm_response": "Formal response", "usage": {"total_tokens": 15}},
           {"llm_response": "Casual response", "usage": {"total_tokens": 15}}
       ]
       
       agent = StatelessAgent(mock_llm, "You are an assistant")
       
       # Formal context
       formal_response = await agent.process(IAgentInput(
           message="Hello",
           metadata={"preferences": {"tone": "formal"}}
       ))
       
       # Casual context
       casual_response = await agent.process(IAgentInput(
           message="Hello",
           metadata={"preferences": {"tone": "casual"}}
       ))
       
       assert formal_response != casual_response
       
       # Verify different system prompts were used
       call_args = mock_llm.chat.call_args_list
       assert "formal" in call_args[0][0][0].system_prompt
       assert "casual" in call_args[1][0][0].system_prompt

Common Pitfalls and Solutions
-----------------------------

**Pitfall 1: Accidental State Storage**

.. code-block:: python

   # ❌ Easy to accidentally store state
   class AccidentallyStatefulAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt):
           super().__init__(llm_client, system_prompt)
           self.cache = {}  # ❌ This becomes shared state
       
       async def process(self, input: IAgentInput) -> str:
           user_id = input.metadata.get("user_id")
           
           # ❌ Storing user-specific data in instance
           if user_id not in self.cache:
               self.cache[user_id] = []
           self.cache[user_id].append(input.message)
           
           # ... rest of processing

   # ✅ Solution: Use external storage or pass in metadata
   class StatelessCachingAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt, cache_manager=None):
           super().__init__(llm_client, system_prompt)
           self.cache_manager = cache_manager  # External dependency
       
       async def process(self, input: IAgentInput) -> str:
           user_id = input.metadata.get("user_id")
           
           # ✅ Use external cache
           if self.cache_manager and user_id:
               await self.cache_manager.add_to_user_cache(user_id, input.message)

**Pitfall 2: Mutable Default Arguments**

.. code-block:: python

   # ❌ Mutable defaults create shared state
   class BadDefaultAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt, tools={}):  # ❌ Mutable default
           super().__init__(llm_client, system_prompt)
           self.tools = tools  # All instances share the same dict
       
       async def process(self, input: IAgentInput) -> str:
           # ❌ Modifying shared state
           self.tools["dynamic_tool"] = lambda: "new tool"

   # ✅ Solution: Use None and create new instances
   class GoodDefaultAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt, tools=None):  # ✅ Immutable default
           super().__init__(llm_client, system_prompt)
           self.tools = tools or {}  # Each instance gets its own dict

**Pitfall 3: Shared External Resources**

.. code-block:: python

   # ❌ Modifying shared resources
   class SharedResourceAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt, shared_config):
           super().__init__(llm_client, system_prompt)
           self.config = shared_config  # Shared reference
       
       async def process(self, input: IAgentInput) -> str:
           # ❌ Modifying shared configuration
           self.config["last_user"] = input.metadata.get("user_id")

   # ✅ Solution: Treat external resources as read-only
   class ImmutableResourceAgent(BaseAgent):
       def __init__(self, llm_client, system_prompt, config):
           super().__init__(llm_client, system_prompt)
           self.config = config  # Read-only reference
       
       async def process(self, input: IAgentInput) -> str:
           # ✅ Only read from shared configuration
           max_tokens = self.config.get("max_tokens", 500)
           # Don't modify self.config

Benefits of Stateless Design
-----------------------------

**Easier Debugging**
   Issues are isolated to individual requests rather than accumulated state.

**Better Performance**
   No state synchronization overhead or memory leaks from accumulated state.

**Horizontal Scaling**
   Multiple agent instances can handle requests independently.

**Improved Testing**
   Predictable behavior and no test isolation issues.

**Reduced Complexity**
   No need to manage state lifecycles or cleanup.

**Better Error Recovery**
   Errors don't corrupt persistent state affecting future requests.

**Concurrent Safety**
   Multiple requests can safely use the same agent instance.

Next Steps
----------

- :doc:`base-agent` - Understand the foundation of stateless agents
- :doc:`creating-agents` - Implement stateless agents step-by-step
- :doc:`agent-patterns` - See stateless patterns in action
- :doc:`examples/index` - Real examples of stateless agent implementations