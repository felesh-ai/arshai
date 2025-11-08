Building a Custom Agentic System
==================================

This tutorial teaches you how to build a custom multi-agent system from scratch using Arshai's core building blocks. You'll create a customer support system with specialized agents, custom orchestration, and advanced coordination patterns.

**What You'll Build:**

- Multi-agent customer support system
- Custom orchestration without workflows
- Agent specialization and routing
- State management and conversation tracking
- Complete production-ready application

**What You'll Learn:**

- Multi-agent system design
- Custom orchestration patterns
- Agent coordination strategies
- Building without framework constraints
- Production deployment patterns

**Prerequisites:**

- Completion of :doc:`simple-chatbot` tutorial
- Python 3.9+
- Arshai installed: ``pip install arshai[openai]``
- Understanding of async Python

**Time to Complete:** 90-120 minutes

System Architecture
-------------------

We'll build a customer support system with:

**Agents:**

1. **Triage Agent** - Analyzes requests and routes to specialists
2. **Technical Support Agent** - Handles technical issues
3. **Billing Agent** - Handles billing and account questions
4. **General Support Agent** - Handles general inquiries
5. **Escalation Agent** - Manages escalated cases

**Custom Orchestration:**

- Dynamic routing based on request analysis
- Multi-turn conversations with context
- Automatic escalation handling
- State management across agents

Project Setup
-------------

Create Project Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mkdir custom-support-system
   cd custom-support-system

   python -m venv venv
   source venv/bin/activate

   pip install arshai[openai] python-dotenv

   # Create structure
   mkdir -p agents orchestration utils
   touch .env
   touch main.py
   touch agents/{__init__,triage,technical,billing,general,escalation}.py
   touch orchestration/{__init__,coordinator,state_manager}.py
   touch utils/{__init__,conversation,analytics}.py

Step 1: Define System State
----------------------------

Create state management:

.. code-block:: python

   # orchestration/state_manager.py
   from dataclasses import dataclass, field
   from typing import Dict, List, Any, Optional
   from datetime import datetime
   from enum import Enum

   class RequestPriority(Enum):
       """Request priority levels."""
       LOW = "low"
       MEDIUM = "medium"
       HIGH = "high"
       CRITICAL = "critical"

   class RequestStatus(Enum):
       """Request processing status."""
       NEW = "new"
       IN_PROGRESS = "in_progress"
       RESOLVED = "resolved"
       ESCALATED = "escalated"

   @dataclass
   class ConversationTurn:
       """Single conversation turn."""
       timestamp: datetime
       agent_type: str
       user_message: str
       agent_response: str
       metadata: Dict[str, Any] = field(default_factory=dict)

   @dataclass
   class RequestState:
       """State for a customer support request."""
       request_id: str
       user_id: str
       initial_message: str
       category: Optional[str] = None
       priority: RequestPriority = RequestPriority.MEDIUM
       status: RequestStatus = RequestStatus.NEW
       current_agent: Optional[str] = None
       conversation_history: List[ConversationTurn] = field(default_factory=list)
       escalation_reason: Optional[str] = None
       resolution_summary: Optional[str] = None
       metadata: Dict[str, Any] = field(default_factory=dict)
       created_at: datetime = field(default_factory=datetime.now)
       updated_at: datetime = field(default_factory=datetime.now)

       def add_turn(
           self,
           agent_type: str,
           user_message: str,
           agent_response: str,
           **metadata
       ):
           """Add conversation turn to history."""
           turn = ConversationTurn(
               timestamp=datetime.now(),
               agent_type=agent_type,
               user_message=user_message,
               agent_response=agent_response,
               metadata=metadata
           )
           self.conversation_history.append(turn)
           self.updated_at = datetime.now()

       def get_conversation_context(self, last_n: int = 5) -> str:
           """Get recent conversation context."""
           recent_turns = self.conversation_history[-last_n:]
           context_lines = []

           for turn in recent_turns:
               context_lines.append(f"User: {turn.user_message}")
               context_lines.append(f"{turn.agent_type}: {turn.agent_response}")

           return "\n".join(context_lines)

   class StateManager:
       """Manages request states."""

       def __init__(self):
           self.states: Dict[str, RequestState] = {}

       def create_request(
           self,
           user_id: str,
           initial_message: str
       ) -> RequestState:
           """Create new request state."""
           from uuid import uuid4

           request_id = f"req_{uuid4().hex[:8]}"

           state = RequestState(
               request_id=request_id,
               user_id=user_id,
               initial_message=initial_message
           )

           self.states[request_id] = state
           return state

       def get_request(self, request_id: str) -> Optional[RequestState]:
           """Get request state."""
           return self.states.get(request_id)

       def update_request(self, state: RequestState):
           """Update request state."""
           state.updated_at = datetime.now()
           self.states[state.request_id] = state

       def get_active_requests(self) -> List[RequestState]:
           """Get all active requests."""
           return [
               s for s in self.states.values()
               if s.status in [RequestStatus.NEW, RequestStatus.IN_PROGRESS]
           ]

Step 2: Build Specialized Agents
---------------------------------

Create Triage Agent:

.. code-block:: python

   # agents/triage.py
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.core.interfaces.illm import ILLMInput
   from typing import Dict, Any

   class TriageAgent(BaseAgent):
       """Analyzes requests and routes to specialists."""

       def __init__(self, llm_client):
           super().__init__(
               llm_client,
               system_prompt="""You are a customer support triage specialist.
               Analyze customer requests and categorize them.

               Categories:
               - technical: Technical issues, bugs, errors, system problems
               - billing: Payment, invoices, subscriptions, refunds
               - general: General questions, information, how-to
               - escalation: Urgent issues, complaints, VIP customers

               Also determine priority: low, medium, high, or critical.

               Respond with JSON format:
               {
                   "category": "technical|billing|general|escalation",
                   "priority": "low|medium|high|critical",
                   "reasoning": "brief explanation",
                   "suggested_agent": "TechnicalSupport|Billing|GeneralSupport|Escalation"
               }"""
           )

       async def analyze_request(self, message: str) -> Dict[str, Any]:
           """Analyze request and return routing decision."""
           import json

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=f"Analyze this customer request:\n{message}"
           )

           result = await self.llm_client.chat(llm_input)
           response = result.get('llm_response', '{}')

           try:
               # Parse JSON response
               analysis = json.loads(response)
               return analysis
           except json.JSONDecodeError:
               # Fallback to general if parsing fails
               return {
                   "category": "general",
                   "priority": "medium",
                   "reasoning": "Unable to parse analysis",
                   "suggested_agent": "GeneralSupport"
               }

Create Technical Support Agent:

.. code-block:: python

   # agents/technical.py
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.illm import ILLMInput

   class TechnicalSupportAgent(BaseAgent):
       """Handles technical support requests."""

       def __init__(self, llm_client):
           super().__init__(
               llm_client,
               system_prompt="""You are a technical support specialist.
               Help customers with technical issues, bugs, and system problems.

               Guidelines:
               - Ask clarifying questions when needed
               - Provide step-by-step troubleshooting
               - Explain technical concepts clearly
               - Escalate if issue is beyond your scope
               - Always be patient and professional"""
           )

       async def handle_request(
           self,
           message: str,
           context: str = ""
       ) -> dict:
           """Handle technical support request."""

           enhanced_prompt = message
           if context:
               enhanced_prompt = f"""Previous conversation:
   {context}

   Current request: {message}"""

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=enhanced_prompt
           )

           result = await self.llm_client.chat(llm_input)

           return {
               'response': result.get('llm_response', ''),
               'agent_type': 'TechnicalSupport',
               'needs_escalation': self._check_escalation_needed(
                   result.get('llm_response', '')
               )
           }

       def _check_escalation_needed(self, response: str) -> bool:
           """Check if response indicates need for escalation."""
           escalation_phrases = [
               'escalate',
               'beyond my scope',
               'senior engineer',
               'cannot resolve'
           ]

           response_lower = response.lower()
           return any(phrase in response_lower for phrase in escalation_phrases)

Create Billing Agent:

.. code-block:: python

   # agents/billing.py
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.illm import ILLMInput

   class BillingAgent(BaseAgent):
       """Handles billing and account questions."""

       def __init__(self, llm_client):
           super().__init__(
               llm_client,
               system_prompt="""You are a billing specialist.
               Help customers with:
               - Payment questions
               - Invoice inquiries
               - Subscription management
               - Refund requests
               - Account upgrades/downgrades

               Guidelines:
               - Be clear about policies
               - Verify account details (simulate)
               - Explain charges clearly
               - Process refunds when appropriate
               - Escalate complex financial issues"""
           )

       async def handle_request(
           self,
           message: str,
           context: str = ""
       ) -> dict:
           """Handle billing request."""

           enhanced_prompt = message
           if context:
               enhanced_prompt = f"""Previous conversation:
   {context}

   Current request: {message}"""

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=enhanced_prompt
           )

           result = await self.llm_client.chat(llm_input)

           return {
               'response': result.get('llm_response', ''),
               'agent_type': 'Billing',
               'needs_escalation': 'refund' in message.lower() and 'large' in message.lower()
           }

Step 3: Build Custom Orchestrator
----------------------------------

Create the orchestrator:

.. code-block:: python

   # orchestration/coordinator.py
   import asyncio
   from typing import Dict, Any, Optional
   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig

   from agents.triage import TriageAgent
   from agents.technical import TechnicalSupportAgent
   from agents.billing import BillingAgent
   from agents.general import GeneralSupportAgent
   from agents.escalation import EscalationAgent
   from orchestration.state_manager import StateManager, RequestPriority, RequestStatus

   class SupportSystemCoordinator:
       """Custom orchestrator for multi-agent support system."""

       def __init__(self, llm_client):
           """Initialize coordinator with all agents."""
           self.llm_client = llm_client

           # Initialize all agents
           self.triage_agent = TriageAgent(llm_client)
           self.technical_agent = TechnicalSupportAgent(llm_client)
           self.billing_agent = BillingAgent(llm_client)
           self.general_agent = GeneralSupportAgent(llm_client)
           self.escalation_agent = EscalationAgent(llm_client)

           # State manager
           self.state_manager = StateManager()

           # Agent routing map
           self.agent_map = {
               'TechnicalSupport': self.technical_agent,
               'Billing': self.billing_agent,
               'GeneralSupport': self.general_agent,
               'Escalation': self.escalation_agent
           }

       async def process_new_request(
           self,
           user_id: str,
           message: str
       ) -> Dict[str, Any]:
           """Process new customer support request."""

           # Create request state
           state = self.state_manager.create_request(user_id, message)

           # Triage the request
           analysis = await self.triage_agent.analyze_request(message)

           # Update state with triage results
           state.category = analysis.get('category')
           state.priority = RequestPriority(analysis.get('priority', 'medium'))
           state.status = RequestStatus.IN_PROGRESS

           # Route to appropriate agent
           suggested_agent = analysis.get('suggested_agent', 'GeneralSupport')
           state.current_agent = suggested_agent

           # Get initial response from specialist
           specialist_response = await self._route_to_agent(
               suggested_agent,
               message,
               state
           )

           # Add to conversation history
           state.add_turn(
               agent_type=suggested_agent,
               user_message=message,
               agent_response=specialist_response['response']
           )

           # Check for escalation
           if specialist_response.get('needs_escalation'):
               await self._handle_escalation(state, "Agent requested escalation")

           # Update state
           self.state_manager.update_request(state)

           return {
               'request_id': state.request_id,
               'response': specialist_response['response'],
               'category': state.category,
               'priority': state.priority.value,
               'current_agent': state.current_agent,
               'status': state.status.value
           }

       async def continue_conversation(
           self,
           request_id: str,
           message: str
       ) -> Dict[str, Any]:
           """Continue existing conversation."""

           # Get request state
           state = self.state_manager.get_request(request_id)
           if not state:
               return {'error': 'Request not found'}

           # Get conversation context
           context = state.get_conversation_context(last_n=5)

           # Route to current agent
           agent_response = await self._route_to_agent(
               state.current_agent,
               message,
               state,
               context
           )

           # Add to conversation history
           state.add_turn(
               agent_type=state.current_agent,
               user_message=message,
               agent_response=agent_response['response']
           )

           # Check for escalation
           if agent_response.get('needs_escalation'):
               await self._handle_escalation(state, "Complex issue detected")

           # Update state
           self.state_manager.update_request(state)

           return {
               'request_id': state.request_id,
               'response': agent_response['response'],
               'current_agent': state.current_agent,
               'status': state.status.value,
               'conversation_turns': len(state.conversation_history)
           }

       async def _route_to_agent(
           self,
           agent_type: str,
           message: str,
           state,
           context: str = ""
       ) -> Dict[str, Any]:
           """Route request to specified agent."""

           agent = self.agent_map.get(agent_type)
           if not agent:
               # Fallback to general support
               agent = self.general_agent

           # Call agent's handle_request method
           return await agent.handle_request(message, context)

       async def _handle_escalation(self, state, reason: str):
           """Handle request escalation."""

           state.status = RequestStatus.ESCALATED
           state.escalation_reason = reason
           state.current_agent = 'Escalation'

           # Get escalation response
           context = state.get_conversation_context()
           escalation_response = await self.escalation_agent.handle_request(
               f"Escalated issue: {reason}\n\nOriginal request: {state.initial_message}",
               context
           )

           state.add_turn(
               agent_type='Escalation',
               user_message=f"[System] Escalated: {reason}",
               agent_response=escalation_response['response']
           )

       def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
           """Get current status of a request."""

           state = self.state_manager.get_request(request_id)
           if not state:
               return None

           return {
               'request_id': state.request_id,
               'user_id': state.user_id,
               'category': state.category,
               'priority': state.priority.value,
               'status': state.status.value,
               'current_agent': state.current_agent,
               'conversation_turns': len(state.conversation_history),
               'created_at': state.created_at.isoformat(),
               'updated_at': state.updated_at.isoformat()
           }

Step 4: Create Main Application
--------------------------------

Build the CLI application:

.. code-block:: python

   # main.py
   import asyncio
   import os
   from dotenv import load_dotenv

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig
   from orchestration.coordinator import SupportSystemCoordinator

   load_dotenv()

   class SupportSystemCLI:
       """Command-line interface for support system."""

       def __init__(self, coordinator: SupportSystemCoordinator):
           self.coordinator = coordinator
           self.current_request_id = None
           self.user_id = "user_001"  # Simulated user ID

       def print_welcome(self):
           """Print welcome message."""
           print("\n" + "=" * 60)
           print("🎧 Customer Support System")
           print("=" * 60)
           print("\nMulti-Agent Support with:")
           print("  • Intelligent request routing")
           print("  • Specialized support agents")
           print("  • Automatic escalation handling")
           print("\nCommands:")
           print("  /new     - Start a new support request")
           print("  /status  - Check current request status")
           print("  /quit    - Exit the system")
           print("\nDescribe your issue to get started!\n")

       async def run(self):
           """Run the CLI."""
           self.print_welcome()

           while True:
               try:
                   user_input = input("\n💬 You: ").strip()

                   if not user_input:
                       continue

                   # Handle commands
                   if user_input.startswith('/'):
                       await self.handle_command(user_input.lower())
                       continue

                   # Process message
                   await self.handle_message(user_input)

               except KeyboardInterrupt:
                   print("\n\n👋 Thank you for contacting support!")
                   break
               except Exception as e:
                   print(f"\n❌ Error: {e}")

       async def handle_command(self, command: str):
           """Handle system commands."""
           if command == '/quit':
               print("\n👋 Thank you for contacting support!")
               exit(0)

           elif command == '/new':
               self.current_request_id = None
               print("\n✓ Ready for new support request")

           elif command == '/status':
               if self.current_request_id:
                   status = self.coordinator.get_request_status(
                       self.current_request_id
                   )
                   if status:
                       self._print_status(status)
               else:
                   print("\n⚠️  No active request")

           else:
               print(f"❌ Unknown command: {command}")

       async def handle_message(self, message: str):
           """Handle user message."""
           try:
               print("🤔 Processing...", end="", flush=True)

               if self.current_request_id:
                   # Continue existing conversation
                   result = await self.coordinator.continue_conversation(
                       self.current_request_id,
                       message
                   )
               else:
                   # New request
                   result = await self.coordinator.process_new_request(
                       self.user_id,
                       message
                   )
                   self.current_request_id = result.get('request_id')

               # Clear processing message
               print("\r" + " " * 40 + "\r", end="")

               # Display response
               agent = result.get('current_agent', 'Support')
               response = result.get('response', '')

               print(f"🎧 {agent}: {response}\n")

               # Show status indicators
               if 'category' in result:
                   print(f"📋 Category: {result['category']} | "
                         f"Priority: {result['priority']} | "
                         f"Status: {result['status']}")

           except Exception as e:
               print(f"\n❌ Error: {e}")

       def _print_status(self, status: Dict):
           """Print request status."""
           print("\n" + "=" * 60)
           print(f"Request ID: {status['request_id']}")
           print(f"Category: {status['category']}")
           print(f"Priority: {status['priority']}")
           print(f"Status: {status['status']}")
           print(f"Current Agent: {status['current_agent']}")
           print(f"Conversation Turns: {status['conversation_turns']}")
           print("=" * 60)

   async def main():
       """Main application entry point."""

       # Check API key
       if not os.getenv("OPENAI_API_KEY"):
           print("❌ Please set OPENAI_API_KEY environment variable")
           return

       # Create LLM client
       llm_client = OpenAIClient(
           ILLMConfig(
               model="gpt-3.5-turbo",
               temperature=0.7,
               max_tokens=500
           )
       )

       # Create coordinator
       coordinator = SupportSystemCoordinator(llm_client)

       # Run CLI
       cli = SupportSystemCLI(coordinator)
       await cli.run()

   if __name__ == "__main__":
       asyncio.run(main())

Step 5: Add Remaining Agents
-----------------------------

General Support Agent:

.. code-block:: python

   # agents/general.py
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.illm import ILLMInput

   class GeneralSupportAgent(BaseAgent):
       """Handles general inquiries."""

       def __init__(self, llm_client):
           super().__init__(
               llm_client,
               system_prompt="""You are a general customer support agent.
               Help with general questions, product information, and how-to guides.
               Be friendly, informative, and professional."""
           )

       async def handle_request(self, message: str, context: str = "") -> dict:
           """Handle general support request."""
           enhanced_prompt = f"{context}\n\nCurrent: {message}" if context else message

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=enhanced_prompt
           )

           result = await self.llm_client.chat(llm_input)

           return {
               'response': result.get('llm_response', ''),
               'agent_type': 'GeneralSupport',
               'needs_escalation': False
           }

Escalation Agent:

.. code-block:: python

   # agents/escalation.py
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.illm import ILLMInput

   class EscalationAgent(BaseAgent):
       """Handles escalated cases."""

       def __init__(self, llm_client):
           super().__init__(
               llm_client,
               system_prompt="""You are a senior support specialist handling escalated cases.
               You have authority to:
               - Make exceptions to standard policies
               - Offer compensation when appropriate
               - Involve engineering or management
               - Provide priority support

               Be empathetic, take ownership, and resolve issues decisively."""
           )

       async def handle_request(self, message: str, context: str = "") -> dict:
           """Handle escalated request."""
           enhanced_prompt = f"{context}\n\nEscalated Issue: {message}" if context else message

           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=enhanced_prompt
           )

           result = await self.llm_client.chat(llm_input)

           return {
               'response': result.get('llm_response', ''),
               'agent_type': 'Escalation',
               'needs_escalation': False  # Already escalated
           }

Step 6: Test the System
------------------------

Run the application:

.. code-block:: bash

   python main.py

Test scenarios:

.. code-block:: text

   🎧 Customer Support System
   ============================================================

   💬 You: My payment failed and I can't access my account

   🎧 Billing: I'm sorry to hear you're having trouble accessing your
   account due to a payment issue. Let me help you resolve this...

   📋 Category: billing | Priority: high | Status: in_progress

   💬 You: I need this fixed immediately, I have a deadline!

   🎧 Escalation: I understand the urgency of your situation and I'm
   here to help immediately. Let me personally ensure we resolve this
   right away...

   📋 Category: billing | Priority: critical | Status: escalated

Step 7: Add Analytics and Monitoring
-------------------------------------

Create analytics module:

.. code-block:: python

   # utils/analytics.py
   from typing import Dict, List
   from collections import Counter
   from datetime import datetime, timedelta

   class SupportAnalytics:
       """Analytics for support system."""

       def __init__(self, state_manager):
           self.state_manager = state_manager

       def get_metrics(self) -> Dict:
           """Get system metrics."""
           states = self.state_manager.states.values()

           return {
               'total_requests': len(states),
               'by_status': self._count_by_status(states),
               'by_category': self._count_by_category(states),
               'by_priority': self._count_by_priority(states),
               'avg_response_time': self._calc_avg_response_time(states),
               'escalation_rate': self._calc_escalation_rate(states)
           }

       def _count_by_status(self, states) -> Dict:
           """Count requests by status."""
           statuses = [s.status.value for s in states]
           return dict(Counter(statuses))

       def _count_by_category(self, states) -> Dict:
           """Count requests by category."""
           categories = [s.category for s in states if s.category]
           return dict(Counter(categories))

       def _count_by_priority(self, states) -> Dict:
           """Count requests by priority."""
           priorities = [s.priority.value for s in states]
           return dict(Counter(priorities))

       def _calc_avg_response_time(self, states) -> float:
           """Calculate average response time."""
           times = []
           for state in states:
               if state.conversation_history:
                   first_turn = state.conversation_history[0]
                   response_time = (first_turn.timestamp - state.created_at).total_seconds()
                   times.append(response_time)

           return sum(times) / len(times) if times else 0.0

       def _calc_escalation_rate(self, states) -> float:
           """Calculate escalation rate."""
           from orchestration.state_manager import RequestStatus

           total = len(states)
           if total == 0:
               return 0.0

           escalated = sum(1 for s in states if s.status == RequestStatus.ESCALATED)
           return (escalated / total) * 100

Production Enhancements
-----------------------

Add Persistent Storage:

.. code-block:: python

   import json
   from pathlib import Path

   class PersistentStateManager(StateManager):
       """State manager with disk persistence."""

       def __init__(self, storage_path: str = "data/requests.json"):
           super().__init__()
           self.storage_path = Path(storage_path)
           self.storage_path.parent.mkdir(exist_ok=True)
           self._load_states()

       def _load_states(self):
           """Load states from disk."""
           if self.storage_path.exists():
               with open(self.storage_path, 'r') as f:
                   data = json.load(f)
                   # Deserialize states
                   # Implementation details...

       def _save_states(self):
           """Save states to disk."""
           with open(self.storage_path, 'w') as f:
               # Serialize states
               json.dump(self._serialize_states(), f, indent=2)

       def update_request(self, state):
           """Update and persist request."""
           super().update_request(state)
           self._save_states()

Add Real-time Notifications:

.. code-block:: python

   class NotificationService:
       """Send notifications for important events."""

       async def notify_escalation(self, request_state):
           """Notify team of escalation."""
           print(f"🚨 ESCALATION: Request {request_state.request_id}")
           print(f"   Priority: {request_state.priority.value}")
           print(f"   Reason: {request_state.escalation_reason}")

       async def notify_resolution(self, request_state):
           """Notify of request resolution."""
           print(f"✅ RESOLVED: Request {request_state.request_id}")

Add Performance Monitoring:

.. code-block:: python

   import time
   from functools import wraps

   def monitor_performance(func):
       """Decorator to monitor function performance."""
       @wraps(func)
       async def wrapper(*args, **kwargs):
           start_time = time.time()
           result = await func(*args, **kwargs)
           duration = time.time() - start_time

           print(f"⏱️  {func.__name__}: {duration:.2f}s")
           return result

       return wrapper

   # Use on coordinator methods:
   @monitor_performance
   async def process_new_request(self, user_id, message):
       # ... implementation

Next Steps
----------

**Enhance the system:**

- Add more specialized agents (sales, feedback, etc.)
- Implement agent collaboration (multi-agent consultation)
- Add natural language understanding for better routing
- Implement learning from past interactions

**Scale for production:**

- Use Redis for state management: :doc:`../implementations/memory/redis-memory`
- Add authentication and user management
- Implement real notification system (email, SMS, Slack)
- Add comprehensive logging and monitoring
- Deploy with load balancing

**Learn more:**

- :doc:`../implementations/orchestration/building-your-own` - More orchestration patterns
- :doc:`../framework/building-systems/index` - System building concepts
- :doc:`../implementations/agents/index` - Agent implementation patterns

Congratulations! You've built a complete custom multi-agent system from scratch! 🎉
