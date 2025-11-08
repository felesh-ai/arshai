Building a Simple Chatbot
==========================

This tutorial walks you through building a complete chatbot from scratch using Arshai's core framework. You'll learn how to create a conversational agent with memory, handle user interactions, and deploy it as a working application.

**What You'll Build:**

- A conversational chatbot with memory
- Command handling (help, clear, quit)
- Conversation history tracking
- Simple web interface (optional)

**What You'll Learn:**

- Direct instantiation of framework components
- Memory integration for conversations
- Agent design patterns
- Error handling and user experience

**Prerequisites:**

- Python 3.9+
- Arshai installed: ``pip install arshai[openai]``
- OpenAI API key

**Time to Complete:** 30-45 minutes

Project Setup
-------------

Create Project Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create project directory
   mkdir simple-chatbot
   cd simple-chatbot

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install arshai[openai] python-dotenv

   # Create project files
   touch .env
   touch chatbot.py
   touch requirements.txt

Configure Environment
~~~~~~~~~~~~~~~~~~~~~

Create ``.env`` file:

.. code-block:: bash

   # .env
   OPENAI_API_KEY=your-api-key-here

Create ``requirements.txt``:

.. code-block:: text

   arshai[openai]
   python-dotenv

Step 1: Build the Core Chatbot Agent
-------------------------------------

Create the main chatbot agent:

.. code-block:: python

   # chatbot.py
   import asyncio
   import os
   from datetime import datetime
   from typing import Dict, Any
   from dotenv import load_dotenv

   from arshai.llms.openai import OpenAIClient
   from arshai.core.interfaces.illm import ILLMConfig, ILLMInput
   from arshai.agents.base import BaseAgent
   from arshai.core.interfaces.iagent import IAgentInput
   from arshai.memory.working_memory.in_memory_manager import InMemoryManager
   from arshai.core.interfaces.imemorymanager import IMemoryInput, IWorkingMemory
   from arshai.memory.memory_types import ConversationMemoryType

   # Load environment variables
   load_dotenv()

   class SimpleChatbot(BaseAgent):
       """A simple chatbot with memory and conversation management."""

       def __init__(self, llm_client, memory_manager, conversation_id: str):
           super().__init__(
               llm_client,
               system_prompt="""You are a helpful and friendly AI assistant.
               You remember previous conversations and provide personalized responses.
               Keep your answers concise but informative."""
           )
           self.memory_manager = memory_manager
           self.conversation_id = conversation_id
           self.interaction_count = 0

       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           """Process user message with memory."""
           self.interaction_count += 1

           # Retrieve conversation memory
           memory_context = self._get_conversation_memory()

           # Build enhanced prompt with memory
           enhanced_prompt = self._build_prompt_with_memory(
               input.message,
               memory_context
           )

           # Call LLM
           llm_input = ILLMInput(
               system_prompt=self.system_prompt,
               user_message=enhanced_prompt
           )

           result = await self.llm_client.chat(llm_input)
           response = result.get('llm_response', '')

           # Store conversation turn in memory
           self._store_conversation_turn(input.message, response)

           return {
               "response": response,
               "interaction_count": self.interaction_count,
               "timestamp": datetime.now().isoformat()
           }

       def _get_conversation_memory(self) -> str:
           """Retrieve conversation history from memory."""
           try:
               memory_input = IMemoryInput(
                   conversation_id=self.conversation_id,
                   memory_type=ConversationMemoryType.WORKING_MEMORY
               )

               memories = self.memory_manager.retrieve(memory_input)

               if memories:
                   return memories[0].working_memory
               return ""

           except Exception as e:
               print(f"Error retrieving memory: {e}")
               return ""

       def _store_conversation_turn(self, user_message: str, bot_response: str):
           """Store conversation turn in memory."""
           try:
               # Get existing memory
               existing_memory = self._get_conversation_memory()

               # Append new turn
               conversation_turn = f"\nUser: {user_message}\nAssistant: {bot_response}"

               # Keep last 10 turns to avoid token limits
               memory_lines = existing_memory.split('\n')
               if len(memory_lines) > 40:  # 10 turns * 4 lines average
                   memory_lines = memory_lines[-40:]

               updated_memory = '\n'.join(memory_lines) + conversation_turn

               # Store updated memory
               memory_input = IMemoryInput(
                   conversation_id=self.conversation_id,
                   memory_type=ConversationMemoryType.WORKING_MEMORY,
                   data=[IWorkingMemory(working_memory=updated_memory)]
               )

               self.memory_manager.store(memory_input)

           except Exception as e:
               print(f"Error storing memory: {e}")

       def _build_prompt_with_memory(self, current_message: str, memory: str) -> str:
           """Build prompt that includes conversation history."""
           if memory:
               return f"""Previous conversation:
   {memory}

   Current message: {current_message}"""
           return current_message

       def clear_memory(self):
           """Clear conversation memory."""
           try:
               memory_input = IMemoryInput(
                   conversation_id=self.conversation_id,
                   memory_type=ConversationMemoryType.WORKING_MEMORY
               )
               self.memory_manager.delete(memory_input)
               self.interaction_count = 0
               print("✓ Conversation memory cleared")
           except Exception as e:
               print(f"Error clearing memory: {e}")

Step 2: Build the CLI Interface
--------------------------------

Add command handling and user interface:

.. code-block:: python

   class ChatbotCLI:
       """Command-line interface for the chatbot."""

       def __init__(self, chatbot: SimpleChatbot):
           self.chatbot = chatbot
           self.running = False

       def print_welcome(self):
           """Print welcome message."""
           print("\n" + "=" * 60)
           print("🤖 Simple Chatbot")
           print("=" * 60)
           print("\nCommands:")
           print("  /help   - Show this help message")
           print("  /clear  - Clear conversation history")
           print("  /quit   - Exit the chatbot")
           print("\nStart chatting! Type your message and press Enter.\n")

       def print_help(self):
           """Print help message."""
           print("\n📖 Available Commands:")
           print("  /help   - Show available commands")
           print("  /clear  - Clear conversation memory")
           print("  /quit   - Exit the application")
           print("\nJust type normally to chat with the bot!\n")

       async def handle_message(self, user_input: str):
           """Handle user message."""
           try:
               # Show typing indicator
               print("🤖 Bot: ", end="", flush=True)

               # Process message
               result = await self.chatbot.process(
                   IAgentInput(message=user_input)
               )

               # Display response
               print(result['response'])

               # Show metadata (optional)
               if result.get('interaction_count'):
                   print(f"\n💬 Turn {result['interaction_count']}", end=" ")
                   print(f"• {result['timestamp'][:19]}")

           except Exception as e:
               print(f"\n❌ Error: {e}")
               print("Please try again.\n")

       async def run(self):
           """Run the chatbot CLI."""
           self.running = True
           self.print_welcome()

           while self.running:
               try:
                   # Get user input
                   user_input = input("\n👤 You: ").strip()

                   # Handle empty input
                   if not user_input:
                       continue

                   # Handle commands
                   if user_input.startswith('/'):
                       await self.handle_command(user_input.lower())
                       continue

                   # Handle regular message
                   await self.handle_message(user_input)

               except KeyboardInterrupt:
                   print("\n\n👋 Goodbye!")
                   self.running = False
               except EOFError:
                   print("\n\n👋 Goodbye!")
                   self.running = False
               except Exception as e:
                   print(f"\n❌ Unexpected error: {e}")

       async def handle_command(self, command: str):
           """Handle chatbot commands."""
           if command == '/quit':
               print("\n👋 Goodbye!")
               self.running = False

           elif command == '/clear':
               self.chatbot.clear_memory()

           elif command == '/help':
               self.print_help()

           else:
               print(f"❌ Unknown command: {command}")
               print("Type /help for available commands")

Step 3: Create the Main Application
------------------------------------

Wire everything together:

.. code-block:: python

   async def create_chatbot(conversation_id: str = None) -> SimpleChatbot:
       """Create and configure the chatbot."""

       # Generate conversation ID if not provided
       if conversation_id is None:
           from uuid import uuid4
           conversation_id = f"chat_{uuid4().hex[:8]}"

       # Check for API key
       api_key = os.getenv("OPENAI_API_KEY")
       if not api_key:
           raise ValueError(
               "OPENAI_API_KEY not found. "
               "Please set it in your .env file or environment."
           )

       # Create LLM client
       llm_config = ILLMConfig(
           model="gpt-3.5-turbo",
           temperature=0.7,
           max_tokens=500
       )
       llm_client = OpenAIClient(llm_config)

       # Create memory manager
       memory_manager = InMemoryManager(ttl=3600)  # 1 hour TTL

       # Create chatbot
       chatbot = SimpleChatbot(
           llm_client=llm_client,
           memory_manager=memory_manager,
           conversation_id=conversation_id
       )

       return chatbot

   async def main():
       """Main application entry point."""
       try:
           # Create chatbot
           chatbot = await create_chatbot()

           # Create and run CLI
           cli = ChatbotCLI(chatbot)
           await cli.run()

       except ValueError as e:
           print(f"❌ Configuration Error: {e}")
       except Exception as e:
           print(f"❌ Error starting chatbot: {e}")

   if __name__ == "__main__":
       asyncio.run(main())

Step 4: Run and Test
--------------------

Run the chatbot:

.. code-block:: bash

   python chatbot.py

Test conversation:

.. code-block:: text

   🤖 Simple Chatbot
   ================================================================

   Commands:
     /help   - Show this help message
     /clear  - Clear conversation history
     /quit   - Exit the chatbot

   Start chatting! Type your message and press Enter.

   👤 You: Hello! My name is Alice.
   🤖 Bot: Hello Alice! It's nice to meet you. How can I help you today?

   💬 Turn 1 • 2024-01-15 10:30:45

   👤 You: What's my name?
   🤖 Bot: Your name is Alice, as you just told me!

   💬 Turn 2 • 2024-01-15 10:30:52

   👤 You: /clear
   ✓ Conversation memory cleared

   👤 You: What's my name?
   🤖 Bot: I don't know your name yet. Would you like to tell me?

   💬 Turn 1 • 2024-01-15 10:31:05

   👤 You: /quit
   👋 Goodbye!

Step 5: Add Enhancements (Optional)
------------------------------------

Enhance with Typing Indicator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import sys
   import time
   import threading

   def show_typing_indicator():
       """Show animated typing indicator."""
       chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
       for _ in range(10):
           for char in chars:
               sys.stdout.write(f'\r🤖 Bot: {char} thinking...')
               sys.stdout.flush()
               time.sleep(0.1)
       sys.stdout.write('\r' + ' ' * 40 + '\r')
       sys.stdout.flush()

   # Use in handle_message:
   # threading.Thread(target=show_typing_indicator, daemon=True).start()

Add Conversation Export
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def export_conversation(self, filename: str = None):
       """Export conversation to file."""
       if filename is None:
           filename = f"conversation_{self.conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

       memory = self._get_conversation_memory()

       with open(filename, 'w') as f:
           f.write(f"Conversation Export\n")
           f.write(f"ID: {self.conversation_id}\n")
           f.write(f"Date: {datetime.now().isoformat()}\n")
           f.write(f"Turns: {self.interaction_count}\n")
           f.write("=" * 60 + "\n\n")
           f.write(memory)

       print(f"✓ Conversation exported to {filename}")

   # Add command in handle_command:
   elif command == '/export':
       self.chatbot.export_conversation()

Add Conversation Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def get_statistics(self) -> Dict[str, Any]:
       """Get conversation statistics."""
       memory = self._get_conversation_memory()

       return {
           "conversation_id": self.conversation_id,
           "total_turns": self.interaction_count,
           "memory_size": len(memory),
           "memory_lines": len(memory.split('\n')) if memory else 0
       }

   # Add command:
   elif command == '/stats':
       stats = self.chatbot.get_statistics()
       print("\n📊 Conversation Statistics:")
       print(f"  ID: {stats['conversation_id']}")
       print(f"  Turns: {stats['total_turns']}")
       print(f"  Memory Size: {stats['memory_size']} characters")
       print(f"  Memory Lines: {stats['memory_lines']}")

Step 6: Add Web Interface (Optional)
-------------------------------------

Create simple web interface using Flask:

.. code-block:: python

   # web_chatbot.py
   from flask import Flask, render_template, request, jsonify
   import asyncio

   app = Flask(__name__)
   chatbot = None

   @app.route('/')
   def index():
       return render_template('chat.html')

   @app.route('/chat', methods=['POST'])
   def chat():
       user_message = request.json.get('message')

       # Process message
       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)
       result = loop.run_until_complete(
           chatbot.process(IAgentInput(message=user_message))
       )

       return jsonify(result)

   @app.route('/clear', methods=['POST'])
   def clear():
       chatbot.clear_memory()
       return jsonify({"status": "cleared"})

   if __name__ == '__main__':
       # Initialize chatbot
       loop = asyncio.get_event_loop()
       chatbot = loop.run_until_complete(create_chatbot("web_session"))

       app.run(debug=True, port=5000)

Create ``templates/chat.html``:

.. code-block:: html

   <!DOCTYPE html>
   <html>
   <head>
       <title>Simple Chatbot</title>
       <style>
           body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
           #chat-container { border: 1px solid #ccc; height: 400px; overflow-y: auto; padding: 10px; }
           .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
           .user { background: #e3f2fd; text-align: right; }
           .bot { background: #f5f5f5; }
           #input-container { margin-top: 10px; }
           input { width: 80%; padding: 10px; }
           button { padding: 10px 20px; }
       </style>
   </head>
   <body>
       <h1>🤖 Simple Chatbot</h1>
       <div id="chat-container"></div>
       <div id="input-container">
           <input type="text" id="user-input" placeholder="Type your message...">
           <button onclick="sendMessage()">Send</button>
           <button onclick="clearChat()">Clear</button>
       </div>

       <script>
           async function sendMessage() {
               const input = document.getElementById('user-input');
               const message = input.value.trim();
               if (!message) return;

               addMessage('user', message);
               input.value = '';

               const response = await fetch('/chat', {
                   method: 'POST',
                   headers: {'Content-Type': 'application/json'},
                   body: JSON.stringify({message: message})
               });

               const data = await response.json();
               addMessage('bot', data.response);
           }

           function addMessage(type, text) {
               const container = document.getElementById('chat-container');
               const div = document.createElement('div');
               div.className = `message ${type}`;
               div.textContent = text;
               container.appendChild(div);
               container.scrollTop = container.scrollHeight;
           }

           async function clearChat() {
               await fetch('/clear', {method: 'POST'});
               document.getElementById('chat-container').innerHTML = '';
           }

           document.getElementById('user-input').addEventListener('keypress', (e) => {
               if (e.key === 'Enter') sendMessage();
           });
       </script>
   </body>
   </html>

Production Considerations
-------------------------

Error Handling
~~~~~~~~~~~~~~

Add robust error handling:

.. code-block:: python

   class RobustChatbot(SimpleChatbot):
       """Chatbot with enhanced error handling."""

       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           max_retries = 3

           for attempt in range(max_retries):
               try:
                   return await super().process(input)

               except Exception as e:
                   if attempt == max_retries - 1:
                       return {
                           "response": "I'm having trouble processing your request. Please try again.",
                           "error": str(e),
                           "interaction_count": self.interaction_count
                       }

                   await asyncio.sleep(1)  # Wait before retry

Rate Limiting
~~~~~~~~~~~~~

.. code-block:: python

   from datetime import datetime, timedelta

   class RateLimitedChatbot(SimpleChatbot):
       """Chatbot with rate limiting."""

       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.request_times = []
           self.max_requests = 10  # per minute

       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Check rate limit
           now = datetime.now()
           self.request_times = [
               t for t in self.request_times
               if now - t < timedelta(minutes=1)
           ]

           if len(self.request_times) >= self.max_requests:
               return {
                   "response": "Rate limit exceeded. Please wait a moment.",
                   "error": "rate_limit"
               }

           self.request_times.append(now)
           return await super().process(input)

Logging
~~~~~~~

.. code-block:: python

   import logging

   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('chatbot.log'),
           logging.StreamHandler()
       ]
   )

   logger = logging.getLogger('SimpleChatbot')

   # Use in chatbot:
   logger.info(f"Processing message for conversation {self.conversation_id}")
   logger.error(f"Error: {e}", exc_info=True)

Next Steps
----------

**Enhance the chatbot:**

- Add personality customization
- Implement sentiment analysis
- Add multi-language support
- Integrate external APIs as tools

**Scale the application:**

- Use Redis for persistent memory: :doc:`../implementations/memory/redis-memory`
- Add user authentication
- Deploy to cloud platforms
- Implement conversation analytics

**Learn more:**

- :doc:`rag-system` - Add document retrieval capabilities
- :doc:`custom-system` - Build advanced orchestration
- :doc:`../implementations/memory/index` - Explore memory options

Congratulations! You've built a complete chatbot with Arshai's framework. 🎉
