"""
Example plugin demonstrating how to extend Arshai.
"""

from typing import Dict, Any, List
import asyncio

from arshai.extensions.base import Plugin, PluginMetadata
from arshai.extensions.hooks import hook, HookType, HookContext
from arshai.core.interfaces import IAgent, IAgentInput


# Function-based tool implementatio
def word_count_tool(text: str) -> str:
    """
    Count the number of words in a text.

    Args:
        text: The text to count words in

    Returns:
        A string with the word count
    """
    word_count = len(text.split())
    return f"The text contains {word_count} words."


class ExamplePlugin(Plugin):
    """
    Example plugin that demonstrates various extension capabilities.
    
    This plugin:
    - Adds a custom tool (word counter)
    - Registers hooks for agent processing
    - Demonstrates configuration handling
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            author="Arshai Team",
            description="Example plugin demonstrating Arshai extensions",
            requires=[],  # No dependencies
            tags=["example", "tools", "hooks"]
        )
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        print(f"Initializing {self._metadata.name} v{self._metadata.version}")

        # Store tool function for later use
        self.word_count_func = word_count_tool

        # Register hooks
        self._register_hooks()

        # Use configuration
        if self.config.get("verbose", False):
            print("Verbose mode enabled")
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        print(f"Shutting down {self._metadata.name}")
        # Clean up resources if needed
    
    def _register_hooks(self) -> None:
        """Register plugin hooks."""
        # We'll use the decorator approach in a method
        from arshai.extensions.hooks import get_hook_manager, Hook
        
        manager = get_hook_manager()
        
        # Hook to log agent processing
        manager.register_hook(Hook(
            name="example_agent_logger",
            hook_type=HookType.BEFORE_AGENT_PROCESS,
            callback=self._log_agent_input,
            priority=10
        ))
        
        # Hook to add metadata after agent processing
        manager.register_hook(Hook(
            name="example_metadata_adder", 
            hook_type=HookType.AFTER_AGENT_PROCESS,
            callback=self._add_metadata,
            priority=5
        ))
    
    async def _log_agent_input(self, context: HookContext) -> None:
        """Log agent input (example hook)."""
        if self.config.get("log_inputs", True):
            agent_input = context.data.get("input")
            if agent_input and hasattr(agent_input, "message"):
                print(f"[ExamplePlugin] Agent input: {agent_input.message}")
    
    async def _add_metadata(self, context: HookContext) -> Dict[str, Any]:
        """Add metadata to agent response (example hook)."""
        response = context.data.get("response")
        if response:
            # Add plugin metadata to response
            return {
                "modified_data": {
                    "response": response,
                    "plugin_metadata": {
                        "processed_by": self._metadata.name,
                        "version": self._metadata.version
                    }
                }
            }
        return {}
    
    def get_word_count_tool(self) -> callable:
        """
        Get the word count tool function for use with agents.

        Returns:
            Callable function that can be used in regular_functions
        """
        return self.word_count_func


# Example of using hooks with decorators (outside the class)
@hook(HookType.BEFORE_LLM_CALL, name="example_llm_logger", priority=20)
async def log_llm_calls(context: HookContext):
    """Example hook that logs LLM calls."""
    model = context.data.get("model", "unknown")
    messages = context.data.get("messages", [])
    print(f"[ExamplePlugin] LLM call to {model} with {len(messages)} messages")


# Make the plugin discoverable
Plugin = ExamplePlugin