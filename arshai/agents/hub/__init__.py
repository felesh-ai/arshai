"""
Agent Hub - Example Agent Implementations

This directory contains reference implementations of agents that demonstrate
how to build custom agents using the Arshai framework. These are examples
showing "our experience" with the framework, not prescriptive implementations.

You are encouraged to:
- Use these agents as-is if they fit your needs
- Modify them for your specific requirements  
- Build completely different implementations
- Ignore them entirely and create your own

The core framework is in the parent directory (base.py). Everything in this
hub directory represents example implementations, not core framework features.
"""

from .working_memory import WorkingMemoryAgent

__all__ = [
    "WorkingMemoryAgent",
]