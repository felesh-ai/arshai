"""
Core interfaces for the Arshai framework.
"""

# Agent interfaces
from .iagent import IAgent, IAgentInput

# LLM interfaces  
from .illm import ILLM, ILLMConfig, ILLMInput

# Memory interfaces
from .imemorymanager import IMemoryManager, IWorkingMemory, ConversationMemoryType, IMemoryInput

# Workflow interfaces
from .iworkflow import IWorkflowState, IUserContext, IWorkflowOrchestrator, IWorkflowConfig, INode
from .iworkflowrunner import IWorkflowRunner

# Document interfaces
from .idocument import Document

# Other interfaces
from .iembedding import IEmbedding
from .ivector_db_client import IVectorDBClient
from .idto import IDTO, IStreamDTO
from .ireranker import IReranker, IRerankInput
from .iwebsearch import IWebSearchClient, IWebSearchResult, IWebSearchConfig
from .inotification import INotificationState, INotificationAttempt

# All available interfaces
__all__ = [
    # Agent
    "IAgent", "IAgentInput",
    # LLM
    "ILLM", "ILLMConfig", "ILLMInput",
    # Memory
    "IMemoryManager", "IWorkingMemory", "ConversationMemoryType", "IMemoryInput",
    # Workflow
    "IWorkflowState", "IUserContext", "IWorkflowOrchestrator", "IWorkflowConfig", "INode", "IWorkflowRunner",
    # Document
    "Document",
    # Other
    "IEmbedding", "IVectorDBClient", "IDTO", "IStreamDTO",
    # Reranker
    "IReranker", "IRerankInput",
    # Web Search
    "IWebSearchClient", "IWebSearchResult", "IWebSearchConfig",
    # Notification
    "INotificationState", "INotificationAttempt",
]

# Backward compatibility
IWorkflow = IWorkflowConfig
__all__.append("IWorkflow")