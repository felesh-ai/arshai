"""
Test harness for workflow testing.
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock
from arshai.workflows import BaseWorkflowOrchestrator, BaseNode
from arshai.core.interfaces.iworkflow import IWorkflowState
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
import logging

logger = logging.getLogger(__name__)


class WorkflowTestHarness:
    """
    Test harness for workflow testing.

    Provides utilities for testing workflows with mocked nodes,
    execution tracking, and result verification.

    Example:
        harness = WorkflowTestHarness()

        # Test with mocked nodes
        result = await harness.test_workflow(
            workflow=my_workflow,
            input_state=test_state,
            mock_nodes={
                "external_api": {"status": "success"},
                "database": {"data": [1, 2, 3]}
            }
        )

        # Verify execution path
        assert harness.executed_nodes == ["input", "external_api", "database", "output"]
    """

    def __init__(self):
        self.executed_nodes: List[str] = []
        self.node_inputs: Dict[str, Any] = {}
        self.node_outputs: Dict[str, Any] = {}
        self.execution_order: List[str] = []

    async def test_workflow(
        self,
        workflow: BaseWorkflowOrchestrator,
        input_state: IWorkflowState,
        mock_nodes: Optional[Dict[str, Any]] = None,
        record_execution: bool = True
    ) -> IWorkflowState:
        """
        Test workflow with optional mocked nodes.

        Args:
            workflow: Workflow to test
            input_state: Input state
            mock_nodes: Dict of node names to mock outputs
            record_execution: Record execution details

        Returns:
            Workflow result
        """
        # Replace nodes with mocks if specified
        if mock_nodes:
            for node_name, mock_output in mock_nodes.items():
                if node_name in workflow.nodes:
                    # Create mock callable for the node
                    mock_callable = self._create_mock_callable(
                        node_name,
                        mock_output,
                        record_execution
                    )
                    workflow.nodes[node_name] = mock_callable

        # Execute workflow
        result = await workflow.execute(input_state)

        return result

    def _create_mock_callable(
        self,
        name: str,
        output: Any,
        record: bool
    ) -> callable:
        """Create a mock callable for testing"""

        async def mock_node(state: IWorkflowState) -> Dict[str, Any]:
            """Mock node execution"""
            if record:
                self.executed_nodes.append(name)
                self.execution_order.append(name)
                self.node_inputs[name] = state
                self.node_outputs[name] = output

            # Return the mock output in the expected format
            return {"state": output} if not isinstance(output, dict) or "state" not in output else output

        return mock_node

    def create_mock_agent(self, name: str, response: str = "mock response") -> BaseAgent:
        """
        Create a mock agent for testing.

        Args:
            name: Agent name
            response: Mock response string

        Returns:
            Mock agent instance
        """

        class MockAgent(BaseAgent):
            def __init__(self, mock_name: str, mock_response: str):
                super().__init__()
                self._name = mock_name
                self._response = mock_response

            async def process_message(self, input_data: IAgentInput) -> tuple:
                """Return mock response"""
                return self._response, {"mocked": True}

        return MockAgent(name, response)

    def assert_node_executed(self, node_name: str):
        """
        Assert that a specific node was executed.

        Args:
            node_name: Name of the node to check

        Raises:
            AssertionError: If node was not executed
        """
        assert node_name in self.executed_nodes, \
            f"Node '{node_name}' was not executed. Executed nodes: {self.executed_nodes}"

    def assert_execution_order(self, expected_order: List[str]):
        """
        Assert nodes were executed in specific order.

        Args:
            expected_order: Expected execution order

        Raises:
            AssertionError: If execution order doesn't match
        """
        assert self.execution_order == expected_order, \
            f"Execution order mismatch.\nExpected: {expected_order}\nActual: {self.execution_order}"

    def assert_node_not_executed(self, node_name: str):
        """
        Assert that a specific node was NOT executed.

        Args:
            node_name: Name of the node to check

        Raises:
            AssertionError: If node was executed
        """
        assert node_name not in self.executed_nodes, \
            f"Node '{node_name}' should not have been executed but was"

    def get_node_input(self, node_name: str) -> Any:
        """
        Get the input that was passed to a specific node.

        Args:
            node_name: Name of the node

        Returns:
            Input data passed to the node
        """
        return self.node_inputs.get(node_name)

    def get_node_output(self, node_name: str) -> Any:
        """
        Get the output from a specific node.

        Args:
            node_name: Name of the node

        Returns:
            Output data from the node
        """
        return self.node_outputs.get(node_name)

    def reset(self):
        """Reset the test harness for a new test."""
        self.executed_nodes.clear()
        self.node_inputs.clear()
        self.node_outputs.clear()
        self.execution_order.clear()

    def get_execution_stats(self) -> dict:
        """
        Get statistics about the workflow execution.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_nodes_executed": len(self.executed_nodes),
            "unique_nodes": len(set(self.executed_nodes)),
            "execution_order": self.execution_order,
            "nodes_with_input": list(self.node_inputs.keys()),
            "nodes_with_output": list(self.node_outputs.keys())
        }


class MockMemoryManager:
    """
    Mock memory manager for testing agents with caching.

    Example:
        agent = MyAgent()
        agent.memory = MockMemoryManager()

        # Test caching behavior
        result = await agent.cached_method(data)
    """

    def __init__(self):
        self.storage = {}

    async def get(self, key: str) -> Optional[bytes]:
        """Get value from mock storage."""
        return self.storage.get(key)

    async def set(self, key: str, value: bytes, ttl: int = 300) -> None:
        """Set value in mock storage."""
        self.storage[key] = value

    def get_sync(self, key: str) -> Optional[bytes]:
        """Synchronous get for testing."""
        return self.storage.get(key)

    def set_sync(self, key: str, value: bytes, ttl: int = 300) -> None:
        """Synchronous set for testing."""
        self.storage[key] = value

    def clear(self):
        """Clear all stored values."""
        self.storage.clear()

    def delete_pattern(self, pattern: str):
        """Delete keys matching pattern."""
        if pattern == "*":
            self.storage.clear()
        else:
            # Simple prefix matching
            keys_to_delete = [k for k in self.storage if k.startswith(pattern)]
            for key in keys_to_delete:
                del self.storage[key]