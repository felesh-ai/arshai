"""
Tests for the WorkflowTestHarness.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from arshai.testing import WorkflowTestHarness
from arshai.testing.harness import MockMemoryManager
from arshai.workflows import BaseWorkflowOrchestrator
from arshai.core.interfaces.iworkflow import IWorkflowState, IUserContext
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput


class TestWorkflowTestHarness:
    """Test cases for WorkflowTestHarness."""

    @pytest.mark.asyncio
    async def test_harness_initialization(self):
        """Test that harness initializes with empty tracking."""
        harness = WorkflowTestHarness()

        assert harness.executed_nodes == []
        assert harness.node_inputs == {}
        assert harness.node_outputs == {}
        assert harness.execution_order == []

    @pytest.mark.asyncio
    async def test_mock_node_execution_tracking(self):
        """Test that mock nodes track execution properly."""
        harness = WorkflowTestHarness()
        workflow = BaseWorkflowOrchestrator()

        # Create test state
        state = IWorkflowState(
            user_context=IUserContext(user_id="test_user"),
            workflow_data={"input": "test"}
        )

        # Add mock nodes to workflow
        mock_output = {"result": "success"}
        workflow.nodes["test_node"] = harness._create_mock_callable(
            "test_node", mock_output, record=True
        )

        # Execute the mock node
        result = await workflow.nodes["test_node"](state)

        # Verify tracking
        assert "test_node" in harness.executed_nodes
        assert harness.node_outputs["test_node"] == mock_output
        assert harness.node_inputs["test_node"] == state

    @pytest.mark.asyncio
    async def test_test_workflow_with_mocks(self):
        """Test workflow execution with mocked nodes."""
        harness = WorkflowTestHarness()
        workflow = BaseWorkflowOrchestrator()

        # Setup workflow with real callables
        async def node1(state):
            return {"state": {"node1": "executed"}}

        async def node2(state):
            return {"state": {"node2": "executed"}}

        workflow.nodes = {"node1": node1, "node2": node2}

        # Create test state
        state = IWorkflowState(
            user_context=IUserContext(user_id="test_user"),
            workflow_data={"test": "data"}
        )

        # Test with mocked nodes
        mock_nodes = {
            "node1": {"mocked": True, "value": 1},
            "node2": {"mocked": True, "value": 2}
        }

        result = await harness.test_workflow(
            workflow=workflow,
            input_state=state,
            mock_nodes=mock_nodes,
            record_execution=True
        )

        # Verify mocked nodes were used
        assert "node1" in harness.executed_nodes
        assert "node2" in harness.executed_nodes
        assert harness.node_outputs["node1"] == {"mocked": True, "value": 1}
        assert harness.node_outputs["node2"] == {"mocked": True, "value": 2}

    def test_create_mock_agent(self):
        """Test mock agent creation."""
        harness = WorkflowTestHarness()

        mock_agent = harness.create_mock_agent(
            name="test_agent",
            response="mock response"
        )

        assert isinstance(mock_agent, BaseAgent)
        assert mock_agent._name == "test_agent"
        assert mock_agent._response == "mock response"

    @pytest.mark.asyncio
    async def test_mock_agent_process_message(self):
        """Test that mock agent returns expected response."""
        harness = WorkflowTestHarness()

        mock_agent = harness.create_mock_agent(
            name="test_agent",
            response="test response"
        )

        input_data = IAgentInput(
            message="test message",
            conversation_id="test_conv"
        )

        response, metadata = await mock_agent.process_message(input_data)

        assert response == "test response"
        assert metadata == {"mocked": True}

    def test_assert_node_executed(self):
        """Test node execution assertion."""
        harness = WorkflowTestHarness()
        harness.executed_nodes = ["node1", "node2", "node3"]

        # Should pass
        harness.assert_node_executed("node1")
        harness.assert_node_executed("node2")

        # Should raise
        with pytest.raises(AssertionError) as exc:
            harness.assert_node_executed("node4")
        assert "node4" in str(exc.value)
        assert "was not executed" in str(exc.value)

    def test_assert_execution_order(self):
        """Test execution order assertion."""
        harness = WorkflowTestHarness()
        harness.execution_order = ["start", "process", "end"]

        # Should pass
        harness.assert_execution_order(["start", "process", "end"])

        # Should raise on wrong order
        with pytest.raises(AssertionError) as exc:
            harness.assert_execution_order(["start", "end", "process"])
        assert "Execution order mismatch" in str(exc.value)

    def test_assert_node_not_executed(self):
        """Test assertion for node not executed."""
        harness = WorkflowTestHarness()
        harness.executed_nodes = ["node1", "node2"]

        # Should pass
        harness.assert_node_not_executed("node3")

        # Should raise
        with pytest.raises(AssertionError) as exc:
            harness.assert_node_not_executed("node1")
        assert "should not have been executed" in str(exc.value)

    def test_get_node_input_output(self):
        """Test getting node inputs and outputs."""
        harness = WorkflowTestHarness()

        test_input = {"input": "data"}
        test_output = {"output": "result"}

        harness.node_inputs["test_node"] = test_input
        harness.node_outputs["test_node"] = test_output

        assert harness.get_node_input("test_node") == test_input
        assert harness.get_node_output("test_node") == test_output
        assert harness.get_node_input("nonexistent") is None
        assert harness.get_node_output("nonexistent") is None

    def test_reset_harness(self):
        """Test harness reset functionality."""
        harness = WorkflowTestHarness()

        # Add some data
        harness.executed_nodes.append("node1")
        harness.node_inputs["node1"] = {"test": "input"}
        harness.node_outputs["node1"] = {"test": "output"}
        harness.execution_order.append("node1")

        # Reset
        harness.reset()

        # Verify everything is cleared
        assert len(harness.executed_nodes) == 0
        assert len(harness.node_inputs) == 0
        assert len(harness.node_outputs) == 0
        assert len(harness.execution_order) == 0

    def test_get_execution_stats(self):
        """Test execution statistics generation."""
        harness = WorkflowTestHarness()

        # Setup execution data
        harness.executed_nodes = ["node1", "node2", "node1"]  # node1 executed twice
        harness.execution_order = ["node1", "node2", "node1"]
        harness.node_inputs = {"node1": {}, "node2": {}}
        harness.node_outputs = {"node1": {}, "node2": {}}

        stats = harness.get_execution_stats()

        assert stats["total_nodes_executed"] == 3
        assert stats["unique_nodes"] == 2
        assert stats["execution_order"] == ["node1", "node2", "node1"]
        assert "node1" in stats["nodes_with_input"]
        assert "node2" in stats["nodes_with_output"]


class TestMockMemoryManager:
    """Test cases for MockMemoryManager."""

    @pytest.mark.asyncio
    async def test_async_get_set(self):
        """Test async get/set operations."""
        memory = MockMemoryManager()

        # Set a value
        await memory.set("key1", b"value1", ttl=300)

        # Get the value
        result = await memory.get("key1")
        assert result == b"value1"

        # Get nonexistent key
        result = await memory.get("nonexistent")
        assert result is None

    def test_sync_get_set(self):
        """Test sync get/set operations."""
        memory = MockMemoryManager()

        # Set a value
        memory.set_sync("key1", b"value1", ttl=300)

        # Get the value
        result = memory.get_sync("key1")
        assert result == b"value1"

        # Get nonexistent key
        result = memory.get_sync("nonexistent")
        assert result is None

    def test_clear(self):
        """Test clearing all storage."""
        memory = MockMemoryManager()

        # Add some data
        memory.storage["key1"] = b"value1"
        memory.storage["key2"] = b"value2"

        # Clear
        memory.clear()

        assert len(memory.storage) == 0

    def test_delete_pattern(self):
        """Test pattern-based deletion."""
        memory = MockMemoryManager()

        # Add test data
        memory.storage["prefix:key1"] = b"value1"
        memory.storage["prefix:key2"] = b"value2"
        memory.storage["other:key3"] = b"value3"

        # Delete by pattern
        memory.delete_pattern("prefix:")

        assert "prefix:key1" not in memory.storage
        assert "prefix:key2" not in memory.storage
        assert "other:key3" in memory.storage

        # Delete all
        memory.delete_pattern("*")
        assert len(memory.storage) == 0


# Integration tests
class TestHarnessIntegration:
    """Integration tests for the test harness with real workflows."""

    @pytest.mark.asyncio
    async def test_complex_workflow_testing(self):
        """Test harness with a more complex workflow scenario."""
        harness = WorkflowTestHarness()
        workflow = BaseWorkflowOrchestrator()

        # Create a workflow with conditional execution
        executed_nodes = []

        async def router_node(state):
            executed_nodes.append("router")
            if state.workflow_data.get("path") == "a":
                return {"next": "path_a"}
            else:
                return {"next": "path_b"}

        async def path_a_node(state):
            executed_nodes.append("path_a")
            return {"state": {"result": "a"}}

        async def path_b_node(state):
            executed_nodes.append("path_b")
            return {"state": {"result": "b"}}

        workflow.nodes = {
            "router": router_node,
            "path_a": path_a_node,
            "path_b": path_b_node
        }

        # Test with path A
        state_a = IWorkflowState(
            user_context=IUserContext(user_id="test"),
            workflow_data={"path": "a"}
        )

        # Mock only path_b (shouldn't be executed)
        await harness.test_workflow(
            workflow=workflow,
            input_state=state_a,
            mock_nodes={"path_b": {"mocked": True}},
            record_execution=True
        )

        # Verify path_b was mocked but not executed
        assert "path_b" not in executed_nodes
        assert "router" in executed_nodes
        assert "path_a" in executed_nodes