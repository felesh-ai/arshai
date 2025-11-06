"""
Example test scenarios demonstrating how to use the WorkflowTestHarness.

These examples show common testing patterns for Arshai workflows and agents.
"""

import pytest
import asyncio
from arshai.testing import WorkflowTestHarness
from arshai.workflows import BaseWorkflowOrchestrator
from arshai.workflows.patterns import FallbackWorkflow
from arshai.workflows.extensions import BatchProcessor, BatchResult
from arshai.core.interfaces.iworkflow import IWorkflowState, IUserContext
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput


# Example 1: Testing a simple linear workflow
class TestLinearWorkflow:
    """Example: Testing a simple sequential workflow."""

    @pytest.mark.asyncio
    async def test_linear_workflow_execution(self):
        """Test a workflow that processes data sequentially."""
        harness = WorkflowTestHarness()

        # Create a simple workflow
        class DataProcessingWorkflow(BaseWorkflowOrchestrator):
            async def execute(self, state: IWorkflowState) -> IWorkflowState:
                # Step 1: Validate
                validate_result = await self.nodes["validate"](state)
                state.workflow_data.update(validate_result.get("state", {}))

                # Step 2: Process
                process_result = await self.nodes["process"](state)
                state.workflow_data.update(process_result.get("state", {}))

                # Step 3: Store
                store_result = await self.nodes["store"](state)
                state.workflow_data.update(store_result.get("state", {}))

                return state

        workflow = DataProcessingWorkflow()

        # Mock external dependencies
        mock_nodes = {
            "validate": {"validated": True, "errors": []},
            "process": {"processed_data": "result", "score": 0.95},
            "store": {"stored": True, "id": "12345"}
        }

        # Create test state
        test_state = IWorkflowState(
            user_context=IUserContext(user_id="test_user"),
            workflow_data={"input": "test_data"}
        )

        # Execute with mocks
        result = await harness.test_workflow(
            workflow=workflow,
            input_state=test_state,
            mock_nodes=mock_nodes,
            record_execution=True
        )

        # Verify execution
        harness.assert_execution_order(["validate", "process", "store"])
        harness.assert_node_executed("validate")
        harness.assert_node_executed("process")
        harness.assert_node_executed("store")

        # Verify outputs
        assert harness.get_node_output("validate")["validated"] is True
        assert harness.get_node_output("process")["score"] == 0.95
        assert harness.get_node_output("store")["id"] == "12345"


# Example 2: Testing conditional workflow paths
class TestConditionalWorkflow:
    """Example: Testing workflows with conditional branching."""

    @pytest.mark.asyncio
    async def test_conditional_workflow_path_a(self):
        """Test workflow taking path A based on condition."""
        harness = WorkflowTestHarness()

        # Create workflow with conditional logic
        class ConditionalWorkflow(BaseWorkflowOrchestrator):
            async def execute(self, state: IWorkflowState) -> IWorkflowState:
                # Evaluate condition
                eval_result = await self.nodes["evaluate"](state)

                if eval_result.get("state", {}).get("score", 0) > 0.5:
                    # High score path
                    result = await self.nodes["high_score_handler"](state)
                else:
                    # Low score path
                    result = await self.nodes["low_score_handler"](state)

                state.workflow_data.update(result.get("state", {}))
                return state

        workflow = ConditionalWorkflow()

        # Test high score path
        mock_nodes = {
            "evaluate": {"score": 0.8},
            "high_score_handler": {"result": "premium"},
            "low_score_handler": {"result": "basic"}
        }

        test_state = IWorkflowState(
            user_context=IUserContext(user_id="test_user"),
            workflow_data={"input": "test"}
        )

        await harness.test_workflow(
            workflow=workflow,
            input_state=test_state,
            mock_nodes=mock_nodes,
            record_execution=True
        )

        # Verify correct path was taken
        harness.assert_node_executed("evaluate")
        harness.assert_node_executed("high_score_handler")
        harness.assert_node_not_executed("low_score_handler")

    @pytest.mark.asyncio
    async def test_conditional_workflow_path_b(self):
        """Test workflow taking path B based on condition."""
        harness = WorkflowTestHarness()

        # Same workflow as above
        class ConditionalWorkflow(BaseWorkflowOrchestrator):
            async def execute(self, state: IWorkflowState) -> IWorkflowState:
                eval_result = await self.nodes["evaluate"](state)

                if eval_result.get("state", {}).get("score", 0) > 0.5:
                    result = await self.nodes["high_score_handler"](state)
                else:
                    result = await self.nodes["low_score_handler"](state)

                state.workflow_data.update(result.get("state", {}))
                return state

        workflow = ConditionalWorkflow()

        # Test low score path
        mock_nodes = {
            "evaluate": {"score": 0.3},
            "high_score_handler": {"result": "premium"},
            "low_score_handler": {"result": "basic"}
        }

        test_state = IWorkflowState(
            user_context=IUserContext(user_id="test_user"),
            workflow_data={"input": "test"}
        )

        await harness.test_workflow(
            workflow=workflow,
            input_state=test_state,
            mock_nodes=mock_nodes,
            record_execution=True
        )

        # Verify correct path was taken
        harness.assert_node_executed("evaluate")
        harness.assert_node_not_executed("high_score_handler")
        harness.assert_node_executed("low_score_handler")


# Example 3: Testing error handling with fallback workflows
class TestFallbackWorkflowScenario:
    """Example: Testing fallback workflow patterns."""

    @pytest.mark.asyncio
    async def test_fallback_workflow_primary_success(self):
        """Test that fallback is not used when primary succeeds."""
        harness = WorkflowTestHarness()

        # Create primary and fallback workflows
        primary = BaseWorkflowOrchestrator()
        fallback = BaseWorkflowOrchestrator()

        # Mock successful primary
        async def primary_execute(state):
            harness.executed_nodes.append("primary")
            return state

        primary.execute = primary_execute

        # Mock fallback (shouldn't be called)
        async def fallback_execute(state):
            harness.executed_nodes.append("fallback")
            return state

        fallback.execute = fallback_execute

        # Create fallback workflow
        resilient = FallbackWorkflow(
            primary=primary,
            fallbacks=[fallback]
        )

        # Test execution
        test_state = IWorkflowState(
            user_context=IUserContext(user_id="test_user"),
            workflow_data={"test": "data"}
        )

        result = await resilient.execute(test_state)

        # Verify only primary was executed
        assert "primary" in harness.executed_nodes
        assert "fallback" not in harness.executed_nodes
        assert result.workflow_data.get("_fallback_used") is False

    @pytest.mark.asyncio
    async def test_fallback_workflow_with_failure(self):
        """Test that fallback is used when primary fails."""
        harness = WorkflowTestHarness()

        # Create primary and fallback workflows
        primary = BaseWorkflowOrchestrator()
        fallback = BaseWorkflowOrchestrator()

        # Mock failing primary
        async def primary_execute(state):
            harness.executed_nodes.append("primary")
            raise Exception("Primary failed")

        primary.execute = primary_execute

        # Mock successful fallback
        async def fallback_execute(state):
            harness.executed_nodes.append("fallback")
            return state

        fallback.execute = fallback_execute

        # Create fallback workflow
        resilient = FallbackWorkflow(
            primary=primary,
            fallbacks=[fallback]
        )

        # Test execution
        test_state = IWorkflowState(
            user_context=IUserContext(user_id="test_user"),
            workflow_data={"test": "data"}
        )

        result = await resilient.execute(test_state)

        # Verify fallback was used
        assert "primary" in harness.executed_nodes
        assert "fallback" in harness.executed_nodes
        assert result.workflow_data.get("_fallback_used") is True


# Example 4: Testing batch processing workflows
class TestBatchProcessingScenario:
    """Example: Testing batch processing with workflows."""

    @pytest.mark.asyncio
    async def test_batch_processing_with_mocked_workflow(self):
        """Test batch processing with partial failures."""
        harness = WorkflowTestHarness()

        # Create a workflow that processes items
        class ItemProcessor(BaseWorkflowOrchestrator):
            async def execute(self, state: IWorkflowState) -> IWorkflowState:
                item_id = state.workflow_data.get("item_id")

                # Simulate failure for specific items
                if item_id == 3:
                    raise ValueError(f"Failed to process item {item_id}")

                state.workflow_data["processed"] = True
                state.workflow_data["result"] = f"Processed item {item_id}"
                return state

        workflow = ItemProcessor()

        # Create batch of items
        states = [
            IWorkflowState(
                user_context=IUserContext(user_id="batch_user"),
                workflow_data={"item_id": i}
            )
            for i in range(1, 6)
        ]

        # Process batch with continue_on_error
        processor = BatchProcessor()
        result = await processor.execute_batch(
            workflow=workflow,
            states=states,
            batch_size=2,
            parallel=True,
            continue_on_error=True
        )

        # Verify results
        assert isinstance(result, BatchResult)
        assert result.total == 5
        assert len(result.successful) == 4  # Items 1, 2, 4, 5
        assert len(result.failed) == 1  # Item 3
        assert result.success_rate == 0.8

        # Check failed item
        failed_state, exception = result.failed[0]
        assert failed_state.workflow_data["item_id"] == 3
        assert "Failed to process item 3" in str(exception)


# Example 5: Testing agent mocking
class TestAgentMocking:
    """Example: Testing workflows with mocked agents."""

    @pytest.mark.asyncio
    async def test_workflow_with_mocked_agents(self):
        """Test workflow that uses multiple agents."""
        harness = WorkflowTestHarness()

        # Create workflow that uses agents
        class AgentOrchestrationWorkflow(BaseWorkflowOrchestrator):
            async def execute(self, state: IWorkflowState) -> IWorkflowState:
                # Get user query
                query = state.workflow_data.get("query", "")

                # Ask analysis agent
                analysis_agent = self.nodes["analysis_agent"]
                analysis_result = await analysis_agent(state)

                # Ask summary agent
                state.workflow_data["analysis"] = analysis_result.get("state", {})
                summary_agent = self.nodes["summary_agent"]
                summary_result = await summary_agent(state)

                state.workflow_data["summary"] = summary_result.get("state", {})
                return state

        workflow = AgentOrchestrationWorkflow()

        # Create mock agents
        mock_analysis = harness.create_mock_agent(
            "analysis_agent",
            response="This is a complex topic requiring detailed analysis."
        )

        mock_summary = harness.create_mock_agent(
            "summary_agent",
            response="Summary: The topic is complex and needs careful consideration."
        )

        # Wrap agents to return proper format
        async def analysis_wrapper(state):
            input_data = IAgentInput(
                message=state.workflow_data.get("query", ""),
                conversation_id="test"
            )
            response, metadata = await mock_analysis.process_message(input_data)
            harness.executed_nodes.append("analysis_agent")
            return {"state": {"response": response, "metadata": metadata}}

        async def summary_wrapper(state):
            input_data = IAgentInput(
                message=state.workflow_data.get("analysis", {}).get("response", ""),
                conversation_id="test"
            )
            response, metadata = await mock_summary.process_message(input_data)
            harness.executed_nodes.append("summary_agent")
            return {"state": {"response": response, "metadata": metadata}}

        workflow.nodes = {
            "analysis_agent": analysis_wrapper,
            "summary_agent": summary_wrapper
        }

        # Test execution
        test_state = IWorkflowState(
            user_context=IUserContext(user_id="test_user"),
            workflow_data={"query": "Explain quantum computing"}
        )

        result = await workflow.execute(test_state)

        # Verify execution
        harness.assert_execution_order(["analysis_agent", "summary_agent"])

        # Verify results
        assert "analysis" in result.workflow_data
        assert "summary" in result.workflow_data
        assert "complex topic" in result.workflow_data["analysis"]["response"]
        assert "Summary:" in result.workflow_data["summary"]["response"]