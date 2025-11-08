"""
Migration helpers for transitioning legacy systems to Arshai.
"""

from typing import Any, Dict, Optional, Callable, Type
import inspect
import logging
import asyncio
from arshai.agents.base import BaseAgent
from arshai.core.interfaces.iagent import IAgentInput
from arshai.core.interfaces.iworkflow import IWorkflowState
from arshai.workflows import BaseWorkflowOrchestrator, BaseNode

logger = logging.getLogger(__name__)


class MigrationHelper:
    """
    Utilities for migrating existing systems to Arshai.

    This helper provides tools to gradually migrate legacy code
    to Arshai without breaking existing functionality.

    Example:
        # Wrap legacy agent
        legacy_agent = MyLegacyAgent()
        arshai_agent = MigrationHelper.wrap_legacy_agent(
            legacy_agent,
            name="my_agent"
        )

        # Use in Arshai workflow
        workflow = MyWorkflow()
        workflow.add_node(arshai_agent)
    """

    @staticmethod
    def wrap_legacy_agent(
        legacy_agent: Any,
        name: Optional[str] = None,
        input_converter: Optional[Callable] = None,
        output_converter: Optional[Callable] = None
    ) -> BaseAgent:
        """
        Wrap a legacy agent to be Arshai-compatible.

        Args:
            legacy_agent: Legacy agent instance
            name: Agent name (defaults to class name)
            input_converter: Function to convert IAgentInput to legacy format
            output_converter: Function to convert legacy output to Arshai format

        Returns:
            Arshai-compatible agent
        """

        agent_name = name or legacy_agent.__class__.__name__

        class WrappedAgent(BaseAgent):
            """Dynamically created wrapper for legacy agent"""

            def __init__(self):
                super().__init__()
                self._name = agent_name
                self.legacy_agent = legacy_agent
                self.input_converter = input_converter or MigrationHelper._default_input_converter
                self.output_converter = output_converter or MigrationHelper._default_output_converter

            async def process_message(self, input_data: IAgentInput) -> tuple:
                """Execute legacy agent with conversion"""
                # Convert input
                legacy_input = self.input_converter(input_data)

                # Find and call the main method
                main_method = MigrationHelper._find_main_method(self.legacy_agent)

                # Execute based on method type
                if inspect.iscoroutinefunction(main_method):
                    result = await main_method(legacy_input)
                else:
                    # Run sync method in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, main_method, legacy_input)

                # Convert output
                return self.output_converter(result), {}

        return WrappedAgent()

    @staticmethod
    def _find_main_method(agent: Any) -> Callable:
        """Find the main execution method of legacy agent"""
        # Common method names in order of preference
        method_names = ['execute', 'run', 'analyze', 'process', 'call', '__call__']

        for name in method_names:
            if hasattr(agent, name) and callable(getattr(agent, name)):
                return getattr(agent, name)

        raise ValueError(f"Could not find main method in {agent.__class__.__name__}")

    @staticmethod
    def _default_input_converter(input_data: IAgentInput) -> Dict[str, Any]:
        """Default converter from IAgentInput to dict"""
        # IAgentInput typically has message and conversation_id
        return {
            "message": input_data.message,
            "conversation_id": input_data.conversation_id,
            "metadata": getattr(input_data, 'metadata', {}),
        }

    @staticmethod
    def _default_output_converter(result: Any) -> str:
        """Default converter from any result to string"""
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Try to get message/response/result field
            return result.get("message", result.get("response", result.get("result", str(result))))
        else:
            return str(result)

    @staticmethod
    def create_adapter_workflow(
        legacy_orchestrator: Any,
        name: str = "adapted_workflow"
    ) -> BaseWorkflowOrchestrator:
        """
        Create Arshai workflow from legacy orchestrator.

        Args:
            legacy_orchestrator: Legacy orchestrator/pipeline
            name: Workflow name

        Returns:
            Arshai workflow orchestrator
        """
        # This would need to be implemented based on specific legacy system patterns
        # For now, providing a basic structure
        logger.warning("create_adapter_workflow is a template - customize for your legacy system")

        class AdaptedWorkflow(BaseWorkflowOrchestrator):
            def __init__(self):
                super().__init__()
                self.legacy_orchestrator = legacy_orchestrator

            async def execute(self, state: IWorkflowState) -> IWorkflowState:
                # Adapt execution - this is a template
                logger.info(f"Executing adapted workflow: {name}")
                # Your adaptation logic here
                return state

        return AdaptedWorkflow()


class LegacyAdapter:
    """
    Adapter to use Arshai components in legacy systems.

    This allows gradual migration by using Arshai components
    within existing legacy code.

    Example:
        # Use Arshai agent in legacy system
        arshai_agent = MyArshaiAgent()
        legacy_compatible = LegacyAdapter.adapt_agent(arshai_agent)

        # Call with legacy context
        result = legacy_compatible.process({"data": "value"})
    """

    @staticmethod
    def adapt_agent(arshai_agent: BaseAgent) -> Any:
        """
        Adapt Arshai agent for use in legacy system.

        Args:
            arshai_agent: Arshai agent instance

        Returns:
            Legacy-compatible wrapper
        """

        class LegacyWrapper:
            """Wrapper to make Arshai agent legacy-compatible"""

            def __init__(self):
                self.arshai_agent = arshai_agent

            def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
                """Legacy synchronous interface"""
                # Convert to IAgentInput
                input_data = IAgentInput(
                    message=context.get("message", ""),
                    conversation_id=context.get("conversation_id", "default")
                )

                # Run async agent
                loop = asyncio.new_event_loop()
                try:
                    result, metadata = loop.run_until_complete(
                        self.arshai_agent.process_message(input_data)
                    )
                finally:
                    loop.close()

                # Return as dict
                return {
                    "result": result,
                    "metadata": metadata
                }

            async def process_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
                """Legacy async interface"""
                # Convert to IAgentInput
                input_data = IAgentInput(
                    message=context.get("message", ""),
                    conversation_id=context.get("conversation_id", "default")
                )

                # Run agent
                result, metadata = await self.arshai_agent.process_message(input_data)

                # Return as dict
                return {
                    "result": result,
                    "metadata": metadata
                }

        return LegacyWrapper()

    @staticmethod
    def adapt_workflow(workflow: BaseWorkflowOrchestrator) -> Any:
        """
        Adapt Arshai workflow for use in legacy system.

        Args:
            workflow: Arshai workflow instance

        Returns:
            Legacy-compatible wrapper
        """

        class LegacyWorkflowWrapper:
            """Wrapper to make Arshai workflow legacy-compatible"""

            def __init__(self):
                self.workflow = workflow

            def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """Legacy synchronous interface"""
                # Create minimal IWorkflowState
                from arshai.core.interfaces.iworkflow import IUserContext

                state = IWorkflowState(
                    user_context=IUserContext(user_id=data.get("user_id", "legacy")),
                    workflow_data=data
                )

                # Run workflow
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        self.workflow.execute(state)
                    )
                finally:
                    loop.close()

                # Return workflow data
                return result.workflow_data

            async def execute_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """Legacy async interface"""
                from arshai.core.interfaces.iworkflow import IUserContext

                # Convert to IWorkflowState
                state = IWorkflowState(
                    user_context=IUserContext(user_id=data.get("user_id", "legacy")),
                    workflow_data=data
                )

                # Run workflow
                result = await self.workflow.execute(state)

                # Return workflow data
                return result.workflow_data

        return LegacyWorkflowWrapper()


class CompatibilityValidator:
    """
    Validate compatibility between legacy and Arshai systems.

    Example:
        validator = CompatibilityValidator()

        # Check agent compatibility
        legacy_agent = MyLegacyAgent()
        arshai_agent = MigrationHelper.wrap_legacy_agent(legacy_agent)

        is_compatible = await validator.validate_agent_compatibility(
            legacy_agent,
            arshai_agent,
            test_input={"test": "data"}
        )
    """

    @staticmethod
    async def validate_agent_compatibility(
        legacy_agent: Any,
        arshai_agent: BaseAgent,
        test_input: Dict[str, Any]
    ) -> bool:
        """
        Validate that wrapped agent produces compatible results.

        Args:
            legacy_agent: Original legacy agent
            arshai_agent: Wrapped Arshai agent
            test_input: Test input to validate with

        Returns:
            True if agents produce compatible results
        """
        try:
            # Run legacy agent
            main_method = MigrationHelper._find_main_method(legacy_agent)
            if inspect.iscoroutinefunction(main_method):
                legacy_result = await main_method(test_input)
            else:
                legacy_result = main_method(test_input)

            # Run Arshai agent
            input_data = IAgentInput(
                message=test_input.get("message", "test"),
                conversation_id=test_input.get("conversation_id", "test")
            )
            arshai_result, _ = await arshai_agent.process_message(input_data)

            # Compare results (basic comparison)
            logger.info(f"Legacy result: {legacy_result}")
            logger.info(f"Arshai result: {arshai_result}")

            # Basic compatibility check - customize as needed
            if isinstance(legacy_result, str) and isinstance(arshai_result, str):
                return legacy_result == arshai_result

            return True  # Consider compatible if no errors

        except Exception as e:
            logger.error(f"Compatibility validation failed: {e}")
            return False