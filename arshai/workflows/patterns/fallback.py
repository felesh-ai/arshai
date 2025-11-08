"""
Fallback workflow pattern for resilient execution.
"""

from typing import Optional, List, Callable, Any
from arshai.workflows import BaseWorkflowOrchestrator
from arshai.core.interfaces.iworkflow import IWorkflowState
import logging

logger = logging.getLogger(__name__)


class FallbackWorkflow(BaseWorkflowOrchestrator):
    """
    Workflow with fallback mechanism for resilience.

    This pattern allows defining a primary workflow with one or more
    fallback workflows that execute if the primary fails.

    Example:
        primary = ComplexAnalysisWorkflow()
        fallback = SimpleAnalysisWorkflow()

        resilient = FallbackWorkflow(
            primary=primary,
            fallbacks=[fallback],
            retry_primary=True
        )

        result = await resilient.execute(state)
    """

    def __init__(
        self,
        primary: BaseWorkflowOrchestrator,
        fallbacks: List[BaseWorkflowOrchestrator],
        retry_primary: bool = False,
        fallback_condition: Optional[Callable[[Exception], bool]] = None,
        debug_mode: bool = False
    ):
        """
        Initialize fallback workflow.

        Args:
            primary: Primary workflow to execute
            fallbacks: List of fallback workflows in order
            retry_primary: Whether to retry primary once before fallback
            fallback_condition: Optional function to determine if fallback should be used
            debug_mode: Enable debug logging
        """
        super().__init__(debug_mode=debug_mode)

        self.primary = primary
        self.fallbacks = fallbacks
        self.retry_primary = retry_primary
        self.fallback_condition = fallback_condition or (lambda e: True)

    async def execute(self, state: IWorkflowState) -> IWorkflowState:
        """
        Execute with fallback mechanism.

        Args:
            state: Workflow state

        Returns:
            Result from primary or fallback workflow
        """
        last_exception = None

        # Try primary workflow
        try:
            logger.info(f"Executing primary workflow")
            if self._debug_mode:
                self._logger.debug(f"Primary workflow state: {state}")

            result = await self.primary.execute(state)

            # Mark successful primary execution
            if hasattr(result, 'workflow_data'):
                result.workflow_data["_fallback_used"] = False
                result.workflow_data["_workflow_type"] = "primary"

            return result

        except Exception as e:
            logger.warning(f"Primary workflow failed: {e}")
            last_exception = e

            # Retry primary if configured
            if self.retry_primary:
                try:
                    logger.info("Retrying primary workflow")
                    result = await self.primary.execute(state)

                    # Mark successful retry
                    if hasattr(result, 'workflow_data'):
                        result.workflow_data["_fallback_used"] = False
                        result.workflow_data["_workflow_type"] = "primary_retry"

                    return result

                except Exception as retry_e:
                    logger.warning(f"Primary retry failed: {retry_e}")
                    last_exception = retry_e

        # Check if we should use fallback
        if not self.fallback_condition(last_exception):
            raise last_exception

        # Try fallback workflows in order
        for i, fallback in enumerate(self.fallbacks):
            try:
                logger.info(f"Executing fallback workflow {i+1}")
                if self._debug_mode:
                    self._logger.debug(f"Fallback {i+1} state: {state}")

                result = await fallback.execute(state)

                # Mark as fallback result
                if hasattr(result, 'workflow_data'):
                    result.workflow_data["_fallback_used"] = True
                    result.workflow_data["_fallback_index"] = i
                    result.workflow_data["_workflow_type"] = f"fallback_{i}"

                return result

            except Exception as e:
                logger.warning(f"Fallback {i+1} failed: {e}")
                last_exception = e
                continue

        # All workflows failed
        logger.error("All workflows (primary and fallbacks) failed")

        # Add failure information to state
        if hasattr(state, 'errors'):
            state.errors.append({
                "type": "all_workflows_failed",
                "message": str(last_exception),
                "timestamp": None  # Would need datetime import
            })

        raise last_exception

    def get_stats(self) -> dict:
        """Get statistics about the fallback workflow configuration."""
        return {
            "primary_configured": self.primary is not None,
            "fallback_count": len(self.fallbacks),
            "retry_enabled": self.retry_primary,
            "conditional_fallback": self.fallback_condition is not None
        }