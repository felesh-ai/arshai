"""
Batch processing utilities for workflows.
"""

from typing import List, Optional, Dict, Any, Tuple, Callable
import asyncio
from dataclasses import dataclass
from arshai.workflows import BaseWorkflowOrchestrator
from arshai.core.interfaces.iworkflow import IWorkflowState
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result from batch processing."""
    successful: List[IWorkflowState]
    failed: List[Tuple[IWorkflowState, Exception]]
    total: int
    success_rate: float


class BatchProcessor:
    """
    Utility for batch processing with workflows.

    Example:
        processor = BatchProcessor()

        # Process batch of items
        states = [IWorkflowState(user_context=..., workflow_data=item) for item in items]
        result = await processor.execute_batch(
            workflow=my_workflow,
            states=states,
            batch_size=10,
            parallel=True
        )

        print(f"Success rate: {result.success_rate:.2%}")
    """

    @staticmethod
    async def execute_batch(
        workflow: BaseWorkflowOrchestrator,
        states: List[IWorkflowState],
        batch_size: int = 10,
        parallel: bool = True,
        continue_on_error: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Execute workflow on batch of states.

        Args:
            workflow: Workflow to execute
            states: List of states to process
            batch_size: Number of items per batch
            parallel: Execute batch items in parallel
            continue_on_error: Continue processing on errors
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with successful and failed items
        """
        successful = []
        failed = []
        total = len(states)

        # Process in batches
        for i in range(0, total, batch_size):
            batch = states[i:i + batch_size]

            if parallel:
                # Execute batch in parallel
                results = await asyncio.gather(
                    *[workflow.execute(state) for state in batch],
                    return_exceptions=True
                )

                # Separate successful and failed
                for state, result in zip(batch, results):
                    if isinstance(result, Exception):
                        failed.append((state, result))
                        if not continue_on_error:
                            raise result
                    else:
                        successful.append(result)

            else:
                # Execute batch sequentially
                for state in batch:
                    try:
                        result = await workflow.execute(state)
                        successful.append(result)
                    except Exception as e:
                        failed.append((state, e))
                        if not continue_on_error:
                            raise

            # Progress callback
            if progress_callback:
                processed = len(successful) + len(failed)
                progress_callback(processed, total)

        # Calculate success rate
        success_rate = len(successful) / total if total > 0 else 0

        return BatchResult(
            successful=successful,
            failed=failed,
            total=total,
            success_rate=success_rate
        )

    @staticmethod
    async def execute_batch_with_retries(
        workflow: BaseWorkflowOrchestrator,
        states: List[IWorkflowState],
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        parallel: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Execute batch with automatic retries for failed items.

        Args:
            workflow: Workflow to execute
            states: List of states to process
            batch_size: Number of items per batch
            max_retries: Maximum number of retries for failed items
            retry_delay: Delay in seconds between retries
            parallel: Execute batch items in parallel
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with final successful and failed items
        """
        # First pass
        result = await BatchProcessor.execute_batch(
            workflow=workflow,
            states=states,
            batch_size=batch_size,
            parallel=parallel,
            continue_on_error=True,
            progress_callback=progress_callback
        )

        # Retry failed items
        retry_count = 0
        while result.failed and retry_count < max_retries:
            retry_count += 1
            logger.info(f"Retrying {len(result.failed)} failed items (attempt {retry_count}/{max_retries})")

            # Wait before retry
            await asyncio.sleep(retry_delay)

            # Extract states from failed items
            retry_states = [state for state, _ in result.failed]

            # Retry failed items
            retry_result = await BatchProcessor.execute_batch(
                workflow=workflow,
                states=retry_states,
                batch_size=batch_size,
                parallel=parallel,
                continue_on_error=True
            )

            # Update results
            result.successful.extend(retry_result.successful)
            result.failed = retry_result.failed

            # Recalculate success rate
            result.success_rate = len(result.successful) / result.total

        return result

    @staticmethod
    def create_progress_logger(name: str = "batch_progress") -> Callable[[int, int], None]:
        """
        Create a simple progress logging callback.

        Args:
            name: Name for the logger

        Returns:
            Progress callback function
        """
        def log_progress(processed: int, total: int):
            percentage = (processed / total * 100) if total > 0 else 0
            logger.info(f"[{name}] Progress: {processed}/{total} ({percentage:.1f}%)")

        return log_progress