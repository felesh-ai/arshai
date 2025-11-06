# Testing Best Practices for Arshai Framework

## Overview

This guide provides best practices for testing Arshai workflows, agents, and other components using the built-in testing utilities.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Using WorkflowTestHarness](#using-workflowtestharness)
3. [Mocking Strategies](#mocking-strategies)
4. [Testing Patterns](#testing-patterns)
5. [Performance Testing](#performance-testing)
6. [Common Pitfalls](#common-pitfalls)

## Testing Philosophy

### Key Principles

1. **Test in Isolation**: Mock external dependencies to test components independently
2. **Test Happy Path and Edge Cases**: Cover both successful and failure scenarios
3. **Use Harness for Complex Workflows**: Leverage WorkflowTestHarness for multi-step workflows
4. **Record and Assert**: Track execution paths and verify expected behavior

## Using WorkflowTestHarness

The `WorkflowTestHarness` is your primary tool for testing workflows. It provides:

- Execution tracking
- Node mocking
- Input/output recording
- Assertion helpers

### Basic Setup

```python
from arshai.testing import WorkflowTestHarness
from arshai.workflows import BaseWorkflowOrchestrator
from arshai.core.interfaces.iworkflow import IWorkflowState, IUserContext

@pytest.mark.asyncio
async def test_my_workflow():
    # Create harness
    harness = WorkflowTestHarness()

    # Create workflow
    workflow = MyWorkflow()

    # Create test state
    state = IWorkflowState(
        user_context=IUserContext(user_id="test_user"),
        workflow_data={"input": "test_data"}
    )

    # Mock external dependencies
    mock_nodes = {
        "external_api": {"status": "success", "data": "mocked"},
        "database": {"saved": True, "id": "123"}
    }

    # Execute with mocks
    result = await harness.test_workflow(
        workflow=workflow,
        input_state=state,
        mock_nodes=mock_nodes,
        record_execution=True
    )

    # Assert execution
    harness.assert_execution_order(["validate", "external_api", "database"])
    harness.assert_node_executed("external_api")
```

## Mocking Strategies

### 1. Mock External Services

Always mock external services to avoid:
- Network dependencies
- Rate limiting
- Test data pollution
- Unpredictable failures

```python
mock_nodes = {
    "openai_api": {"response": "AI generated text", "tokens": 100},
    "database_save": {"success": True, "id": "test_123"},
    "email_service": {"sent": True, "message_id": "msg_456"}
}
```

### 2. Mock Agents

Use the harness to create mock agents:

```python
mock_agent = harness.create_mock_agent(
    name="analysis_agent",
    response="This is the analysis result"
)
```

### 3. Mock Memory Managers

For testing caching behavior:

```python
from arshai.testing.harness import MockMemoryManager

agent = MyAgent()
agent.memory = MockMemoryManager()

# Test caching
result1 = await agent.cached_method(data)
result2 = await agent.cached_method(data)  # Should use cache
```

## Testing Patterns

### 1. Test Linear Workflows

For sequential workflows:

```python
@pytest.mark.asyncio
async def test_linear_workflow():
    harness = WorkflowTestHarness()

    # Mock each step
    mock_nodes = {
        "step1": {"result": "processed"},
        "step2": {"result": "analyzed"},
        "step3": {"result": "completed"}
    }

    result = await harness.test_workflow(...)

    # Verify sequential execution
    harness.assert_execution_order(["step1", "step2", "step3"])
```

### 2. Test Conditional Paths

For workflows with branching:

```python
@pytest.mark.asyncio
async def test_conditional_path_a():
    harness = WorkflowTestHarness()

    # Setup condition for path A
    mock_nodes = {
        "condition": {"score": 0.8},  # High score
        "path_a": {"result": "premium"},
        "path_b": {"result": "basic"}
    }

    result = await harness.test_workflow(...)

    # Verify only path A was taken
    harness.assert_node_executed("path_a")
    harness.assert_node_not_executed("path_b")
```

### 3. Test Error Handling

For testing failure scenarios:

```python
@pytest.mark.asyncio
async def test_error_handling():
    harness = WorkflowTestHarness()

    # Create failing node
    async def failing_node(state):
        raise ValueError("Simulated failure")

    workflow.nodes["problematic"] = failing_node

    # Test with fallback
    with pytest.raises(ValueError):
        await harness.test_workflow(...)

    # Or test recovery
    fallback_workflow = FallbackWorkflow(
        primary=workflow,
        fallbacks=[backup_workflow]
    )

    result = await fallback_workflow.execute(state)
    assert result.workflow_data.get("_fallback_used") is True
```

### 4. Test Batch Processing

For testing batch operations:

```python
@pytest.mark.asyncio
async def test_batch_processing():
    processor = BatchProcessor()

    # Create test batch
    states = [create_test_state(i) for i in range(10)]

    # Process with monitoring
    def progress_callback(processed, total):
        print(f"Progress: {processed}/{total}")

    result = await processor.execute_batch(
        workflow=workflow,
        states=states,
        batch_size=3,
        parallel=True,
        progress_callback=progress_callback
    )

    assert result.success_rate >= 0.9  # 90% success rate
```

### 5. Test Circuit Breaker

For testing resilience patterns:

```python
from arshai.plugins.resilience import CircuitBreakerPlugin

@pytest.mark.asyncio
async def test_circuit_breaker():
    cb = CircuitBreakerPlugin(CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=10
    ))

    # Simulate failures
    for i in range(3):
        with pytest.raises(ValueError):
            await cb.call_async(failing_function)

    # Circuit should be open
    with pytest.raises(CircuitBreakerError):
        await cb.call_async(failing_function)

    assert cb.state == CircuitState.OPEN
```

## Performance Testing

### 1. Benchmark Execution Time

```python
import time

@pytest.mark.asyncio
async def test_performance():
    harness = WorkflowTestHarness()

    start_time = time.time()

    # Process large batch
    states = [create_state(i) for i in range(100)]
    result = await processor.execute_batch(
        workflow=workflow,
        states=states,
        batch_size=20,
        parallel=True
    )

    elapsed = time.time() - start_time

    # Assert performance requirements
    assert elapsed < 10.0  # Should complete within 10 seconds
    assert result.success_rate > 0.95
```

### 2. Test Caching Efficiency

```python
@pytest.mark.asyncio
async def test_cache_performance():
    agent = MyAgent()
    agent.memory = MockMemoryManager()

    # First call - no cache
    start = time.time()
    result1 = await agent.expensive_operation(data)
    first_call_time = time.time() - start

    # Second call - should use cache
    start = time.time()
    result2 = await agent.expensive_operation(data)
    cached_call_time = time.time() - start

    # Cache should be significantly faster
    assert cached_call_time < first_call_time * 0.1
    assert result1 == result2
```

## Common Pitfalls

### 1. Not Resetting Harness Between Tests

Always reset the harness if reusing:

```python
def test_multiple_scenarios():
    harness = WorkflowTestHarness()

    # Test 1
    await harness.test_workflow(...)

    # Reset before next test
    harness.reset()

    # Test 2
    await harness.test_workflow(...)
```

### 2. Forgetting to Mock External Services

Never let tests hit real external services:

```python
# ❌ Bad - hits real API
workflow.nodes["openai"] = real_openai_client

# ✅ Good - uses mock
mock_nodes = {
    "openai": {"response": "mocked response"}
}
```

### 3. Not Testing Error Paths

Always test failure scenarios:

```python
# Test both success and failure
@pytest.mark.parametrize("should_fail", [False, True])
async def test_with_failures(should_fail):
    if should_fail:
        mock_nodes["critical_step"] = None  # Will cause error
    else:
        mock_nodes["critical_step"] = {"success": True}

    if should_fail:
        with pytest.raises(Exception):
            await harness.test_workflow(...)
    else:
        result = await harness.test_workflow(...)
        assert result is not None
```

### 4. Ignoring Async Behavior

Remember to handle async properly:

```python
# ❌ Bad - not awaiting
result = workflow.execute(state)  # Returns coroutine!

# ✅ Good - proper async
result = await workflow.execute(state)
```

### 5. Not Verifying State Mutations

Check that state is properly updated:

```python
initial_data = state.workflow_data.copy()

result = await harness.test_workflow(...)

# Verify state was modified as expected
assert result.workflow_data != initial_data
assert "processed_data" in result.workflow_data
```

## Test Organization

### Directory Structure

```
tests/
├── unit/
│   ├── test_agents/
│   ├── test_workflows/
│   └── test_tools/
├── integration/
│   ├── test_workflow_integration.py
│   └── test_agent_integration.py
├── fixtures/
│   ├── workflow_fixtures.py
│   └── agent_fixtures.py
└── conftest.py
```

### Fixture Examples

```python
# fixtures/workflow_fixtures.py
import pytest
from arshai.testing import WorkflowTestHarness

@pytest.fixture
def harness():
    """Provide a fresh test harness."""
    return WorkflowTestHarness()

@pytest.fixture
def test_state():
    """Provide a standard test state."""
    return IWorkflowState(
        user_context=IUserContext(user_id="test_user"),
        workflow_data={"test": True}
    )

@pytest.fixture
def mock_external_services():
    """Standard mocks for external services."""
    return {
        "database": {"connected": True},
        "api": {"status": "healthy"},
        "cache": {"available": True}
    }
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test Arshai Components

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install poetry
        poetry install

    - name: Run tests
      run: |
        poetry run pytest tests/ \
          --cov=arshai \
          --cov-report=xml \
          --cov-report=term

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Summary

Following these best practices will help you:

1. Write reliable, maintainable tests
2. Catch issues early in development
3. Document expected behavior through tests
4. Enable confident refactoring
5. Improve overall code quality

Remember: Good tests are an investment in your project's future maintainability and reliability.