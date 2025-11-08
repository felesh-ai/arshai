"""
Caching decorators for Arshai agents and workflows.
"""

import functools
import hashlib
import json
from typing import Any, Callable, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def cache_result(
    ttl: int = 300,
    key_prefix: str = "",
    key_func: Optional[Callable] = None,
    cache_none: bool = False
):
    """
    Decorator to cache agent/workflow results.

    This decorator provides transparent caching for expensive operations.
    It works with any object that has a 'memory' attribute implementing
    get/set methods.

    Example:
        class MyAgent(BaseAgent):
            @cache_result(ttl=600, key_prefix="analysis")
            async def analyze_data(self, input_data):
                # Expensive operation
                return result

            @cache_result(ttl=3600)
            def process_sync(self, data):
                # Sync method also supported
                return processed_data

    Args:
        ttl: Time to live in seconds (default 300)
        key_prefix: Prefix for cache keys
        key_func: Optional function to generate cache key
        cache_none: Whether to cache None results (default False)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Check if object has memory manager
            if not hasattr(self, 'memory'):
                # No memory manager, execute without caching
                logger.debug(f"No memory manager found, executing {func.__name__} without cache")
                return await func(self, *args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(args, kwargs)
            else:
                # Default key generation
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                key_hash = hashlib.md5(
                    json.dumps(key_data, sort_keys=True).encode(),
                    usedforsecurity=False
                ).hexdigest()
                cache_key = f"{key_prefix}:{func.__name__}:{key_hash}" if key_prefix else f"{func.__name__}:{key_hash}"

            # Try to get from cache
            try:
                cached = await self.memory.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    # Use JSON for safe deserialization
                    try:
                        return json.loads(cached)
                    except (json.JSONDecodeError, TypeError):
                        # Fallback for non-JSON serializable cached data
                        logger.warning(f"Failed to deserialize cached data as JSON for {cache_key}")
                        return cached
            except Exception as e:
                logger.warning(f"Failed to get from cache: {e}")

            # Execute function
            result = await func(self, *args, **kwargs)

            # Cache result
            if result is not None or cache_none:
                try:
                    # Try JSON serialization first for security
                    try:
                        serialized = json.dumps(result)
                    except (TypeError, ValueError):
                        # For non-JSON serializable objects, convert to string representation
                        logger.debug(f"Result not JSON serializable, using string representation for {cache_key}")
                        serialized = json.dumps({"__repr__": str(result), "__type__": type(result).__name__})

                    await self.memory.set(
                        cache_key,
                        serialized,
                        ttl=ttl
                    )
                    logger.debug(f"Cached result for {cache_key} with TTL={ttl}")
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")

            return result

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Check if object has memory manager
            if not hasattr(self, 'memory'):
                # No memory manager, execute without caching
                logger.debug(f"No memory manager found, executing {func.__name__} without cache")
                return func(self, *args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(args, kwargs)
            else:
                # Default key generation
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                key_hash = hashlib.md5(
                    json.dumps(key_data, sort_keys=True).encode(),
                    usedforsecurity=False
                ).hexdigest()
                cache_key = f"{key_prefix}:{func.__name__}:{key_hash}" if key_prefix else f"{func.__name__}:{key_hash}"

            # Try to get from cache (sync version)
            try:
                # Assuming sync memory manager methods
                cached = self.memory.get_sync(cache_key) if hasattr(self.memory, 'get_sync') else None
                if cached is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    # Use JSON for safe deserialization
                    try:
                        return json.loads(cached)
                    except (json.JSONDecodeError, TypeError):
                        # Fallback for non-JSON serializable cached data
                        logger.warning(f"Failed to deserialize cached data as JSON for {cache_key}")
                        return cached
            except Exception as e:
                logger.warning(f"Failed to get from cache: {e}")

            # Execute function
            result = func(self, *args, **kwargs)

            # Cache result (sync version)
            if result is not None or cache_none:
                try:
                    if hasattr(self.memory, 'set_sync'):
                        # Try JSON serialization first for security
                        try:
                            serialized = json.dumps(result)
                        except (TypeError, ValueError):
                            # For non-JSON serializable objects, convert to string representation
                            logger.debug(f"Result not JSON serializable, using string representation for {cache_key}")
                            serialized = json.dumps({"__repr__": str(result), "__type__": type(result).__name__})

                        self.memory.set_sync(
                            cache_key,
                            serialized,
                            ttl=ttl
                        )
                        logger.debug(f"Cached result for {cache_key} with TTL={ttl}")
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")

            return result

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def create_key_func(include_args: bool = True, include_kwargs: bool = True, hash_objects: bool = True):
    """
    Create a custom key generation function.

    Args:
        include_args: Include positional arguments in key
        include_kwargs: Include keyword arguments in key
        hash_objects: Hash complex objects (otherwise use repr)

    Returns:
        Key generation function

    Example:
        @cache_result(key_func=create_key_func(include_kwargs=False))
        def my_func(self, x, y=10):
            # Only x will be used for cache key, y will be ignored
            return x * y
    """

    def key_func(args, kwargs):
        parts = []

        if include_args:
            for arg in args:
                if hash_objects and hasattr(arg, '__dict__'):
                    # Hash complex objects
                    parts.append(hashlib.md5(
                        str(arg.__dict__).encode(),
                        usedforsecurity=False
                    ).hexdigest())
                else:
                    parts.append(str(arg))

        if include_kwargs:
            for key, value in sorted(kwargs.items()):
                if hash_objects and hasattr(value, '__dict__'):
                    value_hash = hashlib.md5(
                        str(value.__dict__).encode(),
                        usedforsecurity=False
                    ).hexdigest()
                    parts.append(f"{key}={value_hash}")
                else:
                    parts.append(f"{key}={value}")

        return ":".join(parts)

    return key_func


def invalidate_cache(self, pattern: Optional[str] = None):
    """
    Helper function to invalidate cache entries.

    Can be used as a method in classes that use caching.

    Args:
        self: Object with memory manager
        pattern: Optional pattern to match keys (prefix)

    Example:
        class MyAgent(BaseAgent):
            def clear_analysis_cache(self):
                invalidate_cache(self, "analysis:")
    """
    if not hasattr(self, 'memory'):
        logger.warning("No memory manager found for cache invalidation")
        return

    try:
        if hasattr(self.memory, 'delete_pattern'):
            # If memory manager supports pattern deletion
            self.memory.delete_pattern(pattern or "*")
            logger.info(f"Invalidated cache with pattern: {pattern or '*'}")
        else:
            logger.warning("Memory manager does not support pattern deletion")
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")