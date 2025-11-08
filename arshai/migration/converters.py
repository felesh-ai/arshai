"""
Context and result converters for migration between different formats.
"""

from typing import Any, Dict, List, Optional
from arshai.core.interfaces.iagent import IAgentInput


class ContextConverter:
    """
    Convert between different context formats.

    Supports conversion between common formats used in
    various agent systems.
    """

    @staticmethod
    def dict_to_agent_input(data: Dict[str, Any]) -> IAgentInput:
        """Convert dictionary to IAgentInput"""
        return IAgentInput(
            message=data.get("message", data.get("text", data.get("input", ""))),
            conversation_id=data.get("conversation_id", data.get("session_id", "default"))
        )

    @staticmethod
    def agent_input_to_dict(input_data: IAgentInput) -> Dict[str, Any]:
        """Convert IAgentInput to dictionary"""
        return {
            "message": input_data.message,
            "conversation_id": input_data.conversation_id,
            "metadata": getattr(input_data, 'metadata', {})
        }

    @staticmethod
    def flatten_nested_context(context: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested dictionary structure.

        Example:
            {"user": {"name": "John", "age": 30}}
            becomes
            {"user.name": "John", "user.age": 30}
        """
        result = {}
        for key, value in context.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(ContextConverter.flatten_nested_context(value, new_key))
            else:
                result[new_key] = value
        return result

    @staticmethod
    def unflatten_context(flat_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unflatten dictionary structure.

        Example:
            {"user.name": "John", "user.age": 30}
            becomes
            {"user": {"name": "John", "age": 30}}
        """
        result = {}
        for key, value in flat_context.items():
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result


class ResultConverter:
    """
    Convert between different result formats.
    """

    @staticmethod
    def normalize_result(result: Any) -> Dict[str, Any]:
        """
        Normalize various result types to dictionary.

        Handles:
        - Strings -> {"message": string}
        - Lists -> {"results": list}
        - Objects -> {"value": object}
        - Dicts -> returned as-is
        """
        if isinstance(result, dict):
            return result
        elif isinstance(result, str):
            return {"message": result}
        elif isinstance(result, list):
            return {"results": result}
        elif hasattr(result, "__dict__"):
            return result.__dict__
        else:
            return {"value": result}

    @staticmethod
    def extract_message(result: Any) -> str:
        """
        Extract a message string from various result formats.

        Tries common field names: message, response, result, text, output
        """
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Try common field names
            for field in ["message", "response", "result", "text", "output", "answer"]:
                if field in result:
                    value = result[field]
                    if isinstance(value, str):
                        return value
                    else:
                        return str(value)
            # If no common field, return string representation
            return str(result)
        elif isinstance(result, list) and result:
            # If list, try to get first item
            return ResultConverter.extract_message(result[0])
        else:
            return str(result)

    @staticmethod
    def merge_results(results: List[Any]) -> Dict[str, Any]:
        """
        Merge multiple results into a single dictionary.

        Useful for combining outputs from multiple agents.
        """
        merged = {
            "merged_results": [],
            "count": len(results)
        }

        for i, result in enumerate(results):
            normalized = ResultConverter.normalize_result(result)
            merged["merged_results"].append({
                "index": i,
                "result": normalized
            })

        return merged