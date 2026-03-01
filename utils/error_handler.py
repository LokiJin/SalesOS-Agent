"""
Centralized error handling for tools and API
"""
from functools import wraps
from typing import Callable
from utils.logger import get_logger

logger = get_logger(__name__)

def tool_error_handler(func: Callable) -> Callable:
    """
    Decorator to handle tool errors gracefully.
    
    Wraps tool functions to catch exceptions and return user-friendly error messages
    instead of crashing the agent.
    
    Example:
        @tool
        @tool_error_handler
        def my_tool(arg: str) -> str:
            # ... tool logic ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            logger.error(
                f"Error in tool '{func.__name__}'",
                exc_info=True
            )

            return (
                f"I encountered an error while using the "
                f"{func.__name__} tool: {str(e)}. "
                f"Please try rephrasing your question or contact support if the issue persists."
            )

    return wrapper


def safe_execute(func: Callable, *args, **kwargs):
    """
    Safely execute a function and return result or error.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        Tuple of (success: bool, result: any, error: str)
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    
    except Exception as e:
        
        logger.error(
            f"Error executing {func.__name__}",
            exc_info=True
        )

        return False, None, str(e)
