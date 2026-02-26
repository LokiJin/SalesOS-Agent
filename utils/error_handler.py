"""
Centralized error handling for tools and API
"""
from functools import wraps
from typing import Callable
import traceback


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
            error_msg = f"Error in {func.__name__}: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Print traceback in debug mode
            if hasattr(e, '__traceback__'):
                print("Traceback:")
                traceback.print_tb(e.__traceback__)
            
            # Return user-friendly error message
            return f"I encountered an error while using the {func.__name__} tool: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
    
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
        error_msg = f"Error executing {func.__name__}: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return False, None, error_msg
