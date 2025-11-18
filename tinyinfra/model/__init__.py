"""
Model wrappers for LLM inference
All wrapper files in this directory are auto-discovered
"""
import importlib
import inspect
from pathlib import Path

# Auto-discover wrapper classes from files in this directory
_wrapper_classes = {}

def _discover_wrappers():
    """Auto-discover all wrapper classes in model/ directory"""
    model_dir = Path(__file__).parent

    # Find all .py files except __init__.py
    for py_file in model_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        # Import the module
        module_name = py_file.stem
        module = importlib.import_module(f".{module_name}", package="tinyinfra.model")

        # Find the main wrapper class - look for class with "generate" method
        wrapper_class = None
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes
            if obj.__module__ != f"tinyinfra.model.{module_name}":
                continue
            # Check if it has a generate method (main wrapper interface)
            if hasattr(obj, 'generate') and callable(getattr(obj, 'generate')):
                wrapper_class = obj
                # Also export the class
                globals()[name] = obj
                break

        if wrapper_class:
            _wrapper_classes[module_name] = wrapper_class

_discover_wrappers()

__all__ = list(_wrapper_classes.keys()) + [cls.__name__ for cls in _wrapper_classes.values()]

def get_wrapper_registry():
    """Get the registry of available wrappers"""
    return _wrapper_classes.copy()

