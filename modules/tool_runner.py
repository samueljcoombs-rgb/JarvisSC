import importlib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = BASE_DIR / "modules"

def run_module(path: str, function_name: str, args: dict) -> dict:
    """
    Executes a function inside a Python module.
    Bot uses this to test its own code creations.
    """

    module_path = MODULE_DIR / path
    if not module_path.exists():
        return {"error": f"Module {path} not found."}

    module_name = f"modules.{path.replace('.py','')}"

    try:
        mod = importlib.import_module(module_name)
        func = getattr(mod, function_name)
        result = func(**args)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
