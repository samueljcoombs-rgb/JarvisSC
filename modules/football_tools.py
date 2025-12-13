# modules/football_tools.py

import ast
import json
import datetime
import importlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODULES_DIR = BASE_DIR / "modules"
REGISTRY_PATH = MODULES_DIR / "bot_registry.json"

# ⚠️ Existing core modules the football bot MUST NOT edit
PROTECTED_FILES = {
    "layout_manager.py",
    "chat_ui.py",
    "weather_panel.py",
    "podcasts_panel.py",
    "athletic_feed.py",
    "todos_panel.py",
    "__init__.py",
}

# ⚽ Data path – you can override with env var FOOTBALL_DATA_PATH
DEFAULT_DATA_PATH = "/Users/SamECee/football_ai/data/raw/football_ai_NNIA.csv"
DATA_PATH = Path(os.getenv("FOOTBALL_DATA_PATH", DEFAULT_DATA_PATH))


# ---------- Registry helpers ----------

def _load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            with REGISTRY_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if "owned" not in data:
                    data["owned"] = []
                return data
        except Exception:
            pass
    return {"owned": []}


def _save_registry(reg: Dict[str, Any]) -> None:
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)


def _is_owned(file_name: str) -> bool:
    reg = _load_registry()
    return file_name in reg.get("owned", [])


def _add_owned(file_name: str) -> None:
    reg = _load_registry()
    if file_name not in reg["owned"]:
        reg["owned"].append(file_name)
        _save_registry(reg)


# ---------- Module management tools ----------

def list_modules() -> Dict[str, Any]:
    """
    List all modules in /modules with flags:
    - owned: created/managed by football bot
    - protected: core Jarvis modules
    """
    modules: List[Dict[str, Any]] = []
    for p in MODULES_DIR.glob("*.py"):
        name = p.name
        modules.append(
            {
                "name": name,
                "owned": _is_owned(name),
                "protected": name in PROTECTED_FILES,
            }
        )
    return {"modules": modules}


def read_module(path: str) -> Dict[str, Any]:
    """
    Read the contents of a module file relative to /modules.
    Example: "my_strategy.py"
    """
    full_path = MODULES_DIR / path
    if not full_path.exists():
        return {"error": f"Module {path} not found."}
    try:
        code = full_path.read_text(encoding="utf-8")
        return {"path": str(full_path), "code": code}
    except Exception as e:
        return {"error": str(e)}


def write_module(path: str, code: str) -> Dict[str, Any]:
    """
    Create or update a module in /modules with strong safety:
    - Cannot modify PROTECTED_FILES
    - Cannot modify non-owned existing modules
    - Can freely create new modules
    - Can edit any module it previously created (owned)
    - Validates syntax before saving
    - Creates timestamped backup on overwrite
    - Registers new modules in bot_registry.json
    """
    # Normalize file name (no directories, ensure .py)
    name = Path(path).name
    if not name.endswith(".py"):
        name = f"{name}.py"

    full_path = MODULES_DIR / name

    # Safety: do not allow touching protected files
    if name in PROTECTED_FILES:
        return {"error": f"Modification blocked: {name} is a protected core module."}

    # If file exists and is not owned by the football bot, block edit
    if full_path.exists() and not _is_owned(name):
        return {
            "error": f"Modification blocked: {name} exists and is not owned by Football Bot."
        }

    # Validate syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    # Backup if file exists
    if full_path.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = MODULES_DIR / f"{name}.bak_{timestamp}"
        backup_path.write_text(full_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Write new code
    full_path.write_text(code, encoding="utf-8")

    # Mark as owned by football bot
    _add_owned(name)

    return {
        "status": "success",
        "path": str(full_path),
        "owned": True,
    }


def run_module(path: str, function_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Import a module from /modules and run a named function with kwargs.
    Example:
      run_module("my_strategy.py", "run_backtest", {"threshold": 0.05})
    """
    args = args or {}
    name = Path(path).name
    if not name.endswith(".py"):
        name = f"{name}.py"

    full_path = MODULES_DIR / name
    if not full_path.exists():
        return {"error": f"Module {name} not found."}

    module_name = f"modules.{name[:-3]}"  # strip .py

    try:
        if module_name in list(importlib.sys.modules.keys()):
            # Reload to pick up changes
            mod = importlib.reload(importlib.sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)

        if not hasattr(mod, function_name):
            return {"error": f"Function {function_name} not found in {name}."}

        fn = getattr(mod, function_name)
        result = fn(**args)
        return {"result": result}
    except Exception as e:
        # Return the error string so the bot can read & fix it
        return {"error": str(e)}


# ---------- Data / ROI tools ----------

def load_data_basic(limit: int = 200) -> Dict[str, Any]:
    """
    Load a preview of the football dataset for the bot:
      - rows (up to `limit`)
      - columns
      - basic info
    """
    if not DATA_PATH.exists():
        return {"error": f"Data file not found at {DATA_PATH}. Set FOOTBALL_DATA_PATH env variable or update DEFAULT_DATA_PATH."}

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    sample = df.head(limit).to_dict(orient="records")
    return {
        "rows": int(len(df)),
        "cols": list(df.columns),
        "sample": sample,
    }


def list_columns() -> Dict[str, Any]:
    """
    Return just the column names for quick inspection.
    """
    if not DATA_PATH.exists():
        return {"error": f"Data file not found at {DATA_PATH}."}
    df = pd.read_csv(DATA_PATH, nrows=5)
    df.columns = [c.strip() for c in df.columns]
    return {"columns": list(df.columns)}


def basic_roi_for_pl_column(pl_column: str) -> Dict[str, Any]:
    """
    Compute a simple PL / ROI summary for a given PL column (e.g. 'BO 2.5 PL').
    Uses NO GAMES if present to normalise.
    """
    if not DATA_PATH.exists():
        return {"error": f"Data file not found at {DATA_PATH}."}

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    if pl_column not in df.columns:
        return {"error": f"Column {pl_column} not found."}

    if "NO GAMES" not in df.columns:
        return {"error": "NO GAMES column missing; cannot compute per-game ROI."}

    df = df[df[pl_column].notna() & df["NO GAMES"].notna()]
    if df.empty:
        return {"error": "No rows with non-null PL and NO GAMES."}

    total_pl = float(df[pl_column].sum())
    total_games = float(df["NO GAMES"].sum())
    roi_per_game = total_pl / total_games if total_games > 0 else None

    return {
        "pl_column": pl_column,
        "total_pl": total_pl,
        "total_games": total_games,
        "roi_per_game": roi_per_game,
        "rows": int(len(df)),
    }
