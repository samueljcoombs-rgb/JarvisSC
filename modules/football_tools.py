# modules/football_tools.py

import ast
import json
import datetime
import importlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
from io import StringIO

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# ================ PATHS =====================

BASE_DIR = Path(__file__).resolve().parents[1]
MODULES_DIR = BASE_DIR / "modules"
REGISTRY_PATH = MODULES_DIR / "bot_registry.json"

# ================ PROTECTED MODULES ===================

PROTECTED_FILES = {
    "layout_manager.py",
    "chat_ui.py",
    "weather_panel.py",
    "podcasts_panel.py",
    "athletic_feed.py",
    "todos_panel.py",
    "__init__.py",
}

# ================ DATA LOADING (MAIN FOOTBALL CSV) =======================

# Google Drive file ID you provided
GDRIVE_FILE_ID = "1aYMC7YJ1qim-132aDc50hhNMdDm20WbC"

DEFAULT_DATA_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
DATA_URL_ENV = os.getenv("FOOTBALL_DATA_URL", "").strip()
DATA_URL = DATA_URL_ENV or DEFAULT_DATA_URL


def _load_full_df() -> pd.DataFrame:
    """
    Download the full football dataset from Google Drive and return as DataFrame.
    """
    if not DATA_URL:
        raise RuntimeError("DATA_URL not configured.")

    resp = requests.get(DATA_URL)
    resp.raise_for_status()

    csv_data = resp.content.decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(csv_data))
    df.columns = [c.strip() for c in df.columns]
    return df


# ================ REGISTRY HELPERS =======================

def _load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"owned": []}


def _save_registry(data: Dict[str, Any]) -> None:
    REGISTRY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _is_owned(fname: str) -> bool:
    return fname in _load_registry().get("owned", [])


def _add_owned(fname: str) -> None:
    reg = _load_registry()
    if fname not in reg["owned"]:
        reg["owned"].append(fname)
        _save_registry(reg)


# ================ MODULE MANAGEMENT =======================

def list_modules() -> Dict[str, Any]:
    listings: List[Dict[str, Any]] = []
    for p in MODULES_DIR.glob("*.py"):
        name = p.name
        listings.append({
            "name": name,
            "owned": _is_owned(name),
            "protected": name in PROTECTED_FILES,
        })
    return {"modules": listings}


def read_module(path: str) -> Dict[str, Any]:
    fp = MODULES_DIR / Path(path).name
    if not fp.exists():
        return {"error": f"{path} not found"}
    return {"path": str(fp), "code": fp.read_text(encoding="utf-8")}


def write_module(path: str, code: str) -> Dict[str, Any]:
    """
    Safely create or update a module under /modules.
    - Cannot touch protected core modules.
    - Can only edit existing files it owns.
    - Can freely create new modules.
    """
    name = Path(path).name
    if not name.endswith(".py"):
        name += ".py"
    fp = MODULES_DIR / name

    # Safety: do not allow touching protected files
    if name in PROTECTED_FILES:
        return {"error": f"{name} is protected and cannot be modified."}

    # Existing file not owned by bot -> block
    if fp.exists() and not _is_owned(name):
        return {"error": f"{name} exists and is not owned by the Football Bot."}

    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    # Backup if overwriting
    if fp.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MODULES_DIR / f"{name}.bak_{ts}"
        backup.write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")

    # Write new code
    fp.write_text(code, encoding="utf-8")
    _add_owned(name)

    return {"status": "success", "path": str(fp), "owned": True}


def run_module(path: str, function_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Import a module from /modules and run a named function with kwargs.
    """
    args = args or {}
    name = Path(path).name
    if not name.endswith(".py"):
        name += ".py"

    fp = MODULES_DIR / name
    if not fp.exists():
        return {"error": f"{name} not found"}

    module_name = f"modules.{name[:-3]}"

    try:
        if module_name in importlib.sys.modules:
            mod = importlib.reload(importlib.sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)

        if not hasattr(mod, function_name):
            return {"error": f"{function_name} not found in {name}"}

        fn = getattr(mod, function_name)
        result = fn(**args)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# ================ DATA ANALYSIS TOOLS =======================

def load_data_basic(limit: int = 200) -> Dict[str, Any]:
    """
    Return shape + sample of the dataset.
    """
    try:
        df = _load_full_df()
    except Exception as e:
        return {"error": f"Load error: {e}"}

    return {
        "rows": int(len(df)),
        "cols": df.columns.tolist(),
        "sample": df.head(limit).to_dict(orient="records"),
    }


def list_columns() -> Dict[str, Any]:
    try:
        df = _load_full_df()
        return {"columns": df.columns.tolist()}
    except Exception as e:
        return {"error": str(e)}


def basic_roi_for_pl_column(pl_column: str) -> Dict[str, Any]:
    """
    Compute total PL, total NO GAMES, ROI per game, and row count
    for a specific PL column.
    """
    try:
        df = _load_full_df()
    except Exception as e:
        return {"error": f"Data error: {e}"}

    if pl_column not in df.columns:
        return {"error": f"{pl_column} missing"}

    if "NO GAMES" not in df.columns:
        return {"error": "NO GAMES missing"}

    df2 = df[df[pl_column].notna() & df["NO GAMES"].notna()]
    if df2.empty:
        return {"error": "No rows with PL + NO GAMES"}

    total_pl = df2[pl_column].sum()
    total_games = df2["NO GAMES"].sum()
    roi = total_pl / total_games if total_games > 0 else None

    return {
        "pl_column": pl_column,
        "total_pl": float(total_pl),
        "total_games": float(total_games),
        "roi_per_game": roi,
        "rows": int(len(df2)),
    }


# ================ PERMANENT RESEARCH MEMORY (GOOGLE SHEETS) =======================

# We'll reuse the same service account JSON as todos_panel.py: GOOGLE_SERVICE_ACCOUNT_JSON
# and a new secret FOOTBALL_MEMORY_SHEET_URL holding the sheet URL.

MEMORY_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _memory_creds() -> Optional[Credentials]:
    """
    Build Credentials object from GOOGLE_SERVICE_ACCOUNT_JSON in secrets/env.
    """
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return Credentials.from_service_account_info(data, scopes=MEMORY_SCOPES)
    except Exception:
        return None


def _get_memory_sheet_url() -> Optional[str]:
    """
    Returns the memory sheet URL from secrets or env.
    """
    return (
        st.secrets.get("FOOTBALL_MEMORY_SHEET_URL")
        or os.getenv("FOOTBALL_MEMORY_SHEET_URL")
    )


def _get_memory_sheet():
    """
    Return a gspread Worksheet object for the football memory sheet.
    """
    url = _get_memory_sheet_url()
    if not url:
        raise RuntimeError("FOOTBALL_MEMORY_SHEET_URL not set in secrets or env.")

    creds = _memory_creds()
    if not creds:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not configured or invalid.")

    gc = gspread.authorize(creds)
    sh = gc.open_by_url(url)
    # For simplicity, use the first sheet
    ws = sh.sheet1
    return ws


def append_research_note(note: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Append a research note as a new row: [timestamp, note, tags].
    Stored permanently in Google Sheets.
    """
    tags = tags or []
    ts = datetime.datetime.utcnow().isoformat()
    tags_str = ", ".join(tags)

    try:
        ws = _get_memory_sheet()
        ws.append_row([ts, note, tags_str])
        return {"status": "ok", "timestamp": ts, "note": note, "tags": tags}
    except Exception as e:
        return {"error": f"Append failed: {e}"}


def get_recent_research_notes(limit: int = 20) -> Dict[str, Any]:
    """
    Return the last N research notes from the sheet (based on row order).
    """
    try:
        ws = _get_memory_sheet()
        # get_all_records returns a list of dicts keyed by header row
        records = ws.get_all_records()
        if not records:
            return {"notes": []}

        notes = records[-limit:]
        return {"notes": notes}
    except Exception as e:
        return {"error": f"Read failed: {e}"}
