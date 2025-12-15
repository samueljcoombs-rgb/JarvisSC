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

# ================= PATHS =================

BASE_DIR = Path(__file__).resolve().parents[1]
MODULES_DIR = BASE_DIR / "modules"
REGISTRY_PATH = MODULES_DIR / "bot_registry.json"

# ================= PROTECTED FILES =================

PROTECTED_FILES = {
    "layout_manager.py",
    "chat_ui.py",
    "weather_panel.py",
    "podcasts_panel.py",
    "athletic_feed.py",
    "todos_panel.py",
    "__init__.py",
}

# ================= DATA LOADING =================

GDRIVE_FILE_ID = "1aYMC7YJ1qim-132aDc50hhNMdDm20WbC"
DATA_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

def _load_full_df() -> pd.DataFrame:
    resp = requests.get(DATA_URL)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.content.decode("utf-8", errors="ignore")))
    df.columns = [c.strip() for c in df.columns]
    return df

# ================= GOOGLE SHEETS (MEMORY + KNOWLEDGE) =================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _creds() -> Credentials:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    data = json.loads(raw) if isinstance(raw, str) else raw
    return Credentials.from_service_account_info(data, scopes=SCOPES)

def _sheet():
    gc = gspread.authorize(_creds())
    url = st.secrets.get("FOOTBALL_MEMORY_SHEET_URL")
    return gc.open_by_url(url)

# ================= KNOWLEDGE READERS =================

def get_dataset_overview() -> Dict[str, Any]:
    ws = _sheet().worksheet("dataset_overview")
    return {"dataset_overview": ws.get_all_records()}

def get_column_definitions() -> Dict[str, Any]:
    ws = _sheet().worksheet("column_definitions")
    return {"column_definitions": ws.get_all_records()}

def get_research_rules() -> Dict[str, Any]:
    ws = _sheet().worksheet("research_rules")
    return {"research_rules": ws.get_all_records()}

# ================= RESEARCH MEMORY =================

def append_research_note(note: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    ws = _sheet().worksheet("research_memory")
    ts = datetime.datetime.utcnow().isoformat()
    ws.append_row([ts, note, ", ".join(tags or [])])
    return {"status": "ok", "timestamp": ts}

def get_recent_research_notes(limit: int = 20) -> Dict[str, Any]:
    ws = _sheet().worksheet("research_memory")
    records = ws.get_all_records()
    return {"notes": records[-limit:]}

# ================= ROI TOOLS =================

def roi_summary_for_pl_columns(pl_columns: List[str]) -> Dict[str, Any]:
    df = _load_full_df()
    results = []

    for col in pl_columns:
        if col not in df.columns:
            continue
        d = df[df[col].notna()]
        rows = len(d)
        total_pl = float(d[col].sum())
        avg_pl = total_pl / rows if rows else None
        results.append({
            "pl_column": col,
            "rows": rows,
            "total_pl": total_pl,
            "avg_pl_per_row": avg_pl,
        })

    results.sort(key=lambda x: (x["avg_pl_per_row"] or -999), reverse=True)
    return {"results": results}
