# modules/football_tools.py
from __future__ import annotations

import ast
import json
import datetime
import importlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from io import StringIO

import pandas as pd
import requests

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

from supabase import create_client


# ============================================================
# Paths / Registry (guardrails for permanent code edits)
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODULES_DIR = BASE_DIR / "modules"
REGISTRY_PATH = MODULES_DIR / "bot_registry.json"

PROTECTED_FILES = {
    "layout_manager.py",
    "chat_ui.py",
    "weather_panel.py",
    "podcasts_panel.py",
    "athletic_feed.py",
    "todos_panel.py",
    "__init__.py",
}

def _load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"owned": []}

def _save_registry(data: Dict[str, Any]) -> None:
    REGISTRY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _is_owned(fname: str) -> bool:
    return fname in _load_registry().get("owned", [])

def _add_owned(fname: str) -> None:
    reg = _load_registry()
    if fname not in reg["owned"]:
        reg["owned"].append(fname)
        _save_registry(reg)


# ============================================================
# Dataset loader (Google Drive CSV)
# ============================================================

# Your CSV file ID (already working)
GDRIVE_FILE_ID = "1aYMC7YJ1qim-132aDc50hhNMdDm20WbC"
DEFAULT_DATA_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
DATA_URL_ENV = os.getenv("FOOTBALL_DATA_URL", "").strip()
DATA_URL = DATA_URL_ENV or DEFAULT_DATA_URL

@st.cache_data(ttl=600, show_spinner=False)
def _load_full_df_cached(data_url: str) -> pd.DataFrame:
    resp = requests.get(data_url, timeout=60)
    resp.raise_for_status()
    csv_data = resp.content.decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(csv_data), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def _load_full_df() -> pd.DataFrame:
    if not DATA_URL:
        raise RuntimeError("FOOTBALL_DATA_URL / DATA_URL not configured.")
    return _load_full_df_cached(DATA_URL)


# ============================================================
# Google Sheets (Knowledge Base + Memory + State)
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _gs_creds() -> Credentials:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in secrets/env.")
    data = json.loads(raw) if isinstance(raw, str) else raw
    return Credentials.from_service_account_info(data, scopes=SCOPES)

def _gs_doc():
    url = st.secrets.get("FOOTBALL_MEMORY_SHEET_URL") or os.getenv("FOOTBALL_MEMORY_SHEET_URL")
    if not url:
        raise RuntimeError("Missing FOOTBALL_MEMORY_SHEET_URL in secrets/env.")
    gc = gspread.authorize(_gs_creds())
    return gc.open_by_url(url)

def _ws(name: str):
    return _gs_doc().worksheet(name)


# ------------------------
# Knowledge base readers
# ------------------------

def get_dataset_overview() -> Dict[str, Any]:
    try:
        return {"dataset_overview": _ws("dataset_overview").get_all_records()}
    except Exception as e:
        return {"error": f"Failed to load dataset_overview: {e}"}

def get_column_definitions() -> Dict[str, Any]:
    try:
        return {"column_definitions": _ws("column_definitions").get_all_records()}
    except Exception as e:
        return {"error": f"Failed to load column_definitions: {e}"}

def get_research_rules() -> Dict[str, Any]:
    try:
        return {"research_rules": _ws("research_rules").get_all_records()}
    except Exception as e:
        return {"error": f"Failed to load research_rules: {e}"}

def get_evaluation_framework() -> Dict[str, Any]:
    try:
        return {"evaluation_framework": _ws("evaluation_framework").get_all_records()}
    except Exception as e:
        return {"error": f"Failed to load evaluation_framework: {e}"}


# ------------------------
# Permanent research memory
# ------------------------

def append_research_note(note: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Appends to 'research_memory' tab: timestamp | note | tags"""
    tags = tags or []
    ts = datetime.datetime.utcnow().isoformat()
    try:
        _ws("research_memory").append_row([ts, note, ", ".join(tags)])
        return {"status": "ok", "timestamp": ts}
    except Exception as e:
        return {"error": f"Append failed: {e}"}

def get_recent_research_notes(limit: int = 20) -> Dict[str, Any]:
    try:
        records = _ws("research_memory").get_all_records()
        return {"notes": records[-limit:]}
    except Exception as e:
        return {"error": f"Read failed: {e}"}


# ------------------------
# Research state (persistent autonomy)
# ------------------------

def get_research_state() -> Dict[str, Any]:
    """Reads 'research_state' tab (key/value) into a dict. Headers: key | value"""
    try:
        rows = _ws("research_state").get_all_records()
        state = {}
        for r in rows:
            k = str(r.get("key", "")).strip()
            v = str(r.get("value", "")).strip()
            if k:
                state[k] = v
        return {"research_state": state}
    except Exception as e:
        return {"error": f"Failed to read research_state: {e}"}

def set_research_state(key: str, value: str) -> Dict[str, Any]:
    """Upserts key/value into 'research_state'."""
    try:
        sheet = _ws("research_state")
        rows = sheet.get_all_records()
        target_row = None
        for i, r in enumerate(rows, start=2):  # row 1 = header
            if str(r.get("key", "")).strip() == key:
                target_row = i
                break

        if target_row is None:
            sheet.append_row([key, value])
        else:
            sheet.update(f"B{target_row}", value)

        return {"status": "ok", "key": key, "value": value}
    except Exception as e:
        return {"error": f"Failed to write research_state: {e}"}


# ============================================================
# Supabase (job queue + results)
# ============================================================

def _sb():
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)

def submit_job(task_type: str, params: dict) -> dict:
    """Insert a queued job into Supabase jobs table."""
    sb = _sb()
    row = sb.table("jobs").insert({"task_type": task_type, "params": params, "status": "queued"}).execute().data[0]
    return {"job_id": row["job_id"], "status": row["status"], "task_type": row["task_type"]}

def get_job(job_id: str) -> dict:
    """Fetch a job row by job_id."""
    sb = _sb()
    data = sb.table("jobs").select("*").eq("job_id", job_id).limit(1).execute().data
    if not data:
        return {"error": "job not found"}
    return data[0]

def download_result(result_path: str) -> dict:
    """Download a JSON result from Supabase Storage bucket football-results."""
    sb = _sb()
    bucket = "football-results"
    raw = sb.storage.from_(bucket).download(result_path)
    return json.loads(raw.decode("utf-8"))


# ============================================================
# Data inspection tools
# ============================================================

def load_data_basic(limit: int = 200) -> Dict[str, Any]:
    try:
        df = _load_full_df()
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}
    return {
        "rows": int(len(df)),
        "cols": list(df.columns),
        "sample": df.head(limit).to_dict(orient="records"),
    }

def list_columns() -> Dict[str, Any]:
    try:
        df = _load_full_df()
        return {"columns": list(df.columns)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Strategy evaluation (Back/Lay ROI + ID streaks + time split)
# ============================================================

# PL -> odds mapping (your confirmed mappings)
PL_ODDS_MAP: Dict[str, str] = {
    # Lay
    "SHG PL": "HT CS Price",
    "SHG 2+ PL": "HT 2 Ahead Odds",
    "LU1.5 PL": "U1.5 Odds",
    "LFGHU0.5 PL": "FHGU0.5Odds",
    # Back
    "BO 2.5 PL": "O2.5 Odds",
    "BO1.5 FHG PL": "FHGO1.5 Odds",
    "BTTS PL": "BTTS Y Odds",
}

PL_SIDE_MAP: Dict[str, str] = {
    "SHG PL": "lay",
    "SHG 2+ PL": "lay",
    "LU1.5 PL": "lay",
    "LFGHU0.5 PL": "lay",
    "BO 2.5 PL": "back",
    "BO1.5 FHG PL": "back",
    "BTTS PL": "back",
}

def _to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _time_split(df: pd.DataFrame, split: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "DATE" not in df.columns:
        raise RuntimeError("DATE column missing; cannot time split.")
    d = df.copy()
    d["DATE"] = _to_datetime_safe(d["DATE"])
    d = d[d["DATE"].notna()].sort_values("DATE").reset_index(drop=True)
    if d.empty:
        return d, d
    idx = int(len(d) * split)
    idx = max(1, min(idx, len(d) - 1)) if len(d) > 1 else 1
    return d.iloc[:idx].copy(), d.iloc[idx:].copy()

def _row_level_roi(df: pd.DataFrame, pl_col: str, side: str, odds_col: str) -> Dict[str, Any]:
    """
    ROI/exposure uses ROWS (bets). Duplicate IDs imply multiple bets.
    back: stake=1 per row -> ROI = PL / bets
    lay: risk per row = (odds-1) -> ROI = PL / sum(risk)
    Units: points
    """
    d = df[df[pl_col].notna()].copy()
    bets = int(len(d))
    if bets == 0:
        return {"bets": 0, "total_pl": 0.0, "roi": None, "total_stake": 0.0, "total_liability": 0.0, "avg_pl_per_bet": None}

    d[pl_col] = pd.to_numeric(d[pl_col], errors="coerce").fillna(0.0)
    total_pl = float(d[pl_col].sum())
    avg_pl = total_pl / bets

    if side == "back":
        total_stake = float(bets)  # 1pt each
        roi = total_pl / total_stake if total_stake > 0 else None
        return {"bets": bets, "total_pl": total_pl, "total_stake": total_stake, "total_liability": 0.0, "roi": roi, "avg_pl_per_bet": avg_pl}

    odds = pd.to_numeric(d.get(odds_col), errors="coerce")
    risk = (odds - 1.0).clip(lower=0.0)
    total_liability = float(risk.fillna(0.0).sum())
    roi = total_pl / total_liability if total_liability > 0 else None
    return {"bets": bets, "total_pl": total_pl, "total_stake": 0.0, "total_liability": total_liability, "roi": roi, "avg_pl_per_bet": avg_pl}

def _id_aggregated_series(df: pd.DataFrame, pl_col: str) -> pd.DataFrame:
    """
    For losing streak / drawdown:
      - aggregate by ID to avoid double-counting same game
      - pl_id = sum(pl_col)
      - date_id = min(DATE)
    """
    d = df[df[pl_col].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["ID", "DATE", "pl_id"])

    if "ID" not in d.columns:
        raise RuntimeError("ID column missing; cannot compute streaks/drawdown.")
    if "DATE" not in d.columns:
        raise RuntimeError("DATE column missing; cannot compute streaks/drawdown.")

    d["DATE"] = _to_datetime_safe(d["DATE"])
    d[pl_col] = pd.to_numeric(d[pl_col], errors="coerce").fillna(0.0)

    g = d.groupby("ID", as_index=False).agg(
        DATE=("DATE", "min"),
        pl_id=(pl_col, "sum"),
    )
    g = g[g["DATE"].notna()].sort_values("DATE").reset_index(drop=True)
    return g

def _longest_losing_streak(pl_series: List[float]) -> Dict[str, Any]:
    """
    Losing streak based on per-game PL series (points):
      - losing game defined as PL < 0
      - track longest consecutive losing run by count and cumulative points
    """
    best_count = 0
    best_pl = 0.0  # most negative cumulative run
    cur_count = 0
    cur_pl = 0.0

    for pl in pl_series:
        if pl < 0:
            cur_count += 1
            cur_pl += float(pl)
            best_count = max(best_count, cur_count)
            if cur_pl < best_pl:
                best_pl = cur_pl
        else:
            cur_count = 0
            cur_pl = 0.0

    return {"longest_losing_streak_bets": int(best_count), "longest_losing_streak_pts": float(best_pl)}

def _max_drawdown(pl_series: List[float]) -> Dict[str, Any]:
    """Max drawdown on cumulative PL (points) for the per-game series."""
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pl in pl_series:
        cum += float(pl)
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd
    return {"max_drawdown_pts": float(max_dd)}

def strategy_performance_summary(
    pl_column: str,
    side: Optional[str] = None,
    odds_column: Optional[str] = None,
    time_split_ratio: float = 0.7,
    compute_streaks: bool = True,
) -> Dict[str, Any]:
    """
    Computes:
      - ROI & exposure at BET level (row level): duplicates IDs count as multiple bets
      - losing streak / drawdown at GAME level (ID aggregated): duplicates collapsed
      - train/test comparison by DATE split

    Units: points
    """
    try:
        df = _load_full_df()
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    if pl_column not in df.columns:
        return {"error": f"PL column not found: {pl_column}"}

    inferred_side = PL_SIDE_MAP.get(pl_column)
    inferred_odds = PL_ODDS_MAP.get(pl_column)

    side_use = (side or inferred_side or "").strip().lower()
    odds_use = (odds_column or inferred_odds or "").strip()

    if side_use not in {"back", "lay"}:
        return {"error": f"side must be 'back' or 'lay'. Got: {side}. (Could not infer)"}
    if not odds_use:
        return {"error": f"odds_column missing and could not infer for {pl_column}."}
    if odds_use not in df.columns:
        return {"error": f"Odds column not found: {odds_use} (for {pl_column})"}

    overall = _row_level_roi(df, pl_column, side_use, odds_use)

    try:
        train_df, test_df = _time_split(df, split=time_split_ratio)
        train = _row_level_roi(train_df, pl_column, side_use, odds_use)
        test = _row_level_roi(test_df, pl_column, side_use, odds_use)
    except Exception as e:
        train, test = {"error": str(e)}, {"error": str(e)}

    out: Dict[str, Any] = {
        "pl_column": pl_column,
        "side": side_use,
        "odds_column": odds_use,
        "overall_bet_level": overall,
        "train_bet_level": train,
        "test_bet_level": test,
        "units": "points",
        "notes": {
            "roi_definition": "back: PL/bets (1pt each); lay: PL/sum(odds-1) liability",
            "duplicate_id_handling": "ROI uses rows (duplicates count as multiple bets). Streaks/drawdown aggregate by ID.",
        },
    }

    if compute_streaks:
        try:
            g = _id_aggregated_series(df, pl_column)
            pl_series = g["pl_id"].astype(float).tolist()
            out["game_level"] = {"games": int(len(g)), **_longest_losing_streak(pl_series), **_max_drawdown(pl_series)}
        except Exception as e:
            out["game_level_error"] = str(e)

        try:
            _, test_df2 = _time_split(df, split=time_split_ratio)
            g2 = _id_aggregated_series(test_df2, pl_column)
            pl_series2 = g2["pl_id"].astype(float).tolist()
            out["test_game_level"] = {"games": int(len(g2)), **_longest_losing_streak(pl_series2), **_max_drawdown(pl_series2)}
        except Exception as e:
            out["test_game_level_error"] = str(e)

    return out

def strategy_performance_batch(
    pl_columns: List[str],
    time_split_ratio: float = 0.7,
    compute_streaks: bool = True,
) -> Dict[str, Any]:
    results = []
    for c in pl_columns:
        results.append(
            strategy_performance_summary(
                pl_column=c,
                side=PL_SIDE_MAP.get(c),
                odds_column=PL_ODDS_MAP.get(c),
                time_split_ratio=time_split_ratio,
                compute_streaks=compute_streaks,
            )
        )
    return {"results": results}


# ============================================================
# Permanent code tools (guardrailed)
# ============================================================

def list_modules() -> Dict[str, Any]:
    modules = []
    for p in MODULES_DIR.glob("*.py"):
        name = p.name
        modules.append({"name": name, "owned": _is_owned(name), "protected": name in PROTECTED_FILES})
    return {"modules": modules}

def read_module(path: str) -> Dict[str, Any]:
    name = Path(path).name
    if not name.endswith(".py"):
        name += ".py"
    full = MODULES_DIR / name
    if not full.exists():
        return {"error": f"Module not found: {name}"}
    try:
        return {"path": str(full), "code": full.read_text(encoding="utf-8")}
    except Exception as e:
        return {"error": str(e)}

def write_module(path: str, code: str) -> Dict[str, Any]:
    """
    Creates/updates /modules/<path> but:
      - blocks protected modules
      - blocks editing non-owned existing modules
      - backs up before overwrite
      - validates syntax
    """
    name = Path(path).name
    if not name.endswith(".py"):
        name += ".py"
    full = MODULES_DIR / name

    if name in PROTECTED_FILES:
        return {"error": f"Blocked: {name} is protected."}

    if full.exists() and not _is_owned(name):
        return {"error": f"Blocked: {name} exists and is not owned by Football Bot."}

    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    if full.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MODULES_DIR / f"{name}.bak_{ts}"
        try:
            backup.write_text(full.read_text(encoding='utf-8'), encoding="utf-8")
        except Exception:
            pass

    try:
        full.write_text(code, encoding="utf-8")
        _add_owned(name)
        return {"status": "success", "path": str(full), "owned": True}
    except Exception as e:
        return {"error": f"Write failed: {e}"}

def run_module(path: str, function_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = args or {}
    name = Path(path).name
    if not name.endswith(".py"):
        name += ".py"
    full = MODULES_DIR / name
    if not full.exists():
        return {"error": f"Module not found: {name}"}

    module_name = f"modules.{name[:-3]}"
    try:
        if module_name in importlib.sys.modules:
            mod = importlib.reload(importlib.sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)

        if not hasattr(mod, function_name):
            return {"error": f"Function {function_name} not found in {name}"}

        fn = getattr(mod, function_name)
        return {"result": fn(**args)}
    except Exception as e:
        return {"error": str(e)}
