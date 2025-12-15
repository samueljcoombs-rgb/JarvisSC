# modules/football_tools.py
from __future__ import annotations

import json
import os
import io
import time
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials

from supabase import create_client


# -----------------------------
# Config
# -----------------------------

# Google Sheets (definitions, rules, memory/state)
DEFAULT_FOOTBALL_SHEET_URL = os.getenv("FOOTBALL_SHEET_URL") or st.secrets.get("FOOTBALL_SHEET_URL", "")

# CSV data source (recommended: Google Drive direct download OR any direct URL)
# Put this in Streamlit secrets:
# DATA_CSV_URL="https://drive.google.com/uc?export=download&id=FILE_ID"
DEFAULT_DATA_CSV_URL = os.getenv("DATA_CSV_URL") or st.secrets.get("DATA_CSV_URL", "")

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Supabase storage bucket for results
RESULTS_BUCKET = os.getenv("FOOTBALL_RESULTS_BUCKET") or st.secrets.get("FOOTBALL_RESULTS_BUCKET", "football-results")

# Google auth scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# -----------------------------
# Google Sheets helpers
# -----------------------------

def _google_creds() -> Optional[Credentials]:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return Credentials.from_service_account_info(data, scopes=SCOPES)
    except Exception:
        return None

def _open_football_sheet():
    if not DEFAULT_FOOTBALL_SHEET_URL:
        raise RuntimeError("Missing FOOTBALL_SHEET_URL (set in Streamlit secrets).")
    creds = _google_creds()
    if not creds:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON (set in Streamlit secrets).")
    gc = gspread.authorize(creds)
    return gc.open_by_url(DEFAULT_FOOTBALL_SHEET_URL)

def _ws(name: str):
    sh = _open_football_sheet()
    try:
        return sh.worksheet(name)
    except Exception:
        # better error
        existing = [w.title for w in sh.worksheets()]
        raise RuntimeError(f"Worksheet '{name}' not found. Existing: {existing}")

def _sheet_rows(ws) -> List[Dict[str, Any]]:
    """Return sheet rows as list[dict] using header row 1."""
    values = ws.get_all_values()
    if not values or len(values) < 1:
        return []
    header = values[0]
    out: List[Dict[str, Any]] = []
    for row in values[1:]:
        d = {}
        for i, h in enumerate(header):
            d[h] = row[i] if i < len(row) else ""
        out.append(d)
    return out


# -----------------------------
# CSV loading
# -----------------------------

def _normalize_drive_url(url: str) -> str:
    """
    Accepts:
      - https://drive.google.com/file/d/<ID>/view?...
      - https://drive.google.com/uc?export=download&id=<ID>
    Returns a direct download URL.
    """
    if not url:
        return ""
    m = re.search(r"/file/d/([^/]+)/", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    if "drive.google.com" in url and "uc?export=download" in url:
        return url
    return url

@st.cache_data(ttl=3600, show_spinner=False)
def _load_df_cached(csv_url: str) -> pd.DataFrame:
    if not csv_url:
        raise RuntimeError("Missing DATA_CSV_URL (set in Streamlit secrets).")
    url = _normalize_drive_url(csv_url)

    r = requests.get(url, timeout=120)
    r.raise_for_status()

    # try utf-8, fall back latin-1
    content = r.content
    try:
        s = content.decode("utf-8")
        bio = io.StringIO(s)
        df = pd.read_csv(bio, low_memory=False)
    except Exception:
        s = content.decode("latin-1", errors="replace")
        bio = io.StringIO(s)
        df = pd.read_csv(bio, low_memory=False)

    return df

def _load_df() -> pd.DataFrame:
    return _load_df_cached(DEFAULT_DATA_CSV_URL)


# -----------------------------
# Supabase helpers
# -----------------------------

def _sb():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# -----------------------------
# Public tools: definitions / rules / framework
# -----------------------------

def get_dataset_overview() -> Dict[str, Any]:
    rows = _sheet_rows(_ws("dataset_overview"))
    return {"rows": rows}

def get_research_rules() -> Dict[str, Any]:
    rows = _sheet_rows(_ws("research_rules"))
    return {"rows": rows}

def get_column_definitions() -> Dict[str, Any]:
    rows = _sheet_rows(_ws("column_definitions"))
    return {"rows": rows}

def get_evaluation_framework() -> Dict[str, Any]:
    rows = _sheet_rows(_ws("evaluation_framework"))
    return {"rows": rows}


# -----------------------------
# Public tools: research memory + state
# -----------------------------

def append_research_note(note: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    ws = _ws("research_memory")
    ts = datetime.utcnow().isoformat()
    tag_str = ", ".join([t.strip() for t in (tags or []) if t and t.strip()])
    ws.append_row([ts, note, tag_str])
    return {"ok": True, "timestamp": ts, "tags": tag_str}

def get_recent_research_notes(limit: int = 25) -> Dict[str, Any]:
    limit = int(limit or 25)
    limit = max(1, min(limit, 200))
    rows = _sheet_rows(_ws("research_memory"))
    return {"rows": rows[-limit:]}

def get_research_state() -> Dict[str, Any]:
    rows = _sheet_rows(_ws("research_state"))
    state = {}
    for r in rows:
        k = (r.get("key") or "").strip()
        v = (r.get("value") or "").strip()
        if k:
            state[k] = v
    return {"state": state}

def set_research_state(key: str, value: str) -> Dict[str, Any]:
    key = (key or "").strip()
    if not key:
        return {"error": "key is required"}
    ws = _ws("research_state")
    rows = ws.get_all_values()
    if not rows:
        ws.append_row(["key", "value"])
        ws.append_row([key, value])
        return {"ok": True, "key": key, "value": value, "action": "insert"}

    header = rows[0]
    # Find columns
    try:
        key_i = header.index("key")
        val_i = header.index("value")
    except ValueError:
        ws.clear()
        ws.append_row(["key", "value"])
        ws.append_row([key, value])
        return {"ok": True, "key": key, "value": value, "action": "reset_insert"}

    # Find existing
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) > key_i and (row[key_i] or "").strip() == key:
            ws.update_cell(idx, val_i + 1, value)
            return {"ok": True, "key": key, "value": value, "action": "update"}

    ws.append_row([key, value])
    return {"ok": True, "key": key, "value": value, "action": "insert"}


# -----------------------------
# Public tools: data inspection
# -----------------------------

def list_columns() -> Dict[str, Any]:
    df = _load_df()
    return {"columns": df.columns.tolist(), "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])}

def load_data_basic(limit: int = 50) -> Dict[str, Any]:
    df = _load_df()
    limit = int(limit or 50)
    limit = max(10, min(limit, 2000))
    preview = df.head(limit).to_dict(orient="records")
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "preview": preview,
    }


# -----------------------------
# Performance / ROI / Streaks helpers
# -----------------------------

def _to_datetime_series(df: pd.DataFrame) -> pd.Series:
    # best-effort parse DATE + TIME
    if "DATE" in df.columns and "TIME" in df.columns:
        dt = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"].astype(str), errors="coerce")
        if dt.notna().any():
            return dt
    if "DATE" in df.columns:
        dt = pd.to_datetime(df["DATE"], errors="coerce")
        if dt.notna().any():
            return dt
    # fallback
    return pd.to_datetime(pd.Series([None] * len(df)), errors="coerce")

def _split_by_time(df: pd.DataFrame, ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratio = float(ratio or 0.7)
    ratio = max(0.5, min(ratio, 0.95))
    tmp = df.copy()
    tmp["_dt"] = _to_datetime_series(tmp)
    tmp = tmp.sort_values("_dt", na_position="last")
    cut = int(math.floor(len(tmp) * ratio))
    train = tmp.iloc[:cut].drop(columns=["_dt"])
    test = tmp.iloc[cut:].drop(columns=["_dt"])
    return train, test

def _bet_level_roi(df: pd.DataFrame, pl_col: str, side: str, odds_col: Optional[str]) -> Dict[str, Any]:
    d = df[df[pl_col].notna()].copy()
    d[pl_col] = pd.to_numeric(d[pl_col], errors="coerce")
    d = d[d[pl_col].notna()]
    n = int(len(d))
    total_pl = float(d[pl_col].sum()) if n else 0.0

    if n == 0:
        return {"bets": 0, "total_pl": 0.0, "roi": 0.0, "avg_pl": 0.0, "stake_or_liability": 0.0}

    side = (side or "").lower().strip()
    if side == "lay":
        if not odds_col or odds_col not in d.columns:
            return {"error": f"Lay ROI needs odds_column. Missing/invalid odds_column: {odds_col}"}
        odds = pd.to_numeric(d[odds_col], errors="coerce").fillna(0.0)
        liability = (odds - 1.0).clip(lower=0.0)
        total_liability = float(liability.sum())
        roi = (total_pl / total_liability) if total_liability > 0 else 0.0
        avg_pl = total_pl / n
        return {
            "bets": n,
            "total_pl": total_pl,
            "roi": roi,
            "avg_pl": avg_pl,
            "stake_or_liability": total_liability,
            "mode": "lay_liability",
            "odds_column": odds_col,
        }

    # default back
    total_stake = float(n)  # 1pt per bet
    roi = total_pl / total_stake
    avg_pl = total_pl / n
    return {
        "bets": n,
        "total_pl": total_pl,
        "roi": roi,
        "avg_pl": avg_pl,
        "stake_or_liability": total_stake,
        "mode": "back_flat_1pt",
    }

def _game_level_metrics(df: pd.DataFrame, pl_col: str) -> Dict[str, Any]:
    # aggregate by ID for streak/drawdown
    if "ID" not in df.columns:
        return {"error": "Missing ID column (required for game-level streak/drawdown)."}

    d = df[df[pl_col].notna()].copy()
    d[pl_col] = pd.to_numeric(d[pl_col], errors="coerce")
    d = d[d[pl_col].notna()]

    if len(d) == 0:
        return {"games": 0, "longest_losing_streak_bets": 0, "longest_losing_streak_pl": 0.0, "max_drawdown": 0.0}

    d["_dt"] = _to_datetime_series(d)

    g = (
        d.groupby("ID", as_index=False)
         .agg(
            game_pl=(pl_col, "sum"),
            date=("_dt", "min"),
         )
         .sort_values("date", na_position="last")
    )

    game_pls = g["game_pl"].tolist()

    # Longest losing streak: by consecutive games with game_pl < 0
    max_streak = 0
    cur_streak = 0

    # longest losing streak in PL (points) = most negative sum across any consecutive losing streak
    worst_streak_pl = 0.0
    cur_streak_pl = 0.0

    for pl in game_pls:
        if pl < 0:
            cur_streak += 1
            cur_streak_pl += float(pl)
            max_streak = max(max_streak, cur_streak)
            worst_streak_pl = min(worst_streak_pl, cur_streak_pl)
        else:
            cur_streak = 0
            cur_streak_pl = 0.0

    # Max drawdown on cumulative PL series (points)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pl in game_pls:
        cum += float(pl)
        peak = max(peak, cum)
        dd = cum - peak  # negative or 0
        max_dd = min(max_dd, dd)

    return {
        "games": int(len(g)),
        "longest_losing_streak_bets": int(max_streak),
        "longest_losing_streak_pl": float(worst_streak_pl),  # negative number in points
        "max_drawdown": float(max_dd),  # negative number in points
    }


# -----------------------------
# Public tools: strategy performance
# -----------------------------

def strategy_performance_summary(
    pl_column: str,
    side: str = "back",
    odds_column: Optional[str] = None,
    time_split_ratio: float = 0.7,
    compute_streaks: bool = True,
) -> Dict[str, Any]:
    df = _load_df()

    if pl_column not in df.columns:
        return {"error": f"PL column not found: {pl_column}"}

    # time split on bet rows
    train_df, test_df = _split_by_time(df[df[pl_column].notna()].copy(), time_split_ratio)

    overall = _bet_level_roi(df, pl_column, side, odds_column)
    train = _bet_level_roi(train_df, pl_column, side, odds_column)
    test = _bet_level_roi(test_df, pl_column, side, odds_column)

    out: Dict[str, Any] = {
        "pl_column": pl_column,
        "side": side,
        "odds_column": odds_column,
        "time_split_ratio": time_split_ratio,
        "overall": overall,
        "train": train,
        "test": test,
    }

    if compute_streaks:
        out["game_level_overall"] = _game_level_metrics(df, pl_column)
        out["game_level_test"] = _game_level_metrics(test_df, pl_column)

    return out

def strategy_performance_batch(
    pl_columns: List[str],
    time_split_ratio: float = 0.7,
    compute_streaks: bool = True,
) -> Dict[str, Any]:
    # Default mappings (based on your notes)
    # Lay:
    #   SHG PL -> HT CS Price
    #   SHG 2+ PL -> HT 2 Ahead Odds
    #   LU1.5 PL -> U1.5 Odds (best available mapping)
    #   LFGHU0.5 PL -> FHGU0.5Odds
    # Back:
    #   BO 2.5 PL -> O2.5 Odds
    #   BO1.5 FHG PL -> FHGO1.5 Odds
    #   BTTS PL -> BTTS Y Odds
    mappings = {
        "SHG PL": ("lay", "HT CS Price"),
        "SHG 2+ PL": ("lay", "HT 2 Ahead Odds"),
        "LU1.5 PL": ("lay", "U1.5 Odds"),
        "LFGHU0.5 PL": ("lay", "FHGU0.5Odds"),
        "BO 2.5 PL": ("back", "O2.5 Odds"),
        "BO1.5 FHG PL": ("back", "FHGO1.5 Odds"),
        "BTTS PL": ("back", "BTTS Y Odds"),
    }

    results = []
    for c in pl_columns:
        side, odds_col = mappings.get(c, ("back", None))
        results.append(strategy_performance_summary(
            pl_column=c,
            side=side,
            odds_column=odds_col,
            time_split_ratio=time_split_ratio,
            compute_streaks=compute_streaks,
        ))
    return {"results": results}


# -----------------------------
# Public tools: Supabase job queue + results
# -----------------------------

def submit_job(task_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Submit a job to public.jobs for Modal worker to process.
    params is OPTIONAL (defaults to {}), so the agent cannot fail by omission.
    """
    if params is None:
        params = {}
    sb = _sb()

    payload = {
        "status": "queued",
        "task_type": task_type,
        "params": params,
    }
    res = sb.table("jobs").insert(payload).execute()

    # supabase-py returns .data as list of inserted rows
    data = getattr(res, "data", None)
    if not data:
        return {"error": "Insert returned no data", "raw": str(res)}

    row = data[0]
    return {
        "job_id": row.get("job_id"),
        "status": row.get("status"),
        "task_type": row.get("task_type"),
        "created_at": row.get("created_at"),
    }

def get_job(job_id: str) -> Dict[str, Any]:
    sb = _sb()
    res = sb.table("jobs").select("*").eq("job_id", job_id).limit(1).execute()
    data = getattr(res, "data", None)
    if not data:
        return {"error": f"Job not found: {job_id}"}
    row = data[0]
    # include key fields
    return {
        "job_id": row.get("job_id"),
        "status": row.get("status"),
        "task_type": row.get("task_type"),
        "params": row.get("params"),
        "result_path": row.get("result_path"),
        "error": row.get("error"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }

def download_result(result_path: str) -> Dict[str, Any]:
    sb = _sb()
    if not result_path:
        return {"error": "result_path is required"}
    try:
        b = sb.storage.from_(RESULTS_BUCKET).download(result_path)
        # b is bytes in supabase-py sync client
        txt = b.decode("utf-8", errors="replace")
        obj = json.loads(txt)
        return {"ok": True, "result_path": result_path, "result": obj}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "result_path": result_path}
