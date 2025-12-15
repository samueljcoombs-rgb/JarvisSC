# modules/football_tools.py
from __future__ import annotations

import os
import json
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
import streamlit as st

from supabase import create_client  # supabase==2.6.0
import gspread
from google.oauth2.service_account import Credentials

DEFAULT_RESULTS_BUCKET = "football-results"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# -----------------
# Small helpers
# -----------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _get_env_or_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        v = st.secrets.get(key)
    except Exception:
        v = None
    if not v:
        v = os.getenv(key, default)
    return v


# -----------------
# Supabase
# -----------------
def _supabase_client():
    url = _get_env_or_secret("SUPABASE_URL")
    key = (
        _get_env_or_secret("SUPABASE_SERVICE_ROLE_KEY")
        or _get_env_or_secret("SUPABASE_ANON_KEY")
        or _get_env_or_secret("SUPABASE_KEY")
    )
    if not url or not key:
        raise RuntimeError(
            "Missing Supabase credentials. Provide SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "(recommended) or SUPABASE_ANON_KEY in Streamlit secrets."
        )
    return create_client(url, key)


# -----------------
# Google Sheets
# -----------------
def _creds() -> Optional[Credentials]:
    raw = _get_env_or_secret("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return Credentials.from_service_account_info(data, scopes=SCOPES)
    except Exception:
        return None


def _sheet_url() -> Optional[str]:
    return _get_env_or_secret("FOOTBALL_SHEET_URL") or _get_env_or_secret("FOOTBALL_MEMORY_SHEET_URL")


def _open_sheet():
    sheet_url = _sheet_url()
    creds = _creds()
    if not sheet_url or not creds:
        raise RuntimeError(
            "Missing Google Sheets config. Provide FOOTBALL_SHEET_URL (or FOOTBALL_MEMORY_SHEET_URL) and "
            "GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets."
        )
    gc = gspread.authorize(creds)
    return gc.open_by_url(sheet_url)


# DEFAULT logical->tab mapping (matches YOUR agreed tabs)
# You can still override any of these via Streamlit secrets like:
# FOOTBALL_WS_RESEARCH_NOTES="research_memory"
_DEFAULT_WS_MAP = {
    "research_notes": "research_memory",
    "dataset_overview": "dataset_overview",
    "research_rules": "research_rules",
    "column_definitions": "column_definitions",
    "research_state": "research_state",
    "evaluation_framework": "evaluation_framework",
}


def _ws_name(logical: str) -> str:
    # allow override: FOOTBALL_WS_<LOGICAL_UPPER>
    key = f"FOOTBALL_WS_{logical.upper()}"
    override = _get_env_or_secret(key)
    if override:
        return override
    return _DEFAULT_WS_MAP.get(logical, logical)


def _ws(logical_name: str):
    sh = _open_sheet()
    tab = _ws_name(logical_name)
    try:
        return sh.worksheet(tab)
    except Exception:
        titles = [w.title for w in sh.worksheets()]
        raise RuntimeError(
            f"Worksheet/tab not found: '{tab}' (logical name '{logical_name}'). "
            f"Available tabs: {titles}. "
            f"Either rename the tab to '{tab}', or set Streamlit secret FOOTBALL_WS_{logical_name.upper()} "
            f"to the correct tab name."
        )


def _read_kv_sheet(sheet_logical: str, key_col: str = "key", value_col: str = "value") -> Dict[str, str]:
    ws = _ws(sheet_logical)
    rows = ws.get_all_records()
    out: Dict[str, str] = {}
    for r in rows:
        k = str(r.get(key_col, "")).strip()
        v = str(r.get(value_col, "")).strip()
        if k:
            out[k] = v
    return out


# -----------------
# Public: sheet-backed tools
# -----------------
def get_dataset_overview() -> Dict[str, Any]:
    return _read_kv_sheet("dataset_overview")


def get_research_rules() -> List[str]:
    ws = _ws("research_rules")
    rows = ws.get_all_records()
    return [str(r.get("rule", "")).strip() for r in rows if str(r.get("rule", "")).strip()]


def get_column_definitions() -> List[Dict[str, Any]]:
    return _ws("column_definitions").get_all_records()


def get_evaluation_framework() -> List[Dict[str, Any]]:
    return _ws("evaluation_framework").get_all_records()


def get_recent_research_notes(limit: int = 50) -> List[Dict[str, Any]]:
    ws = _ws("research_notes")
    rows = ws.get_all_records()
    return rows[-limit:]


def append_research_note(note: str, tags: str = "") -> Dict[str, Any]:
    ws = _ws("research_notes")
    ts = _now_iso()
    # research_memory sheet columns expected: timestamp, note, tags
    ws.append_row([ts, note, tags], value_input_option="RAW")
    return {"ok": True, "timestamp": ts, "tags": tags}


def get_research_state() -> Dict[str, Any]:
    return _read_kv_sheet("research_state")


def set_research_state(key: str, value: str) -> Dict[str, Any]:
    ws = _ws("research_state")
    rows = ws.get_all_records()  # expects columns: key, value
    target_row_idx = None
    for i, r in enumerate(rows, start=2):  # header row is 1
        if str(r.get("key", "")).strip() == key:
            target_row_idx = i
            break
    if target_row_idx is None:
        ws.append_row([key, value], value_input_option="RAW")
        return {"ok": True, "action": "inserted", "key": key, "value": value}
    ws.update(f"B{target_row_idx}", value)
    return {"ok": True, "action": "updated", "key": key, "value": value}


# -----------------
# CSV tools
# -----------------
@st.cache_data(ttl=600, show_spinner=False)
def _load_csv_cached(csv_url: str) -> pd.DataFrame:
    r = requests.get(csv_url, timeout=180)
    r.raise_for_status()
    bio = BytesIO(r.content)
    return pd.read_csv(bio, low_memory=False)


def load_data_basic(csv_url: Optional[str] = None) -> Dict[str, Any]:
    csv_url = csv_url or _get_env_or_secret("DATA_CSV_URL")
    if not csv_url:
        raise RuntimeError("Missing DATA_CSV_URL in Streamlit secrets.")
    df = _load_csv_cached(csv_url)
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "head": df.head(3).to_dict(orient="records")}


def list_columns(csv_url: Optional[str] = None) -> List[str]:
    csv_url = csv_url or _get_env_or_secret("DATA_CSV_URL")
    if not csv_url:
        raise RuntimeError("Missing DATA_CSV_URL in Streamlit secrets.")
    df = _load_csv_cached(csv_url)
    return list(df.columns)


def basic_roi_for_pl_column(pl_column: str, csv_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Basic row-level ROI:
      - filter rows where pl_column is not null
      - total_pl = sum(pl_column)
      - bets = count rows (each row = 1 bet instance)
      - avg_pl_per_bet = total_pl / bets

    NOTE: This is not the lay-liability ROI; we'll compute that in the worker strategy_search.
    """
    csv_url = csv_url or _get_env_or_secret("DATA_CSV_URL")
    if not csv_url:
        raise RuntimeError("Missing DATA_CSV_URL in Streamlit secrets.")
    df = _load_csv_cached(csv_url)
    if pl_column not in df.columns:
        return {"error": "missing_column", "pl_column": pl_column}

    sub = df[df[pl_column].notna()].copy()
    sub[pl_column] = pd.to_numeric(sub[pl_column], errors="coerce")
    sub = sub[sub[pl_column].notna()]

    bets = int(len(sub))
    total_pl = float(sub[pl_column].sum()) if bets else 0.0
    avg = float(total_pl / bets) if bets else 0.0
    return {"pl_column": pl_column, "bets": bets, "total_pl": total_pl, "avg_pl_per_bet": avg}


# -----------------
# Job queue + results
# -----------------
def submit_job(task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    sb = _supabase_client()
    payload = {
        "task_type": task_type,
        "params": params,
        "status": "queued",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    res = sb.table("jobs").insert(payload).execute()
    data = res.data[0] if res.data else None
    if not data:
        raise RuntimeError("Failed to insert job into Supabase.")
    return data


def get_job(job_id: str) -> Dict[str, Any]:
    sb = _supabase_client()
    res = sb.table("jobs").select("*").eq("job_id", job_id).limit(1).execute()
    if not res.data:
        return {"error": "not_found", "job_id": job_id}
    return res.data[0]


def download_result(result_path: str, bucket: Optional[str] = None) -> Dict[str, Any]:
    sb = _supabase_client()
    bucket = bucket or _get_env_or_secret("RESULTS_BUCKET") or DEFAULT_RESULTS_BUCKET
    raw = sb.storage.from_(bucket).download(result_path)
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text, "result_path": result_path, "bucket": bucket}


def wait_for_job(job_id: str, timeout_s: int = 900, poll_s: int = 5, auto_download: bool = True, bucket: Optional[str] = None) -> Dict[str, Any]:
    start = time.time()
    last_job = None
    while True:
        last_job = get_job(job_id)
        stt = (last_job.get("status") if isinstance(last_job, dict) else None) or "unknown"

        if stt == "done":
            out: Dict[str, Any] = {"status": "done", "job": last_job}
            if auto_download and last_job.get("result_path"):
                out["result"] = download_result(last_job["result_path"], bucket=bucket)
            return out

        if stt == "error":
            return {"status": "error", "job": last_job}

        if time.time() - start >= timeout_s:
            return {"status": "timeout", "job": last_job, "message": "Job still queued/running."}

        time.sleep(max(1, int(poll_s)))


# -----------------
# Chat persistence in Supabase Storage (no new tables)
# -----------------
def save_chat(session_id: str, messages: List[Dict[str, Any]], bucket: Optional[str] = None) -> Dict[str, Any]:
    sb = _supabase_client()
    bucket = bucket or _get_env_or_secret("RESULTS_BUCKET") or DEFAULT_RESULTS_BUCKET
    path = f"chats/{session_id}.json"
    payload = json.dumps(
        {"session_id": session_id, "saved_at": _now_iso(), "messages": messages},
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    sb.storage.from_(bucket).upload(
        path=path,
        file=payload,
        file_options={"content-type": "application/json", "upsert": "true"},
    )
    return {"ok": True, "bucket": bucket, "path": path, "session_id": session_id}


def load_chat(session_id: str, bucket: Optional[str] = None) -> Dict[str, Any]:
    sb = _supabase_client()
    bucket = bucket or _get_env_or_secret("RESULTS_BUCKET") or DEFAULT_RESULTS_BUCKET
    path = f"chats/{session_id}.json"
    try:
        raw = sb.storage.from_(bucket).download(path)
    except Exception:
        return {"ok": False, "error": "not_found", "bucket": bucket, "path": path}

    text = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
        return {"ok": True, "bucket": bucket, "path": path, "data": data}
    except Exception:
        return {"ok": False, "error": "bad_json", "bucket": bucket, "path": path, "raw": text}
