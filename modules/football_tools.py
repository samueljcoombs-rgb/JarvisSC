# modules/football_tools.py
from __future__ import annotations

import os
import json
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

from supabase import create_client  # supabase==2.6.0
import gspread
from google.oauth2.service_account import Credentials

# =========================
# Config
# =========================
DEFAULT_RESULTS_BUCKET = "football-results"

# Google Sheets (same pattern as todos_panel.py)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _get_env_or_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit secrets first, then env
    val = None
    try:
        val = st.secrets.get(key)
    except Exception:
        val = None
    if not val:
        val = os.getenv(key, default)
    return val


# =========================
# Supabase helpers
# =========================
def _supabase_client():
    url = _get_env_or_secret("SUPABASE_URL")
    # Prefer SERVICE ROLE key if provided (worker uses it). Otherwise ANON key.
    key = (
        _get_env_or_secret("SUPABASE_SERVICE_ROLE_KEY")
        or _get_env_or_secret("SUPABASE_ANON_KEY")
        or _get_env_or_secret("SUPABASE_KEY")
    )
    if not url or not key:
        raise RuntimeError(
            "Missing Supabase credentials. Provide SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (recommended) "
            "or SUPABASE_ANON_KEY in Streamlit secrets."
        )
    return create_client(url, key)


# =========================
# Google Sheets helpers (for memory/rules/notes)
# =========================
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


def _ws(name: str):
    sh = _open_sheet()
    return sh.worksheet(name)


def _read_kv_sheet(sheet_name: str, key_col: str = "key", value_col: str = "value") -> Dict[str, str]:
    ws = _ws(sheet_name)
    rows = ws.get_all_records()
    out: Dict[str, str] = {}
    for r in rows:
        k = str(r.get(key_col, "")).strip()
        v = str(r.get(value_col, "")).strip()
        if k:
            out[k] = v
    return out


# =========================
# Public: Memory/rules/state/notes tools
# =========================
def get_dataset_overview() -> Dict[str, Any]:
    return _read_kv_sheet("dataset_overview")


def get_research_rules() -> List[str]:
    ws = _ws("research_rules")
    rows = ws.get_all_records()
    rules = []
    for r in rows:
        rule = str(r.get("rule", "")).strip()
        if rule:
            rules.append(rule)
    return rules


def get_column_definitions() -> List[Dict[str, Any]]:
    ws = _ws("column_definitions")
    return ws.get_all_records()


def get_evaluation_framework() -> List[Dict[str, Any]]:
    ws = _ws("evaluation_framework")
    return ws.get_all_records()


def get_recent_research_notes(limit: int = 50) -> List[Dict[str, Any]]:
    ws = _ws("research_notes")
    rows = ws.get_all_records()
    return rows[-limit:]


def append_research_note(note: str, tags: str = "") -> Dict[str, Any]:
    ws = _ws("research_notes")
    ts = _now_iso()
    ws.append_row([ts, note, tags], value_input_option="RAW")
    return {"ok": True, "timestamp": ts, "tags": tags}


def get_research_state() -> Dict[str, Any]:
    # sheet: research_state (columns: key, value)
    return _read_kv_sheet("research_state")


def set_research_state(key: str, value: str) -> Dict[str, Any]:
    ws = _ws("research_state")
    rows = ws.get_all_records()  # expects columns key/value
    # Find existing row
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


# =========================
# Data loading tools (CSV from URL)
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def _load_csv_cached(csv_url: str) -> pd.DataFrame:
    # Download to memory (works for direct Google Drive uc?export=download)
    r = requests.get(csv_url, timeout=120)
    r.raise_for_status()
    bio = BytesIO(r.content)
    df = pd.read_csv(bio, low_memory=False)
    return df


def load_data_basic(csv_url: Optional[str] = None) -> Dict[str, Any]:
    csv_url = csv_url or _get_env_or_secret("DATA_CSV_URL")
    if not csv_url:
        raise RuntimeError("Missing DATA_CSV_URL in Streamlit secrets.")
    df = _load_csv_cached(csv_url)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "head": df.head(3).to_dict(orient="records"),
    }


def list_columns(csv_url: Optional[str] = None) -> List[str]:
    csv_url = csv_url or _get_env_or_secret("DATA_CSV_URL")
    if not csv_url:
        raise RuntimeError("Missing DATA_CSV_URL in Streamlit secrets.")
    df = _load_csv_cached(csv_url)
    return list(df.columns)


# =========================
# Job queue tools (Supabase table 'jobs', storage bucket football-results)
# =========================
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
    # raw is bytes
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text, "result_path": result_path, "bucket": bucket}


def wait_for_job(
    job_id: str,
    timeout_s: int = 900,
    poll_s: int = 5,
    auto_download: bool = True,
    bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """
    IMPORTANT:
    - Returns status 'done' only when job.status == done
    - Returns status 'error' only when job.status == error
    - Returns status 'timeout' if still queued/running when timeout hits
    """
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
            return {
                "status": "timeout",
                "job": last_job,
                "message": "Job still queued/running. Worker may be busy or schedule not firing yet.",
            }

        time.sleep(max(1, int(poll_s)))


# =========================
# Lightweight ROI tool (optional, handy sanity check)
# =========================
def basic_roi_for_pl_column(pl_column: str, csv_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Simple row-level ROI:
      - total_pl = sum(pl_column) over non-null rows
      - bets = count of non-null rows
      - avg_pl_per_bet = total_pl / bets
    NOTE: For lay ROI you will later want liability-weighting; this is basic only.
    """
    csv_url = csv_url or _get_env_or_secret("DATA_CSV_URL")
    if not csv_url:
        raise RuntimeError("Missing DATA_CSV_URL in Streamlit secrets.")
    df = _load_csv_cached(csv_url)

    if pl_column not in df.columns:
        return {"error": "missing_column", "pl_column": pl_column}

    sub = df[df[pl_column].notna()].copy()
    # coerce numeric
    sub[pl_column] = pd.to_numeric(sub[pl_column], errors="coerce")
    sub = sub[sub[pl_column].notna()]

    bets = int(len(sub))
    total_pl = float(sub[pl_column].sum()) if bets else 0.0
    avg = float(total_pl / bets) if bets else 0.0
    return {"pl_column": pl_column, "bets": bets, "total_pl": total_pl, "avg_pl_per_bet": avg}
