from __future__ import annotations

import os
import json
import time
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from google.oauth2.service_account import Credentials
from supabase import create_client


# =========================
# Google Sheets (memory tabs)
# =========================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

SHEET_URL_DEFAULT = os.getenv("FOOTBALL_MEMORY_SHEET_URL") or st.secrets.get("FOOTBALL_MEMORY_SHEET_URL", "")

TAB_RESEARCH_MEMORY = "research_memory"
TAB_DATASET_OVERVIEW = "dataset_overview"
TAB_RESEARCH_RULES = "research_rules"
TAB_COLUMN_DEFS = "column_definitions"
TAB_RESEARCH_STATE = "research_state"
TAB_EVAL_FRAMEWORK = "evaluation_framework"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _gs_creds() -> Credentials:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets/env.")
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return Credentials.from_service_account_info(data, scopes=SCOPES)
    except Exception as e:
        raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON: {e}")


def _sheet_url() -> str:
    url = st.secrets.get("FOOTBALL_MEMORY_SHEET_URL") or os.getenv("FOOTBALL_MEMORY_SHEET_URL") or SHEET_URL_DEFAULT
    url = (url or "").strip()
    if not url:
        raise RuntimeError("Missing FOOTBALL_MEMORY_SHEET_URL in Streamlit secrets/env.")
    return url


def _sh():
    gc = gspread.authorize(_gs_creds())
    return gc.open_by_url(_sheet_url())


def _ws(name: str):
    sh = _sh()
    try:
        return sh.worksheet(name)
    except WorksheetNotFound:
        raise RuntimeError(f"WorksheetNotFound: '{name}'. Create this tab in the Google Sheet.")


def _read_kv_tab(tab_name: str) -> Dict[str, str]:
    ws = _ws(tab_name)
    vals = ws.get_all_values()
    out: Dict[str, str] = {}
    if not vals or len(vals) < 2:
        return out
    for row in vals[1:]:
        if len(row) < 2:
            continue
        k = (row[0] or "").strip()
        v = (row[1] or "").strip()
        if k:
            out[k] = v
    return out


def _read_table_tab(tab_name: str) -> List[Dict[str, str]]:
    ws = _ws(tab_name)
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        return []
    headers = [h.strip() for h in vals[0]]
    rows = []
    for r in vals[1:]:
        if not any((c or "").strip() for c in r):
            continue
        d = {}
        for i, h in enumerate(headers):
            d[h] = (r[i] if i < len(r) else "").strip()
        rows.append(d)
    return rows


# ---------------- Public getters ----------------

def get_dataset_overview() -> Dict[str, Any]:
    return {"tab": TAB_DATASET_OVERVIEW, "data": _read_kv_tab(TAB_DATASET_OVERVIEW)}


def get_research_rules() -> Dict[str, Any]:
    return {"tab": TAB_RESEARCH_RULES, "data": _read_table_tab(TAB_RESEARCH_RULES)}


def get_column_definitions() -> Dict[str, Any]:
    return {"tab": TAB_COLUMN_DEFS, "data": _read_table_tab(TAB_COLUMN_DEFS)}


def get_evaluation_framework() -> Dict[str, Any]:
    return {"tab": TAB_EVAL_FRAMEWORK, "data": _read_table_tab(TAB_EVAL_FRAMEWORK)}


def get_recent_research_notes(limit: int = 20) -> Dict[str, Any]:
    ws = _ws(TAB_RESEARCH_MEMORY)
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        return {"tab": TAB_RESEARCH_MEMORY, "rows": []}
    headers = [h.strip() for h in vals[0]]
    body = vals[1:]
    take = body[-int(limit):] if limit else body[-20:]
    rows = []
    for r in take:
        d = {}
        for i, h in enumerate(headers):
            d[h] = (r[i] if i < len(r) else "")
        rows.append(d)
    return {"tab": TAB_RESEARCH_MEMORY, "rows": rows}


def append_research_note(note: str, tags: str = "") -> Dict[str, Any]:
    ws = _ws(TAB_RESEARCH_MEMORY)
    try:
        ws.append_row([_now_iso(), note, tags or ""], value_input_option="RAW")
        return {"ok": True, "tab": TAB_RESEARCH_MEMORY}
    except APIError as e:
        return {"ok": False, "error": f"gspread APIError append_research_note: {e}"}


def get_research_state() -> Dict[str, Any]:
    return {"tab": TAB_RESEARCH_STATE, "data": _read_kv_tab(TAB_RESEARCH_STATE)}


def set_research_state(key: str, value: str) -> Dict[str, Any]:
    """
    Fault-tolerant:
    - ensures header exists
    - tries to update existing row
    - if update fails (protected range / perms / transient), appends a new row instead
    - never raises (returns ok False with error details)
    """
    try:
        ws = _ws(TAB_RESEARCH_STATE)
        vals = ws.get_all_values()

        if not vals:
            ws.append_row(["key", "value"], value_input_option="RAW")
            ws.append_row([key, value], value_input_option="RAW")
            return {"ok": True, "key": key, "value": value, "created_sheet": True}

        # Ensure header
        header = [c.strip().lower() for c in (vals[0] if vals else [])]
        if len(header) < 2 or header[0] != "key" or header[1] != "value":
            # Don’t try to rewrite header if protected; just append safely.
            try:
                ws.insert_row(["key", "value"], index=1)
            except Exception:
                pass  # ok

        # Find key
        for idx, row in enumerate(vals[1:], start=2):
            if len(row) >= 1 and (row[0] or "").strip() == key:
                try:
                    # update_cell is less error-prone than A1 range strings
                    ws.update_cell(idx, 2, value)
                    return {"ok": True, "key": key, "value": value, "updated": True}
                except APIError as e:
                    # fallback: append
                    try:
                        ws.append_row([key, value], value_input_option="RAW")
                        return {"ok": True, "key": key, "value": value, "appended_fallback": True, "update_error": str(e)}
                    except APIError as e2:
                        return {"ok": False, "error": f"update failed ({e}); append fallback failed ({e2})"}

        # Not found -> append
        try:
            ws.append_row([key, value], value_input_option="RAW")
            return {"ok": True, "key": key, "value": value, "created": True}
        except APIError as e:
            return {"ok": False, "error": f"append failed: {e}"}

    except Exception as e:
        return {"ok": False, "error": f"set_research_state failed: {e}"}


# =========================
# Supabase
# =========================

def _sb():
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets/env.")
    return create_client(url, key)


# =========================
# CSV loading (Supabase Storage preferred)
# =========================

def _download_from_storage(bucket: str, path: str) -> bytes:
    sb = _sb()
    return sb.storage.from_(bucket).download(path)


def _download_from_url(csv_url: str) -> bytes:
    r = requests.get(csv_url, timeout=180)
    r.raise_for_status()
    return r.content


def _load_csv(storage_bucket: Optional[str] = None, storage_path: Optional[str] = None, csv_url: Optional[str] = None) -> pd.DataFrame:
    if storage_bucket and storage_path:
        raw = _download_from_storage(storage_bucket, storage_path)
    elif csv_url:
        raw = _download_from_url(csv_url)
    else:
        raise ValueError("Provide either (storage_bucket + storage_path) OR csv_url.")

    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="replace")

    return pd.read_csv(StringIO(text), low_memory=False)


def load_data_basic(storage_bucket: str = "", storage_path: str = "", csv_url: str = "") -> Dict[str, Any]:
    df = _load_csv(storage_bucket or None, storage_path or None, csv_url or None)
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "head": df.head(5).to_dict(orient="records")}


def list_columns(storage_bucket: str = "", storage_path: str = "", csv_url: str = "") -> Dict[str, Any]:
    df = _load_csv(storage_bucket or None, storage_path or None, csv_url or None)
    return {"columns": df.columns.tolist(), "n": int(len(df.columns))}


# =========================
# ROI helpers (row-level)
# =========================

def _mapping() -> Dict[str, Tuple[str, str]]:
    return {
        "SHG PL": ("lay", "HT CS Price"),
        "SHG 2+ PL": ("lay", "HT 2 Ahead Odds"),
        "LU1.5 PL": ("lay", "U1.5 Odds"),
        "LFGHU0.5 PL": ("lay", "FHGU0.5Odds"),
        "BO 2.5 PL": ("back", "O2.5 Odds"),
        "BO1.5 FHG PL": ("back", "FHGO1.5 Odds"),
        "BTTS PL": ("back", "BTTS Y Odds"),
    }


def basic_roi_for_pl_column(pl_column: str, storage_bucket: str = "", storage_path: str = "", csv_url: str = "") -> Dict[str, Any]:
    df = _load_csv(storage_bucket or None, storage_path or None, csv_url or None)
    if pl_column not in df.columns:
        return {"error": f"Missing PL column: {pl_column}"}

    side, odds_col = _mapping().get(pl_column, ("back", ""))
    d = df[df[pl_column].notna()].copy()
    d[pl_column] = pd.to_numeric(d[pl_column], errors="coerce")
    d = d[d[pl_column].notna()]
    n = int(len(d))
    total_pl = float(d[pl_column].sum()) if n else 0.0

    if n == 0:
        return {"pl_column": pl_column, "bets": 0, "total_pl": 0.0, "roi": 0.0, "avg_pl": 0.0, "side": side}

    if side == "lay":
        if odds_col not in d.columns:
            return {"error": f"Lay ROI requires odds col '{odds_col}' but it is missing."}
        odds = pd.to_numeric(d[odds_col], errors="coerce").fillna(0.0)
        liability = (odds - 1.0).clip(lower=0.0)
        denom = float(liability.sum())
        roi = (total_pl / denom) if denom > 0 else 0.0
        return {"pl_column": pl_column, "side": "lay", "odds_col": odds_col, "bets": n, "total_pl": total_pl,
                "denom_liability": denom, "roi": float(roi), "avg_pl_per_bet": float(total_pl / n)}

    denom = float(n)
    return {"pl_column": pl_column, "side": "back", "odds_col": odds_col or None, "bets": n, "total_pl": total_pl,
            "denom_stake": denom, "roi": float(total_pl / denom), "avg_pl_per_bet": float(total_pl / denom)}


# =========================
# Background jobs (Supabase table + Storage results)
# =========================

def submit_job(task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    sb = _sb()
    row = {"status": "queued", "task_type": task_type, "params": params or {}, "created_at": _now_iso(), "updated_at": _now_iso()}
    res = sb.table("jobs").insert(row).execute()
    data = (res.data or [])
    if not data:
        return {"error": "Insert returned no rows. Check table schema/RLS.", "raw": str(res)}
    return data[0]


def get_job(job_id: str) -> Dict[str, Any]:
    sb = _sb()
    res = sb.table("jobs").select("*").eq("job_id", job_id).limit(1).execute()
    data = (res.data or [])
    if not data:
        return {"error": "job not found", "job_id": job_id}
    return data[0]


def download_result(result_path: str, bucket: str = "") -> Dict[str, Any]:
    sb = _sb()
    b = (bucket or st.secrets.get("RESULTS_BUCKET") or os.getenv("RESULTS_BUCKET") or "football-results").strip()
    raw = sb.storage.from_(b).download(result_path)
    try:
        return {"ok": True, "bucket": b, "result_path": result_path, "result": json.loads(raw.decode("utf-8"))}
    except Exception:
        return {"ok": True, "bucket": b, "result_path": result_path, "raw_text": raw.decode("latin-1", errors="replace")}


def wait_for_job(job_id: str, timeout_s: int = 300, poll_s: int = 5, auto_download: bool = True) -> Dict[str, Any]:
    deadline = time.time() + int(timeout_s or 300)
    poll = max(1, int(poll_s or 5))
    last_job: Dict[str, Any] = {}

    while time.time() < deadline:
        last_job = get_job(job_id)
        if last_job.get("error"):
            return {"status": "error", "error": last_job.get("error"), "job": last_job}

        status = (last_job.get("status") or "").lower()
        if status in ("done", "error"):
            out = {"status": status, "job": last_job}
            if status == "done" and auto_download and last_job.get("result_path"):
                out["result"] = download_result(last_job["result_path"]).get("result")
            return out

        time.sleep(poll)

    return {"status": "timeout", "job": last_job, "job_id": job_id}


# =========================
# Chat persistence (Supabase Storage)
# =========================

def _chat_bucket() -> str:
    return (st.secrets.get("CHAT_BUCKET") or os.getenv("CHAT_BUCKET") or "football-chats").strip()


def save_chat(session_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    sb = _sb()
    bucket = _chat_bucket()
    path = f"sessions/{session_id}.json"
    payload = json.dumps({"session_id": session_id, "messages": messages, "saved_at": _now_iso()}, ensure_ascii=False, indent=2).encode("utf-8")

    sb.storage.from_(bucket).upload(path=path, file=payload, file_options={"content-type": "application/json", "upsert": "true"})
    return {"ok": True, "bucket": bucket, "path": path}


def load_chat(session_id: str) -> Dict[str, Any]:
    sb = _sb()
    bucket = _chat_bucket()
    path = f"sessions/{session_id}.json"
    try:
        raw = sb.storage.from_(bucket).download(path)
        return {"ok": True, "bucket": bucket, "path": path, "data": json.loads(raw.decode("utf-8"))}
    except Exception:
        return {"ok": False, "bucket": bucket, "path": path, "data": {}}
