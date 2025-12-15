# modules/football_tools.py
from __future__ import annotations

import os
import json
import time
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st

import gspread
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

# Your agreed tabs:
TAB_RESEARCH_MEMORY = "research_memory"
TAB_DATASET_OVERVIEW = "dataset_overview"
TAB_RESEARCH_RULES = "research_rules"
TAB_COLUMN_DEFS = "column_definitions"
TAB_RESEARCH_STATE = "research_state"
TAB_EVAL_FRAMEWORK = "evaluation_framework"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _gs_creds() -> Optional[Credentials]:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return Credentials.from_service_account_info(data, scopes=SCOPES)
    except Exception:
        return None


def _sheet_url() -> str:
    url = st.secrets.get("FOOTBALL_MEMORY_SHEET_URL") or os.getenv("FOOTBALL_MEMORY_SHEET_URL") or SHEET_URL_DEFAULT
    return (url or "").strip()


def _sh():
    url = _sheet_url()
    if not url:
        raise RuntimeError("Missing FOOTBALL_MEMORY_SHEET_URL in Streamlit secrets/env.")
    creds = _gs_creds()
    if not creds:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets/env.")
    gc = gspread.authorize(creds)
    return gc.open_by_url(url)


def _ws(name: str):
    sh = _sh()
    return sh.worksheet(name)


def _read_kv_tab(tab_name: str) -> Dict[str, str]:
    ws = _ws(tab_name)
    vals = ws.get_all_values()
    out: Dict[str, str] = {}
    if not vals or len(vals) < 2:
        return out
    header = [h.strip().lower() for h in vals[0]]
    # expect ["key","value"] but be tolerant
    for row in vals[1:]:
        if len(row) < 2:
            continue
        k = row[0].strip()
        v = row[1].strip()
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


# Public getters used by the agent
def get_dataset_overview() -> Dict[str, Any]:
    return {"tab": TAB_DATASET_OVERVIEW, "data": _read_kv_tab(TAB_DATASET_OVERVIEW)}


def get_research_rules() -> Dict[str, Any]:
    # research_rules is a 1-column "rule" table in your spec
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
    ws.append_row([_now_iso(), note, tags or ""], value_input_option="RAW")
    return {"ok": True, "tab": TAB_RESEARCH_MEMORY}


def get_research_state() -> Dict[str, Any]:
    return {"tab": TAB_RESEARCH_STATE, "data": _read_kv_tab(TAB_RESEARCH_STATE)}


def set_research_state(key: str, value: str) -> Dict[str, Any]:
    ws = _ws(TAB_RESEARCH_STATE)
    vals = ws.get_all_values()
    if not vals:
        ws.append_row(["key", "value"])
        ws.append_row([key, value])
        return {"ok": True, "key": key, "value": value}

    # find existing key in col A
    for idx, row in enumerate(vals[1:], start=2):  # 1-based, skip header
        if len(row) >= 1 and (row[0] or "").strip() == key:
            ws.update(f"B{idx}", value)
            return {"ok": True, "key": key, "value": value, "updated": True}

    ws.append_row([key, value])
    return {"ok": True, "key": key, "value": value, "created": True}


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
    b = (bucket or "").strip()
    p = (path or "").strip()
    if not b or not p:
        raise ValueError("storage_bucket and storage_path are required for storage download.")
    data = sb.storage.from_(b).download(p)
    return data


def _download_from_url(csv_url: str) -> bytes:
    r = requests.get(csv_url, timeout=180)
    r.raise_for_status()
    return r.content


def _load_csv(storage_bucket: Optional[str] = None, storage_path: Optional[str] = None, csv_url: Optional[str] = None) -> pd.DataFrame:
    raw: bytes
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

    df = pd.read_csv(StringIO(text), low_memory=False)
    return df


def load_data_basic(storage_bucket: str = "", storage_path: str = "", csv_url: str = "") -> Dict[str, Any]:
    df = _load_csv(storage_bucket or None, storage_path or None, csv_url or None)
    head = df.head(5).to_dict(orient="records")
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "head": head}


def list_columns(storage_bucket: str = "", storage_path: str = "", csv_url: str = "") -> Dict[str, Any]:
    df = _load_csv(storage_bucket or None, storage_path or None, csv_url or None)
    return {"columns": df.columns.tolist(), "n": int(len(df.columns))}


# =========================
# ROI helpers (row-level)
# =========================

def _mapping() -> Dict[str, Tuple[str, str]]:
    # (side, odds_col)
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
            return {"error": f"Lay ROI requires mapped odds col '{odds_col}' but it is missing."}
        odds = pd.to_numeric(d[odds_col], errors="coerce").fillna(0.0)
        liability = (odds - 1.0).clip(lower=0.0)
        denom = float(liability.sum())
        roi = (total_pl / denom) if denom > 0 else 0.0
        return {
            "pl_column": pl_column,
            "side": "lay",
            "odds_col": odds_col,
            "bets": n,
            "total_pl": total_pl,
            "denom_liability": denom,
            "roi": float(roi),
            "avg_pl_per_bet": float(total_pl / n),
        }

    # back 1pt per bet
    denom = float(n)
    return {
        "pl_column": pl_column,
        "side": "back",
        "odds_col": odds_col or None,
        "bets": n,
        "total_pl": total_pl,
        "denom_stake": denom,
        "roi": float(total_pl / denom),
        "avg_pl_per_bet": float(total_pl / denom),
    }


# =========================
# Background jobs (Supabase table + Storage results)
# =========================

def submit_job(task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    sb = _sb()
    # jobs table expected columns:
    # job_id (uuid default), status, task_type, params (jsonb), result_path, error, created_at, updated_at
    row = {
        "status": "queued",
        "task_type": task_type,
        "params": params or {},
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    res = sb.table("jobs").insert(row).execute()
    data = (res.data or [])
    if not data:
        return {"error": "Insert returned no rows. Check Supabase table schema/RLS.", "raw": str(res)}
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
        text = raw.decode("utf-8")
        return {"ok": True, "bucket": b, "result_path": result_path, "result": json.loads(text)}
    except Exception:
        # last resort return raw text
        try:
            text = raw.decode("latin-1", errors="replace")
        except Exception:
            text = str(raw)
        return {"ok": True, "bucket": b, "result_path": result_path, "raw_text": text}


def wait_for_job(job_id: str, timeout_s: int = 300, poll_s: int = 5, auto_download: bool = True) -> Dict[str, Any]:
    deadline = time.time() + int(timeout_s or 300)
    poll = max(1, int(poll_s or 5))

    last_job: Dict[str, Any] = {}
    while time.time() < deadline:
        last_job = get_job(job_id)

        # If get_job itself failed
        if last_job.get("error"):
            return {"status": "error", "error": last_job.get("error"), "job": last_job}

        status = (last_job.get("status") or "").lower()
        if status in ("done", "error"):
            out = {"status": status, "job": last_job}
            if status == "done" and auto_download:
                rp = last_job.get("result_path")
                if rp:
                    dl = download_result(rp)
                    out["result"] = dl.get("result") or dl
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

    sb.storage.from_(bucket).upload(
        path=path,
        file=payload,
        file_options={"content-type": "application/json", "upsert": "true"},
    )
    return {"ok": True, "bucket": bucket, "path": path}


def load_chat(session_id: str) -> Dict[str, Any]:
    sb = _sb()
    bucket = _chat_bucket()
    path = f"sessions/{session_id}.json"
    try:
        raw = sb.storage.from_(bucket).download(path)
        text = raw.decode("utf-8")
        return {"ok": True, "bucket": bucket, "path": path, "data": json.loads(text)}
    except Exception:
        return {"ok": False, "bucket": bucket, "path": path, "data": {}}
