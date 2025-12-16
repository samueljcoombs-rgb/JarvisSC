from __future__ import annotations

import os
import json
import re
import time
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from google.oauth2.service_account import Credentials
from supabase import create_client

try:
    from storage3.utils import StorageException
except Exception:
    StorageException = Exception

# ============================================================
# Google Sheets tabs (agreed names)
# ============================================================

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

# ============================================================
# Google Sheets helpers (fault tolerant)
# ============================================================

def _gs_creds() -> Credentials:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets/env.")
    data = json.loads(raw) if isinstance(raw, str) else raw
    return Credentials.from_service_account_info(data, scopes=SCOPES)

def _sheet_url() -> str:
    url = st.secrets.get("FOOTBALL_MEMORY_SHEET_URL") or os.getenv("FOOTBALL_MEMORY_SHEET_URL") or SHEET_URL_DEFAULT
    url = (url or "").strip()
    if not url:
        raise RuntimeError("Missing FOOTBALL_MEMORY_SHEET_URL in Streamlit secrets/env.")
    return url

@st.cache_resource(show_spinner=False)
def _sh_cached():
    gc = gspread.authorize(_gs_creds())
    return gc.open_by_url(_sheet_url())

def _ws(name: str):
    sh = _sh_cached()
    try:
        return sh.worksheet(name)
    except WorksheetNotFound:
        raise RuntimeError(f"WorksheetNotFound: '{name}'. Create this tab in the Google Sheet.")

def _safe_get_all_values(tab: str) -> List[List[str]]:
    try:
        return _ws(tab).get_all_values()
    except Exception:
        return []

def _ensure_header(tab: str, header: List[str]) -> Dict[str, Any]:
    try:
        ws = _ws(tab)
        vals = ws.get_all_values()
        if not vals:
            ws.append_row(header, value_input_option="RAW")
            return {"ok": True, "changed": True, "created": True}

        first = [c.strip() for c in (vals[0] or [])]
        if len(first) < len(header):
            try:
                ws.insert_row(header, index=1)
                return {"ok": True, "changed": True, "inserted": True}
            except Exception:
                return {"ok": True, "changed": False, "note": "Header short but cannot insert (possibly protected)."}
        return {"ok": True, "changed": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _read_kv_tab(tab_name: str) -> Dict[str, str]:
    vals = _safe_get_all_values(tab_name)
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
    vals = _safe_get_all_values(tab_name)
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
    vals = _safe_get_all_values(TAB_RESEARCH_MEMORY)
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
    _ensure_header(TAB_RESEARCH_MEMORY, ["timestamp", "note", "tags"])
    try:
        ws = _ws(TAB_RESEARCH_MEMORY)
        ws.append_row([_now_iso(), note, tags or ""], value_input_option="RAW")
        return {"ok": True, "tab": TAB_RESEARCH_MEMORY}
    except APIError as e:
        return {"ok": False, "error": f"gspread APIError append_research_note: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"append_research_note failed: {e}"}

def get_research_state() -> Dict[str, Any]:
    return {"tab": TAB_RESEARCH_STATE, "data": _read_kv_tab(TAB_RESEARCH_STATE)}

def set_research_state(key: str, value: str) -> Dict[str, Any]:
    _ensure_header(TAB_RESEARCH_STATE, ["key", "value"])
    try:
        ws = _ws(TAB_RESEARCH_STATE)
        vals = ws.get_all_values() or []
        for idx, row in enumerate(vals[1:], start=2):
            if len(row) >= 1 and (row[0] or "").strip() == key:
                try:
                    ws.update_cell(idx, 2, value)
                    return {"ok": True, "key": key, "value": value, "updated": True}
                except APIError as e:
                    try:
                        ws.append_row([key, value], value_input_option="RAW")
                        return {"ok": True, "key": key, "value": value, "appended_fallback": True, "update_error": str(e)}
                    except Exception as e2:
                        return {"ok": False, "error": f"update failed ({e}); append fallback failed ({e2})"}

        ws.append_row([key, value], value_input_option="RAW")
        return {"ok": True, "key": key, "value": value, "created": True}
    except APIError as e:
        return {"ok": False, "error": f"set_research_state APIError: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"set_research_state failed: {e}"}

# ============================================================
# Source-of-truth context fetch + derived constraints
# ============================================================

def _extract_ignored_columns(rules_rows: List[Dict[str, str]]) -> List[str]:
    ignored: List[str] = []
    for r in rules_rows or []:
        txt = (r.get("rule") or r.get("Rule") or "").strip()
        if not txt:
            continue
        low = txt.lower()
        if "should be ignored" in low or "ignored completely" in low:
            # quick heuristic: pull backtick-quoted or comma-separated column names if present
            cols = re.findall(r"`([^`]+)`", txt)
            if cols:
                ignored.extend([c.strip() for c in cols if c.strip()])
                continue
            if "result" in low:
                ignored.append("Result")
            if "home form" in low:
                ignored.append("HOME FORM")
    # de-dupe preserve order
    seen = set()
    out = []
    for x in ignored:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _extract_outcome_columns(col_defs: List[Dict[str, str]]) -> List[str]:
    out = []
    for r in col_defs or []:
        col = (r.get("column") or r.get("Column") or "").strip()
        role = (r.get("role") or r.get("Role") or "").strip().lower()
        if col and role == "outcome":
            out.append(col)
    return out


def _parse_csv_list(v):
    if v is None:
        return []
    if not isinstance(v, str):
        v = str(v)
    parts = [p.strip() for p in v.replace("\n", ",").split(",")]
    return [p for p in parts if p]

def get_research_context(limit_notes: int = 20) -> Dict[str, Any]:
    overview = get_dataset_overview().get("data") or {}
    rules_rows = get_research_rules().get("data") or []
    col_defs = get_column_definitions().get("data") or []
    eval_fw = get_evaluation_framework().get("data") or []
    state = get_research_state().get("data") or {}

    # Optional feature blacklist: exclude from predictive inputs (but keep for grouping/reporting)
    ignored_feature_columns = []
    for k in ("ignored_feature_columns", "feature_blacklist", "ignored_features"):
        if k in state and state.get(k):
            ignored_feature_columns.extend(_parse_csv_list(state.get(k)))
    ignored_feature_columns = sorted(set([c for c in ignored_feature_columns if c]))
    notes = get_recent_research_notes(limit=limit_notes).get("rows") or []

    ignored_cols = _extract_ignored_columns(rules_rows)
    outcome_cols = _extract_outcome_columns(col_defs)

    return {
        "ok": True,
        "dataset_overview": overview,
        "research_rules": rules_rows,
        "column_definitions": col_defs,
        "evaluation_framework": eval_fw,
        "research_state": state,
        "recent_notes": notes,
        "derived": {
            "ignored_feature_columns": ignored_feature_columns,
"ignored_columns": ignored_cols, "outcome_columns": outcome_cols},
        "ts": _now_iso(),
    }

# ============================================================
# Supabase
# ============================================================

@st.cache_resource(show_spinner=False)
def _sb_cached():
    url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets/env.")
    return create_client(url, key)

def _sb():
    return _sb_cached()

# ============================================================
# CSV loading (Supabase Storage preferred)
# ============================================================

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

# ============================================================
# Jobs (Supabase table + Storage results)
# ============================================================

def submit_job(task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    sb = _sb()
    row = {"status": "queued", "task_type": task_type, "params": params or {}, "created_at": _now_iso(), "updated_at": _now_iso()}
    try:
        res = sb.table("jobs").insert(row).execute()
        data = (res.data or [])
        if not data:
            return {"error": "Insert returned no rows. Check table schema/RLS.", "raw": str(res)}
        return data[0]
    except Exception as e:
        return {"error": f"submit_job failed: {e}"}

def get_job(job_id: str) -> Dict[str, Any]:
    sb = _sb()
    try:
        res = sb.table("jobs").select("*").eq("job_id", job_id).limit(1).execute()
        data = (res.data or [])
        if not data:
            return {"error": "job not found", "job_id": job_id}
        return data[0]
    except Exception as e:
        return {"error": f"get_job failed: {e}", "job_id": job_id}



def get_job_events(job_id: str, since_ts: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    """
    Fetch narrated worker events for a job from Supabase `job_events`.
    since_ts: ISO timestamp string; if provided, only events with ts > since_ts are returned.
    """
    sb = _sb()
    q = sb.table("job_events").select("event_id,job_id,ts,level,message,payload").eq("job_id", job_id).order("ts", desc=False).limit(limit)
    if since_ts:
        q = q.gt("ts", since_ts)
    res = q.execute()
    rows = res.data or []
    return {"job_id": job_id, "events": rows}

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

# ============================================================
# PL lab starter
# ============================================================

def start_pl_lab(
    duration_minutes: int = 300,
    pl_column: str = "BTTS PL",
    do_hyperopt: bool = False,
    hyperopt_iter: int = 12,
    enforcement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx = get_research_context(limit_notes=10)
    derived = (ctx.get("derived") or {})
    ignored_columns = derived.get("ignored_columns") or []
    outcome_columns = derived.get("outcome_columns") or []

    storage_bucket = (st.secrets.get("DATA_STORAGE_BUCKET") or os.getenv("DATA_STORAGE_BUCKET") or "football-data").strip()
    storage_path = (st.secrets.get("DATA_STORAGE_PATH") or os.getenv("DATA_STORAGE_PATH") or "football_ai_NNIA.csv").strip()
    results_bucket = (st.secrets.get("RESULTS_BUCKET") or os.getenv("RESULTS_BUCKET") or "football-results").strip()

    params = {
        "storage_bucket": storage_bucket,
        "storage_path": storage_path,
        "_results_bucket": results_bucket,
        "pl_column": pl_column,
        "duration_minutes": int(duration_minutes),
        "top_fracs": [0.05, 0.1, 0.2],
        "do_hyperopt": bool(do_hyperopt),
        "hyperopt_iter": int(hyperopt_iter),
        "top_n": 12,
        "ignored_columns": ignored_columns,
        "outcome_columns": outcome_columns,
        "enforcement": enforcement or {},
    }
    return submit_job("pl_lab", params)

# ============================================================
# Chat sessions (Supabase Storage)
# ============================================================

def _chat_bucket() -> str:
    return (st.secrets.get("CHAT_BUCKET") or os.getenv("CHAT_BUCKET") or "football-chats").strip()

def _chat_index_path() -> str:
    return "sessions/index.json"

def _load_chat_index(sb) -> Dict[str, Any]:
    bucket = _chat_bucket()
    try:
        raw = sb.storage.from_(bucket).download(_chat_index_path())
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {"sessions": []}

def _save_chat_index(sb, index: Dict[str, Any]) -> None:
    bucket = _chat_bucket()
    payload = json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8")
    sb.storage.from_(bucket).upload(
        path=_chat_index_path(),
        file=payload,
        file_options={"content-type": "application/json", "upsert": "true"},
    )

def list_chats(limit: int = 200) -> Dict[str, Any]:
    sb = _sb()
    bucket = _chat_bucket()
    idx = _load_chat_index(sb)
    sessions = idx.get("sessions") or []
    sessions = sessions[-int(limit):] if limit else sessions[-200:]
    return {"ok": True, "bucket": bucket, "sessions": sessions}

def save_chat(session_id: str, messages: List[Dict[str, Any]], title: str = "") -> Dict[str, Any]:
    sb = _sb()
    bucket = _chat_bucket()
    path = f"sessions/{session_id}.json"
    payload = json.dumps({"session_id": session_id, "title": title or "", "messages": messages, "saved_at": _now_iso()}, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        sb.storage.from_(bucket).upload(path=path, file=payload, file_options={"content-type": "application/json", "upsert": "true"})
    except StorageException as e:
        return {"ok": False, "error": f"Storage upload failed. Create bucket '{bucket}' in Supabase Storage. Details: {e}", "bucket": bucket, "path": path}
    except Exception as e:
        return {"ok": False, "error": f"save_chat failed: {e}", "bucket": bucket, "path": path}

    # update index best-effort
    try:
        idx = _load_chat_index(sb)
        sessions = idx.get("sessions") or []
        existing = next((s for s in sessions if s.get("session_id") == session_id), None)
        if existing:
            existing["saved_at"] = _now_iso()
            if title:
                existing["title"] = title
        else:
            sessions.append({"session_id": session_id, "title": title or f"Session {session_id[:8]}", "saved_at": _now_iso()})
        idx["sessions"] = sessions
        _save_chat_index(sb, idx)
    except Exception:
        pass

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

def rename_chat(session_id: str, title: str) -> Dict[str, Any]:
    sb = _sb()
    idx = _load_chat_index(sb)
    sessions = idx.get("sessions") or []
    found = False
    for s in sessions:
        if s.get("session_id") == session_id:
            s["title"] = title
            s["saved_at"] = _now_iso()
            found = True
            break
    if not found:
        sessions.append({"session_id": session_id, "title": title, "saved_at": _now_iso()})
    idx["sessions"] = sessions
    try:
        _save_chat_index(sb, idx)
    except Exception:
        pass
    return {"ok": True, "session_id": session_id, "title": title}

def delete_chat(session_id: str) -> Dict[str, Any]:
    sb = _sb()
    bucket = _chat_bucket()
    try:
        sb.storage.from_(bucket).remove([f"sessions/{session_id}.json"])
    except Exception:
        pass
    idx = _load_chat_index(sb)
    sessions = idx.get("sessions") or []
    idx["sessions"] = [s for s in sessions if s.get("session_id") != session_id]
    try:
        _save_chat_index(sb, idx)
    except Exception:
        pass
    return {"ok": True, "deleted": True, "session_id": session_id}
