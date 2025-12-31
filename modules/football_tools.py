"""
Football Tools v2 - Autonomous Agent Edition

New in v2:
- query_data: Submit data exploration jobs
- test_filter: Submit quick hypothesis testing jobs
- regime_check: Submit stability analysis jobs
- Enhanced research context with full Bible loading

All tools are designed to be called by the autonomous agent.
"""

from __future__ import annotations

import os
import json
import re
import time
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Union

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
# Google Sheets configuration
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

TAB_RESEARCH_MEMORY = "research_memory"
TAB_DATASET_OVERVIEW = "dataset_overview"
TAB_RESEARCH_RULES = "research_rules"
TAB_COLUMN_DEFS = "column_definitions"
TAB_RESEARCH_STATE = "research_state"
TAB_EVAL_FRAMEWORK = "evaluation_framework"


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


# ============================================================
# Google Sheets helpers
# ============================================================

def _gs_creds() -> Credentials:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets/env.")
    data = json.loads(raw) if isinstance(raw, str) else raw
    return Credentials.from_service_account_info(data, scopes=SCOPES)


def _sheet_url() -> str:
    url = st.secrets.get("FOOTBALL_MEMORY_SHEET_URL") or os.getenv("FOOTBALL_MEMORY_SHEET_URL", "")
    if not url:
        raise RuntimeError("Missing FOOTBALL_MEMORY_SHEET_URL in Streamlit secrets/env.")
    return url.strip()


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
                return {"ok": True, "changed": False}
        return {"ok": True, "changed": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _read_kv_tab(tab_name: str) -> Dict[str, str]:
    vals = _safe_get_all_values(tab_name)
    out = {}
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


# ============================================================
# Public getters for Bible tabs
# ============================================================

def get_dataset_overview() -> Dict[str, Any]:
    """Get the dataset_overview tab from the Bible."""
    return {"tab": TAB_DATASET_OVERVIEW, "data": _read_kv_tab(TAB_DATASET_OVERVIEW)}


def get_research_rules() -> Dict[str, Any]:
    """Get the research_rules tab from the Bible (THE LAW)."""
    return {"tab": TAB_RESEARCH_RULES, "data": _read_table_tab(TAB_RESEARCH_RULES)}


def get_column_definitions() -> Dict[str, Any]:
    """Get column definitions - crucial for understanding what features mean."""
    return {"tab": TAB_COLUMN_DEFS, "data": _read_table_tab(TAB_COLUMN_DEFS)}


def get_evaluation_framework() -> Dict[str, Any]:
    """Get the evaluation framework - how to judge strategies."""
    return {"tab": TAB_EVAL_FRAMEWORK, "data": _read_table_tab(TAB_EVAL_FRAMEWORK)}


def get_recent_research_notes(limit: int = 20) -> Dict[str, Any]:
    """Get recent research notes - memory of past sessions."""
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
    """Append a research note to the Bible's research_memory tab."""
    _ensure_header(TAB_RESEARCH_MEMORY, ["timestamp", "note", "tags"])
    try:
        ws = _ws(TAB_RESEARCH_MEMORY)
        ws.append_row([_now_iso(), note, tags or ""], value_input_option="RAW")
        return {"ok": True, "tab": TAB_RESEARCH_MEMORY}
    except APIError as e:
        return {"ok": False, "error": f"gspread APIError: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"append_research_note failed: {e}"}


def get_research_state() -> Dict[str, Any]:
    """Get the research_state tab - key/value persistent state."""
    return {"tab": TAB_RESEARCH_STATE, "data": _read_kv_tab(TAB_RESEARCH_STATE)}


def set_research_state(key: str, value: str) -> Dict[str, Any]:
    """Set a key-value pair in research_state."""
    _ensure_header(TAB_RESEARCH_STATE, ["key", "value"])
    try:
        ws = _ws(TAB_RESEARCH_STATE)
        vals = ws.get_all_values() or []
        for idx, row in enumerate(vals[1:], start=2):
            if len(row) >= 1 and (row[0] or "").strip() == key:
                try:
                    ws.update_cell(idx, 2, value)
                    return {"ok": True, "key": key, "value": value, "updated": True}
                except Exception:
                    ws.append_row([key, value], value_input_option="RAW")
                    return {"ok": True, "key": key, "value": value, "appended_fallback": True}
        ws.append_row([key, value], value_input_option="RAW")
        return {"ok": True, "key": key, "value": value, "created": True}
    except Exception as e:
        return {"ok": False, "error": f"set_research_state failed: {e}"}


# ============================================================
# Derived context extraction
# ============================================================

def _extract_ignored_columns(rules_rows: List[Dict[str, str]]) -> List[str]:
    ignored = []
    for r in rules_rows or []:
        txt = (r.get("rule") or r.get("Rule") or "").strip()
        if not txt:
            continue
        low = txt.lower()
        if "should be ignored" in low or "ignored completely" in low:
            cols = re.findall(r"`([^`]+)`", txt)
            if cols:
                ignored.extend([c.strip() for c in cols if c.strip()])
                continue
            if "result" in low:
                ignored.append("Result")
            if "home form" in low:
                ignored.append("HOME FORM")
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


def _extract_feature_columns(col_defs: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Extract features by type from column definitions."""
    numeric = []
    categorical = []
    for r in col_defs or []:
        col = (r.get("column") or r.get("Column") or "").strip()
        role = (r.get("role") or r.get("Role") or "").strip().lower()
        if not col:
            continue
        if "numeric" in role or "feature" in role:
            if "categorical" in role:
                categorical.append(col)
            else:
                numeric.append(col)
        elif "categorical" in role:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical}


def _parse_csv_list(v) -> List[str]:
    if v is None:
        return []
    if not isinstance(v, str):
        v = str(v)
    parts = [p.strip() for p in v.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def get_research_context(limit_notes: int = 20) -> Dict[str, Any]:
    """
    Get the FULL Bible context - everything the agent needs to make decisions.
    
    Returns:
        - dataset_overview: Goal, output format, key advice
        - research_rules: THE LAW - what to never do, what to always do
        - column_definitions: What each column means
        - evaluation_framework: How to judge strategies
        - research_state: Current gates, recent actions
        - recent_notes: Memory from past sessions
        - derived: Extracted constraints (ignored cols, outcome cols, features)
    """
    overview = get_dataset_overview().get("data") or {}
    rules_rows = get_research_rules().get("data") or []
    col_defs = get_column_definitions().get("data") or []
    eval_fw = get_evaluation_framework().get("data") or []
    state = get_research_state().get("data") or {}
    notes = get_recent_research_notes(limit=limit_notes).get("rows") or []

    # Extract derived constraints
    ignored_feature_columns = []
    for k in ("ignored_feature_columns", "feature_blacklist", "ignored_features"):
        if k in state and state.get(k):
            ignored_feature_columns.extend(_parse_csv_list(state.get(k)))
    ignored_feature_columns = sorted(set([c for c in ignored_feature_columns if c]))

    ignored_cols = _extract_ignored_columns(rules_rows)
    outcome_cols = _extract_outcome_columns(col_defs)
    feature_cols = _extract_feature_columns(col_defs)

    # Parse gates from state with safe defaults
    def _safe_int(v, default):
        try:
            return int(float(v))
        except:
            return default
    
    def _safe_float(v, default):
        try:
            return float(v)
        except:
            return default

    gates = {
        "min_train_rows": _safe_int(state.get("min_train_rows"), 300),
        "min_val_rows": _safe_int(state.get("min_val_rows"), 60),
        "min_test_rows": _safe_int(state.get("min_test_rows"), 60),
        "max_train_val_gap_roi": _safe_float(state.get("max_train_val_gap_roi"), 0.4),
        "max_test_drawdown": _safe_float(state.get("max_test_drawdown"), -50),
        "max_test_losing_streak_bets": _safe_int(state.get("max_test_losing_streak_bets"), 50),
    }

    # Parse recent actions from state
    recent_actions = []
    try:
        actions_json = state.get("recent_actions", "[]")
        if actions_json:
            recent_actions = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
    except Exception:
        pass

    return {
        "ok": True,
        "dataset_overview": overview,
        "research_rules": rules_rows,
        "column_definitions": col_defs,
        "evaluation_framework": eval_fw,
        "research_state": state,
        "recent_notes": notes,
        "gates": gates,
        "recent_actions": recent_actions,
        "derived": {
            "ignored_feature_columns": ignored_feature_columns,
            "ignored_columns": ignored_cols,
            "outcome_columns": outcome_cols,
            "feature_columns": feature_cols,
        },
        "ts": _now_iso(),
    }


# ============================================================
# Supabase client
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
# Storage helpers
# ============================================================

def _get_storage_config() -> Dict[str, str]:
    """Get storage bucket/path configuration."""
    return {
        "storage_bucket": (st.secrets.get("DATA_STORAGE_BUCKET") or os.getenv("DATA_STORAGE_BUCKET") or "football-data").strip(),
        "storage_path": (st.secrets.get("DATA_STORAGE_PATH") or os.getenv("DATA_STORAGE_PATH") or "football_ai_NNIA.csv").strip(),
        "results_bucket": (st.secrets.get("RESULTS_BUCKET") or os.getenv("RESULTS_BUCKET") or "football-results").strip(),
    }


def _download_from_storage(bucket: str, path: str) -> bytes:
    sb = _sb()
    return sb.storage.from_(bucket).download(path)


def _load_csv(storage_bucket: Optional[str] = None, storage_path: Optional[str] = None) -> pd.DataFrame:
    cfg = _get_storage_config()
    bucket = storage_bucket or cfg["storage_bucket"]
    path = storage_path or cfg["storage_path"]
    raw = _download_from_storage(bucket, path)
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(text), low_memory=False)


# ============================================================
# Jobs (Supabase table + Storage results)
# ============================================================

def submit_job(task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a job to the queue."""
    sb = _sb()
    row = {
        "status": "queued",
        "task_type": task_type,
        "params": params or {},
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    try:
        res = sb.table("jobs").insert(row).execute()
        data = res.data or []
        if not data:
            return {"error": "Insert returned no rows. Check table schema/RLS.", "raw": str(res)}
        return data[0]
    except Exception as e:
        return {"error": f"submit_job failed: {e}"}


def get_job(job_id: str) -> Dict[str, Any]:
    """Get job status and details."""
    sb = _sb()
    try:
        res = sb.table("jobs").select("*").eq("job_id", job_id).limit(1).execute()
        data = res.data or []
        if not data:
            return {"error": "job not found", "job_id": job_id}
        return data[0]
    except Exception as e:
        return {"error": f"get_job failed: {e}", "job_id": job_id}


def get_job_events(job_id: str, since_ts: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    """Fetch job events for progress tracking."""
    sb = _sb()
    q = sb.table("job_events").select("*").eq("job_id", job_id).order("ts", desc=False).limit(limit)
    if since_ts:
        q = q.gt("ts", since_ts)
    res = q.execute()
    return {"job_id": job_id, "events": res.data or []}


def download_result(result_path: str, bucket: str = "") -> Dict[str, Any]:
    """Download job result JSON."""
    sb = _sb()
    cfg = _get_storage_config()
    b = bucket or cfg["results_bucket"]
    raw = sb.storage.from_(b).download(result_path)
    try:
        return {"ok": True, "bucket": b, "result_path": result_path, "result": json.loads(raw.decode("utf-8"))}
    except Exception:
        return {"ok": True, "bucket": b, "result_path": result_path, "raw_text": raw.decode("latin-1", errors="replace")}


def wait_for_job(job_id: str, timeout_s: int = 300, poll_s: int = 5, auto_download: bool = True) -> Dict[str, Any]:
    """Wait for a job to complete."""
    deadline = time.time() + int(timeout_s)
    last_job = {}
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
        time.sleep(poll_s)
    return {"status": "timeout", "job": last_job, "job_id": job_id}


# ============================================================
# Job starters - Existing tools
# ============================================================

def start_pl_lab(
    pl_column: str = "BO 2.5 PL",
    duration_minutes: int = 10,
    do_hyperopt: bool = False,
    hyperopt_iter: int = 12,
    enforcement: Optional[Dict[str, Any]] = None,
    row_filters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Start a PL Lab job - full ML pipeline with distillation."""
    ctx = get_research_context(limit_notes=10)
    derived = ctx.get("derived") or {}
    cfg = _get_storage_config()

    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "duration_minutes": int(duration_minutes),
        "top_fracs": [0.05, 0.1, 0.2],
        "do_hyperopt": bool(do_hyperopt),
        "hyperopt_iter": int(hyperopt_iter),
        "ignored_columns": derived.get("ignored_columns") or [],
        "outcome_columns": derived.get("outcome_columns") or [],
        "ignored_feature_columns": derived.get("ignored_feature_columns") or [],
        "row_filters": row_filters or [],
        "enforcement": enforcement or ctx.get("gates") or {},
    }
    return submit_job("pl_lab", params)


def start_subgroup_scan(
    pl_column: str,
    duration_minutes: int = 15,
    group_cols: Optional[List[str]] = None,
    max_groups: int = 50,
    enforcement: Optional[Dict[str, Any]] = None,
    row_filters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Start a subgroup scan - find stable categorical buckets."""
    ctx = get_research_context(limit_notes=10)
    derived = ctx.get("derived") or {}
    cfg = _get_storage_config()

    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "duration_minutes": int(duration_minutes),
        "group_cols": group_cols or ["MODE", "MARKET", "LEAGUE", "BRACKET"],
        "max_groups": int(max_groups),
        "ignored_columns": derived.get("ignored_columns") or [],
        "outcome_columns": derived.get("outcome_columns") or [],
        "row_filters": row_filters or [],
        "enforcement": enforcement or ctx.get("gates") or {},
    }
    return submit_job("subgroup_scan", params)


def start_bracket_sweep(
    pl_column: str,
    duration_minutes: int = 15,
    sweep_cols: Optional[List[str]] = None,
    n_bins: int = 12,
    max_results: int = 50,
    enforcement: Optional[Dict[str, Any]] = None,
    row_filters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Start a bracket sweep - find profitable numeric ranges."""
    ctx = get_research_context(limit_notes=10)
    derived = ctx.get("derived") or {}
    cfg = _get_storage_config()

    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "duration_minutes": int(duration_minutes),
        "sweep_cols": sweep_cols or [],
        "n_bins": int(n_bins),
        "max_results": int(max_results),
        "ignored_columns": derived.get("ignored_columns") or [],
        "outcome_columns": derived.get("outcome_columns") or [],
        "row_filters": row_filters or [],
        "enforcement": enforcement or ctx.get("gates") or {},
    }
    return submit_job("bracket_sweep", params)


def start_hyperopt_pl_lab(
    pl_column: str,
    duration_minutes: int = 30,
    hyperopt_trials: int = 10,
    top_fracs: Optional[List[float]] = None,
    enforcement: Optional[Dict[str, Any]] = None,
    row_filters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Start hyperopt PL lab - Optuna-driven optimization."""
    ctx = get_research_context(limit_notes=10)
    derived = ctx.get("derived") or {}
    cfg = _get_storage_config()

    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "duration_minutes": int(duration_minutes),
        "hyperopt_trials": int(hyperopt_trials),
        "top_fracs": top_fracs or [0.05, 0.1, 0.2],
        "ignored_columns": derived.get("ignored_columns") or [],
        "outcome_columns": derived.get("outcome_columns") or [],
        "row_filters": row_filters or [],
        "enforcement": enforcement or ctx.get("gates") or {},
    }
    return submit_job("hyperopt_pl_lab", params)


# ============================================================
# NEW Job starters - Agent exploration tools
# ============================================================

def start_query_data(
    query_type: str = "aggregate",
    filters: Optional[List[Dict[str, Any]]] = None,
    group_by: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    pl_column: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Start a data exploration query.
    
    query_type: "aggregate", "filter", "sample", "describe"
    filters: List of filter conditions
    group_by: Columns to group by
    metrics: ["count", "sum:PL_COL", "mean:PL_COL"]
    """
    cfg = _get_storage_config()
    
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "query_type": query_type,
        "filters": filters or [],
        "group_by": group_by or [],
        "metrics": metrics or ["count"],
        "pl_column": pl_column,
        "limit": int(limit),
    }
    return submit_job("query_data", params)


def start_test_filter(
    filters: List[Dict[str, Any]],
    pl_column: str,
    enforcement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Quick hypothesis test - test a specific filter combination.
    
    filters: List of filter conditions like:
        [{"col": "LEAGUE", "op": "in", "values": ["EPL"]},
         {"col": "ACTUAL ODDS", "op": "between", "min": 1.8, "max": 2.5}]
    """
    ctx = get_research_context(limit_notes=5)
    cfg = _get_storage_config()
    
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "filters": filters,
        "pl_column": pl_column,
        "enforcement": enforcement or ctx.get("gates") or {},
    }
    return submit_job("test_filter", params)


def start_regime_check(
    filters: List[Dict[str, Any]],
    pl_column: str,
    split_by: str = "month",
) -> Dict[str, Any]:
    """
    Check time-period stability of a filter combination.
    
    split_by: "week", "month", "quarter"
    """
    cfg = _get_storage_config()
    
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "filters": filters,
        "pl_column": pl_column,
        "split_by": split_by,
    }
    return submit_job("regime_check", params)


# ============================================================
# Convenience aliases
# ============================================================

def subgroup_scan(pl_column: str, **kwargs) -> Dict[str, Any]:
    """Alias for start_subgroup_scan."""
    return start_subgroup_scan(pl_column=pl_column, **kwargs)


def bracket_sweep(pl_column: str, **kwargs) -> Dict[str, Any]:
    """Alias for start_bracket_sweep."""
    return start_bracket_sweep(pl_column=pl_column, **kwargs)


def hyperopt_pl_lab(pl_column: str, **kwargs) -> Dict[str, Any]:
    """Alias for start_hyperopt_pl_lab."""
    return start_hyperopt_pl_lab(pl_column=pl_column, **kwargs)


def query_data(**kwargs) -> Dict[str, Any]:
    """Alias for start_query_data."""
    return start_query_data(**kwargs)


def test_filter(filters: List[Dict], pl_column: str, **kwargs) -> Dict[str, Any]:
    """Alias for start_test_filter."""
    return start_test_filter(filters=filters, pl_column=pl_column, **kwargs)


def regime_check(filters: List[Dict], pl_column: str, **kwargs) -> Dict[str, Any]:
    """Alias for start_regime_check."""
    return start_regime_check(filters=filters, pl_column=pl_column, **kwargs)


# ============================================================
# NEW v4: Additional analysis tools
# ============================================================

def start_combination_scan(
    pl_column: str,
    base_filters: Optional[List[Dict]] = None,
    scan_cols: Optional[List[str]] = None,
    max_combinations: int = 50,
    enforcement: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Start a combination scan job to find synergistic filter combinations.
    
    Args:
        pl_column: Target P&L column
        base_filters: Starting filters to combine with
        scan_cols: Columns to scan for additional filters
        max_combinations: Max combinations to test
        enforcement: Gate parameters
    """
    cfg = _get_storage_config()
    params = {
        "storage_bucket": kwargs.get("storage_bucket") or cfg["storage_bucket"],
        "storage_path": kwargs.get("storage_path") or cfg["storage_path"],
        "_results_bucket": kwargs.get("results_bucket") or cfg["results_bucket"],
        "pl_column": pl_column,
        "base_filters": base_filters or [],
        "scan_cols": scan_cols or ["MODE", "MARKET", "DRIFT IN / OUT", "LEAGUE"],
        "max_combinations": max_combinations,
        "enforcement": enforcement or {},
    }
    return submit_job("combination_scan", params)


def start_forward_walk(
    pl_column: str,
    filters: Optional[List[Dict]] = None,
    n_windows: int = 6,
    train_months: int = 4,
    test_months: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """
    Start a walk-forward validation job.
    
    Args:
        pl_column: Target P&L column
        filters: Filter criteria to test
        n_windows: Number of walk-forward windows
        train_months: Months per training window
        test_months: Months per test window
    """
    cfg = _get_storage_config()
    params = {
        "storage_bucket": kwargs.get("storage_bucket") or cfg["storage_bucket"],
        "storage_path": kwargs.get("storage_path") or cfg["storage_path"],
        "_results_bucket": kwargs.get("results_bucket") or cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters or [],
        "n_windows": n_windows,
        "train_months": train_months,
        "test_months": test_months,
    }
    return submit_job("forward_walk", params)


def start_monte_carlo_sim(
    pl_column: str,
    filters: Optional[List[Dict]] = None,
    n_simulations: int = 1000,
    sample_frac: float = 0.7,
    **kwargs
) -> Dict[str, Any]:
    """
    Start a Monte Carlo simulation for confidence intervals.
    
    Args:
        pl_column: Target P&L column
        filters: Filter criteria
        n_simulations: Number of bootstrap samples
        sample_frac: Fraction of data to sample each iteration
    """
    cfg = _get_storage_config()
    params = {
        "storage_bucket": kwargs.get("storage_bucket") or cfg["storage_bucket"],
        "storage_path": kwargs.get("storage_path") or cfg["storage_path"],
        "_results_bucket": kwargs.get("results_bucket") or cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters or [],
        "n_simulations": n_simulations,
        "sample_frac": sample_frac,
    }
    return submit_job("monte_carlo_sim", params)


def start_correlation_check(
    filters: Optional[List[Dict]] = None,
    outcome_columns: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Start a correlation check to detect potential data leakage.
    
    Args:
        filters: Filter criteria to check
        outcome_columns: Known outcome columns to check against
    """
    cfg = _get_storage_config()
    params = {
        "storage_bucket": kwargs.get("storage_bucket") or cfg["storage_bucket"],
        "storage_path": kwargs.get("storage_path") or cfg["storage_path"],
        "_results_bucket": kwargs.get("results_bucket") or cfg["results_bucket"],
        "filters": filters or [],
        "outcome_columns": outcome_columns or [],
    }
    return submit_job("correlation_check", params)


# Aliases for convenience
def combination_scan(pl_column: str, **kwargs) -> Dict[str, Any]:
    """Alias for start_combination_scan."""
    return start_combination_scan(pl_column=pl_column, **kwargs)


def forward_walk(pl_column: str, filters: List[Dict], **kwargs) -> Dict[str, Any]:
    """Alias for start_forward_walk."""
    return start_forward_walk(pl_column=pl_column, filters=filters, **kwargs)


def monte_carlo_sim(pl_column: str, filters: List[Dict], **kwargs) -> Dict[str, Any]:
    """Alias for start_monte_carlo_sim."""
    return start_monte_carlo_sim(pl_column=pl_column, filters=filters, **kwargs)


def correlation_check(filters: List[Dict], **kwargs) -> Dict[str, Any]:
    """Alias for start_correlation_check."""
    return start_correlation_check(filters=filters, **kwargs)


# ============================================================
# Data loading (for quick local checks)
# ============================================================

def load_data_basic(storage_bucket: str = "", storage_path: str = "") -> Dict[str, Any]:
    """Load basic data info."""
    df = _load_csv(storage_bucket or None, storage_path or None)
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "head": df.head(5).to_dict(orient="records")}


def list_columns(storage_bucket: str = "", storage_path: str = "") -> Dict[str, Any]:
    """List all columns in the dataset."""
    df = _load_csv(storage_bucket or None, storage_path or None)
    return {"columns": df.columns.tolist(), "n": int(len(df.columns))}


# ============================================================
# Chat session management
# ============================================================

def _chat_bucket() -> str:
    return (st.secrets.get("CHAT_BUCKET") or os.getenv("CHAT_BUCKET") or "football-chats").strip()


def _chat_index_path() -> str:
    return "sessions/index.json"


def _load_chat_index(sb) -> Dict[str, Any]:
    try:
        raw = sb.storage.from_(_chat_bucket()).download(_chat_index_path())
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {"sessions": []}


def _save_chat_index(sb, index: Dict[str, Any]) -> None:
    payload = json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8")
    sb.storage.from_(_chat_bucket()).upload(
        path=_chat_index_path(),
        file=payload,
        file_options={"content-type": "application/json", "upsert": "true"},
    )


def list_chats(limit: int = 200) -> Dict[str, Any]:
    """List saved chat sessions."""
    sb = _sb()
    idx = _load_chat_index(sb)
    sessions = idx.get("sessions") or []
    sessions = sessions[-int(limit):]
    return {"ok": True, "bucket": _chat_bucket(), "sessions": sessions}


def save_chat(session_id: str, messages: List[Dict[str, Any]], title: str = "") -> Dict[str, Any]:
    """Save a chat session."""
    sb = _sb()
    path = f"sessions/{session_id}.json"
    payload = json.dumps({
        "session_id": session_id,
        "title": title or "",
        "messages": messages,
        "saved_at": _now_iso()
    }, ensure_ascii=False, indent=2).encode("utf-8")
    
    try:
        sb.storage.from_(_chat_bucket()).upload(
            path=path,
            file=payload,
            file_options={"content-type": "application/json", "upsert": "true"}
        )
    except Exception as e:
        return {"ok": False, "error": f"save_chat failed: {e}"}

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

    return {"ok": True, "bucket": _chat_bucket(), "path": path}


def load_chat(session_id: str) -> Dict[str, Any]:
    """Load a chat session."""
    sb = _sb()
    path = f"sessions/{session_id}.json"
    try:
        raw = sb.storage.from_(_chat_bucket()).download(path)
        return {"ok": True, "data": json.loads(raw.decode("utf-8"))}
    except Exception:
        return {"ok": False, "data": {}}


def delete_chat(session_id: str) -> Dict[str, Any]:
    """Delete a chat session."""
    sb = _sb()
    try:
        sb.storage.from_(_chat_bucket()).remove([f"sessions/{session_id}.json"])
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


# ============================================================
# Action logging for agent memory
# ============================================================

def log_agent_action(
    action_type: str,
    params: Dict[str, Any],
    result_summary: Dict[str, Any],
    tags: str = "",
) -> Dict[str, Any]:
    """
    Log an agent action to research_memory for learning.
    
    This helps the agent remember what it tried and what worked/didn't.
    """
    note = {
        "ts_utc": _now_iso(),
        "action_type": action_type,
        "params": params,
        "result": result_summary,
    }
    
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    tag_list.append(action_type)
    
    return append_research_note(
        note=json.dumps(note, ensure_ascii=False),
        tags=",".join(tag_list)
    )


def update_recent_actions(action: Dict[str, Any]) -> Dict[str, Any]:
    """Update the recent_actions list in research_state."""
    state = get_research_state().get("data") or {}
    
    try:
        actions_json = state.get("recent_actions", "[]")
        recent = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
    except Exception:
        recent = []
    
    if not isinstance(recent, list):
        recent = []
    
    recent.append(action)
    recent = recent[-50:]  # Keep last 50
    
    return set_research_state("recent_actions", json.dumps(recent))


# =====================================================================
# NEW: Feature Importance Starter
# =====================================================================

def start_feature_importance(
    pl_column: str = "PL",
    top_n: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze which features correlate most with profitability.
    
    For each numeric column, computes:
    - Correlation with PL column
    - Mean PL for high/low halves
    - Quartile analysis
    
    This helps identify which features are worth exploring for filter strategies.
    
    Args:
        pl_column: Target profit/loss column
        top_n: Number of top features to return
    
    Returns:
        Job submission result with job_id
    """
    params = {
        "pl_column": pl_column,
        "top_n": top_n,
        **kwargs,
    }
    return submit_job("feature_importance", params)


def feature_importance(pl_column: str = "PL", **kwargs) -> Dict[str, Any]:
    """Convenience wrapper that starts job and waits for result."""
    job = start_feature_importance(pl_column=pl_column, **kwargs)
    if not job.get("job_id"):
        return job
    return wait_for_job(job["job_id"])


# =====================================================================
# NEW: Univariate Scan Starter
# =====================================================================

def start_univariate_scan(
    pl_column: str = "PL",
    scan_cols: Optional[List[str]] = None,
    min_rows: int = 200,
    n_bins: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    For each column, find the single best filter value/range.
    
    This is simpler than bracket_sweep - it tests each column independently
    and finds what single filter works best for that column alone.
    
    Args:
        pl_column: Target profit/loss column
        scan_cols: Specific columns to scan (None = all non-outcome columns)
        min_rows: Minimum rows required for a column to be analyzed
        n_bins: Number of bins for numeric columns
    
    Returns:
        Job submission result with job_id
    """
    params = {
        "pl_column": pl_column,
        "min_rows": min_rows,
        "n_bins": n_bins,
        **kwargs,
    }
    if scan_cols:
        params["scan_cols"] = scan_cols
    
    return submit_job("univariate_scan", params)


def univariate_scan(pl_column: str = "PL", **kwargs) -> Dict[str, Any]:
    """Convenience wrapper that starts job and waits for result."""
    job = start_univariate_scan(pl_column=pl_column, **kwargs)
    if not job.get("job_id"):
        return job
    return wait_for_job(job["job_id"])


# =====================================================================
# NEW v6: Session, Checkpoint, Strategy, Learning Tools
# =====================================================================

def start_create_session(pl_column: str = "", config: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """Create a new research session."""
    params = {"pl_column": pl_column, "config": config or {}, **kwargs}
    return submit_job("create_session", params)


def start_save_checkpoint(session_id: str, state: Dict, iteration: int = 0, **kwargs) -> Dict[str, Any]:
    """Save checkpoint to Supabase."""
    params = {"session_id": session_id, "state": state, "iteration": iteration, **kwargs}
    return submit_job("save_checkpoint", params)


def start_load_checkpoint(session_id: str, **kwargs) -> Dict[str, Any]:
    """Load checkpoint from Supabase."""
    params = {"session_id": session_id, **kwargs}
    return submit_job("load_checkpoint", params)


def start_save_strategy(
    pl_column: str,
    filters: List[Dict],
    train: Optional[Dict] = None,
    val: Optional[Dict] = None,
    test: Optional[Dict] = None,
    status: str = "draft",
    **kwargs
) -> Dict[str, Any]:
    """Save strategy to Supabase."""
    params = {
        "pl_column": pl_column,
        "filters": filters,
        "train": train or {},
        "val": val or {},
        "test": test or {},
        "status": status,
        **kwargs
    }
    return submit_job("save_strategy", params)


def start_query_strategies(
    status: Optional[str] = None,
    pl_column: Optional[str] = None,
    min_test_roi: Optional[float] = None,
    limit: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Query strategies from Supabase."""
    params = {"status": status, "pl_column": pl_column, "min_test_roi": min_test_roi, "limit": limit, **kwargs}
    return submit_job("query_strategies", params)


def start_promote_strategy(
    strategy_id: Optional[str] = None,
    filter_hash: Optional[str] = None,
    new_status: str = "candidate",
    **kwargs
) -> Dict[str, Any]:
    """Promote strategy to next lifecycle stage."""
    params = {"strategy_id": strategy_id, "filter_hash": filter_hash, "new_status": new_status, **kwargs}
    return submit_job("promote_strategy", params)


def start_save_learning(
    insight: str,
    category: Optional[str] = None,
    confidence: str = "low",
    evidence: Optional[List] = None,
    **kwargs
) -> Dict[str, Any]:
    """Save learning to Supabase."""
    params = {
        "insight": insight,
        "category": category,
        "confidence": confidence,
        "evidence": evidence or [],
        **kwargs
    }
    return submit_job("save_learning", params)


def start_query_learnings(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Query learnings from Supabase."""
    params = {"category": category, "search": search, "limit": limit, **kwargs}
    return submit_job("query_learnings", params)


def start_get_research_context(pl_column: str = "BO 2.5 PL", **kwargs) -> Dict[str, Any]:
    """Get research context (Bible)."""
    params = {"pl_column": pl_column, **kwargs}
    return submit_job("get_research_context", params)


def start_statistical_significance(
    pl_column: str,
    filters: List[Dict],
    **kwargs
) -> Dict[str, Any]:
    """Calculate statistical significance."""
    cfg = _get_storage_config()
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters,
        **kwargs
    }
    return submit_job("statistical_significance", params)


def start_time_decay_analysis(
    pl_column: str,
    filters: List[Dict],
    **kwargs
) -> Dict[str, Any]:
    """Analyze alpha decay over time."""
    cfg = _get_storage_config()
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters,
        **kwargs
    }
    return submit_job("time_decay_analysis", params)


# Convenience aliases
def create_session(**kwargs) -> Dict[str, Any]:
    job = start_create_session(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def save_checkpoint(**kwargs) -> Dict[str, Any]:
    job = start_save_checkpoint(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def load_checkpoint(**kwargs) -> Dict[str, Any]:
    job = start_load_checkpoint(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def save_strategy(**kwargs) -> Dict[str, Any]:
    job = start_save_strategy(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def query_strategies_sync(**kwargs) -> Dict[str, Any]:
    job = start_query_strategies(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def promote_strategy(**kwargs) -> Dict[str, Any]:
    job = start_promote_strategy(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def save_learning(**kwargs) -> Dict[str, Any]:
    job = start_save_learning(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def query_learnings_sync(**kwargs) -> Dict[str, Any]:
    job = start_query_learnings(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def get_research_context(**kwargs) -> Dict[str, Any]:
    job = start_get_research_context(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def statistical_significance(**kwargs) -> Dict[str, Any]:
    job = start_statistical_significance(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def time_decay_analysis(**kwargs) -> Dict[str, Any]:
    job = start_time_decay_analysis(**kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job


# =====================================================================
# NEW v6: ML Training Tools (Dual-Track Research)
# =====================================================================

def start_train_catboost(
    pl_column: str,
    filters: Optional[List[Dict]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Train CatBoost model and get feature importance."""
    cfg = _get_storage_config()
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters or [],
        **kwargs
    }
    return submit_job("train_catboost", params)


def start_train_xgboost(
    pl_column: str,
    filters: Optional[List[Dict]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Train XGBoost model and get feature importance."""
    cfg = _get_storage_config()
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters or [],
        **kwargs
    }
    return submit_job("train_xgboost", params)


def start_train_lightgbm(
    pl_column: str,
    filters: Optional[List[Dict]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Train LightGBM model and get feature importance."""
    cfg = _get_storage_config()
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters or [],
        **kwargs
    }
    return submit_job("train_lightgbm", params)


def start_shap_explain(
    pl_column: str,
    filters: Optional[List[Dict]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate SHAP explanations and filter suggestions."""
    cfg = _get_storage_config()
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters or [],
        **kwargs
    }
    return submit_job("shap_explain", params)


# Convenience wrappers
def train_catboost(pl_column: str, **kwargs) -> Dict[str, Any]:
    job = start_train_catboost(pl_column=pl_column, **kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def train_xgboost(pl_column: str, **kwargs) -> Dict[str, Any]:
    job = start_train_xgboost(pl_column=pl_column, **kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def train_lightgbm(pl_column: str, **kwargs) -> Dict[str, Any]:
    job = start_train_lightgbm(pl_column=pl_column, **kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job

def shap_explain(pl_column: str, **kwargs) -> Dict[str, Any]:
    job = start_shap_explain(pl_column=pl_column, **kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job


def start_train_logistic(
    pl_column: str,
    filters: Optional[List[Dict]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Train Logistic Regression model - interpretable with coefficients."""
    cfg = _get_storage_config()
    params = {
        "storage_bucket": cfg["storage_bucket"],
        "storage_path": cfg["storage_path"],
        "_results_bucket": cfg["results_bucket"],
        "pl_column": pl_column,
        "filters": filters or [],
        **kwargs
    }
    return submit_job("train_logistic", params)


def train_logistic(pl_column: str, **kwargs) -> Dict[str, Any]:
    """Train Logistic Regression and wait for result."""
    job = start_train_logistic(pl_column=pl_column, **kwargs)
    return wait_for_job(job["job_id"]) if job.get("job_id") else job
