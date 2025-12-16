from __future__ import annotations

import os
import json
import uuid
import re
from datetime import datetime
from typing import Any, Dict, List, Set

import streamlit as st
from openai import OpenAI
from openai import BadRequestError

from modules import football_tools as functions


# ============================================================
# OpenAI client + model
# ============================================================
def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


client = _init_client()

PREFERRED = (os.getenv("PREFERRED_OPENAI_MODEL") or st.secrets.get("PREFERRED_OPENAI_MODEL") or "").strip()
MODEL = PREFERRED or "gpt-5.1"

DEFAULT_STORAGE_BUCKET = os.getenv("DATA_STORAGE_BUCKET") or st.secrets.get("DATA_STORAGE_BUCKET", "football-data")
DEFAULT_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH") or st.secrets.get("DATA_STORAGE_PATH", "football_ai_NNIA.csv")
DEFAULT_RESULTS_BUCKET = os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET", "football-results")

MAX_MESSAGES_TO_KEEP = int(os.getenv("MAX_CHAT_MESSAGES") or st.secrets.get("MAX_CHAT_MESSAGES", 220))


# ============================================================
# Tool runner
# ============================================================
def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**args)


# ============================================================
# Tools schema
# NOTE: every array must include `items` in JSON schema
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_research_context",
            "description": "Fetch all Google Sheet tabs + derived constraints (ignored_columns, outcome_columns).",
            "parameters": {"type": "object", "properties": {"limit_notes": {"type": "integer"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_research_note",
            "description": "Append to research_memory.",
            "parameters": {"type": "object", "properties": {"note": {"type": "string"}, "tags": {"type": "string"}}, "required": ["note"]},
        },
    },
    {
        "type": "function",
        "function": {"name": "get_research_state", "description": "Get research_state KV.", "parameters": {"type": "object", "properties": {}, "required": []}},
    },
    {
        "type": "function",
        "function": {
            "name": "set_research_state",
            "description": "Set research_state KV.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_data_basic",
            "description": "Load CSV preview.",
            "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List CSV columns.",
            "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_job",
            "description": "Submit Modal worker job.",
            "parameters": {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]},
        },
    },
    {
        "type": "function",
        "function": {"name": "get_job", "description": "Get job status by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_job",
            "description": "Wait for completion; optionally downloads results.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_result",
            "description": "Download a result JSON from storage by path.",
            "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}, "bucket": {"type": "string"}}, "required": ["result_path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_pl_lab",
            "description": "Start best-practice ML lab for any PL column (Sheet rules enforced; categoricals included; distills explicit rules).",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_minutes": {"type": "integer"},
                    "pl_column": {"type": "string"},
                    "do_hyperopt": {"type": "boolean"},
                    "hyperopt_iter": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_btts_lab",
            "description": "Convenience wrapper around start_pl_lab for BTTS PL.",
            "parameters": {"type": "object", "properties": {"duration_minutes": {"type": "integer"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {"name": "list_chats", "description": "List saved chat sessions.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}},
    },
    {
        "type": "function",
        "function": {
            "name": "save_chat",
            "description": "Save chat session.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}, "messages": {"type": "array", "items": {"type": "object"}}, "title": {"type": "string"}},
                "required": ["session_id", "messages"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "load_chat", "description": "Load chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
    },
    {
        "type": "function",
        "function": {
            "name": "rename_chat",
            "description": "Rename chat session.",
            "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "title": {"type": "string"}}, "required": ["session_id", "title"]},
        },
    },
    {
        "type": "function",
        "function": {"name": "delete_chat", "description": "Delete chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
    },
]


# ============================================================
# System prompt (world class explicit)
# ============================================================
SYSTEM_PROMPT = """You are FootballResearcher — an autonomous strategy R&D agent.

NON-NEGOTIABLE SOURCE OF TRUTH
- You MUST use Google Sheet tabs via get_research_context():
  dataset_overview, research_rules, column_definitions, evaluation_framework, research_state, research_memory.
- Treat research_rules as ENFORCEMENT rules, not suggestions.

MISSION / PURPOSE
- Discover strategies that are REPEATABLE on future matches.
- Avoid overfitting and leakage. You are judged on out-of-sample stability and risk:
  - P&L, ROI
  - max drawdown and longest losing streak (in POINTS)
- Output strategies as explicit filters:
  - Numeric ranges + categorical constraints (categoricals are REQUIRED where useful)
  - With minimum sample sizes on train/val/test (and unique IDs where possible)
- You MUST separate:
  - discovery (train/val) from final confirmation (test)
  - do NOT tune thresholds using test results

CAPABILITIES (WHAT YOU CAN DO)
- Read the “rules + definitions + evaluation framework” from the Google Sheet.
- Submit compute jobs to a Modal worker via start_pl_lab (recommended) or submit_job.
- Download and summarise results, distilling explicit strategies that include categoricals.

CONSTRAINTS (IMPORTANT)
- Each dataset row is a SCAN OUTCOME, not a single match.
- PL columns are OUTCOMES ONLY and must never be used as predictive features.
- Result / HOME FORM columns must be ignored per sheet rules.
- Splits MUST be time-based and leakage-safe (the worker uses time-ordered group splits by ID).

OPERATING PROCEDURE (AUTOPILOT)
1) Load Sheet context first (get_research_context).
2) Decide target PL column from the user's request (default: BTTS PL if unspecified).
3) Run experiments with start_pl_lab using 3-way time split (train/val/test).
4) Select strategies using VAL ONLY; then report TEST stability + risk.
5) Log significant findings to research_memory as compact structured JSON.
6) Iterate: simplify, re-test, penalise fragile strategies.

CHAT MODE RULE
- In chat mode, do not call tools unless the user explicitly requests.

AUTOPILOT MODE RULE
- In autopilot, be decisive: pick the next experiment yourself. Only ask the user if blocked.
"""


# ============================================================
# Streamlit UI setup
# ============================================================
st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Football Researcher")


# ============================================================
# Session management
# ============================================================
def _load_sessions() -> List[Dict[str, Any]]:
    out = _run_tool("list_chats", {"limit": 200})
    if not out.get("ok", True):
        return []
    return out.get("sessions") or []


def _new_session_id() -> str:
    return str(uuid.uuid4())


if "session_id" not in st.session_state:
    qp = st.query_params.get("sid")
    st.session_state.session_id = qp if qp else _new_session_id()

SESSION_ID = st.session_state.session_id


def _set_session(sid: str):
    st.session_state.session_id = sid
    st.query_params["sid"] = sid


def _init_messages_if_needed():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def _trim_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(messages) <= MAX_MESSAGES_TO_KEEP:
        return messages
    system = messages[0:1]
    tail = messages[-(MAX_MESSAGES_TO_KEEP - 1):]
    return system + tail


def _persist_chat(title: str = ""):
    st.session_state.messages = _trim_messages(st.session_state.messages)
    out = _run_tool("save_chat", {"session_id": st.session_state.session_id, "messages": st.session_state.messages, "title": title})
    if isinstance(out, dict) and out.get("ok") is False:
        st.session_state.last_chat_save_error = out


def _try_load_chat(sid: str) -> bool:
    loaded = _run_tool("load_chat", {"session_id": sid})
    if loaded.get("ok") and loaded.get("data", {}).get("messages"):
        st.session_state.messages = _trim_messages(loaded["data"]["messages"])
        return True
    return False


def _reset_to_new_chat(title: str = ""):
    """
    Creates a brand new chat session (new sid), resets messages, and persists immediately.
    """
    new_sid = _new_session_id()
    _set_session(new_sid)
    st.session_state.loaded_for_sid = new_sid
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    _persist_chat(title=title or f"Session {new_sid[:8]}")


def _duplicate_current_to_new_chat(title: str = ""):
    """
    Forks the current chat into a new session id (keeps current messages).
    """
    cur_msgs = st.session_state.get("messages") or [{"role": "system", "content": SYSTEM_PROMPT}]
    new_sid = _new_session_id()
    _set_session(new_sid)
    st.session_state.loaded_for_sid = new_sid
    st.session_state.messages = _trim_messages(list(cur_msgs))
    _persist_chat(title=title or f"Fork {new_sid[:8]}")


_init_messages_if_needed()

if "loaded_for_sid" not in st.session_state or st.session_state.loaded_for_sid != SESSION_ID:
    st.session_state.loaded_for_sid = SESSION_ID
    if not _try_load_chat(SESSION_ID):
        _persist_chat(title=f"Session {SESSION_ID[:8]}")

if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "chat"


# ============================================================
# Sanitise history for OpenAI
# ============================================================
def _sanitize_history_for_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    out: List[Dict[str, Any]] = []
    first = messages[0]
    if first.get("role") != "system":
        out.append({"role": "system", "content": SYSTEM_PROMPT})
    else:
        out.append({"role": "system", "content": first.get("content", SYSTEM_PROMPT)})

    expecting_tool_ids: Set[str] = set()

    for m in messages[1:]:
        role = (m.get("role") or "").strip()

        if role == "assistant":
            expecting_tool_ids = set()
            clean_assistant: Dict[str, Any] = {"role": "assistant", "content": m.get("content", "") or ""}
            tc = m.get("tool_calls")
            if isinstance(tc, list) and tc:
                cleaned_tool_calls = []
                for call in tc:
                    cid = call.get("id")
                    fn = call.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments", "{}")
                    if cid and name:
                        cleaned_tool_calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
                        expecting_tool_ids.add(cid)
                if cleaned_tool_calls:
                    clean_assistant["tool_calls"] = cleaned_tool_calls
            out.append(clean_assistant)
            continue

        if role == "tool":
            tcid = m.get("tool_call_id")
            if tcid and expecting_tool_ids and tcid in expecting_tool_ids:
                out.append({"role": "tool", "tool_call_id": tcid, "content": m.get("content", "") or ""})
            continue

        if role == "user":
            out.append({"role": "user", "content": m.get("content", "") or ""})
            continue

    return out


# ============================================================
# LLM call
# ============================================================
def _call_llm(messages: List[Dict[str, Any]]):
    mode = st.session_state.agent_mode
    safe_messages = _sanitize_history_for_llm(messages)

    if mode == "chat":
        chat_only: List[Dict[str, Any]] = []
        for m in safe_messages:
            if m.get("role") == "tool":
                continue
            if m.get("role") == "assistant" and "tool_calls" in m:
                mm = dict(m)
                mm.pop("tool_calls", None)
                chat_only.append(mm)
            else:
                chat_only.append(m)
        return client.chat.completions.create(model=MODEL, messages=chat_only)

    return client.chat.completions.create(model=MODEL, messages=safe_messages, tools=TOOLS, tool_choice="auto")


# ============================================================
# Parsing helpers
# ============================================================
def _minutes_from_text(t: str, default_minutes: int = 300) -> int:
    t = (t or "").lower().strip()
    minutes = default_minutes
    m = re.search(r"next\s+(\d+)\s*(hour|hours|hr|hrs|h)\b", t)
    if m:
        minutes = int(m.group(1)) * 60
    m2 = re.search(r"(\d+)\s*(minute|minutes|min)\b", t)
    if m2:
        minutes = int(m2.group(1))
    return max(5, min(minutes, 360))


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\(\)\[\]\{\}\,\;\:]+", " ", s)
    s = s.replace("  ", " ")
    return re.sub(r"\s+", " ", s).strip()


def _resolve_pl_column(user_text: str, ctx: Dict[str, Any]) -> str:
    t = _normalize(user_text)
    derived = (ctx.get("derived") or {})
    outcomes: List[str] = derived.get("outcome_columns") or []

    aliases = {
        "btts": "BTTS PL",
        "both teams to score": "BTTS PL",
        "over 2.5": "BO 2.5 PL",
        "o 2.5": "BO 2.5 PL",
        "o2.5": "BO 2.5 PL",
        "bo 2.5": "BO 2.5 PL",
        "bo2.5": "BO 2.5 PL",
        "fh over 1.5": "BO1.5 FHG PL",
        "fh o1.5": "BO1.5 FHG PL",
        "shg": "SHG PL",
    }
    for k, v in aliases.items():
        if k in t:
            return v

    norm_map = {_normalize(c): c for c in outcomes}
    for c in outcomes:
        if _normalize(c) in t:
            return c

    t_compact = t.replace(" ", "")
    for c in outcomes:
        if _normalize(c).replace(" ", "") in t_compact:
            return c

    m = re.search(r"\bfor\s+(.+?\bpl)\b", t)
    if m:
        guess = m.group(1).strip()
        if _normalize(guess) in norm_map:
            return norm_map[_normalize(guess)]
        return guess

    return "BTTS PL"


# ============================================================
# Rendering distilled rules
# ============================================================
def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def _render_rule_spec(spec: Dict[str, Any]) -> str:
    parts: List[str] = []

    for c in (spec.get("categorical") or []):
        col = c.get("col")
        if not col:
            continue
        if "in" in c and c["in"]:
            parts.append(f"**{col} IN** {c['in']}")
        if "not_in" in c and c["not_in"]:
            parts.append(f"**{col} NOT IN** {c['not_in']}")

    for n in (spec.get("numeric") or []):
        col = n.get("col")
        if not col:
            continue
        mn = n.get("min", None)
        mx = n.get("max", None)
        if mn is not None and mx is not None:
            parts.append(f"**{col}** in [{_fmt_num(mn)}, {_fmt_num(mx)}]")
        elif mn is not None:
            parts.append(f"**{col}** >= {_fmt_num(mn)}")
        elif mx is not None:
            parts.append(f"**{col}** <= {_fmt_num(mx)}")

    return "  \n".join(parts) if parts else "(no constraints)"


def _render_distilled_top(result_obj: Dict[str, Any], top_n: int = 3) -> str:
    payload = (result_obj or {}).get("result") or {}
    distilled = payload.get("distilled") or {}
    picked = payload.get("picked") or {}

    if not distilled or "top_distilled_rules" not in distilled:
        return "No distilled strategies found in this result."

    rules = (distilled.get("top_distilled_rules") or [])[: max(1, int(top_n))]

    out: List[str] = []
    out.append(f"### ✅ Distilled strategies (top {len(rules)})")
    out.append(f"- **Picked:** `{picked}`")
    base = distilled.get("best_base_model") or {}
    out.append(f"- **Base model:** `{base}`")
    out.append("")

    for i, rr in enumerate(rules, start=1):
        spec = rr.get("spec") or {}
        tr = rr.get("train") or {}
        va = rr.get("val") or {}
        te = rr.get("test") or {}
        gl = rr.get("test_game_level") or {}
        gap = rr.get("gap_train_minus_val")

        out.append(f"#### Strategy #{i}")
        out.append(_render_rule_spec(spec))
        out.append("")
        out.append(f"- **Train:** rows={tr.get('rows')} roi={_fmt_num(tr.get('roi'))} total_pl={_fmt_num(tr.get('total_pl'))}")
        out.append(f"- **Val:** rows={va.get('rows')} roi={_fmt_num(va.get('roi'))} total_pl={_fmt_num(va.get('total_pl'))}")
        out.append(f"- **Test:** rows={te.get('rows')} roi={_fmt_num(te.get('roi'))} total_pl={_fmt_num(te.get('total_pl'))}")
        out.append(f"- **Stability:** gap(train−val)={_fmt_num(gap)}")
        out.append(f"- **Test risk (by ID):** unique_ids={gl.get('unique_ids')} max_dd={_fmt_num(gl.get('max_dd'))} losing_streak={gl.get('losing_streak')}")
        out.append("")

    return "\n".join(out)


# ============================================================
# Sheet logging (compact but useful)
# ============================================================
def _context_snapshot_text(ctx: Dict[str, Any]) -> str:
    ov = (ctx.get("dataset_overview") or {})
    derived = (ctx.get("derived") or {})
    ignored = derived.get("ignored_columns") or []
    outcomes = derived.get("outcome_columns") or []
    primary_goal = ov.get("primary_goal", "")
    fmt = ov.get("strategy_output_format", "")
    return (
        f"Context loaded.\n"
        f"- primary_goal: {primary_goal}\n"
        f"- strategy_output_format: {fmt}\n"
        f"- ignored_columns: {ignored}\n"
        f"- outcome_columns: {outcomes}\n"
    )


def _log_lab_to_sheet(job_id: str, result_obj: Dict[str, Any], tags: str):
    try:
        payload = (result_obj or {}).get("result") or {}
        if not payload:
            return

        picked = payload.get("picked") or {}
        pl_col = str((picked.get("pl_column") or "")).strip()

        distilled_rules = ((payload.get("distilled") or {}).get("top_distilled_rules") or [])[:3]

        note = {
            "ts": datetime.utcnow().isoformat(),
            "kind": "pl_lab_result",
            "job_id": job_id,
            "picked": picked,
            "sheet_enforcement": payload.get("sheet_enforcement"),
            "splits": payload.get("splits"),
            "features": payload.get("features"),
            "baseline": payload.get("baseline"),
            "top_distilled_rules": [
                {
                    "spec": rr.get("spec"),
                    "train": rr.get("train"),
                    "val": rr.get("val"),
                    "test": rr.get("test"),
                    "test_game_level": rr.get("test_game_level"),
                    "gap_train_minus_val": rr.get("gap_train_minus_val"),
                    "samples": rr.get("samples"),
                }
                for rr in distilled_rules
            ],
        }

        tag_bits = [t.strip() for t in (tags or "").split(",") if t.strip()]
        if pl_col:
            tag_bits.append(pl_col.replace(" ", "_"))
        final_tags = ",".join(tag_bits)

        _run_tool("append_research_note", {"note": json.dumps(note, ensure_ascii=False), "tags": final_tags})
    except Exception:
        return


# ============================================================
# Autopilot command intercepts
# ============================================================
def _maybe_start_pl_lab(user_text: str) -> bool:
    if st.session_state.agent_mode != "autopilot":
        return False

    t = _normalize(user_text)
    wants_strategy = ("strategy" in t) or ("build" in t and ("pl" in t or "btts" in t or "over" in t or "shg" in t))
    if not wants_strategy:
        return False

    ctx = _run_tool("get_research_context", {"limit_notes": 10})
    st.session_state.last_context = ctx

    minutes = _minutes_from_text(user_text, default_minutes=180)
    do_hyperopt = any(k in t for k in ["hyperopt", "grid", "cv", "bayes", "optuna"])

    pl_col = _resolve_pl_column(user_text, ctx)

    submitted = _run_tool(
        "start_pl_lab",
        {"duration_minutes": minutes, "pl_column": pl_col, "do_hyperopt": do_hyperopt, "hyperopt_iter": 12},
    )
    job_id = submitted.get("job_id")

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": _context_snapshot_text(ctx)})

    if job_id:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"✅ Started **PL Lab** for **{pl_col}** for **{minutes} minutes**.\n\n"
                    f"**Job ID:** `{job_id}`\n\n"
                    "Use:\n"
                    f"- `Check job {job_id}`\n"
                    f"- `Show results for {job_id}`"
                ),
            }
        )
        _run_tool("set_research_state", {"key": "last_pl_lab_job_id", "value": job_id})
        _run_tool("set_research_state", {"key": "last_pl_lab_started_at", "value": datetime.utcnow().isoformat()})
        _run_tool("set_research_state", {"key": "last_pl_lab_pl_column", "value": pl_col})
    else:
        st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to start PL lab: {submitted}"})

    _persist_chat()
    return True


def _maybe_handle_job_queries(user_text: str) -> bool:
    t = (user_text or "").strip().lower()

    m = re.search(r"\bcheck job\b\s+([0-9a-f\-]{10,})", t)
    if m:
        job_id = m.group(1)
        job = _run_tool("get_job", {"job_id": job_id})
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.messages.append({"role": "assistant", "content": f"```json\n{json.dumps(job, indent=2)}\n```"})
        _persist_chat()
        return True

    m2 = re.search(r"\bshow results\b.*\b([0-9a-f\-]{10,})", t)
    if m2:
        job_id = m2.group(1)

        waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 1, "poll_s": 1, "auto_download": False})
        job = waited.get("job") or {}
        rp = job.get("result_path")
        params = job.get("params") or {}
        bucket = (params.get("_results_bucket") or DEFAULT_RESULTS_BUCKET).strip()

        st.session_state.messages.append({"role": "user", "content": user_text})

        if not rp:
            st.session_state.messages.append({"role": "assistant", "content": f"No result_path yet.\n```json\n{json.dumps(job, indent=2)}\n```"})
            _persist_chat()
            return True

        res = _run_tool("download_result", {"bucket": bucket, "result_path": rp})
        result_obj = res.get("result") or {}

        md = _render_distilled_top(result_obj, top_n=3)
        st.session_state.messages.append({"role": "assistant", "content": md})

        st.session_state.messages.append(
            {"role": "assistant", "content": f"```json\n{json.dumps(result_obj, indent=2)[:12000]}\n```"}
        )

        try:
            task_type = (result_obj.get("task_type") or "")
            if task_type in ("pl_lab", "btts_lab"):
                _log_lab_to_sheet(job_id, result_obj, tags="pl_lab,ml,distilled,autopilot")
        except Exception:
            pass

        _persist_chat()
        return True

    return False


# ============================================================
# Main chat loop
# ============================================================
def _chat_with_tools(user_text: str, max_rounds: int = 6):
    if st.session_state.agent_mode == "autopilot":
        if _maybe_start_pl_lab(user_text):
            return
        if _maybe_handle_job_queries(user_text):
            return

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages = _trim_messages(st.session_state.messages)

    for _ in range(max_rounds):
        try:
            resp = _call_llm(st.session_state.messages)
        except BadRequestError as e:
            st.error("OpenAI BadRequestError (full details below).")
            try:
                st.json(e.response.json())
            except Exception:
                st.exception(e)
            raise

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ]

        st.session_state.messages.append(assistant_msg)

        if st.session_state.agent_mode == "chat":
            break
        if not tool_calls:
            break

        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            out = _run_tool(name, args)
            st.session_state.messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)})

        st.session_state.messages = _trim_messages(st.session_state.messages)

    _persist_chat()


# ============================================================
# Sidebar UI
# ============================================================
with st.sidebar:
    st.caption(f"Requested model: `{MODEL}`")
    st.caption(f"Data: `{DEFAULT_STORAGE_BUCKET}/{DEFAULT_STORAGE_PATH}`")
    st.caption(f"Results: `{DEFAULT_RESULTS_BUCKET}`")

    st.radio("Agent mode", ["chat", "autopilot"], key="agent_mode")
    st.divider()

    # --- NEW CHAT controls (restored) ---
    st.subheader("🧠 Session Controls")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🆕 New chat", use_container_width=True):
            _reset_to_new_chat()
            st.rerun()
    with colB:
        if st.button("🧬 Fork chat", use_container_width=True, help="Duplicate current chat into a new session id"):
            _duplicate_current_to_new_chat()
            st.rerun()

    with st.expander("Rename / Delete", expanded=False):
        new_title = st.text_input("Rename this chat", value=f"Session {st.session_state.session_id[:8]}")
        if st.button("Save title"):
            _run_tool("rename_chat", {"session_id": st.session_state.session_id, "title": new_title})
            _persist_chat(title=new_title)
            st.success("Renamed.")
        if st.button("Delete this chat", type="secondary"):
            _run_tool("delete_chat", {"session_id": st.session_state.session_id})
            _reset_to_new_chat()
            st.rerun()

    st.divider()

    if st.session_state.get("last_chat_save_error"):
        st.warning(f"Chat persistence warning: {st.session_state.last_chat_save_error}")
        if st.button("Clear chat save warning"):
            st.session_state.last_chat_save_error = None

    st.subheader("💾 Chat Sessions")
    sessions = _load_sessions()
    sessions_display = list(reversed(sessions))

    options = [{"session_id": st.session_state.session_id, "title": f"(current) {st.session_state.session_id[:8]}"}] + [
        {"session_id": s.get("session_id"), "title": s.get("title") or s.get("session_id")[:8]}
        for s in sessions_display
        if s.get("session_id") and s.get("session_id") != st.session_state.session_id
    ]

    labels = [f"{o['title']} — {o['session_id'][:8]}" for o in options]
    chosen = st.selectbox("Select session", options=list(range(len(options))), format_func=lambda i: labels[i], index=0)

    if options[chosen]["session_id"] != st.session_state.session_id:
        _set_session(options[chosen]["session_id"])
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        _try_load_chat(st.session_state.session_id)
        st.rerun()


# ============================================================
# Main
# ============================================================
st.subheader("💬 Chat")
for m in st.session_state.messages:
    if m.get("role") == "system":
        continue
    with st.chat_message(m.get("role", "assistant")):
        st.markdown(m.get("content", ""))

user_msg = st.chat_input("Ask the researcher…")
if user_msg:
    _chat_with_tools(user_msg)
    st.rerun()
