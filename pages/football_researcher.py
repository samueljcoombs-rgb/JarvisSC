from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Set, Optional

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


# ============================================================
# Defaults
# ============================================================

DEFAULT_STORAGE_BUCKET = os.getenv("DATA_STORAGE_BUCKET") or st.secrets.get("DATA_STORAGE_BUCKET", "football-data")
DEFAULT_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH") or st.secrets.get("DATA_STORAGE_PATH", "football_ai_NNIA.csv")
DEFAULT_RESULTS_BUCKET = os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET", "football-results")

MAX_MESSAGES_TO_KEEP = int(os.getenv("MAX_CHAT_MESSAGES") or st.secrets.get("MAX_CHAT_MESSAGES", 220))


# ============================================================
# Tool runner (APP INTERNAL)
# ============================================================

def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**args)


# ============================================================
# Tools schema for OpenAI function calling
# IMPORTANT: only include tools the LLM should call.
# Do NOT include chat persistence tools here.
# ============================================================

LLM_TOOLS = [
    {"type": "function", "function": {"name": "get_dataset_overview", "description": "Get dataset_overview tab.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_research_rules", "description": "Get research_rules tab.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_column_definitions", "description": "Get column_definitions tab.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_evaluation_framework", "description": "Get evaluation_framework tab.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_recent_research_notes", "description": "Get recent research_memory rows.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "append_research_note", "description": "Append to research_memory.", "parameters": {"type": "object", "properties": {"note": {"type": "string"}, "tags": {"type": "string"}}, "required": ["note"]}}},
    {"type": "function", "function": {"name": "get_research_state", "description": "Get research_state KV.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "set_research_state", "description": "Set research_state KV.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},

    # Dataset (Supabase Storage only)
    {"type": "function", "function": {"name": "load_data_basic", "description": "Load CSV preview from Supabase Storage.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": ["storage_bucket", "storage_path"]}}},
    {"type": "function", "function": {"name": "list_columns", "description": "List CSV columns from Supabase Storage.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": ["storage_bucket", "storage_path"]}}},
    {"type": "function", "function": {"name": "basic_roi_for_pl_column", "description": "Row-level ROI for PL column (outcome only).", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}, "storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": ["pl_column", "storage_bucket", "storage_path"]}}},

    # Jobs
    {"type": "function", "function": {"name": "submit_job", "description": "Submit Modal worker job.", "parameters": {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]}}},
    {"type": "function", "function": {"name": "get_job", "description": "Get job status by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for completion; optionally downloads results.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "download_result", "description": "Download a result JSON from storage by path.", "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}, "bucket": {"type": "string"}}, "required": ["result_path"]}}},
]


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """You are FootballResearcher — an autonomous research agent that discovers profitable, robust football trading strategy criteria.

Source of truth:
- Use Google Sheet tabs: dataset_overview, research_rules, column_definitions, evaluation_framework, research_state, research_memory.

Hard constraints:
- PL columns are outcomes only and MUST NOT be used as predictive features.
- Avoid overfitting: time-based splits; never tune thresholds on final test.
- Always report sample sizes and stability (train vs test gap) and drawdown/losing streak in POINTS.
- Prefer simple rules that generalise; penalise fragile, tiny samples.

Objective:
- When user asks to “design a strategy”, you should plan, run experiments (using worker jobs when needed), log results into research_memory, and present progress updates.
- Be decisive: pick what to test next. Only ask user if blocked by missing config.

Important:
- In CHAT mode: reply conversationally (no tools).
- In AUTOPILOT mode: tools enabled, but still explain briefly.
- Dataset must be loaded from Supabase Storage ONLY (bucket/path). Never use URLs.
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
    if not isinstance(out, dict) or not out.get("ok", True):
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

    # Avoid starting trimmed history on a tool message (can orphan tool call context)
    while tail and tail[0].get("role") == "tool":
        # pull one more message from earlier if possible
        idx_start = len(messages) - len(tail) - 1
        if idx_start <= 0:
            break
        tail = [messages[idx_start]] + tail

    return system + tail


def _persist_chat(title: str = ""):
    st.session_state.messages = _trim_messages(st.session_state.messages)
    out = _run_tool("save_chat", {"session_id": SESSION_ID, "messages": st.session_state.messages, "title": title})
    if isinstance(out, dict) and out.get("ok") is False:
        st.session_state.last_chat_save_error = out


def _try_load_chat(sid: str) -> bool:
    loaded = _run_tool("load_chat", {"session_id": sid})
    if isinstance(loaded, dict) and loaded.get("ok") and loaded.get("data", {}).get("messages"):
        st.session_state.messages = _trim_messages(loaded["data"]["messages"])
        return True
    return False


_init_messages_if_needed()

if "loaded_for_sid" not in st.session_state or st.session_state.loaded_for_sid != SESSION_ID:
    st.session_state.loaded_for_sid = SESSION_ID
    if not _try_load_chat(SESSION_ID):
        _persist_chat(title=f"Session {SESSION_ID[:8]}")


# ============================================================
# Agent mode
# ============================================================

if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "chat"  # chat | autopilot


# ============================================================
# History repair + sanitization
# This FIXES:
# "assistant tool_calls must be followed by tool messages..."
# ============================================================

def _repair_tool_call_chains(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures no dangling assistant.tool_calls exist without matching tool responses.
    If a tool_call chain is incomplete, we strip tool_calls from that assistant
    and remove any orphan tool messages for that chain.
    """
    if not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    repaired: List[Dict[str, Any]] = []
    repaired.append(messages[0] if messages[0].get("role") == "system" else {"role": "system", "content": SYSTEM_PROMPT})

    pending_ids: Set[str] = set()
    pending_assistant_index: Optional[int] = None
    pending_tool_msg_indices: List[int] = []

    def _drop_pending_chain():
        nonlocal pending_ids, pending_assistant_index, pending_tool_msg_indices
        if pending_assistant_index is not None and 0 <= pending_assistant_index < len(repaired):
            repaired[pending_assistant_index].pop("tool_calls", None)
        # remove any tool messages we attached for that pending chain (orphan now)
        for idx in sorted(pending_tool_msg_indices, reverse=True):
            if 0 <= idx < len(repaired) and repaired[idx].get("role") == "tool":
                repaired.pop(idx)
        pending_ids = set()
        pending_assistant_index = None
        pending_tool_msg_indices = []

    for m in messages[1:]:
        role = (m.get("role") or "").strip()

        if role == "assistant":
            # If there was a pending tool chain and we're seeing a new assistant,
            # it means it wasn't completed -> drop it.
            if pending_ids:
                _drop_pending_chain()

            clean = {"role": "assistant", "content": m.get("content", "") or ""}
            tc = m.get("tool_calls")
            if isinstance(tc, list) and tc:
                cleaned_tool_calls = []
                ids = set()
                for call in tc:
                    try:
                        cid = call.get("id")
                        fn = call.get("function") or {}
                        name = fn.get("name")
                        args = fn.get("arguments", "{}")
                        if cid and name:
                            cleaned_tool_calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
                            ids.add(cid)
                    except Exception:
                        continue

                if cleaned_tool_calls:
                    clean["tool_calls"] = cleaned_tool_calls
                    pending_ids = set(ids)
                    pending_assistant_index = len(repaired)
                    pending_tool_msg_indices = []

            repaired.append(clean)
            continue

        if role == "tool":
            tcid = (m.get("tool_call_id") or "").strip()
            if pending_ids and tcid and tcid in pending_ids:
                repaired.append({"role": "tool", "tool_call_id": tcid, "content": m.get("content", "") or ""})
                pending_tool_msg_indices.append(len(repaired) - 1)
                pending_ids.remove(tcid)
                if not pending_ids:
                    pending_assistant_index = None
                    pending_tool_msg_indices = []
            # Drop unexpected/orphan tool messages
            continue

        if role == "user":
            # If user arrives before tool chain completes -> drop pending chain
            if pending_ids:
                _drop_pending_chain()
            repaired.append({"role": "user", "content": m.get("content", "") or ""})
            continue

        # drop other roles
        continue

    # End of history: if tool chain still pending -> drop it
    if pending_ids:
        _drop_pending_chain()

    return repaired


def _sanitize_history_for_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Repair first so OpenAI never sees invalid sequences.
    repaired = _repair_tool_call_chains(messages)

    # Guarantee leading system message
    out: List[Dict[str, Any]] = []
    first = repaired[0] if repaired else {"role": "system", "content": SYSTEM_PROMPT}
    if first.get("role") != "system":
        out.append({"role": "system", "content": SYSTEM_PROMPT})
    else:
        out.append({"role": "system", "content": first.get("content", SYSTEM_PROMPT)})

    # Copy through already-repaired content
    for m in repaired[1:]:
        role = (m.get("role") or "").strip()
        if role in ("user", "assistant", "tool"):
            out.append(m)
    return out


# ============================================================
# LLM call (never send tool_choice unless tools present)
# ============================================================

def _call_llm(messages: List[Dict[str, Any]]):
    # Repair the stored history in-session too, so the error doesn't come back.
    st.session_state.messages = _repair_tool_call_chains(st.session_state.messages)

    mode = st.session_state.agent_mode
    safe_messages = _sanitize_history_for_llm(st.session_state.messages)

    if mode == "chat":
        # CHAT MODE: no tools, no tool_choice
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

    # AUTOPILOT MODE: tools enabled
    return client.chat.completions.create(model=MODEL, messages=safe_messages, tools=LLM_TOOLS, tool_choice="auto")


# ============================================================
# Chat loop
# ============================================================

def _chat_with_tools(user_text: str, max_rounds: int = 6):
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
# Autopilot receipt + structured logging (manual non-blocking)
# ============================================================

def _structured_note(worker_payload: Dict[str, Any], job_id: str, result_path: str) -> str:
    result = worker_payload.get("result", {}) if isinstance(worker_payload, dict) else {}
    picked = result.get("picked") or {}
    search = result.get("search") or {}
    top_rules = (search.get("top_rules") or [])[:3]
    note = {
        "ts": datetime.utcnow().isoformat(),
        "kind": "strategy_search_result",
        "job_id": job_id,
        "result_path": result_path,
        "picked": picked,
        "top_rules": top_rules,
    }
    return json.dumps(note, ensure_ascii=False)


def _submit_autopilot_job() -> Dict[str, Any]:
    params = {
        "storage_bucket": DEFAULT_STORAGE_BUCKET,
        "storage_path": DEFAULT_STORAGE_PATH,
        "_results_bucket": DEFAULT_RESULTS_BUCKET,
        "time_split_ratio": 0.7,
    }
    submitted = _run_tool("submit_job", {"task_type": "strategy_search", "params": params})
    job_id = submitted.get("job_id")

    if job_id:
        _run_tool("set_research_state", {"key": "last_job_id", "value": job_id})
        _run_tool("set_research_state", {"key": "last_job_submitted_at", "value": datetime.utcnow().isoformat()})
    return {"submitted": submitted, "job_id": job_id}


def _poll_latest_job_and_log_if_done() -> Dict[str, Any]:
    rs = _run_tool("get_research_state", {}).get("data", {}) or {}
    job_id = (rs.get("last_job_id") or "").strip()
    if not job_id:
        return {"ok": True, "note": "No last_job_id in research_state."}

    job = _run_tool("get_job", {"job_id": job_id})
    if job.get("error"):
        return {"ok": False, "error": job.get("error"), "job": job}

    status = (job.get("status") or "").lower()
    out: Dict[str, Any] = {"ok": True, "job_id": job_id, "status": status, "job": job}

    if status == "done" and job.get("result_path"):
        last_logged = (rs.get("last_logged_job_id") or "").strip()
        if last_logged == job_id:
            out["logged"] = False
            out["note"] = "Result already logged."
            return out

        res = _run_tool("download_result", {"result_path": job["result_path"], "bucket": DEFAULT_RESULTS_BUCKET})
        result_json = res.get("result") if isinstance(res, dict) else None

        note = _structured_note(result_json or {}, job_id, job["result_path"])
        _run_tool("append_research_note", {"note": note, "tags": "autopilot,worker,structured"})
        _run_tool("set_research_state", {"key": "last_logged_job_id", "value": job_id})
        _run_tool("set_research_state", {"key": "last_autopilot_logged_at", "value": datetime.utcnow().isoformat()})
        out["logged"] = True

    return out


# ============================================================
# Sidebar UI
# ============================================================

with st.sidebar:
    st.caption(f"Model: `{MODEL}`")
    st.caption(f"Data: `{DEFAULT_STORAGE_BUCKET}/{DEFAULT_STORAGE_PATH}`")
    st.caption(f"Results: `{DEFAULT_RESULTS_BUCKET}`")

    st.radio("Agent mode", ["chat", "autopilot"], key="agent_mode")
    st.divider()

    if st.session_state.get("last_chat_save_error"):
        st.warning(f"Chat persistence warning: {st.session_state.last_chat_save_error}")
        if st.button("Clear chat save warning"):
            st.session_state.last_chat_save_error = None

    st.subheader("💾 Chat Sessions")
    sessions = _load_sessions()
    sessions_display = list(reversed(sessions))

    options = [{"session_id": SESSION_ID, "title": f"(current) {SESSION_ID[:8]}"}] + [
        {"session_id": s.get("session_id"), "title": s.get("title") or s.get("session_id")[:8]}
        for s in sessions_display
        if s.get("session_id") and s.get("session_id") != SESSION_ID
    ]

    labels = [f"{o['title']} — {o['session_id'][:8]}" for o in options]
    chosen = st.selectbox("Select session", options=list(range(len(options))), format_func=lambda i: labels[i], index=0)

    if options[chosen]["session_id"] != SESSION_ID:
        _set_session(options[chosen]["session_id"])
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        _try_load_chat(st.session_state.session_id)
        st.rerun()

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ New"):
            sid = _new_session_id()
            _set_session(sid)
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            _persist_chat(title=f"Session {sid[:8]}")
            st.rerun()

    with colB:
        if st.button("💾 Save"):
            _persist_chat()
            st.success("Saved.")

    new_title = st.text_input("Rename current session", value="")
    if st.button("Rename"):
        if new_title.strip():
            _run_tool("rename_chat", {"session_id": SESSION_ID, "title": new_title.strip()})
            _persist_chat(title=new_title.strip())
            st.success("Renamed.")
            st.rerun()

    if st.button("🗑️ Delete current session"):
        _run_tool("delete_chat", {"session_id": SESSION_ID})
        sid = _new_session_id()
        _set_session(sid)
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        _persist_chat(title=f"Session {sid[:8]}")
        st.success("Deleted + created new.")
        st.rerun()

    st.divider()

    st.subheader("🤖 Autopilot (Manual)")
    if st.button("Submit 1 strategy_search job"):
        submitted = _submit_autopilot_job()
        st.session_state.last_autopilot = {"stage": "submitted", **submitted}
        st.rerun()

    if st.button("Poll latest job + log if done"):
        receipt = _poll_latest_job_and_log_if_done()
        st.session_state.last_autopilot = {"stage": "polled", "receipt": receipt}
        st.rerun()


# ============================================================
# Main: Autopilot receipt + Chat
# ============================================================

st.subheader("🧾 Autopilot Receipt")
last = st.session_state.get("last_autopilot")
if not last:
    st.info("No autopilot run in this session yet.")
else:
    st.json(last)

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
