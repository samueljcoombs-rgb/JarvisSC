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
# Defaults (Supabase Storage dataset location)
# ============================================================

DEFAULT_STORAGE_BUCKET = os.getenv("DATA_STORAGE_BUCKET") or st.secrets.get("DATA_STORAGE_BUCKET", "football-data")
DEFAULT_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH") or st.secrets.get("DATA_STORAGE_PATH", "football_ai_NNIA.csv")
DEFAULT_RESULTS_BUCKET = os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET", "football-results")

MAX_MESSAGES_TO_KEEP = int(os.getenv("MAX_CHAT_MESSAGES") or st.secrets.get("MAX_CHAT_MESSAGES", 220))


DATA_TOOLS = {"load_data_basic", "list_columns", "basic_roi_for_pl_column"}


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
    {"type": "function", "function": {"name": "load_data_basic", "description": "Load CSV preview from Supabase Storage.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "list_columns", "description": "List CSV columns from Supabase Storage.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "basic_roi_for_pl_column", "description": "Row-level ROI for PL column (outcome only).", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}, "storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": ["pl_column"]}}},

    # Jobs
    {"type": "function", "function": {"name": "submit_job", "description": "Submit Modal worker job.", "parameters": {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]}}},
    {"type": "function", "function": {"name": "get_job", "description": "Get job status by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for completion; optionally downloads results.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "download_result", "description": "Download a result JSON from storage by path.", "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}, "bucket": {"type": "string"}}, "required": ["result_path"]}}},
]


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = f"""You are FootballResearcher — an autonomous research agent that discovers profitable, robust football trading strategy criteria.

Source of truth:
- Use Google Sheet tabs: dataset_overview, research_rules, column_definitions, evaluation_framework, research_state, research_memory.

Hard constraints:
- PL columns are outcomes only and MUST NOT be used as predictive features.
- Avoid overfitting: time-based splits; never tune thresholds on final test.
- Always report sample sizes and stability (train vs test gap) and drawdown/losing streak in POINTS.
- Prefer simple rules that generalise; penalise fragile, tiny samples.

Data access:
- Dataset MUST be loaded from Supabase Storage ONLY.
- Default dataset is: bucket="{DEFAULT_STORAGE_BUCKET}", path="{DEFAULT_STORAGE_PATH}".
- Do NOT invent bucket/path like "dataset/master.csv". If user doesn’t specify bucket/path, always use the default dataset.

Mode rules:
- In CHAT mode: conversational; do not call tools unless explicitly asked.
- In AUTOPILOT mode: tools enabled; still explain briefly.
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
    while tail and tail[0].get("role") == "tool":
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
# History repair (prevents dangling tool_calls)
# ============================================================

def _repair_tool_call_chains(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    repaired: List[Dict[str, Any]] = []
    repaired.append(messages[0] if messages[0].get("role") == "system" else {"role": "system", "content": SYSTEM_PROMPT})

    pending_ids: Set[str] = set()
    pending_assistant_index: Optional[int] = None
    pending_tool_indices: List[int] = []

    def _drop_pending_chain():
        nonlocal pending_ids, pending_assistant_index, pending_tool_indices
        if pending_assistant_index is not None and 0 <= pending_assistant_index < len(repaired):
            repaired[pending_assistant_index].pop("tool_calls", None)
        for idx in sorted(pending_tool_indices, reverse=True):
            if 0 <= idx < len(repaired) and repaired[idx].get("role") == "tool":
                repaired.pop(idx)
        pending_ids = set()
        pending_assistant_index = None
        pending_tool_indices = []

    for m in messages[1:]:
        role = (m.get("role") or "").strip()

        if role == "assistant":
            if pending_ids:
                _drop_pending_chain()

            clean = {"role": "assistant", "content": m.get("content", "") or ""}
            tc = m.get("tool_calls")
            if isinstance(tc, list) and tc:
                cleaned = []
                ids: Set[str] = set()
                for call in tc:
                    try:
                        cid = call.get("id")
                        fn = call.get("function") or {}
                        name = fn.get("name")
                        args = fn.get("arguments", "{}")
                        if cid and name:
                            cleaned.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
                            ids.add(cid)
                    except Exception:
                        continue
                if cleaned:
                    clean["tool_calls"] = cleaned
                    pending_ids = set(ids)
                    pending_assistant_index = len(repaired)
                    pending_tool_indices = []
            repaired.append(clean)
            continue

        if role == "tool":
            tcid = (m.get("tool_call_id") or "").strip()
            if pending_ids and tcid in pending_ids:
                repaired.append({"role": "tool", "tool_call_id": tcid, "content": m.get("content", "") or ""})
                pending_tool_indices.append(len(repaired) - 1)
                pending_ids.remove(tcid)
                if not pending_ids:
                    pending_assistant_index = None
                    pending_tool_indices = []
            continue

        if role == "user":
            if pending_ids:
                _drop_pending_chain()
            repaired.append({"role": "user", "content": m.get("content", "") or ""})
            continue

    if pending_ids:
        _drop_pending_chain()

    return repaired


def _sanitize_history_for_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _repair_tool_call_chains(messages)


# ============================================================
# LLM call
# ============================================================

def _call_llm(messages: List[Dict[str, Any]]):
    st.session_state.messages = _repair_tool_call_chains(st.session_state.messages)
    mode = st.session_state.agent_mode
    safe_messages = _sanitize_history_for_llm(st.session_state.messages)

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

    return client.chat.completions.create(model=MODEL, messages=safe_messages, tools=LLM_TOOLS, tool_choice="auto")


# ============================================================
# Tool argument guard: force correct dataset location
# ============================================================

def _apply_default_dataset_args(tool_name: str, args: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    """
    If the model calls dataset tools with missing/incorrect bucket/path,
    force the configured DEFAULT_STORAGE_BUCKET/DEFAULT_STORAGE_PATH.
    Only allow override if the user explicitly mentions a bucket/path in the message.
    """
    if tool_name not in DATA_TOOLS:
        return args

    user_lower = (user_text or "").lower()
    user_specified = ("storage_bucket" in user_lower) or ("storage_path" in user_lower) or (DEFAULT_STORAGE_BUCKET.lower() in user_lower) or (DEFAULT_STORAGE_PATH.lower() in user_lower)

    sb = (args.get("storage_bucket") or "").strip()
    sp = (args.get("storage_path") or "").strip()

    if user_specified:
        # user is intentionally controlling location; just fill blanks if missing
        if not sb:
            args["storage_bucket"] = DEFAULT_STORAGE_BUCKET
        if not sp:
            args["storage_path"] = DEFAULT_STORAGE_PATH
        return args

    # otherwise: force defaults (prevents dataset/master.csv mistakes)
    args["storage_bucket"] = DEFAULT_STORAGE_BUCKET
    args["storage_path"] = DEFAULT_STORAGE_PATH
    return args


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

            # ✅ enforce correct dataset location
            args = _apply_default_dataset_args(name, args, user_text)

            try:
                out = _run_tool(name, args)
            except Exception as e:
                out = {"ok": False, "tool": name, "args": args, "error": str(e)}

            st.session_state.messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)})

        st.session_state.messages = _trim_messages(st.session_state.messages)

    _persist_chat()


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
            sid = str(uuid.uuid4())
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
        sid = str(uuid.uuid4())
        _set_session(sid)
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        _persist_chat(title=f"Session {sid[:8]}")
        st.success("Deleted + created new.")
        st.rerun()


# ============================================================
# Main UI
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
