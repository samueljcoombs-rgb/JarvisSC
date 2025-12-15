from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import streamlit as st
from openai import OpenAI, BadRequestError

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
# Tool runner
# ============================================================

def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**(args or {}))


# ============================================================
# Tools exposed to the LLM (SAFE SET ONLY)
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

    {"type": "function", "function": {"name": "load_data_basic", "description": "Load CSV preview (Supabase Storage).", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "list_columns", "description": "List CSV columns (Supabase Storage).", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "basic_roi_for_pl_column", "description": "ROI for a PL column (outcome only).", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}, "storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}}, "required": ["pl_column"]}}},

    # ✅ SAFE job tools
    {"type": "function", "function": {"name": "submit_strategy_search", "description": "Submit a strategy_search job safely. Optional target_pl_column; otherwise worker picks best market.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "results_bucket": {"type": "string"}, "time_split_ratio": {"type": "number"}, "target_pl_column": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_job", "description": "Get job status by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for completion; optionally downloads results.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "download_result", "description": "Download a result JSON from storage by path.", "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}, "bucket": {"type": "string"}}, "required": ["result_path"]}}},
]


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = f"""You are FootballResearcher — an autonomous research agent that discovers profitable, robust football trading strategy criteria.

Hard constraints:
- PL columns are outcomes only and MUST NOT be used as predictive features.
- Avoid overfitting: time-based splits; never tune thresholds on final test.
- Always report sample sizes, train vs test gap, and drawdown/losing streak in POINTS.
- Prefer simple rules that generalise; penalise fragile, tiny samples.

Source of truth:
- Use Google Sheet tabs: dataset_overview, research_rules, column_definitions, evaluation_framework, research_state, research_memory.

Data access:
- Default dataset is bucket="{DEFAULT_STORAGE_BUCKET}", path="{DEFAULT_STORAGE_PATH}".
- Do NOT invent bucket/path.

Jobs:
- Only submit worker jobs using submit_strategy_search (do NOT invent task types).

Mode rules:
- CHAT mode: conversational (no tools).
- AUTOPILOT mode: tools enabled; keep explanations short.
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
# LLM call (chat mode has no tools)
# ============================================================

def _call_llm(messages: List[Dict[str, Any]]):
    mode = st.session_state.agent_mode

    if mode == "chat":
        chat_only = []
        for m in messages:
            if m.get("role") == "tool":
                continue
            if m.get("role") == "assistant" and "tool_calls" in m:
                mm = dict(m)
                mm.pop("tool_calls", None)
                chat_only.append(mm)
            else:
                chat_only.append(m)

        return client.chat.completions.create(model=MODEL, messages=chat_only)

    return client.chat.completions.create(model=MODEL, messages=messages, tools=LLM_TOOLS, tool_choice="auto")


def _apply_default_dataset_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    args = dict(args or {})
    # Default dataset for data tools + strategy submission
    if tool_name in {"load_data_basic", "list_columns", "basic_roi_for_pl_column", "submit_strategy_search"}:
        if not (args.get("storage_bucket") and args.get("storage_path")):
            args["storage_bucket"] = DEFAULT_STORAGE_BUCKET
            args["storage_path"] = DEFAULT_STORAGE_PATH
    if tool_name == "submit_strategy_search":
        if not args.get("results_bucket"):
            args["results_bucket"] = DEFAULT_RESULTS_BUCKET
        if args.get("time_split_ratio") is None:
            args["time_split_ratio"] = 0.7
    return args


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
            args = _apply_default_dataset_args(name, args)

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

    if st.button("➕ New session"):
        sid = _new_session_id()
        _set_session(sid)
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        _persist_chat(title=f"Session {sid[:8]}")
        st.rerun()

    st.divider()
    st.subheader("🤖 Quick Test Buttons")

    if st.button("Test: load_data_basic"):
        out = _run_tool("load_data_basic", {"storage_bucket": DEFAULT_STORAGE_BUCKET, "storage_path": DEFAULT_STORAGE_PATH})
        st.session_state.last_test = {"tool": "load_data_basic", "out": out}
        st.rerun()

    if st.button("Test: submit strategy_search (no target)"):
        out = _run_tool("submit_strategy_search", {"storage_bucket": DEFAULT_STORAGE_BUCKET, "storage_path": DEFAULT_STORAGE_PATH, "results_bucket": DEFAULT_RESULTS_BUCKET, "time_split_ratio": 0.7})
        st.session_state.last_test = {"tool": "submit_strategy_search", "out": out}
        st.session_state.last_job_id = (out or {}).get("job_id")
        st.rerun()

    if st.button("Test: submit strategy_search (BTTS PL)"):
        out = _run_tool("submit_strategy_search", {"storage_bucket": DEFAULT_STORAGE_BUCKET, "storage_path": DEFAULT_STORAGE_PATH, "results_bucket": DEFAULT_RESULTS_BUCKET, "time_split_ratio": 0.7, "target_pl_column": "BTTS PL"})
        st.session_state.last_test = {"tool": "submit_strategy_search", "out": out}
        st.session_state.last_job_id = (out or {}).get("job_id")
        st.rerun()

    if st.button("Test: wait last job (10 min)"):
        job_id = st.session_state.get("last_job_id")
        if not job_id:
            st.session_state.last_test = {"error": "No last_job_id in session."}
        else:
            waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 600, "poll_s": 3, "auto_download": True})
            st.session_state.last_test = {"tool": "wait_for_job", "job_id": job_id, "out": waited}
        st.rerun()


# ============================================================
# Main UI
# ============================================================

st.subheader("🧾 Last test output")
st.json(st.session_state.get("last_test") or {"note": "No tests run yet in this session."})

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
