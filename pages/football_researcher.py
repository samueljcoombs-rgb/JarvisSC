from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI

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
MODEL = PREFERRED or "gpt-5.1"  # pinned


# ============================================================
# Defaults
# ============================================================

DEFAULT_STORAGE_BUCKET = os.getenv("DATA_STORAGE_BUCKET") or st.secrets.get("DATA_STORAGE_BUCKET", "football-data")
DEFAULT_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH") or st.secrets.get("DATA_STORAGE_PATH", "football_ai_NNIA.csv")
DEFAULT_RESULTS_BUCKET = os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET", "football-results")

# Prevent giant conversations (browser crashes)
MAX_MESSAGES_TO_KEEP = int(os.getenv("MAX_CHAT_MESSAGES") or st.secrets.get("MAX_CHAT_MESSAGES", 220))


def _dataset_locator() -> Dict[str, str]:
    return {"storage_bucket": DEFAULT_STORAGE_BUCKET, "storage_path": DEFAULT_STORAGE_PATH}


# ============================================================
# Tool runner
# ============================================================

def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**args)


# ============================================================
# Tools schema for OpenAI function calling
# ============================================================

TOOLS = [
    {"type": "function", "function": {"name": "get_dataset_overview", "description": "Get dataset_overview tab.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_research_rules", "description": "Get research_rules tab.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_column_definitions", "description": "Get column_definitions tab.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_evaluation_framework", "description": "Get evaluation_framework tab.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_recent_research_notes", "description": "Get recent research_memory rows.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "append_research_note", "description": "Append to research_memory.", "parameters": {"type": "object", "properties": {"note": {"type": "string"}, "tags": {"type": "string"}}, "required": ["note"]}}},
    {"type": "function", "function": {"name": "get_research_state", "description": "Get research_state KV.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "set_research_state", "description": "Set research_state KV.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},

    {"type": "function", "function": {"name": "load_data_basic", "description": "Load CSV preview.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "list_columns", "description": "List CSV columns.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "basic_roi_for_pl_column", "description": "Row-level ROI for PL column.", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}, "storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": ["pl_column"]}}},

    {"type": "function", "function": {"name": "submit_job", "description": "Submit Modal worker job.", "parameters": {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]}}},
    {"type": "function", "function": {"name": "get_job", "description": "Get job status by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for completion; optionally downloads results.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "download_result", "description": "Download a result JSON from storage by path.", "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}, "bucket": {"type": "string"}}, "required": ["result_path"]}}},

    # chat sessions (restored)
    {"type": "function", "function": {"name": "list_chats", "description": "List saved chat sessions.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "save_chat", "description": "Save chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "messages": {"type": "array"}, "title": {"type": "string"}}, "required": ["session_id", "messages"]}}},
    {"type": "function", "function": {"name": "load_chat", "description": "Load chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}}},
    {"type": "function", "function": {"name": "rename_chat", "description": "Rename chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "title": {"type": "string"}}, "required": ["session_id", "title"]}}},
    {"type": "function", "function": {"name": "delete_chat", "description": "Delete chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}}},
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
- Strategies must be explicit filters: MARKET/PL column + ranges on numeric fields + optional categorical constraints.
- Be decisive: pick what to test next. Only ask user if blocked by missing config.

Important:
- In CHAT mode you must reply conversationally and should not call tools unless explicitly asked.
- In AUTOPILOT mode you may call tools and run jobs, but still write a short explanation of what you did.
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
        # non-fatal: show later in sidebar
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
    # keep first system message + last N-1 messages
    system = messages[0:1]
    tail = messages[-(MAX_MESSAGES_TO_KEEP - 1):]
    return system + tail


def _persist_chat(title: str = ""):
    st.session_state.messages = _trim_messages(st.session_state.messages)
    out = _run_tool("save_chat", {"session_id": SESSION_ID, "messages": st.session_state.messages, "title": title})
    # non-fatal storage failures shouldn’t crash app
    if isinstance(out, dict) and out.get("ok") is False:
        st.session_state.last_chat_save_error = out


def _try_load_chat(sid: str) -> bool:
    loaded = _run_tool("load_chat", {"session_id": sid})
    if loaded.get("ok") and loaded.get("data", {}).get("messages"):
        st.session_state.messages = loaded["data"]["messages"]
        st.session_state.messages = _trim_messages(st.session_state.messages)
        return True
    return False


_init_messages_if_needed()

if "loaded_for_sid" not in st.session_state or st.session_state.loaded_for_sid != SESSION_ID:
    st.session_state.loaded_for_sid = SESSION_ID
    if not _try_load_chat(SESSION_ID):
        _persist_chat(title=f"Session {SESSION_ID[:8]}")


# ============================================================
# Agent mode (restores “mind”)
# ============================================================

if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "chat"  # chat | autopilot


# ============================================================
# LLM call + tool loop
# ============================================================

def _call_llm(messages: List[Dict[str, Any]]):
    mode = st.session_state.agent_mode
    if mode == "chat":
        # Chat mode: always speak, no tools.
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tool_choice="none",
        )

    # Autopilot mode: tools enabled.
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )


def _chat_with_tools(user_text: str, max_rounds: int = 6):
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages = _trim_messages(st.session_state.messages)

    for _ in range(max_rounds):
        resp = _call_llm(st.session_state.messages)
        msg = resp.choices[0].message

        # Always append assistant message, even if empty (so UI doesn’t feel “dead”)
        if msg.content:
            st.session_state.messages.append({"role": "assistant", "content": msg.content})
        else:
            # In chat mode, empty content is undesirable; add a friendly fallback
            if st.session_state.agent_mode == "chat":
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I’m here — ask me anything about the model, the system, or strategies. What do you want to do next?"}
                )

        tool_calls = getattr(msg, "tool_calls", None)

        # In chat mode we explicitly disabled tools, so we stop here.
        if st.session_state.agent_mode == "chat":
            break

        if not tool_calls:
            break

        # Execute tool calls
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            out = _run_tool(name, args)

            # MUST be response to tool_call_id
            st.session_state.messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)}
            )

        st.session_state.messages = _trim_messages(st.session_state.messages)

    _persist_chat()


# ============================================================
# Autopilot receipt + structured logging
# ============================================================

def _format_rule(rule_obj: Dict[str, Any]) -> str:
    parts = []
    for cond in rule_obj.get("rule", []):
        parts.append(f"{cond.get('col')} in [{cond.get('min')}, {cond.get('max')}]")
    return " AND ".join(parts) if parts else "(no rule)"


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


def _autopilot_one_cycle():
    params = {
        "storage_bucket": DEFAULT_STORAGE_BUCKET,
        "storage_path": DEFAULT_STORAGE_PATH,
        "_results_bucket": DEFAULT_RESULTS_BUCKET,
        "time_split_ratio": 0.7,
    }

    submitted = _run_tool("submit_job", {"task_type": "strategy_search", "params": params})
    job_id = submitted.get("job_id")

    st.session_state.last_autopilot = {"stage": "submitted", "submitted": submitted}

    if not job_id:
        st.session_state.last_autopilot["stage"] = "error"
        st.session_state.last_autopilot["error"] = "No job_id returned from submit_job"
        return

    waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 900, "poll_s": 5, "auto_download": True})
    st.session_state.last_autopilot["waited"] = waited

    # state writes are non-fatal
    _run_tool("set_research_state", {"key": "last_autopilot_ran_at", "value": datetime.utcnow().isoformat()})
    _run_tool("set_research_state", {"key": "last_job_id", "value": job_id})

    if waited.get("status") != "done":
        st.session_state.last_autopilot["stage"] = waited.get("status")
        return

    job = waited.get("job") or {}
    result_path = job.get("result_path") or ""
    result_json = waited.get("result") or {}

    note = _structured_note(result_json, job_id, result_path)
    _run_tool("append_research_note", {"note": note, "tags": "autopilot,worker,structured"})

    # additional state helpful for “learning”
    try:
        picked = (result_json.get("result") or {}).get("picked") or {}
        top = ((result_json.get("result") or {}).get("search") or {}).get("top_rules") or []
        sig = ""
        if top:
            sig = "|".join(
                [f"{c.get('col')}:[{c.get('min')},{c.get('max')}]" for c in (top[0].get("rule") or [])]
            )
        _run_tool("set_research_state", {"key": "last_market", "value": str(picked.get("pl_column", ""))})
        _run_tool("set_research_state", {"key": "last_rule_signature", "value": sig})
        _run_tool("set_research_state", {"key": "last_result_path", "value": result_path})
    except Exception:
        pass

    st.session_state.last_autopilot["stage"] = "done"


# ============================================================
# Sidebar UI
# ============================================================

with st.sidebar:
    st.caption(f"Model: `{MODEL}`")
    st.caption(f"Data: `{DEFAULT_STORAGE_BUCKET}/{DEFAULT_STORAGE_PATH}`")
    st.caption(f"Results: `{DEFAULT_RESULTS_BUCKET}`")

    st.radio("Agent mode", ["chat", "autopilot"], key="agent_mode")
    st.divider()

    # Chat bucket issues (non-fatal warning)
    if st.session_state.get("last_chat_save_error"):
        st.warning(f"Chat persistence warning: {st.session_state.last_chat_save_error}")
        if st.button("Clear chat save warning"):
            st.session_state.last_chat_save_error = None

    # --- sessions ---
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

    # --- autopilot ---
    st.subheader("🤖 Autopilot")
    if st.button("Run 1 autopilot cycle"):
        _autopilot_one_cycle()
        st.rerun()

    st.divider()
    st.subheader("🧪 Quick tools")
    if st.button("Recent research_memory (10)"):
        st.json(_run_tool("get_recent_research_notes", {"limit": 10}))

    if st.button("Research state"):
        st.json(_run_tool("get_research_state", {}))

    if st.button("List CSV columns"):
        loc = _dataset_locator()
        st.json(_run_tool("list_columns", {**loc}))


# ============================================================
# Main panels: Autopilot receipt + Chat
# ============================================================

st.subheader("🧾 Autopilot Receipt")
last = st.session_state.get("last_autopilot")
if not last:
    st.info("No autopilot run in this session yet.")
else:
    st.json(last)
    try:
        waited = last.get("waited") or {}
        if waited.get("status") == "done":
            payload = waited.get("result") or {}
            picked = (payload.get("result") or {}).get("picked") or {}
            top_rules = ((payload.get("result") or {}).get("search") or {}).get("top_rules") or []
            if top_rules:
                st.markdown(
                    f"**Picked:** `{picked.get('pl_column')}` (side={picked.get('side')}, odds_col={picked.get('odds_col')})  \n"
                    f"**Top rule:** `{_format_rule(top_rules[0])}`  \n"
                    f"**Test ROI:** `{top_rules[0].get('test', {}).get('roi')}` | **Test bets:** `{top_rules[0].get('test', {}).get('bets')}`  \n"
                    f"**Game-level DD (pts):** `{top_rules[0].get('test_game_level', {}).get('max_dd')}` | "
                    f"**Losing streak (bets/pts):** `{top_rules[0].get('test_game_level', {}).get('losing_streak', {}).get('bets')}` / "
                    f"`{top_rules[0].get('test_game_level', {}).get('losing_streak', {}).get('pl')}`"
                )
    except Exception:
        pass


st.subheader("💬 Chat")
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m.get("content", ""))

user_msg = st.chat_input("Ask the researcher…")
if user_msg:
    _chat_with_tools(user_msg)
    st.rerun()
