from __future__ import annotations

import os
import json
import uuid
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

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
# (Fix: arrays must declare items)
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

    # SAFE wrappers
    {"type": "function", "function": {"name": "submit_strategy_search", "description": "Submit strategy_search job (safe pl_column resolution).", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "results_bucket": {"type": "string"}, "time_split_ratio": {"type": "number"}, "target_pl_column": {"type": "string"}}, "required": ["target_pl_column"]}}},
    {"type": "function", "function": {"name": "submit_feature_audit", "description": "Submit feature_audit job.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "results_bucket": {"type": "string"}, "target_pl_column": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "submit_feature_rank", "description": "Submit feature_rank job.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "results_bucket": {"type": "string"}, "target_pl_column": {"type": "string"}, "time_split_ratio": {"type": "number"}, "max_rows": {"type": "integer"}}, "required": ["target_pl_column"]}}},

    {"type": "function", "function": {"name": "submit_job", "description": "Submit Modal worker job (raw).", "parameters": {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]}}},
    {"type": "function", "function": {"name": "get_job", "description": "Get job status by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for completion; optionally downloads results.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "download_result", "description": "Download a result JSON from storage by path.", "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}, "bucket": {"type": "string"}}, "required": ["result_path"]}}},

    {"type": "function", "function": {"name": "list_chats", "description": "List saved chat sessions.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {"name": "save_chat", "description": "Save chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "messages": {"type": "array", "items": {"type": "object"}}, "title": {"type": "string"}}, "required": ["session_id", "messages"]}}},
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

Critical:
- Never pass raw user sentences as a column name.
- Always resolve target_pl_column to an exact CSV column (e.g. 'BTTS PL') before submitting jobs.
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
    out = _run_tool("save_chat", {"session_id": SESSION_ID, "messages": st.session_state.messages, "title": title})
    if isinstance(out, dict) and out.get("ok") is False:
        st.session_state.last_chat_save_error = out


def _try_load_chat(sid: str) -> bool:
    loaded = _run_tool("load_chat", {"session_id": sid})
    if loaded.get("ok") and loaded.get("data", {}).get("messages"):
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
            clean_assistant: Dict[str, Any] = {
                "role": "assistant",
                "content": m.get("content", "") or "",
            }

            tc = m.get("tool_calls")
            if isinstance(tc, list) and tc:
                cleaned_tool_calls = []
                for call in tc:
                    try:
                        cid = call.get("id")
                        fn = call.get("function") or {}
                        name = fn.get("name")
                        args = fn.get("arguments", "{}")
                        if cid and name:
                            cleaned_tool_calls.append(
                                {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
                            )
                            expecting_tool_ids.add(cid)
                    except Exception:
                        continue

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

        return client.chat.completions.create(
            model=MODEL,
            messages=chat_only,
        )

    return client.chat.completions.create(
        model=MODEL,
        messages=safe_messages,
        tools=TOOLS,
        tool_choice="auto",
    )


# ============================================================
# Local PL resolver (never pass raw user text)
# ============================================================

def _resolve_pl_from_user_text(user_text: str) -> Optional[str]:
    # pull columns once
    cols_out = _run_tool("list_columns", {"storage_bucket": DEFAULT_STORAGE_BUCKET, "storage_path": DEFAULT_STORAGE_PATH})
    cols = cols_out.get("columns") or []
    t = (user_text or "").lower()

    # prefer exact column matches present in the user text
    for c in cols:
        if c and c.lower() in t:
            return c

    # fuzzy keywords -> canonical
    if "btts" in t:
        for c in cols:
            if c.lower().strip() == "btts pl":
                return c

    if ("2.5" in t or "o2.5" in t or "over 2.5" in t) and "pl" in t:
        for c in cols:
            if c.lower().strip() == "bo 2.5 pl":
                return c

    # last resort: any column containing 'pl' and keyword
    if "pl" in t:
        for c in cols:
            cl = c.lower()
            if "pl" in cl and ("btts" in t and "btts" in cl):
                return c

    return None


# ============================================================
# One-shot pipeline runner (BTTS etc)
# ============================================================

def _run_strategy_pipeline(pl_column: str, split_ratio: float = 0.7) -> Dict[str, Any]:
    # 1) run strategy_search (safe wrapper resolves mapping properly)
    submitted = _run_tool(
        "submit_strategy_search",
        {
            "storage_bucket": DEFAULT_STORAGE_BUCKET,
            "storage_path": DEFAULT_STORAGE_PATH,
            "results_bucket": DEFAULT_RESULTS_BUCKET,
            "time_split_ratio": float(split_ratio),
            "target_pl_column": pl_column,
        },
    )

    job_id = submitted.get("job_id")
    if not job_id:
        return {"ok": False, "stage": "submit", "submitted": submitted}

    waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 1800, "poll_s": 5, "auto_download": True})
    if waited.get("status") != "done":
        return {"ok": False, "stage": "wait", "waited": waited}

    job = waited.get("job") or {}
    result_path = job.get("result_path") or ""
    # fetch full result (wait_for_job may or may not include parsed)
    result = _run_tool("download_result", {"bucket": DEFAULT_RESULTS_BUCKET, "result_path": result_path})
    return {"ok": True, "job_id": job_id, "result_path": result_path, "result": result}


def _summarise_top3(result_obj: Dict[str, Any]) -> str:
    try:
        payload = (result_obj.get("result") or {}).get("result") or {}
        inner = payload.get("result") or payload  # tolerate nesting
        picked = inner.get("picked") or {}
        search = inner.get("search") or {}
        if "error" in search:
            return f"Picked: {picked}\n\nERROR: {search.get('error')}"
        top = (search.get("top_rules") or [])[:3]
        lines = []
        lines.append(f"Picked: {picked}")
        lines.append("")
        for i, r in enumerate(top, start=1):
            rule = r.get("rule")
            te = r.get("test")
            tr = r.get("train")
            gap = r.get("gap_train_minus_test")
            games = (r.get("test_game_level") or {})
            lines.append(f"Top #{i}")
            lines.append(f"  Rule: {rule}")
            lines.append(f"  Train: {tr}")
            lines.append(f"  Test:  {te}")
            lines.append(f"  Gap(train-test): {gap}")
            lines.append(f"  Test drawdown/losing: {games}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Could not summarise result: {e}\nRaw: {result_obj}"


# ============================================================
# Chat loop
# ============================================================

def _chat_with_tools(user_text: str, max_rounds: int = 6):
    # AUTOPILOT fast-path: if user asks for a strategy, resolve PL locally and run pipeline
    if st.session_state.agent_mode == "autopilot":
        if re.search(r"\b(build|design|create)\b.*\bstrategy\b", user_text.lower()) or ("strategy" in user_text.lower() and "pl" in user_text.lower()):
            pl = _resolve_pl_from_user_text(user_text)
            if not pl:
                st.session_state.messages.append({"role": "assistant", "content": "I couldn’t resolve the PL column from your message. Try: `Build a strategy for BTTS PL` (exact)."})
                _persist_chat()
                return

            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.messages.append({"role": "assistant", "content": f"Running strategy_search pipeline for **{pl}** (time_split_ratio=0.7) using {DEFAULT_STORAGE_BUCKET}/{DEFAULT_STORAGE_PATH}…"})
            out = _run_strategy_pipeline(pl_column=pl, split_ratio=0.7)
            st.session_state.messages.append({"role": "assistant", "content": _summarise_top3(out.get("result", {}))})
            _run_tool("append_research_note", {"note": json.dumps({"ts": datetime.utcnow().isoformat(), "kind": "strategy_pipeline", "pl_column": pl, "job_id": out.get("job_id"), "result_path": out.get("result_path")}, ensure_ascii=False), "tags": "pipeline,autopilot"})
            _persist_chat()
            return

    # Normal LLM loop
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

        if assistant_msg["content"]:
            st.session_state.messages.append(assistant_msg)
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I’m here — what do you want to do next?"})

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


# ============================================================
# Main: Chat
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
