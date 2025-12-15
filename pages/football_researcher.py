# pages/football_researcher.py
from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI

from modules import football_tools as functions


# =========================
# OpenAI client + model selection
# =========================
def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


client = _init_client()

PREFERRED = (os.getenv("PREFERRED_OPENAI_MODEL") or st.secrets.get("PREFERRED_OPENAI_MODEL") or "").strip()
MODEL = PREFERRED or "gpt-5.2-thinking"  # will fall back only if your account can't access it


# =========================
# Tool registry (OpenAI function calling)
# =========================
TOOLS = [
    {"type": "function", "function": {"name": "get_dataset_overview", "description": "Get dataset overview from Google Sheet.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_research_rules", "description": "Get research rules from Google Sheet.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_column_definitions", "description": "Get column definitions table from Google Sheet.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_evaluation_framework", "description": "Get evaluation framework from Google Sheet.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_recent_research_notes", "description": "Get last N research notes from Google Sheet.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}}, "required": []}}},
    {"type": "function", "function": {"name": "append_research_note", "description": "Append a research note to research_memory sheet.", "parameters": {"type": "object", "properties": {"note": {"type": "string"}, "tags": {"type": "string"}}, "required": ["note"]}}},
    {"type": "function", "function": {"name": "get_research_state", "description": "Get persistent key/value research state.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "set_research_state", "description": "Set persistent key/value research state.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
    {"type": "function", "function": {"name": "load_data_basic", "description": "Load a preview of the CSV and return rows/cols/head.", "parameters": {"type": "object", "properties": {"csv_url": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "list_columns", "description": "List column names in the CSV.", "parameters": {"type": "object", "properties": {"csv_url": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "basic_roi_for_pl_column", "description": "Basic row-level ROI for a PL column.", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}, "csv_url": {"type": "string"}}, "required": ["pl_column"]}}},
    {"type": "function", "function": {"name": "submit_job", "description": "Submit a background job (Modal worker).", "parameters": {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]}}},
    {"type": "function", "function": {"name": "get_job", "description": "Get job record by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for job completion; returns done/error/timeout.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "download_result", "description": "Download JSON result from storage.", "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}}, "required": ["result_path"]}}},
    {"type": "function", "function": {"name": "save_chat", "description": "Persist the current chat session to Supabase Storage.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "messages": {"type": "array"}}, "required": ["session_id", "messages"]}}},
    {"type": "function", "function": {"name": "load_chat", "description": "Load chat session from Supabase Storage.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}}},
]


def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**args)


# =========================
# System prompt (strong + autonomous)
# =========================
SYSTEM_PROMPT = """You are FootballResearcher — an autonomous research agent that discovers profitable, robust football trading strategy criteria.

Absolute rules:
- Use the Google Sheet tabs as your source of truth for dataset definitions, research rules, and evaluation framework.
- PL columns are outcomes only and MUST NOT be used as predictive features.
- Avoid overfitting: use time-based split (train/validation/test) and never tune thresholds on the final test.
- Always report sample size and stability (train vs test gap).
- Prefer simple rule sets that generalise.

Primary objective:
- Propose explicit strategy criteria usable on future matches (e.g. MODE=XG, MARKET=BTTS, xG diff in [a,b], odds in [x,y], drift constraints).
- When appropriate, submit background jobs to the Modal worker to compute heavy evaluations.
- Log significant findings with append_research_note.

Be decisive: choose what to test next; do not ask the user what to do unless blocked by missing config or data.
"""


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Football Researcher")

# Persistent session id (so refresh can reload from storage)
if "football_session_id" not in st.session_state:
    st.session_state.football_session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.football_session_id

# Chat init
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Try auto-load on first render
if "loaded_from_storage" not in st.session_state:
    st.session_state.loaded_from_storage = True
    loaded = _run_tool("load_chat", {"session_id": SESSION_ID})
    if loaded.get("ok") and loaded.get("data", {}).get("messages"):
        st.session_state.messages = loaded["data"]["messages"]


def _get_csv_url_or_stop() -> str:
    csv_url = os.getenv("DATA_CSV_URL") or st.secrets.get("DATA_CSV_URL")
    if not csv_url:
        st.error("Missing DATA_CSV_URL in Streamlit secrets.")
        st.stop()
    return csv_url


def _persist_chat():
    # Save current messages to storage so refresh doesn't wipe conversation
    _run_tool("save_chat", {"session_id": SESSION_ID, "messages": st.session_state.messages})


# =========================
# LLM tool-calling loop
# =========================
def _call_llm(messages: List[Dict[str, Any]]):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )


def _chat_with_tools(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})

    resp = _call_llm(st.session_state.messages)
    msg = resp.choices[0].message

    if msg.content:
        st.session_state.messages.append({"role": "assistant", "content": msg.content})

    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            out = _run_tool(name, args)

            st.session_state.messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)}
            )

        follow = _call_llm(st.session_state.messages)
        follow_msg = follow.choices[0].message
        st.session_state.messages.append({"role": "assistant", "content": follow_msg.content or "(no content)"})

    _persist_chat()


# =========================
# Autopilot (1 cycle)
# =========================
def _autopilot_one_cycle():
    csv_url = _get_csv_url_or_stop()
    params = {
        "csv_url": csv_url,
        "_results_bucket": os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET") or "football-results",
        "time_split_ratio": 0.7,
        # no market specified: worker chooses (future step)
    }

    submitted = _run_tool("submit_job", {"task_type": "strategy_search", "params": params})
    st.success("Autopilot: submitted strategy_search.")
    st.json(submitted)

    job_id = submitted.get("job_id")
    if not job_id:
        st.error("No job_id returned from submit_job.")
        return

    waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 900, "poll_s": 5, "auto_download": True})
    st.info("Autopilot: wait_for_job output.")
    st.json(waited)

    _run_tool("set_research_state", {"key": "last_autopilot_ran_at", "value": datetime.utcnow().isoformat()})

    if waited.get("status") == "timeout":
        st.warning("Job still queued/running. Not a failure — worker may be busy.")
        return
    if waited.get("status") == "error":
        st.error("Job ended in error. See job.error in output.")
        return

    result = waited.get("result") or {}
    note = (
        "Autopilot cycle complete.\n\n"
        f"- job_id: {job_id}\n"
        f"- status: {waited.get('status')}\n"
        f"- result_message: {result.get('message') if isinstance(result, dict) else ''}\n"
        f"- computed_at: {result.get('computed_at') if isinstance(result, dict) else ''}\n"
    )
    _run_tool("append_research_note", {"note": note, "tags": "autopilot,worker"})
    st.success("Appended an autopilot research note.")


# =========================
# UI
# =========================
with st.sidebar:
    st.caption(f"Model: `{MODEL}`")
    st.caption(f"Session: `{SESSION_ID}`")

    if st.button("💾 Save chat now"):
        _persist_chat()
        st.success("Saved.")

    if st.button("🧹 New chat (fresh system prompt)"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        _persist_chat()
        st.success("Reset chat.")
        st.rerun()

    st.divider()
    st.subheader("🤖 Autopilot")
    if st.button("Run 1 autopilot cycle"):
        _autopilot_one_cycle()

    st.divider()
    st.subheader("🧠 Context")
    if st.button("Dataset overview"):
        st.json(_run_tool("get_dataset_overview", {}))
    if st.button("Research rules"):
        st.json(_run_tool("get_research_rules", {}))
    if st.button("Evaluation framework"):
        st.json(_run_tool("get_evaluation_framework", {}))
    if st.button("Column definitions"):
        st.json(_run_tool("get_column_definitions", {}))
    if st.button("Recent research notes"):
        st.json(_run_tool("get_recent_research_notes", {"limit": 20}))

    st.divider()
    st.subheader("📦 Data")
    if st.button("Load data basic"):
        st.json(_run_tool("load_data_basic", {}))
    if st.button("List CSV columns"):
        st.json(_run_tool("list_columns", {}))

    st.divider()
    st.subheader("🧰 Manual worker jobs")
    task_type = st.text_input("task_type", value="ping")
    params_raw = st.text_area("params (JSON)", value='{"hello":"world"}', height=110)
    if st.button("Submit job"):
        try:
            params = json.loads(params_raw or "{}")
        except Exception:
            st.error("params must be valid JSON")
            params = None
        if params is not None:
            st.json(_run_tool("submit_job", {"task_type": task_type, "params": params}))

    job_id = st.text_input("job_id to check")
    if st.button("Get job"):
        if job_id.strip():
            st.json(_run_tool("get_job", {"job_id": job_id.strip()}))

    if st.button("Wait for job"):
        if job_id.strip():
            st.json(_run_tool("wait_for_job", {"job_id": job_id.strip(), "timeout_s": 120, "poll_s": 5, "auto_download": True}))


st.subheader("💬 Chat")
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask the researcher…")
if user_msg:
    _chat_with_tools(user_msg)
    st.rerun()
