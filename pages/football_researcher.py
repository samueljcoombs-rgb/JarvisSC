# pages/football_researcher.py
from __future__ import annotations

import os
import json
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
MODEL = PREFERRED or "gpt-5.2-thinking"  # will error only if your account truly cannot access it


# =========================
# Tool registry (OpenAI function calling)
# =========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": "Get high-level dataset overview and intent from Google Sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_research_rules",
            "description": "Get research rules/principles from Google Sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_definitions",
            "description": "Get column definitions table from Google Sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evaluation_framework",
            "description": "Get evaluation framework (P&L/ROI/streaks etc.) from Google Sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_research_notes",
            "description": "Get last N research notes from Google Sheet.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_research_note",
            "description": "Append a research note to Google Sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "tags": {"type": "string"},
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_research_state",
            "description": "Get persistent key/value research state from Google Sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_research_state",
            "description": "Set persistent key/value research state in Google Sheet.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_data_basic",
            "description": "Load a preview of the CSV from DATA_CSV_URL (or provided csv_url) and return rows/cols/head.",
            "parameters": {
                "type": "object",
                "properties": {"csv_url": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List column names in the CSV.",
            "parameters": {
                "type": "object",
                "properties": {"csv_url": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "basic_roi_for_pl_column",
            "description": "Compute simple row-level P&L and avg P&L per bet for a given PL column (basic tool).",
            "parameters": {
                "type": "object",
                "properties": {"pl_column": {"type": "string"}, "csv_url": {"type": "string"}},
                "required": ["pl_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_job",
            "description": "Submit a background job to Supabase jobs table for Modal worker to pick up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["task_type", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job",
            "description": "Get a job record from Supabase by job_id.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_job",
            "description": "Poll Supabase for job completion; returns done/error/timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "timeout_s": {"type": "integer", "minimum": 10, "maximum": 7200},
                    "poll_s": {"type": "integer", "minimum": 1, "maximum": 60},
                    "auto_download": {"type": "boolean"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_result",
            "description": "Download a JSON result from Supabase Storage via result_path.",
            "parameters": {
                "type": "object",
                "properties": {"result_path": {"type": "string"}},
                "required": ["result_path"],
            },
        },
    },
]


def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**args)


# =========================
# System prompt (kept short, but strict)
# =========================
SYSTEM_PROMPT = """You are FootballResearcher, an autonomous research agent for building profitable, robust football trading strategies.
You MUST follow the research rules and evaluation framework from the Google Sheet.
You SHOULD use the tools to fetch dataset definitions, rules, notes, then propose and test strategies.
Do not use PL columns as predictive features.
Always use time-based splits and guard against overfitting.
When you run tools, use OpenAI tool calling (function calls)."""


# =========================
# Streamlit UI state
# =========================
st.set_page_config(page_title="Football Researcher", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

st.title("⚽ Football Researcher")


# =========================
# Chat renderer
# =========================
def _render_chat():
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


# =========================
# LLM + tool loop
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

    # 1) first call (may contain tool_calls)
    resp = _call_llm(st.session_state.messages)
    msg = resp.choices[0].message

    # assistant content (can be empty if it immediately tool-calls)
    if msg.content:
        st.session_state.messages.append({"role": "assistant", "content": msg.content})

    # 2) handle tool calls (if any)
    if getattr(msg, "tool_calls", None):
        tool_calls = msg.tool_calls
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            out = _run_tool(name, args)

            st.session_state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(out, ensure_ascii=False),
                }
            )

        # 3) follow-up call to let model see tool outputs
        follow = _call_llm(st.session_state.messages)
        follow_msg = follow.choices[0].message
        st.session_state.messages.append(
            {"role": "assistant", "content": follow_msg.content or "(no content)"}
        )


# =========================
# Autopilot (1 cycle)
# =========================
def _autopilot_one_cycle():
    csv_url = _get_csv_url_or_stop()
    params = {
        "csv_url": csv_url,
        "_results_bucket": os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET") or "football-results",
        "time_split_ratio": 0.7,
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

    # Persist autopilot timestamp regardless
    _run_tool("set_research_state", {"key": "last_autopilot_ran_at", "value": datetime.utcnow().isoformat()})

    if waited.get("status") == "timeout":
        st.warning(
            "Job is still queued/running. This is NOT a failure. "
            "If it stays queued for a long time, check Modal logs / schedule."
        )
        return

    if waited.get("status") == "error":
        st.error("Job ended in error. See job.error in output.")
        return

    # done case
    result = waited.get("result") or {}
    note = (
        "Autopilot cycle complete.\n\n"
        f"- job_id: {job_id}\n"
        f"- worker_status: {waited.get('status')}\n"
        f"- result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}\n"
        f"- message: {result.get('message') if isinstance(result, dict) else ''}\n"
    )
    _run_tool("append_research_note", {"note": note, "tags": "autopilot,worker"})
    st.success("Appended a research note for this autopilot cycle.")


def _get_csv_url_or_stop() -> str:
    csv_url = os.getenv("DATA_CSV_URL") or st.secrets.get("DATA_CSV_URL")
    if not csv_url:
        st.error("Missing DATA_CSV_URL in Streamlit secrets.")
        st.stop()
    return csv_url


# =========================
# Layout
# =========================
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("🤖 Autopilot")
    if st.button("Autopilot: run 1 cycle"):
        _autopilot_one_cycle()

    st.divider()
    st.subheader("🧠 Context tools")
    if st.button("Load dataset overview"):
        st.json(_run_tool("get_dataset_overview", {}))
    if st.button("Load research rules"):
        st.json(_run_tool("get_research_rules", {}))
    if st.button("List columns"):
        st.json(_run_tool("list_columns", {}))

with right:
    st.subheader("💬 Chat")
    _render_chat()
    user_msg = st.chat_input("Ask the researcher…")
    if user_msg:
        _chat_with_tools(user_msg)
        st.rerun()
