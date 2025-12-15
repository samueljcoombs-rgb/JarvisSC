# pages/football_researcher.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI

# IMPORTANT: This must exist in your repo: modules/football_tools.py
from modules import football_tools as functions


# ---------------------------
# OpenAI client + model select
# ---------------------------

def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY.")
        st.stop()
    return OpenAI(api_key=api_key)

client = _init_client()

def _select_best_model(c: OpenAI) -> str:
    """
    Prefer GPT-5 if available, otherwise fall back.
    (This is as close as we can get to 'always GPT-5.2 Thinking' programmatically,
     since the exact variant name may not appear in model listing.)
    """
    preferred = os.getenv("PREFERRED_OPENAI_MODEL", "").strip() or st.secrets.get("PREFERRED_OPENAI_MODEL", "")
    if preferred:
        return preferred

    # Try to list models; if unavailable, fall back
    try:
        names = {m.id for m in c.models.list().data}
        for candidate in ["gpt-5", "gpt-latest", "gpt-4.1", "gpt-4o", "gpt-4.1-mini"]:
            if candidate in names:
                return candidate
    except Exception:
        pass

    return "gpt-4o"

MODEL = _select_best_model(client)


# ---------------------------
# Tool registry
# ---------------------------

TOOL_FUNCS: Dict[str, Any] = {
    # Knowledge base
    "get_dataset_overview": functions.get_dataset_overview,
    "get_column_definitions": functions.get_column_definitions,
    "get_research_rules": functions.get_research_rules,
    "get_evaluation_framework": functions.get_evaluation_framework,

    # Research memory/state
    "append_research_note": functions.append_research_note,
    "get_recent_research_notes": functions.get_recent_research_notes,
    "get_research_state": functions.get_research_state,
    "set_research_state": functions.set_research_state,

    # Data inspection
    "load_data_basic": functions.load_data_basic,
    "list_columns": functions.list_columns,

    # Strategy evaluation
    "strategy_performance_summary": functions.strategy_performance_summary,
    "strategy_performance_batch": functions.strategy_performance_batch,

    # Offloaded compute (Supabase job queue + results)
    "submit_job": functions.submit_job,
    "get_job": functions.get_job,
    "download_result": functions.download_result,
}

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    # ---- KB ----
    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": "Load dataset_overview sheet (key/value style rows).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_definitions",
            "description": "Load column_definitions sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_research_rules",
            "description": "Load research_rules sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evaluation_framework",
            "description": "Load evaluation_framework sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },

    # ---- memory/state ----
    {
        "type": "function",
        "function": {
            "name": "append_research_note",
            "description": "Append a research note into research_memory sheet. Provide note and optional tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_research_notes",
            "description": "Fetch last N research notes from research_memory.",
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
            "name": "get_research_state",
            "description": "Fetch key/value state dictionary from research_state sheet.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_research_state",
            "description": "Upsert a key/value into research_state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },

    # ---- data ----
    {
        "type": "function",
        "function": {
            "name": "load_data_basic",
            "description": "Load dataset preview + column list.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 10, "maximum": 2000}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List all columns in the dataset.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },

    # ---- evaluation ----
    {
        "type": "function",
        "function": {
            "name": "strategy_performance_summary",
            "description": "Compute bet-level ROI and ID-aggregated streak/drawdown for a PL column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "side": {"type": "string", "enum": ["back", "lay"]},
                    "odds_column": {"type": "string"},
                    "time_split_ratio": {"type": "number", "minimum": 0.5, "maximum": 0.95},
                    "compute_streaks": {"type": "boolean"},
                },
                "required": ["pl_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "strategy_performance_batch",
            "description": "Compute performance summaries for multiple PL columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_columns": {"type": "array", "items": {"type": "string"}},
                    "time_split_ratio": {"type": "number", "minimum": 0.5, "maximum": 0.95},
                    "compute_streaks": {"type": "boolean"},
                },
                "required": ["pl_columns"],
            },
        },
    },

    # ---- offloaded compute ----
    {
        "type": "function",
        "function": {
            "name": "submit_job",
            "description": "Submit a heavy compute job to Supabase queue (processed by Modal worker).",
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
            "description": "Get status + fields for a Supabase job_id.",
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
            "name": "download_result",
            "description": "Download a JSON result from Supabase Storage path.",
            "parameters": {
                "type": "object",
                "properties": {"result_path": {"type": "string"}},
                "required": ["result_path"],
            },
        },
    },
]


# ---------------------------
# Agent prompt
# ---------------------------

SYSTEM_PROMPT = """You are Football Researcher, an autonomous research agent.

Mission:
- Find strategy criteria (explicit filters/ranges) that generalise to future matches and produce profit.
- Use strict anti-overfitting rules: time-based splits, minimum sample sizes, simple rules first.
- Never use outcome columns as predictive features.

You have tools to:
- load dataset overview, column definitions, research rules, evaluation framework
- run evaluations on PL columns (bet-level ROI; game-level streak/drawdown in points)
- submit heavy jobs to an offloaded compute worker (Supabase queue processed by Modal)
- store persistent research notes and state in Google Sheets.

When you need heavy computation, submit a job via submit_job and then poll get_job until done,
then download_result and interpret. Always log key findings using append_research_note and update research_state.
"""


# ---------------------------
# Tool execution loop
# ---------------------------

def _call_llm(messages: List[Dict[str, Any]]) -> Any:
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )

def _run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    fn = TOOL_FUNCS.get(name)
    if not fn:
        return {"error": f"Tool not found: {name}"}
    try:
        return fn(**args) if args else fn()
    except TypeError:
        # If function takes no kwargs
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

def _chat_with_tools(user_text: str, max_rounds: int = 6) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    st.session_state.messages.append({"role": "user", "content": user_text})

    for _ in range(max_rounds):
        resp = _call_llm(st.session_state.messages)
        msg = resp.choices[0].message

        # If the model returns tool calls, execute them
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # Add assistant message indicating it is calling tools
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")
                out = _run_tool(tool_name, tool_args)

                # Append tool output
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": json.dumps(out, ensure_ascii=False),
                })

            continue

        # Otherwise it's a normal assistant response — end
        st.session_state.messages.append({"role": "assistant", "content": msg.content or ""})
        break


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Football Researcher", layout="wide")

st.title("⚽ Football Researcher (Autonomous)")

st.caption(f"Model: {MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Render chat
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    if m["role"] == "tool":
        with st.expander(f"🛠 Tool output: {m.get('name')}", expanded=False):
            st.code(m.get("content", ""), language="json")
        continue
    with st.chat_message(m["role"]):
        st.write(m.get("content", ""))

# Input
user_msg = st.chat_input("Ask the researcher…")
if user_msg:
    _chat_with_tools(user_msg)
    st.rerun()

# Debug helpers
with st.expander("Debug / Quick tests", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Test: list_columns"):
            _chat_with_tools("Call list_columns and show the result.")
            st.rerun()
    with col2:
        if st.button("Test: submit ping job"):
            _chat_with_tools('Submit a job: task_type="ping", params={"hello":"world"}. Then show me the job_id.')
            st.rerun()
    with col3:
        if st.button("Clear chat"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()
