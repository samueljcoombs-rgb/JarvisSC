# pages/football_researcher.py
from __future__ import annotations

import json
import os
import inspect
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI

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
    Exact '5.2 Thinking' string may not be exposed; we prefer gpt-5 / gpt-latest.
    """
    preferred = (os.getenv("PREFERRED_OPENAI_MODEL", "").strip()
                 or st.secrets.get("PREFERRED_OPENAI_MODEL", ""))
    if preferred:
        return preferred

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

    # Offloaded compute (Supabase queue + results)
    "submit_job": functions.submit_job,
    "get_job": functions.get_job,
    "download_result": functions.download_result,
}

def _tool_schema(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    # ---- KB ----
    _tool_schema(
        "get_dataset_overview",
        "Load dataset_overview sheet (key/value style rows).",
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool_schema(
        "get_column_definitions",
        "Load column_definitions sheet.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool_schema(
        "get_research_rules",
        "Load research_rules sheet.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool_schema(
        "get_evaluation_framework",
        "Load evaluation_framework sheet.",
        {"type": "object", "properties": {}, "required": []},
    ),

    # ---- memory/state ----
    _tool_schema(
        "append_research_note",
        "Append a research note into research_memory sheet. Provide note and optional tags.",
        {
            "type": "object",
            "properties": {
                "note": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["note"],
        },
    ),
    _tool_schema(
        "get_recent_research_notes",
        "Fetch last N research notes from research_memory.",
        {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            "required": [],
        },
    ),
    _tool_schema(
        "get_research_state",
        "Fetch key/value state dictionary from research_state sheet.",
        {"type": "object", "properties": {}, "required": []},
    ),
    _tool_schema(
        "set_research_state",
        "Upsert a key/value into research_state.",
        {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    ),

    # ---- data ----
    _tool_schema(
        "load_data_basic",
        "Load dataset preview + column list.",
        {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 10, "maximum": 2000}},
            "required": [],
        },
    ),
    _tool_schema(
        "list_columns",
        "List all columns in the dataset.",
        {"type": "object", "properties": {}, "required": []},
    ),

    # ---- evaluation ----
    _tool_schema(
        "strategy_performance_summary",
        "Compute bet-level ROI and ID-aggregated streak/drawdown for a PL column.",
        {
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
    ),
    _tool_schema(
        "strategy_performance_batch",
        "Compute performance summaries for multiple PL columns.",
        {
            "type": "object",
            "properties": {
                "pl_columns": {"type": "array", "items": {"type": "string"}},
                "time_split_ratio": {"type": "number", "minimum": 0.5, "maximum": 0.95},
                "compute_streaks": {"type": "boolean"},
            },
            "required": ["pl_columns"],
        },
    ),

    # ---- offloaded compute ----
    _tool_schema(
        "submit_job",
        "Submit a heavy compute job to Supabase queue (processed by Modal worker).",
        {
            "type": "object",
            "properties": {
                "task_type": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["task_type", "params"],
        },
    ),
    _tool_schema(
        "get_job",
        "Get status + fields for a Supabase job_id.",
        {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    ),
    _tool_schema(
        "download_result",
        "Download a JSON result from Supabase Storage path.",
        {
            "type": "object",
            "properties": {"result_path": {"type": "string"}},
            "required": ["result_path"],
        },
    ),
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
- load dataset overview/definitions/rules/framework from Google Sheets
- evaluate strategies on PL columns (bet-level ROI; game-level streak/drawdown in points)
- submit heavy jobs to a Supabase queue (processed by a Modal worker), then poll+download results
- store persistent research notes and state in Google Sheets

When you need heavy computation, submit via submit_job then poll get_job until done, then download_result.
Always log key findings via append_research_note and update research_state.
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
    """
    Robust tool runner:
    - Filters args to match function signature
    - Never calls a function with missing required args
    - Returns clear errors instead of crashing Streamlit
    """
    fn = TOOL_FUNCS.get(name)
    if not fn:
        return {"error": f"Tool not found: {name}"}

    args = args or {}

    try:
        sig = inspect.signature(fn)
        params = sig.parameters

        accepted = {k: v for k, v in args.items() if k in params}

        missing_required = []
        for p_name, p in params.items():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if p.default is inspect._empty and p_name not in accepted:
                missing_required.append(p_name)

        if missing_required:
            return {
                "error": f"Missing required args for {name}: {missing_required}",
                "provided_args": list(args.keys()),
                "accepted_args": list(accepted.keys()),
            }

        return fn(**accepted)

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tool": name, "args": args}

def _chat_with_tools(user_text: str, max_rounds: int = 6) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    st.session_state.messages.append({"role": "user", "content": user_text})

    for _ in range(max_rounds):
        resp = _call_llm(st.session_state.messages)
        msg = resp.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")

                out = _run_tool(tool_name, tool_args)

                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": json.dumps(out, ensure_ascii=False),
                })

            continue

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

user_msg = st.chat_input("Ask the researcher…")
if user_msg:
    _chat_with_tools(user_msg)
    st.rerun()

with st.expander("Debug / Quick tests", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Test: list_columns"):
            _chat_with_tools("Call list_columns and show the result.")
            st.rerun()
    with c2:
        if st.button("Test: submit ping job"):
            _chat_with_tools('Call submit_job with task_type="ping" and params={"hello":"world"} and show me the tool output.')
            st.rerun()
    with c3:
        if st.button("Clear chat"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()
