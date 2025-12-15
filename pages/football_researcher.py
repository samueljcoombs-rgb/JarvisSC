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
    Prefer GPT-5. If you set Streamlit secret PREFERRED_OPENAI_MODEL="gpt-5",
    this will lock it.
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
    # definitions / rules / framework
    "get_dataset_overview": functions.get_dataset_overview,
    "get_column_definitions": functions.get_column_definitions,
    "get_research_rules": functions.get_research_rules,
    "get_evaluation_framework": functions.get_evaluation_framework,

    # memory/state
    "append_research_note": functions.append_research_note,
    "get_recent_research_notes": functions.get_recent_research_notes,
    "get_research_state": functions.get_research_state,
    "set_research_state": functions.set_research_state,

    # data
    "load_data_basic": functions.load_data_basic,
    "list_columns": functions.list_columns,

    # evaluation
    "strategy_performance_summary": functions.strategy_performance_summary,
    "strategy_performance_batch": functions.strategy_performance_batch,

    # offloaded compute
    "submit_job": functions.submit_job,
    "get_job": functions.get_job,
    "download_result": functions.download_result,
}

def _tool_schema(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    _tool_schema("get_dataset_overview", "Load dataset_overview sheet.", {"type": "object", "properties": {}, "required": []}),
    _tool_schema("get_column_definitions", "Load column_definitions sheet.", {"type": "object", "properties": {}, "required": []}),
    _tool_schema("get_research_rules", "Load research_rules sheet.", {"type": "object", "properties": {}, "required": []}),
    _tool_schema("get_evaluation_framework", "Load evaluation_framework sheet.", {"type": "object", "properties": {}, "required": []}),

    _tool_schema("append_research_note", "Append a research note (note + optional tags).",
                 {"type": "object", "properties": {"note": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["note"]}),
    _tool_schema("get_recent_research_notes", "Fetch last N research notes.",
                 {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}}, "required": []}),
    _tool_schema("get_research_state", "Fetch research_state key/value map.", {"type": "object", "properties": {}, "required": []}),
    _tool_schema("set_research_state", "Upsert key/value into research_state.",
                 {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}),

    _tool_schema("load_data_basic", "Load dataset preview.", {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 10, "maximum": 2000}}, "required": []}),
    _tool_schema("list_columns", "List all dataset columns.", {"type": "object", "properties": {}, "required": []}),

    _tool_schema("strategy_performance_summary", "ROI + streak/drawdown for a PL column.",
                 {"type": "object",
                  "properties": {"pl_column": {"type": "string"},
                                 "side": {"type": "string", "enum": ["back", "lay"]},
                                 "odds_column": {"type": "string"},
                                 "time_split_ratio": {"type": "number", "minimum": 0.5, "maximum": 0.95},
                                 "compute_streaks": {"type": "boolean"}},
                  "required": ["pl_column"]}),
    _tool_schema("strategy_performance_batch", "Batch performance summaries.",
                 {"type": "object",
                  "properties": {"pl_columns": {"type": "array", "items": {"type": "string"}},
                                 "time_split_ratio": {"type": "number", "minimum": 0.5, "maximum": 0.95},
                                 "compute_streaks": {"type": "boolean"}},
                  "required": ["pl_columns"]}),

    # submit_job params OPTIONAL (prevents model failing by omission)
    _tool_schema("submit_job", "Submit heavy job to Supabase queue (Modal worker).",
                 {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type"]}),
    _tool_schema("get_job", "Fetch job status for job_id.",
                 {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}),
    _tool_schema("download_result", "Download JSON result from Supabase Storage path.",
                 {"type": "object", "properties": {"result_path": {"type": "string"}}, "required": ["result_path"]}),
]


SYSTEM_PROMPT = """You are Football Researcher, an autonomous research agent.

Mission:
- Find explicit strategy criteria (filters/ranges) that generalise to future matches and produce profit.
- Apply anti-overfitting rules (time split, minimum sample sizes, simple rules first).
- Never use outcome columns (PL, RETURN, BET RESULT, etc.) as predictive features.

Tools:
- Google Sheets: definitions/rules/framework + persistent research notes/state
- Evaluation: ROI (back=PL per 1pt bet, lay=PL per total liability using mapped odds column), plus game-level streak/drawdown in points
- Offload heavy compute: submit_job -> poll get_job -> download_result (Modal worker)

Always log meaningful conclusions using append_research_note and maintain progress in research_state.
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
            return {"error": f"Missing required args for {name}: {missing_required}", "provided_args": list(args.keys())}

        return fn(**accepted)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tool": name, "args": args}

def _minimal_tool_call_dict(tc: Any) -> Dict[str, Any]:
    # Strict minimal shape OpenAI expects in messages history
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments or "{}",
        },
    }

def _chat_with_tools(user_text: str, max_rounds: int = 6) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    st.session_state.messages.append({"role": "user", "content": user_text})

    for _ in range(max_rounds):
        try:
            resp = _call_llm(st.session_state.messages)
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"OpenAI request failed: {type(e).__name__}: {e}"})
            return

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [_minimal_tool_call_dict(tc) for tc in tool_calls],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")

                out = _run_tool(tool_name, tool_args)

                # Tool response MUST be: role=tool + tool_call_id + content
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
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
        with st.expander("🛠 Tool output", expanded=False):
            st.code(m.get("content", ""), language="json")
        continue
    with st.chat_message(m["role"]):
        st.write(m.get("content", ""))

# Chat input
user_msg = st.chat_input("Ask the researcher…")
if user_msg:
    _chat_with_tools(user_msg)
    st.rerun()

# Debug section
with st.expander("Debug / Quick tests", expanded=False):
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Test: list_columns"):
            _chat_with_tools("Call list_columns and show the result.")
            st.rerun()

    with c2:
        if st.button("Test: submit ping job (DIRECT)"):
            # bypass model so we ALWAYS get a job_id
            out = _run_tool("submit_job", {"task_type": "ping", "params": {"hello": "world"}})
            st.session_state.messages.append({"role": "assistant", "content": "Submitted ping job (direct tool call)."})
            st.session_state.messages.append({"role": "tool", "tool_call_id": "debug_submit_job", "content": json.dumps(out, ensure_ascii=False)})
            st.rerun()

    with c3:
        if st.button("Clear chat"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()
