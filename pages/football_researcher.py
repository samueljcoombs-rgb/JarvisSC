# pages/football_researcher.py
from datetime import datetime
from __future__ import annotations

import json
import os
import inspect
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI

from modules import football_tools as functions


def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY.")
        st.stop()
    return OpenAI(api_key=api_key)

client = _init_client()


def _select_best_model(c: OpenAI) -> str:
    # Prefer GPT-5.2 Thinking explicitly, then GPT-5.2, then fallbacks
    preferred = (os.getenv("PREFERRED_OPENAI_MODEL", "").strip()
                 or st.secrets.get("PREFERRED_OPENAI_MODEL", "")).strip()
    if preferred:
        return preferred

    try:
        names = {m.id for m in c.models.list().data}
        for candidate in [
            "gpt-5.2-thinking",
            "gpt-5.2",
            "gpt-5",
            "gpt-latest",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4.1-mini",
        ]:
            if candidate in names:
                return candidate
    except Exception:
        pass

    # final fallback
    return "gpt-4o"


MODEL = _select_best_model(client)


TOOL_FUNCS: Dict[str, Any] = {
    "get_dataset_overview": functions.get_dataset_overview,
    "get_column_definitions": functions.get_column_definitions,
    "get_research_rules": functions.get_research_rules,
    "get_evaluation_framework": functions.get_evaluation_framework,

    "append_research_note": functions.append_research_note,
    "get_recent_research_notes": functions.get_recent_research_notes,
    "get_research_state": functions.get_research_state,
    "set_research_state": functions.set_research_state,

    "load_data_basic": functions.load_data_basic,
    "list_columns": functions.list_columns,

    "strategy_performance_summary": functions.strategy_performance_summary,
    "strategy_performance_batch": functions.strategy_performance_batch,

    "submit_job": functions.submit_job,
    "get_job": functions.get_job,
    "download_result": functions.download_result,
    "wait_for_job": functions.wait_for_job,
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

    _tool_schema("submit_job", "Submit heavy job to Supabase queue (Modal worker).",
                 {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]}),
    _tool_schema("get_job", "Fetch job status for job_id.",
                 {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}),
    _tool_schema("download_result", "Download JSON result from Supabase Storage path.",
                 {"type": "object", "properties": {"result_path": {"type": "string"}}, "required": ["result_path"]}),
    _tool_schema("wait_for_job", "Poll job until done/error/timeout. Optionally auto-download result.",
                 {"type": "object",
                  "properties": {"job_id": {"type": "string"},
                                 "timeout_s": {"type": "integer"},
                                 "poll_s": {"type": "integer"},
                                 "auto_download": {"type": "boolean"}},
                  "required": ["job_id"]}),
]


SYSTEM_PROMPT = """You are Football Researcher, an autonomous research agent.

Mission:
- Find explicit strategy criteria (filters/ranges) that generalise to future matches and produce profit.
- Apply anti-overfitting rules (time split, minimum sample sizes, simple rules first).
- Never use outcome columns (PL, RETURN, BET RESULT, etc.) as predictive features.
- When offloading compute: submit_job -> wait_for_job -> download_result.
- Always log meaningful conclusions (append_research_note) and keep progress (set_research_state).
"""


def _sanitize_history(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    last_assistant_had_tool_calls = False
    for m in msgs:
        role = m.get("role")
        if role == "assistant":
            last_assistant_had_tool_calls = bool(m.get("tool_calls"))
            cleaned.append(m)
            continue
        if role == "tool":
            if last_assistant_had_tool_calls and m.get("tool_call_id"):
                cleaned.append(m)
            continue
        cleaned.append(m)
        last_assistant_had_tool_calls = False
    return cleaned


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


def _chat_with_tools(user_text: str, max_rounds: int = 6) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.messages = _sanitize_history(st.session_state.messages)
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
                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}} for tc in tool_calls],
            })

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")
                out = _run_tool(tool_name, tool_args)
                st.session_state.messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)})

            continue

        st.session_state.messages.append({"role": "assistant", "content": msg.content or ""})
        break


def _autopilot_one_cycle():
    """
    One autonomous loop:
      - submit strategy_search job (no market specified; worker picks)
      - wait + download
      - log note + update research_state
    """
    csv_url = os.getenv("DATA_CSV_URL") or st.secrets.get("DATA_CSV_URL", "")
    if not csv_url:
        st.session_state.messages.append({"role": "assistant", "content": "Missing DATA_CSV_URL in Streamlit secrets."})
        return

    # submit
    submit_out = _run_tool("submit_job", {
        "task_type": "strategy_search",
        "params": {
            "csv_url": csv_url,
            "time_split_ratio": 0.7,
            "_results_bucket": "football-results",
        }
    })
    st.session_state.messages.append({"role": "assistant", "content": "Autopilot: submitted strategy_search.\n```json\n" + json.dumps(submit_out, indent=2) + "\n```"})

    job_id = submit_out.get("job_id")
    if not job_id:
        return

    # wait
    waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 240, "poll_s": 3, "auto_download": True})
    st.session_state.messages.append({"role": "assistant", "content": "Autopilot: wait_for_job output.\n```json\n" + json.dumps(waited, indent=2) + "\n```"})

    download = (waited.get("download") or {})
    result = (download.get("result") or {})
    payload = result.get("result") or {}

    # log note (clean summary)
    picked = payload.get("picked") or {}
    search = payload.get("search") or {}
    top = (search.get("top_rules") or [])[:5]

    note_lines = []
    note_lines.append("Autopilot cycle: strategy_search")
    if picked:
        note_lines.append(f"Picked market: {picked.get('pl_column')} (side={picked.get('side')}, odds_col={picked.get('odds_col')}).")
        note_lines.append(f"Picked by best test ROI among mapped markets (current baseline).")
    if search.get("error"):
        note_lines.append(f"Search error: {search.get('error')}")
    else:
        note_lines.append(f"Searched rules tried: {search.get('searched_rules')}")
        note_lines.append("Top candidate criteria (first 5):")
        for idx, r in enumerate(top, start=1):
            rule = r.get("rule", [])
            tr = r.get("train", {})
            te = r.get("test", {})
            gl = r.get("test_game_level", {})
            note_lines.append(
                f"{idx}) {rule} | test_roi={te.get('roi'):.4f} | train_roi={tr.get('roi'):.4f} | "
                f"gap={r.get('gap_train_minus_test'):.4f} | test_bets={te.get('bets')} | "
                f"test_dd={gl.get('max_dd')} | test_ls_bets={((gl.get('losing_streak') or {}).get('bets'))}"
            )

    note = "\n".join(note_lines)
    _run_tool("append_research_note", {"note": note, "tags": ["autopilot", "strategy_search"]})
    _run_tool("set_research_state", {"key": "last_autopilot_job_id", "value": str(job_id)})
    _run_tool("set_research_state", {"key": "last_autopilot_ran_at", "value": datetime.utcnow().isoformat()})


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Football Researcher (Autonomous)")
st.caption(f"Model: {MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
st.session_state.messages = _sanitize_history(st.session_state.messages)

for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    if m["role"] == "tool":
        with st.expander("🛠 Tool output", expanded=False):
            st.code(m.get("content", ""), language="json")
        continue
    with st.chat_message(m["role"]):
        st.write(m.get("content", ""))

user_msg = st.chat_input("Ask the researcher…")
if user_msg:
    _chat_with_tools(user_msg)
    st.rerun()

with st.expander("Autopilot", expanded=False):
    st.write("Runs one autonomous cycle: submit heavy strategy search, wait, download, log conclusions.")
    if st.button("🤖 Autopilot: run 1 cycle"):
        _autopilot_one_cycle()
        st.rerun()

with st.expander("Debug / Quick tests", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Test: list_columns"):
            _chat_with_tools("Call list_columns and show the result.")
            st.rerun()
    with c2:
        if st.button("Clear chat"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()
