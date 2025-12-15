# pages/football_researcher.py
from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

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
MODEL = PREFERRED or "gpt-5.2-thinking"  # if account lacks access, OpenAI will error: then set PREFERRED_OPENAI_MODEL


# =========================
# Tool registry (OpenAI tool calling)
# =========================
TOOLS = [
    {"type": "function", "function": {"name": "get_dataset_overview", "description": "Get dataset overview from Google Sheet.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_research_rules", "description": "Get research rules from Google Sheet.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_column_definitions", "description": "Get column definitions table from Google Sheet.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_evaluation_framework", "description": "Get evaluation framework from Google Sheet.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_recent_research_notes", "description": "Get last N research notes from research_memory.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}}, "required": []}}},
    {"type": "function", "function": {"name": "append_research_note", "description": "Append a research note row to research_memory sheet.", "parameters": {"type": "object", "properties": {"note": {"type": "string"}, "tags": {"type": "string"}}, "required": ["note"]}}},
    {"type": "function", "function": {"name": "get_research_state", "description": "Get persistent key/value research_state.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "set_research_state", "description": "Set persistent key/value research_state.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
    {"type": "function", "function": {"name": "load_data_basic", "description": "Load a preview of the dataset CSV (from Supabase storage or URL).", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "list_columns", "description": "List column names in the dataset CSV.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "basic_roi_for_pl_column", "description": "Basic row-level ROI for a PL column (uses correct lay/back stake rules if mapping exists).", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}, "storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": ["pl_column"]}}},
    {"type": "function", "function": {"name": "submit_job", "description": "Submit a background job for Modal worker (writes to Supabase jobs table).", "parameters": {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]}}},
    {"type": "function", "function": {"name": "get_job", "description": "Get a job record by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for job completion; optionally downloads result JSON.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "download_result", "description": "Download JSON result from Supabase Storage.", "parameters": {"type": "object", "properties": {"bucket": {"type": "string"}, "result_path": {"type": "string"}}, "required": ["result_path"]}}},
    {"type": "function", "function": {"name": "save_chat", "description": "Persist chat messages to Supabase Storage.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "messages": {"type": "array"}}, "required": ["session_id", "messages"]}}},
    {"type": "function", "function": {"name": "load_chat", "description": "Load chat session from Supabase Storage.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}}},
]


def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**args)


# =========================
# System prompt (autonomous)
# =========================
SYSTEM_PROMPT = """You are FootballResearcher — an autonomous research agent that discovers profitable, robust football trading strategy criteria.

Hard constraints:
- Use the Google Sheet tabs as your source of truth: dataset_overview, research_rules, column_definitions, evaluation_framework.
- PL columns are outcomes only and MUST NOT be used as predictive features.
- Avoid overfitting: use time splits; do not tune thresholds on final test.
- Always report sample sizes and stability (train vs test gap).
- Prefer simple rules that generalise.

Objective:
- Propose explicit strategy criteria usable on future matches (ranges + categorical filters).
- When needed, submit background jobs to the Modal worker for heavy evaluation.
- Log significant findings into research_memory with structured JSON (so we can learn and avoid repeats).
- Be decisive: choose what to test next; do not ask the user what to do unless blocked by missing config.
"""


# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Football Researcher")


# =========================
# Dataset location defaults (Supabase Storage)
# =========================
DEFAULT_STORAGE_BUCKET = os.getenv("DATA_STORAGE_BUCKET") or st.secrets.get("DATA_STORAGE_BUCKET", "football-data")
DEFAULT_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH") or st.secrets.get("DATA_STORAGE_PATH", "football_ai_NNIA.csv")
DEFAULT_RESULTS_BUCKET = os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET", "football-results")


def _dataset_locator() -> Dict[str, str]:
    return {"storage_bucket": DEFAULT_STORAGE_BUCKET, "storage_path": DEFAULT_STORAGE_PATH}


# =========================
# Chat persistence (Supabase Storage)
# =========================
if "football_session_id" not in st.session_state:
    st.session_state.football_session_id = str(uuid.uuid4())

SESSION_ID = st.session_state.football_session_id

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

if "loaded_from_storage" not in st.session_state:
    st.session_state.loaded_from_storage = True
    loaded = _run_tool("load_chat", {"session_id": SESSION_ID})
    if loaded.get("ok") and loaded.get("data", {}).get("messages"):
        st.session_state.messages = loaded["data"]["messages"]


def _persist_chat():
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

            # Tool messages must correspond to tool_calls
            st.session_state.messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)}
            )

        follow = _call_llm(st.session_state.messages)
        follow_msg = follow.choices[0].message
        st.session_state.messages.append({"role": "assistant", "content": follow_msg.content or "(no content)"})

    _persist_chat()


# =========================
# Result parsing + structured memory
# =========================
def _safe_get(d: Any, path: List[Any], default=None):
    cur = d
    for p in path:
        try:
            if isinstance(p, int) and isinstance(cur, list) and 0 <= p < len(cur):
                cur = cur[p]
            elif isinstance(p, str) and isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return default
        except Exception:
            return default
    return cur if cur is not None else default


def _render_top_rules(result_obj: Dict[str, Any]):
    """
    Expects worker result structure:
      result_obj["result"]["picked"], result_obj["result"]["search"]["top_rules"]
    """
    picked = _safe_get(result_obj, ["result", "picked"], {}) or {}
    search = _safe_get(result_obj, ["result", "search"], {}) or {}
    top_rules = search.get("top_rules") or []

    st.subheader("✅ Worker Result Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Picked PL column", str(picked.get("pl_column", "")))
    with c2:
        st.metric("Side", str(picked.get("side", "")))
    with c3:
        st.metric("Odds col", str(picked.get("odds_col", "")))
    with c4:
        st.metric("Rules searched", str(search.get("searched_rules", "")))

    st.divider()
    st.subheader("🏆 Top Rules (explicit criteria)")

    if not top_rules:
        st.info("No top_rules returned (worker may have found no stable rules under constraints).")
        return

    for idx, r in enumerate(top_rules[:3], start=1):
        rule_list = r.get("rule") or []
        train = r.get("train") or {}
        test = r.get("test") or {}
        gap = r.get("gap_train_minus_test")
        samples = r.get("samples") or {}
        game = r.get("test_game_level") or {}

        # human readable
        parts = []
        for cond in rule_list:
            col = cond.get("col")
            mn = cond.get("min")
            mx = cond.get("max")
            parts.append(f"{col} ∈ [{mn:.4g}, {mx:.4g}]")
        english = " AND ".join(parts) if parts else "(missing rule conditions)"

        with st.expander(f"Rule #{idx} — score={r.get('score', None)}", expanded=(idx == 1)):
            st.markdown(f"**Criteria:** {english}")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Train ROI", f"{train.get('roi', 0):.2%}" if isinstance(train.get("roi"), (int, float)) else str(train.get("roi")))
                st.caption(f"Train bets: {train.get('bets', '')}")
            with m2:
                st.metric("Test ROI", f"{test.get('roi', 0):.2%}" if isinstance(test.get("roi"), (int, float)) else str(test.get("roi")))
                st.caption(f"Test bets: {test.get('bets', '')}")
            with m3:
                st.metric("Gap (train-test)", f"{gap:.2%}" if isinstance(gap, (int, float)) else str(gap))
                st.caption(f"Train rows: {samples.get('train_rows','')}, Test rows: {samples.get('test_rows','')}")
            with m4:
                dd = game.get("max_dd", 0.0)
                ls = game.get("losing_streak", {}) or {}
                st.metric("Test Max DD (pts)", f"{dd:.2f}")
                st.caption(f"Losing streak: {ls.get('bets','')} bets, {ls.get('pl','')} pts")

            st.json(r)


def _structured_memory_payload(worker_result: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    picked = _safe_get(worker_result, ["result", "picked"], {}) or {}
    search = _safe_get(worker_result, ["result", "search"], {}) or {}
    top = (search.get("top_rules") or [])[:3]

    def _cond_to_dict(cond: Dict[str, Any]) -> Dict[str, Any]:
        return {"col": cond.get("col"), "min": cond.get("min"), "max": cond.get("max")}

    parsed_rules = []
    for r in top:
        parsed_rules.append({
            "score": r.get("score"),
            "rule": [_cond_to_dict(c) for c in (r.get("rule") or [])],
            "train": r.get("train"),
            "test": r.get("test"),
            "gap_train_minus_test": r.get("gap_train_minus_test"),
            "test_game_level": r.get("test_game_level"),
            "samples": r.get("samples"),
        })

    return {
        "ts": datetime.utcnow().isoformat(),
        "kind": "strategy_search_result",
        "job_id": job_id,
        "picked": {
            "pl_column": picked.get("pl_column"),
            "side": picked.get("side"),
            "odds_col": picked.get("odds_col"),
            "test_roi": picked.get("test_roi"),
            "test_bets": picked.get("test_bets"),
        },
        "searched_rules": search.get("searched_rules"),
        "top_rules": parsed_rules,
        "notes": "Structured memory: store top rules + stability metrics so autopilot can avoid repeats.",
    }


def _log_structured_memory(worker_result: Dict[str, Any], job_id: str, tags: str = "autopilot,worker,structured"):
    payload = _structured_memory_payload(worker_result, job_id)
    note_json = json.dumps(payload, ensure_ascii=False)
    _run_tool("append_research_note", {"note": note_json, "tags": tags})


# =========================
# Autopilot (1 cycle)
# =========================
def _autopilot_one_cycle():
    locator = _dataset_locator()
    params = {
        "storage_bucket": locator["storage_bucket"],
        "storage_path": locator["storage_path"],
        "_results_bucket": DEFAULT_RESULTS_BUCKET,
        "time_split_ratio": 0.7,
        # IMPORTANT: no market specified — worker chooses
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

    # waited["result"] should be the downloaded JSON
    worker_result = waited.get("result") or {}
    st.success("✅ Downloaded worker result. Rendering top rules...")
    _render_top_rules(worker_result)

    # Structured memory append
    try:
        _log_structured_memory(worker_result, job_id)
        st.success("🧠 Saved structured research_memory row.")
    except Exception as e:
        st.error(f"Failed to write structured memory: {e}")


# =========================
# UI
# =========================
with st.sidebar:
    st.caption(f"Model: `{MODEL}`")
    st.caption(f"Session: `{SESSION_ID}`")

    st.divider()
    st.subheader("📦 Dataset (Supabase Storage)")
    st.write(f"**Bucket:** `{DEFAULT_STORAGE_BUCKET}`")
    st.write(f"**Path:** `{DEFAULT_STORAGE_PATH}`")
    st.write(f"**Results bucket:** `{DEFAULT_RESULTS_BUCKET}`")

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
    if st.button("Recent research_memory"):
        st.json(_run_tool("get_recent_research_notes", {"limit": 20}))

    st.divider()
    st.subheader("📦 Data tools")
    if st.button("Load data basic"):
        st.json(_run_tool("load_data_basic", _dataset_locator()))
    if st.button("List CSV columns"):
        st.json(_run_tool("list_columns", _dataset_locator()))

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
            st.json(_run_tool("wait_for_job", {"job_id": job_id.strip(), "timeout_s": 180, "poll_s": 5, "auto_download": True}))


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
