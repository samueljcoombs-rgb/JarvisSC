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


def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


client = _init_client()

PREFERRED = (os.getenv("PREFERRED_OPENAI_MODEL") or st.secrets.get("PREFERRED_OPENAI_MODEL") or "").strip()
MODEL = PREFERRED or "gpt-5.2-thinking"

DEFAULT_STORAGE_BUCKET = os.getenv("DATA_STORAGE_BUCKET") or st.secrets.get("DATA_STORAGE_BUCKET", "football-data")
DEFAULT_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH") or st.secrets.get("DATA_STORAGE_PATH", "football_ai_NNIA.csv")
DEFAULT_RESULTS_BUCKET = os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET", "football-results")


def _dataset_locator() -> Dict[str, str]:
    return {"storage_bucket": DEFAULT_STORAGE_BUCKET, "storage_path": DEFAULT_STORAGE_PATH}


def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**args)


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
- Log significant findings into research_memory with structured JSON.
- Be decisive: choose what to test next; do not ask the user what to do unless blocked by missing config.
"""


st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Football Researcher")

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


def _call_llm(messages: List[Dict[str, Any]]):
    # keep tool calling enabled
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tool_choice="auto",
    )


def _structured_note(worker_result: Dict[str, Any], job_id: str) -> str:
    payload = {
        "ts": datetime.utcnow().isoformat(),
        "kind": "strategy_search_result",
        "job_id": job_id,
        "picked": worker_result.get("result", {}).get("picked"),
        "top_rules": (worker_result.get("result", {}).get("search", {}).get("top_rules") or [])[:3],
        "note": "Stored top rules + stability metrics.",
    }
    return json.dumps(payload, ensure_ascii=False)


def _autopilot_one_cycle():
    params = {
        "storage_bucket": DEFAULT_STORAGE_BUCKET,
        "storage_path": DEFAULT_STORAGE_PATH,
        "_results_bucket": DEFAULT_RESULTS_BUCKET,
        "time_split_ratio": 0.7,
    }

    submitted = _run_tool("submit_job", {"task_type": "strategy_search", "params": params})
    st.success("Autopilot: submitted strategy_search.")
    st.json(submitted)

    job_id = submitted.get("job_id")
    if not job_id:
        st.error("No job_id returned.")
        return

    waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 900, "poll_s": 5, "auto_download": True})
    st.info("Autopilot: wait_for_job output.")
    st.json(waited)

    # ✅ DO NOT LET GOOGLE SHEETS WRITE CRASH THE APP
    state_out = _run_tool("set_research_state", {"key": "last_autopilot_ran_at", "value": datetime.utcnow().isoformat()})
    if not state_out.get("ok", True):
        st.warning(f"research_state write failed (non-fatal): {state_out}")

    if waited.get("status") != "done":
        st.warning("Job not done yet (or error).")
        return

    worker_result = waited.get("result") or {}
    # store structured memory (also non-fatal)
    note = _structured_note(worker_result, job_id)
    mem_out = _run_tool("append_research_note", {"note": note, "tags": "autopilot,worker,structured"})
    if not mem_out.get("ok", True):
        st.warning(f"research_memory write failed (non-fatal): {mem_out}")
    else:
        st.success("Saved structured research_memory row.")


with st.sidebar:
    st.caption(f"Model: `{MODEL}`")
    st.caption(f"Session: `{SESSION_ID}`")
    st.caption(f"Data: `{DEFAULT_STORAGE_BUCKET}/{DEFAULT_STORAGE_PATH}`")
    st.caption(f"Results bucket: `{DEFAULT_RESULTS_BUCKET}`")

    if st.button("Run 1 autopilot cycle"):
        _autopilot_one_cycle()

    st.divider()
    if st.button("Recent research_memory"):
        st.json(_run_tool("get_recent_research_notes", {"limit": 10}))


st.subheader("💬 Chat")
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m.get("content", ""))

user_msg = st.chat_input("Ask the researcher…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    _persist_chat()
    st.rerun()
