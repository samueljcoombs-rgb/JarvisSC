from __future__ import annotations

import os
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Set, Optional

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


# ============================================================
# Tool runner
# ============================================================

def _run_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = getattr(functions, name, None)
    if not fn:
        raise RuntimeError(f"Unknown tool: {name}")
    return fn(**args)


# ============================================================
# LLM Tools (ONLY what the model should call)
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

    # SAFE job wrappers
    {"type": "function", "function": {"name": "submit_strategy_search", "description": "Submit strategy_search safely.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "results_bucket": {"type": "string"}, "time_split_ratio": {"type": "number"}, "target_pl_column": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "submit_feature_audit", "description": "Submit feature_audit safely.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "results_bucket": {"type": "string"}, "target_pl_column": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "submit_feature_rank", "description": "Submit feature_rank safely.", "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "results_bucket": {"type": "string"}, "target_pl_column": {"type": "string"}, "time_split_ratio": {"type": "number"}, "max_rows": {"type": "integer"}}, "required": ["target_pl_column"]}}},

    {"type": "function", "function": {"name": "get_job", "description": "Get job status by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for completion; optionally downloads results.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "download_result", "description": "Download a result JSON from storage by path.", "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}, "bucket": {"type": "string"}}, "required": ["result_path"]}}},
]


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """You are FootballResearcher — an autonomous research agent that discovers profitable, robust football trading strategy criteria.

Source of truth:
- Google Sheet tabs: dataset_overview, research_rules, column_definitions, evaluation_framework, research_state, research_memory.

Hard constraints:
- PL columns are outcomes only and MUST NOT be used as predictive features.
- Avoid overfitting: time-based splits; never tune thresholds on final test.
- Always report sample sizes, train vs test gap, drawdown + losing streak in POINTS.
- Prefer simple rules that generalise; penalise fragile, tiny samples.

Mode:
- CHAT = conversational only (no tools unless user explicitly asks).
- AUTOPILOT = tools/jobs allowed, keep outputs concrete.

When asked what model you are, do NOT guess; rely on sidebar “OpenAI returned model”.
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
# Context snapshot injection (AUTOPILOT)
# ============================================================

def _build_context_snapshot(max_notes: int = 8) -> str:
    try:
        overview = _run_tool("get_dataset_overview", {}) or {}
        rules = _run_tool("get_research_rules", {}) or {}
        evalfw = _run_tool("get_evaluation_framework", {}) or {}
        state = _run_tool("get_research_state", {}) or {}
        notes = _run_tool("get_recent_research_notes", {"limit": int(max_notes)}) or {}
    except Exception as e:
        return f"[context_snapshot_error] {e}"

    snap = {
        "dataset_overview": overview.get("data", overview),
        "research_rules": rules.get("data", rules),
        "evaluation_framework": evalfw.get("data", evalfw),
        "research_state": state.get("data", state),
        "recent_research_memory": notes.get("rows", notes),
    }
    s = json.dumps(snap, ensure_ascii=False)
    return s if len(s) <= 9000 else (s[:9000] + "…")


def _autopilot_system_prompt() -> str:
    now = datetime.utcnow().timestamp()
    cached = st.session_state.get("_ctx_snapshot")
    cached_at = st.session_state.get("_ctx_snapshot_at", 0.0)
    if cached and (now - float(cached_at)) < 180:
        snap = cached
    else:
        snap = _build_context_snapshot(max_notes=10)
        st.session_state["_ctx_snapshot"] = snap
        st.session_state["_ctx_snapshot_at"] = now

    return SYSTEM_PROMPT + "\n\nAUTOPILOT_CONTEXT_SNAPSHOT_JSON:\n" + snap


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
            clean_assistant: Dict[str, Any] = {"role": "assistant", "content": m.get("content", "") or ""}

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
                            cleaned_tool_calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
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

        return client.chat.completions.create(model=MODEL, messages=chat_only)

    # AUTOPILOT: inject sheet snapshot
    safe_messages[0]["content"] = _autopilot_system_prompt()
    return client.chat.completions.create(model=MODEL, messages=safe_messages, tools=TOOLS, tool_choice="auto")


# ============================================================
# Pipeline router: "build strategy for <X PL>"
# ============================================================

def _extract_pl_column(user_text: str) -> Optional[str]:
    m = re.search(r"([A-Za-z0-9\.\s]+?\sPL)\b", user_text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _wait_and_download(job_id: str, bucket: str) -> Dict[str, Any]:
    waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 1800, "poll_s": 5, "auto_download": True})
    if waited.get("status") != "done":
        return {"error": f"Job not done: {waited.get('status')}", "waited": waited}
    job = waited.get("job") or {}
    rp = job.get("result_path") or ""
    result = waited.get("result")
    if not result and rp:
        dl = _run_tool("download_result", {"result_path": rp, "bucket": bucket})
        result = dl.get("result")
    return {"job": job, "result": result}


def _summarise_feature_audit(payload: Dict[str, Any]) -> str:
    if not payload or not isinstance(payload, dict):
        return "feature_audit: no payload"
    res = payload.get("result") if "result" in payload else payload
    inner = res.get("result") if isinstance(res, dict) and "result" in res else res
    if isinstance(inner, dict) and inner.get("error"):
        return f"feature_audit error: {inner['error']}"
    if not isinstance(inner, dict):
        return "feature_audit: malformed result"
    lines = []
    lines.append("### 1) Feature audit")
    lines.append(f"- Rows/Cols: **{inner.get('rows')}** / **{inner.get('cols')}**")
    lines.append(f"- Target PL resolved: `{inner.get('target_pl_column_resolved')}`")
    lines.append(f"- PL columns found: **{len(inner.get('pl_columns') or [])}**")
    mt = inner.get("missingness_top") or []
    if mt:
        lines.append("- Top missingness:")
        for r in mt[:8]:
            lines.append(f"  - {r.get('col')}: {round(float(r.get('missing_pct',0))*100,1)}%")
    return "\n".join(lines)


def _summarise_feature_rank(payload: Dict[str, Any]) -> str:
    if not payload or not isinstance(payload, dict):
        return "feature_rank: no payload"
    res = payload.get("result") if "result" in payload else payload
    inner = res.get("result") if isinstance(res, dict) and "result" in res else res
    if isinstance(inner, dict) and inner.get("error"):
        return f"feature_rank error: {inner['error']}"
    if not isinstance(inner, dict):
        return "feature_rank: malformed result"
    m = inner.get("metrics") or {}
    lines = []
    lines.append("### 2) Feature ranking (Logit + Permutation)")
    lines.append(f"- Target: `{inner.get('target_pl_column')}` (binary: PL>0)")
    lines.append(f"- Test ROC AUC: **{m.get('test_roc_auc')}**, LogLoss: **{m.get('test_logloss')}**, Brier: **{m.get('test_brier')}**")
    lines.append(f"- Features used: **{m.get('n_features')}** | Train rows: **{m.get('train_rows')}** | Test rows: **{m.get('test_rows')}**")
    top = inner.get("top_features_permutation_auc") or []
    if top:
        lines.append("- Top permutation features (AUC impact):")
        for r in top[:12]:
            lines.append(f"  - {r.get('feature')}: {round(float(r.get('importance_mean',0)),6)}")
    return "\n".join(lines)


def _summarise_strategy_search(payload: Dict[str, Any]) -> str:
    if not payload or not isinstance(payload, dict):
        return "strategy_search: no payload"
    res = payload.get("result") if "result" in payload else payload
    inner = res.get("result") if isinstance(res, dict) and "result" in res else res
    if isinstance(inner, dict) and inner.get("error"):
        return f"strategy_search error: {inner['error']}"
    if not isinstance(inner, dict):
        return "strategy_search: malformed result"

    picked = inner.get("picked") or {}
    search = inner.get("search") or {}
    if isinstance(search, dict) and search.get("error"):
        return f"Picked: `{picked}`\n\nSearch error: {search.get('error')}"

    rules = (search.get("top_rules") or [])[:3]
    lines = []
    lines.append("### 3) Rule search (top 3)")
    lines.append(f"- Picked: `{picked}`")
    if not rules:
        lines.append("- No rules returned.")
        return "\n".join(lines)

    for i, r in enumerate(rules, start=1):
        lines.append(f"\n**Rule #{i}**")
        lines.append(f"- Rule: `{r.get('rule')}`")
        lines.append(f"- Samples: `{r.get('samples')}`")
        lines.append(f"- Train: `{r.get('train')}`")
        lines.append(f"- Test: `{r.get('test')}`")
        lines.append(f"- Gap train−test: `{r.get('gap_train_minus_test')}`")
        lines.append(f"- Test game-level risk: `{r.get('test_game_level')}`")
    return "\n".join(lines)


def _run_btts_pipeline(pl_column: str, split_ratio: float = 0.7) -> str:
    # 1) audit
    a = _run_tool("submit_feature_audit", {
        "storage_bucket": DEFAULT_STORAGE_BUCKET,
        "storage_path": DEFAULT_STORAGE_PATH,
        "results_bucket": DEFAULT_RESULTS_BUCKET,
        "target_pl_column": pl_column,
    })
    aj = a.get("job_id") if isinstance(a, dict) else None
    if not aj:
        return f"feature_audit submit failed: {a}"
    ares = _wait_and_download(aj, DEFAULT_RESULTS_BUCKET)
    if ares.get("error"):
        return f"feature_audit failed: {ares}"

    # 2) rank
    r = _run_tool("submit_feature_rank", {
        "storage_bucket": DEFAULT_STORAGE_BUCKET,
        "storage_path": DEFAULT_STORAGE_PATH,
        "results_bucket": DEFAULT_RESULTS_BUCKET,
        "target_pl_column": pl_column,
        "time_split_ratio": float(split_ratio),
        "max_rows": 250000,
    })
    rj = r.get("job_id") if isinstance(r, dict) else None
    if not rj:
        return f"feature_rank submit failed: {r}"
    rres = _wait_and_download(rj, DEFAULT_RESULTS_BUCKET)
    if rres.get("error"):
        return f"feature_rank failed: {rres}"

    # 3) strategy search
    s = _run_tool("submit_strategy_search", {
        "storage_bucket": DEFAULT_STORAGE_BUCKET,
        "storage_path": DEFAULT_STORAGE_PATH,
        "results_bucket": DEFAULT_RESULTS_BUCKET,
        "time_split_ratio": float(split_ratio),
        "target_pl_column": pl_column,
    })
    sj = s.get("job_id") if isinstance(s, dict) else None
    if not sj:
        return f"strategy_search submit failed: {s}"
    sres = _wait_and_download(sj, DEFAULT_RESULTS_BUCKET)
    if sres.get("error"):
        return f"strategy_search failed: {sres}"

    # Log structured note
    note = json.dumps({
        "ts": datetime.utcnow().isoformat(),
        "kind": "btts_pipeline_run",
        "target_pl": pl_column,
        "time_split_ratio": float(split_ratio),
        "jobs": {"feature_audit": aj, "feature_rank": rj, "strategy_search": sj},
    }, ensure_ascii=False)
    _run_tool("append_research_note", {"note": note, "tags": "autopilot,pipeline,btts"})

    parts = []
    parts.append(_summarise_feature_audit(ares.get("result") or {}))
    parts.append(_summarise_feature_rank(rres.get("result") or {}))
    parts.append(_summarise_strategy_search(sres.get("result") or {}))
    return "\n\n".join(parts)


# ============================================================
# Chat loop
# ============================================================

def _chat_with_tools(user_text: str, max_rounds: int = 6):
    # AUTOPILOT shortcut: if user asks for a PL strategy, run the pipeline deterministically
    if st.session_state.agent_mode == "autopilot":
        pl = _extract_pl_column(user_text)
        if pl:
            summary = _run_btts_pipeline(pl_column=pl, split_ratio=0.7)
            st.session_state.messages.append({"role": "user", "content": user_text})
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.messages = _trim_messages(st.session_state.messages)
            _persist_chat()
            return

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages = _trim_messages(st.session_state.messages)

    for _ in range(max_rounds):
        try:
            resp = _call_llm(st.session_state.messages)
            st.session_state.last_openai_model_used = getattr(resp, "model", None)
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
            if st.session_state.agent_mode == "chat":
                st.session_state.messages.append({"role": "assistant", "content": "I’m here — what do you want to do next (strategy, debugging, or analysis)?"})
            else:
                st.session_state.messages.append(assistant_msg)

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
    st.caption(f"Requested MODEL var: `{MODEL}`")
    st.caption(f"Data: `{DEFAULT_STORAGE_BUCKET}/{DEFAULT_STORAGE_PATH}`")
    st.caption(f"Results: `{DEFAULT_RESULTS_BUCKET}`")

    used = st.session_state.get("last_openai_model_used")
    if used:
        st.caption(f"OpenAI returned model: `{used}`")
    else:
        st.caption("OpenAI returned model: `(no call yet)`")

    st.divider()
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
