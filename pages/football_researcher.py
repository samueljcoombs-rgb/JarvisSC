from __future__ import annotations

import os
import json
import uuid
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Set

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


def _sid() -> str:
    return st.session_state.session_id


# ============================================================
# Tools schema
# NOTE: every array must include `items` in JSON schema
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_research_context",
            "description": "Fetch all Google Sheet tabs + derived constraints (ignored_columns, outcome_columns, enforcement).",
            "parameters": {"type": "object", "properties": {"limit_notes": {"type": "integer"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_research_note",
            "description": "Append to research_memory.",
            "parameters": {"type": "object", "properties": {"note": {"type": "string"}, "tags": {"type": "string"}}, "required": ["note"]},
        },
    },
    {
        "type": "function",
        "function": {"name": "get_research_state", "description": "Get research_state KV.", "parameters": {"type": "object", "properties": {}, "required": []}},
    },
    {
        "type": "function",
        "function": {
            "name": "set_research_state",
            "description": "Set research_state KV.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_data_basic",
            "description": "Load CSV preview.",
            "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List CSV columns.",
            "parameters": {"type": "object", "properties": {"storage_bucket": {"type": "string"}, "storage_path": {"type": "string"}, "csv_url": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_job",
            "description": "Submit Modal worker job.",
            "parameters": {"type": "object", "properties": {"task_type": {"type": "string"}, "params": {"type": "object"}}, "required": ["task_type", "params"]},
        },
    },
    {
        "type": "function",
        "function": {"name": "get_job", "description": "Get job status by job_id.", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_job",
            "description": "Wait for completion; optionally downloads results.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}, "poll_s": {"type": "integer"}, "auto_download": {"type": "boolean"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_result",
            "description": "Download a result JSON from storage by path.",
            "parameters": {"type": "object", "properties": {"result_path": {"type": "string"}, "bucket": {"type": "string"}}, "required": ["result_path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_pl_lab",
            "description": "Start best-practice ML lab for any PL column (Sheet rules enforced; categoricals included; distills explicit rules).",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_minutes": {"type": "integer"},
                    "pl_column": {"type": "string"},
                    "do_hyperopt": {"type": "boolean"},
                    "hyperopt_iter": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_btts_lab",
            "description": "Convenience wrapper around start_pl_lab for BTTS PL.",
            "parameters": {"type": "object", "properties": {"duration_minutes": {"type": "integer"}}, "required": []},
        },
    },
    # NEW: Small primitives for autonomous iteration
    {
        "type": "function",
        "function": {
            "name": "start_feature_profile",
            "description": "Start a feature profiling job (missingness, cardinality, time-window stability summary).",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_minutes": {"type": "integer"},
                    "pl_column": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_rule_validate",
            "description": "Validate a proposed explicit rule spec (numeric ranges + categorical constraints) on train/val/test with risk + stability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_minutes": {"type": "integer"},
                    "pl_column": {"type": "string"},
                    "spec": {"type": "object"},
                },
                "required": ["spec"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "list_chats", "description": "List saved chat sessions.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []}},
    },
    {
        "type": "function",
        "function": {
            "name": "save_chat",
            "description": "Save chat session.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}, "messages": {"type": "array", "items": {"type": "object"}}, "title": {"type": "string"}},
                "required": ["session_id", "messages"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "load_chat", "description": "Load chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
    },
    {
        "type": "function",
        "function": {
            "name": "rename_chat",
            "description": "Rename chat session.",
            "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}, "title": {"type": "string"}}, "required": ["session_id", "title"]},
        },
    },
    {
        "type": "function",
        "function": {"name": "delete_chat", "description": "Delete chat session.", "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}},
    },
]


SYSTEM_PROMPT = """You are FootballResearcher — an autonomous strategy R&D agent.

NON-NEGOTIABLE SOURCE OF TRUTH (THE BIBLE)
- You MUST use Google Sheet tabs via get_research_context():
  dataset_overview, research_rules, column_definitions, evaluation_framework, research_state, research_memory.
- Treat research_rules as ENFORCEMENT rules, not suggestions.
- You MUST read recent research_memory before starting new analysis.
- You MUST use derived.enforcement limits (min samples, max gaps, risk limits) when choosing/validating strategies.

MISSION / PURPOSE
- Discover strategies that are REPEATABLE on future matches.
- The dataset is POST-SCAN. Each row = the historical outcome of a scan condition, NOT a match/bet.
- PL columns are OUTCOMES ONLY: never use PL columns (or proxies) as predictive features.
- NO GAMES is an aggregated count: must NOT be summed across rows; treat carefully.
- You are judged on out-of-sample stability and risk (points):
  - P&L and ROI
  - max drawdown and longest losing streak (by ID aggregation)
- Output strategies as explicit filters:
  - Numeric ranges + categorical constraints (MODE/MARKET/DRIFT etc.)
  - With minimum sample sizes on train/val/test and unique IDs where possible

VALIDATION DISCIPLINE
- Always time-split: train on earlier data; validate on later; final test untouched.
- Never tune thresholds using test.
- Prefer simple, stable strategies over complex, fragile ones.
- Log significant findings using append_research_note.
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
    out = _run_tool("save_chat", {"session_id": _sid(), "messages": st.session_state.messages, "title": title})
    if isinstance(out, dict) and out.get("ok") is False:
        st.session_state.last_chat_save_error = out


def _try_load_chat(sid: str) -> bool:
    loaded = _run_tool("load_chat", {"session_id": sid})
    if loaded.get("ok") and loaded.get("data", {}).get("messages"):
        st.session_state.messages = _trim_messages(loaded["data"]["messages"])
        return True
    return False


if "session_id" not in st.session_state:
    qp = st.query_params.get("sid")
    st.session_state.session_id = qp if qp else _new_session_id()

_init_messages_if_needed()

if "loaded_for_sid" not in st.session_state or st.session_state.loaded_for_sid != _sid():
    st.session_state.loaded_for_sid = _sid()
    if not _try_load_chat(_sid()):
        _persist_chat(title=f"Session {_sid()[:8]}")

if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "chat"


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
                    cid = call.get("id")
                    fn = call.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments", "{}")
                    if cid and name:
                        cleaned_tool_calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
                        expecting_tool_ids.add(cid)
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
# LLM calls
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

    return client.chat.completions.create(model=MODEL, messages=safe_messages, tools=TOOLS, tool_choice="auto")


def _call_llm_tools(messages: List[Dict[str, Any]]):
    safe_messages = _sanitize_history_for_llm(messages)
    return client.chat.completions.create(model=MODEL, messages=safe_messages, tools=TOOLS, tool_choice="auto")


def _call_llm_no_tools(messages: List[Dict[str, Any]]):
    safe_messages = _sanitize_history_for_llm(messages)
    return client.chat.completions.create(model=MODEL, messages=safe_messages, tools=TOOLS, tool_choice="none")


# ============================================================
# Parsing helpers
# ============================================================
def _minutes_from_text(t: str, default_minutes: int = 30) -> int:
    """
    Supports:
      - "30", "for 30", "30m", "30 min", "30 minutes"
      - "2h", "2 hours", "next 2 hours"
      - "0.5h"
    """
    s = (t or "").lower().strip()

    # hours (float)
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)\b", s)
    if m:
        mins = int(round(float(m.group(1)) * 60))
        return max(5, min(mins, 360))

    # minutes
    m2 = re.search(r"\b(\d+)\s*(m|min|mins|minute|minutes)\b", s)
    if m2:
        mins = int(m2.group(1))
        return max(5, min(mins, 360))

    # "for 30" (assume minutes)
    m3 = re.search(r"\bfor\s+(\d{1,3})\b", s)
    if m3:
        mins = int(m3.group(1))
        return max(5, min(mins, 360))

    # bare "30" anywhere (only if it's the ONLY number-ish thing)
    nums = re.findall(r"\b(\d{1,3})\b", s)
    if len(nums) == 1:
        mins = int(nums[0])
        if 5 <= mins <= 360:
            return mins

    return max(5, min(int(default_minutes), 360))


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\(\)\[\]\{\}\,\;\:]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _resolve_pl_column(user_text: str, ctx: Dict[str, Any]) -> str:
    t = _normalize(user_text)
    outcomes: List[str] = ((ctx.get("derived") or {}).get("outcome_columns") or [])

    aliases = {
        "btts": "BTTS PL",
        "both teams to score": "BTTS PL",
        "over 2.5": "BO 2.5 PL",
        "o2.5": "BO 2.5 PL",
        "bo 2.5": "BO 2.5 PL",
        "fh over 1.5": "BO1.5 FHG PL",
        "shg": "SHG PL",
        "lu1.5": "LU1.5 PL",
        "lfghu0.5": "LFGHU0.5 PL",
    }
    for k, v in aliases.items():
        if k in t:
            return v

    def nn(x: str) -> str:
        return _normalize(x).replace(" ", "")

    t_compact = t.replace(" ", "")
    for col in outcomes:
        if nn(col) in t_compact:
            return col

    return "BTTS PL"


# ============================================================
# Rendering distilled rules (unchanged)
# ============================================================
def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def _render_rule_spec(spec: Dict[str, Any]) -> str:
    parts: List[str] = []

    for c in (spec.get("categorical") or []):
        col = c.get("col")
        if not col:
            continue
        if "in" in c and c["in"]:
            parts.append(f"**{col} IN** {c['in']}")
        if "not_in" in c and c["not_in"]:
            parts.append(f"**{col} NOT IN** {c['not_in']}")

    for n in (spec.get("numeric") or []):
        col = n.get("col")
        if not col:
            continue
        mn = n.get("min", None)
        mx = n.get("max", None)
        if mn is not None and mx is not None:
            parts.append(f"**{col}** in [{_fmt_num(mn)}, {_fmt_num(mx)}]")
        elif mn is not None:
            parts.append(f"**{col}** >= {_fmt_num(mn)}")
        elif mx is not None:
            parts.append(f"**{col}** <= {_fmt_num(mx)}")

    return "  \n".join(parts) if parts else "(no constraints)"


def _render_distilled_top(result_obj: Dict[str, Any], top_n: int = 3) -> str:
    payload = (result_obj or {}).get("result") or {}
    distilled = payload.get("distilled") or {}
    picked = payload.get("picked") or {}

    if not distilled or "top_distilled_rules" not in distilled:
        return "No distilled strategies found in this result."

    rules = (distilled.get("top_distilled_rules") or [])[: max(1, int(top_n))]

    out: List[str] = []
    out.append(f"### ✅ Distilled strategies (top {len(rules)})")
    out.append(f"- **Picked:** `{picked}`")
    base = distilled.get("best_base_model") or {}
    out.append(f"- **Base model:** `{base}`")
    out.append("")

    for i, rr in enumerate(rules, start=1):
        spec = rr.get("spec") or {}
        tr = rr.get("train") or {}
        va = rr.get("val") or {}
        te = rr.get("test") or {}
        gl = rr.get("test_game_level") or {}
        gap = rr.get("gap_train_minus_val")

        out.append(f"#### Strategy #{i}")
        out.append(_render_rule_spec(spec))
        out.append("")
        out.append(f"- **Train:** rows={tr.get('rows')} roi={_fmt_num(tr.get('roi'))} total_pl={_fmt_num(tr.get('total_pl'))}")
        out.append(f"- **Val:** rows={va.get('rows')} roi={_fmt_num(va.get('roi'))} total_pl={_fmt_num(va.get('total_pl'))}")
        out.append(f"- **Test:** rows={te.get('rows')} roi={_fmt_num(te.get('roi'))} total_pl={_fmt_num(te.get('total_pl'))}")
        out.append(f"- **Stability:** gap(train−val)={_fmt_num(gap)}")
        out.append(f"- **Test risk (by ID):** unique_ids={gl.get('unique_ids')} max_dd={_fmt_num(gl.get('max_dd'))} losing_streak={gl.get('losing_streak')}")
        out.append("")

    return "\n".join(out)


def _context_snapshot_text(ctx: Dict[str, Any]) -> str:
    ov = (ctx.get("dataset_overview") or {})
    derived = (ctx.get("derived") or {})
    ignored = derived.get("ignored_columns") or []
    outcomes = derived.get("outcome_columns") or []
    enforcement = derived.get("enforcement") or {}
    primary_goal = ov.get("primary_goal", "")
    fmt = ov.get("strategy_output_format", "")
    row_def = ov.get("row_definition", "")
    warn = ov.get("analysis_warning", "")

    return (
        f"Context loaded (Sheet Bible).\n"
        f"- primary_goal: {primary_goal}\n"
        f"- row_definition: {row_def}\n"
        f"- warning: {warn}\n"
        f"- strategy_output_format: {fmt}\n"
        f"- ignored_columns: {ignored}\n"
        f"- outcome_columns: {outcomes}\n"
        f"- enforcement: {enforcement}\n"
    )


def _log_lab_to_sheet(job_id: str, result_obj: Dict[str, Any], tags: str):
    try:
        payload = (result_obj or {}).get("result") or {}
        if not payload:
            return
        picked = payload.get("picked") or {}
        pl_col = str((picked.get("pl_column") or "")).strip()

        distilled_rules = ((payload.get("distilled") or {}).get("top_distilled_rules") or [])[:3]

        note = {
            "ts": datetime.utcnow().isoformat(),
            "kind": "pl_lab_result",
            "job_id": job_id,
            "picked": picked,
            "sheet_enforcement": payload.get("sheet_enforcement"),
            "splits": payload.get("splits"),
            "features": payload.get("features"),
            "baseline": payload.get("baseline"),
            "top_distilled_rules": [
                {
                    "spec": rr.get("spec"),
                    "train": rr.get("train"),
                    "val": rr.get("val"),
                    "test": rr.get("test"),
                    "test_game_level": rr.get("test_game_level"),
                    "gap_train_minus_val": rr.get("gap_train_minus_val"),
                    "samples": rr.get("samples"),
                }
                for rr in distilled_rules
            ],
        }

        tag_bits = [t.strip() for t in (tags or "").split(",") if t.strip()]
        if pl_col:
            tag_bits.append(pl_col.replace(" ", "_"))
        final_tags = ",".join(tag_bits)

        _run_tool("append_research_note", {"note": json.dumps(note, ensure_ascii=False), "tags": final_tags})
    except Exception:
        return


# ============================================================
# NEW: Autonomous research loop
# ============================================================
def _run_autonomous_research_loop(user_text: str, budget_minutes: int, max_rounds: int = 80):
    budget_minutes = max(5, min(int(budget_minutes), 360))
    deadline = time.time() + budget_minutes * 60

    # Add user request once
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages = _trim_messages(st.session_state.messages)

    # Load sheet context immediately (Bible) and inject it to the model
    ctx = _run_tool("get_research_context", {"limit_notes": 40})
    st.session_state.last_context = ctx

    st.session_state.messages.append({"role": "assistant", "content": _context_snapshot_text(ctx)})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "🧠 AUTONOMOUS RESEARCH LOOP START\n"
                f"- Budget: {budget_minutes} minutes\n"
                "- You must iterate: run jobs → interpret results → decide next job params.\n"
                "- Always obey Sheet rules: no leakage, 3-way time split, never tune on test.\n"
                "- Prefer short jobs so you can iterate multiple times.\n"
                "- Log significant findings to research_memory.\n"
                "- When remaining time < 120s, stop tools and write final strategies.\n"
            ),
        }
    )

    rounds = 0
    while rounds < max_rounds and time.time() < deadline:
        rounds += 1
        remaining_s = int(deadline - time.time())

        st.session_state.messages.append(
            {"role": "assistant", "content": f"[Loop clock] Remaining seconds: {remaining_s}. If < 120, start final write-up."}
        )

        try:
            resp = _call_llm_tools(st.session_state.messages)
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
        st.session_state.messages.append(assistant_msg)

        if not tool_calls:
            break

        # Execute tools with guardrails
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            # Clamp waits to keep loop responsive
            if name == "wait_for_job":
                args["timeout_s"] = min(int(args.get("timeout_s") or 60), max(5, min(120, int(deadline - time.time()))))
                args["poll_s"] = min(int(args.get("poll_s") or 5), 10)

            # Keep job durations short inside the loop for iterative control
            if name in ("start_pl_lab", "start_feature_profile", "start_rule_validate"):
                remaining_min = max(1, int((deadline - time.time()) // 60))
                args["duration_minutes"] = min(int(args.get("duration_minutes") or 8), max(5, min(12, remaining_min)))

            out = _run_tool(name, args)
            st.session_state.messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False)})

        st.session_state.messages = _trim_messages(st.session_state.messages)

        # If nearly out of time, break to final
        if int(deadline - time.time()) < 90:
            break

    # Final write-up without tools
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "🧾 FINAL OUTPUT NOW.\n"
                "Produce your best 1–3 strategies as explicit rules.\n"
                "For each: train/val/test sample sizes + ROI + total PL + max drawdown + losing streak (by ID).\n"
                "Explain why it makes logical sense (not just random), referencing MODE/MARKET/DRIFT/DIFF logic.\n"
                "If nothing is robust, say so and explain what failed and what to try next."
            ),
        }
    )

    resp_final = _call_llm_no_tools(st.session_state.messages)
    msg_final = resp_final.choices[0].message
    st.session_state.messages.append({"role": "assistant", "content": msg_final.content or ""})

    _persist_chat()


# ============================================================
# Autopilot intercepts
# ============================================================
def _maybe_run_loop(user_text: str) -> bool:
    if st.session_state.agent_mode != "autopilot":
        return False

    t = _normalize(user_text)

    # Trigger phrases for autonomous multi-step loop
    wants_loop = any(
        k in t
        for k in [
            "run research loop",
            "research loop",
            "autonomous loop",
            "autonomously",
            "iterate",
            "keep iterating",
            "full loop",
        ]
    )
    if not wants_loop:
        return False

    minutes = _minutes_from_text(user_text, default_minutes=30)
    _run_autonomous_research_loop(user_text=user_text, budget_minutes=minutes, max_rounds=90)
    return True


def _maybe_start_pl_lab(user_text: str) -> bool:
    if st.session_state.agent_mode != "autopilot":
        return False

    t = _normalize(user_text)
    wants_strategy = ("strategy" in t) or ("build" in t and ("pl" in t or "btts" in t or "over" in t or "shg" in t))
    if not wants_strategy:
        return False

    ctx = _run_tool("get_research_context", {"limit_notes": 10})
    st.session_state.last_context = ctx

    minutes = _minutes_from_text(user_text, default_minutes=30)
    do_hyperopt = any(k in t for k in ["hyperopt", "grid", "cv", "bayes", "optuna"])

    pl_col = _resolve_pl_column(user_text, ctx)

    submitted = _run_tool(
        "start_pl_lab",
        {"duration_minutes": minutes, "pl_column": pl_col, "do_hyperopt": do_hyperopt, "hyperopt_iter": 12},
    )
    job_id = submitted.get("job_id")

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": _context_snapshot_text(ctx)})

    if job_id:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"✅ Started **PL Lab** for **{pl_col}** for **{minutes} minutes**.\n\n"
                    f"**Job ID:** `{job_id}`\n\n"
                    "Use:\n"
                    f"- `Check job {job_id}`\n"
                    f"- `Show results for {job_id}`"
                ),
            }
        )
        _run_tool("set_research_state", {"key": "last_pl_lab_job_id", "value": job_id})
        _run_tool("set_research_state", {"key": "last_pl_lab_started_at", "value": datetime.utcnow().isoformat()})
        _run_tool("set_research_state", {"key": "last_pl_lab_pl_column", "value": pl_col})
    else:
        st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to start PL lab: {submitted}"})

    _persist_chat()
    return True


def _maybe_handle_job_queries(user_text: str) -> bool:
    t = (user_text or "").strip().lower()

    m = re.search(r"\bcheck job\b\s+([0-9a-f\-]{10,})", t)
    if m:
        job_id = m.group(1)
        job = _run_tool("get_job", {"job_id": job_id})
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.messages.append({"role": "assistant", "content": f"```json\n{json.dumps(job, indent=2)}\n```"})
        _persist_chat()
        return True

    m2 = re.search(r"\bshow results\b.*\b([0-9a-f\-]{10,})", t)
    if m2:
        job_id = m2.group(1)

        waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 1, "poll_s": 1, "auto_download": False})
        job = waited.get("job") or {}
        rp = job.get("result_path")
        params = job.get("params") or {}
        bucket = (params.get("_results_bucket") or DEFAULT_RESULTS_BUCKET).strip()

        st.session_state.messages.append({"role": "user", "content": user_text})

        if not rp:
            st.session_state.messages.append({"role": "assistant", "content": f"No result_path yet.\n```json\n{json.dumps(job, indent=2)}\n```"})
            _persist_chat()
            return True

        res = _run_tool("download_result", {"bucket": bucket, "result_path": rp})
        result_obj = res.get("result") or {}

        # Render distilled strategies if present, otherwise show raw
        st.session_state.messages.append({"role": "assistant", "content": _render_distilled_top(result_obj, top_n=3)})
        st.session_state.messages.append({"role": "assistant", "content": f"```json\n{json.dumps(result_obj, indent=2)[:12000]}\n```"})

        try:
            task_type = (result_obj.get("task_type") or "")
            if task_type in ("pl_lab", "btts_lab"):
                _log_lab_to_sheet(job_id, result_obj, tags="pl_lab,ml,distilled,autopilot")
        except Exception:
            pass

        _persist_chat()
        return True

    return False


def _chat_with_tools(user_text: str, max_rounds: int = 6):
    if st.session_state.agent_mode == "autopilot":
        if _maybe_run_loop(user_text):
            return
        if _maybe_start_pl_lab(user_text):
            return
        if _maybe_handle_job_queries(user_text):
            return

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
    st.caption(f"Requested model: `{MODEL}`")
    st.caption(f"Data: `{DEFAULT_STORAGE_BUCKET}/{DEFAULT_STORAGE_PATH}`")
    st.caption(f"Results: `{DEFAULT_RESULTS_BUCKET}`")

    st.radio("Agent mode", ["chat", "autopilot"], key="agent_mode")
    st.divider()

    if st.button("➕ New chat"):
        new_id = _new_session_id()
        _set_session(new_id)
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.loaded_for_sid = new_id
        _persist_chat(title=f"Session {new_id[:8]}")
        st.rerun()

    st.divider()

    if st.session_state.get("last_chat_save_error"):
        st.warning(f"Chat persistence warning: {st.session_state.last_chat_save_error}")
        if st.button("Clear chat save warning"):
            st.session_state.last_chat_save_error = None

    st.subheader("💾 Chat Sessions")
    sessions = _load_sessions()
    sessions_display = list(reversed(sessions))

    options = [{"session_id": _sid(), "title": f"(current) {_sid()[:8]}"}] + [
        {"session_id": s.get("session_id"), "title": s.get("title") or s.get("session_id")[:8]}
        for s in sessions_display
        if s.get("session_id") and s.get("session_id") != _sid()
    ]

    labels = [f"{o['title']} — {o['session_id'][:8]}" for o in options]
    chosen = st.selectbox("Select session", options=list(range(len(options))), format_func=lambda i: labels[i], index=0)

    if options[chosen]["session_id"] != _sid():
        _set_session(options[chosen]["session_id"])
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        _try_load_chat(st.session_state.session_id)
        st.rerun()


# ============================================================
# Main
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
