from __future__ import annotations

import os
import json
import uuid
import re
import time
import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

# Optional dependency for non-blocking auto refresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

from openai import OpenAI, BadRequestError

import sys
from pathlib import Path as _Path

# Ensure `jarvissc/` is on sys.path so `from modules ...` resolves to `jarvissc/modules/...`
_THIS_FILE = _Path(__file__).resolve()
_JARVISSC_DIR = _THIS_FILE.parents[1]  # .../jarvissc
if str(_JARVISSC_DIR) not in sys.path:
    sys.path.insert(0, str(_JARVISSC_DIR))

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
# Autopilot diagnostics helpers
# ============================================================

def _safe_json_load(s: str, default):
    try:
        return json.loads(s)
    except Exception:
        return default


def _summarize_near_misses(near_misses: list, max_items: int = 8) -> dict:
    """Summarise why candidate rules failed so the agent can pick the next experiment."""
    if not near_misses:
        return {"count": 0, "by_reason": {}, "top": []}

    by_reason = {}
    for nm in near_misses:
        reason = (nm or {}).get("reason", "unknown")
        by_reason[reason] = by_reason.get(reason, 0) + 1

    top = []
    for nm in near_misses[:max_items]:
        spec = (nm or {}).get("rule", [])
        gate = (nm or {}).get("gate", {})
        top.append({"rule": spec, "reason": (nm or {}).get("reason"), "gate": gate})

    return {"count": len(near_misses), "by_reason": by_reason, "top": top}


def _record_action_in_state(ctx: dict, action: dict, result_summary: dict) -> None:
    """Persist a compact action log into research_state (key/value sheet) to avoid repetition."""
    rs = (ctx or {}).get("research_state") or {}
    key = "agent_recent_actions_json"
    hist = _safe_json_load(str(rs.get(key, "[]")), [])
    if not isinstance(hist, list):
        hist = []

    entry = {
        "ts": datetime.utcnow().isoformat(),
        "action": action,
        "result": result_summary,
    }
    hist.append(entry)
    # Keep last 50
    hist = hist[-50:]

    try:
        _run_tool("set_research_state", {"key": key, "value": json.dumps(hist)})
    except Exception:
        # Don't hard-fail UI if the sheet write is temporarily unavailable
        pass

# ============================================================
# Tool runner (signature-safe, never hard-crashes on kwargs drift)
# ============================================================







def _run_tool(name: str, args: Optional[Dict[str, Any]] = None) -> Any:
    """Execute a tool function safely.

    Goals:
    - Never crash the UI on unknown tool names.
    - Allow shorthand tool names (aliases).
    - Be robust to argument-name drift between:
        * tool schema (what the LLM calls)
        * python function signatures (what we implement)

    This is critical for autonomy: the agent must keep going even if a
    tool-call includes an unexpected kwarg.
    """
    args = args or {}

    # Tool name aliases (LLM sometimes uses shorthand)
    name_aliases = {
        "bracket_sweep": "start_bracket_sweep",
        "subgroup_scan": "start_subgroup_scan",
        "hyperopt_pl_lab": "start_hyperopt_pl_lab",
    }
    resolved_name = name_aliases.get(name, name)

    fn = getattr(functions, resolved_name, None)
    if not callable(fn):
        available = sorted(
            [
                n
                for n in dir(functions)
                if not n.startswith("_") and callable(getattr(functions, n, None))
            ]
        )
        return {
            "ok": False,
            "error": f"Unknown tool: {name}",
            "resolved_name": resolved_name,
            "available_tools": available,
        }

    # Argument aliases (schema ↔ implementation drift)
    arg_aliases_by_tool = {
        # Older schema used timeout_seconds; our implementation uses timeout_s
        "wait_for_job": {"timeout_seconds": "timeout_s"},
        # Older schema used path; our implementation uses result_path
        "download_result": {"path": "result_path"},
    }

    fixed_args = dict(args)
    alias_map = arg_aliases_by_tool.get(resolved_name) or arg_aliases_by_tool.get(name) or {}
    for old_k, new_k in alias_map.items():
        if old_k in fixed_args and new_k not in fixed_args:
            fixed_args[new_k] = fixed_args.pop(old_k)

    # Filter args to the function signature unless it accepts **kwargs
    try:
        sig = inspect.signature(fn)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not accepts_kwargs:
            fixed_args = {k: v for k, v in fixed_args.items() if k in sig.parameters}
    except Exception:
        # If signature inspection fails, best-effort call with provided args
        pass

    try:
        return fn(**fixed_args)
    except Exception as e:
        return {
            "ok": False,
            "error": f"Tool {resolved_name} failed: {type(e).__name__}: {e}",
        }


def _coerce_row_filters(filters_any: Any) -> List[Dict[str, Any]]:
    """Normalize various filter shapes into worker-friendly row_filters (list of dicts).

    Accepts:
      - None / empty -> []
      - list[dict] already in {col,op,value} form (passes through)
      - dict[str, Any] -> equality/membership filters
    """
    if not filters_any:
        return []
    if isinstance(filters_any, list):
        return [f for f in filters_any if isinstance(f, dict)]
    if isinstance(filters_any, dict):
        out: List[Dict[str, Any]] = []
        for k, v in filters_any.items():
            if k is None:
                continue
            col = str(k)
            if isinstance(v, (list, tuple, set)):
                out.append({"col": col, "op": "in", "value": list(v)})
            else:
                out.append({"col": col, "op": "==", "value": v})
        return out
    return []
# =================================
# Tools schema

# ============================================================
# Tools schema
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_research_context",
            "description": "Fetch the current governance/bible context, column definitions, and recent notes from Google Sheets and the app state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit_notes": {"type": "integer", "default": 20}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_research_note",
            "description": "Append a research note to the research_memory tab in Google Sheets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"}
                },
                "required": ["note"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_research_state",
            "description": "Set key-value pairs in the Google Sheets research_state tab (used for defaults and gates).",
            "parameters": {
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "object",
                        "additionalProperties": True
                    }
                },
                "required": ["updates"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_job",
            "description": "Submit a raw job to the Supabase jobs queue (advanced; prefer start_* helpers).",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string"},
                    "params": {"type": "object"}
                },
                "required": ["task_type", "params"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_pl_lab",
            "description": "Queue a PL Lab job that trains models and distills explicit filter rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "duration_minutes": {"type": "integer", "default": 30},
                    "do_hyperopt": {"type": "boolean", "default": False},
                    "hyperopt_iter": {"type": "integer", "default": 0},
                    "enforcement": {"type": "object"},
                    "row_filters": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["pl_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_subgroup_scan",
            "description": "Queue a subgroup scan job to find profitable stable categorical buckets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "duration_minutes": {"type": "integer", "default": 30},
                    "group_cols": {"type": "array", "items": {"type": "string"}},
                    "max_groups": {"type": "integer", "default": 50},
                    "enforcement": {"type": "object"},
                    "row_filters": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["pl_column"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "subgroup_scan",
            "description": "Run a subgroup scan (alias for start_subgroup_scan).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "duration_minutes": {"type": "integer", "default": 30},
                    "group_cols": {"type": "array", "items": {"type": "string"}},
                    "max_groups": {"type": "integer", "default": 50},
                    "enforcement": {"type": "object"},
                    "row_filters": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["pl_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_bracket_sweep",
            "description": "Queue a bracket sweep job to search numeric quantile brackets for stable profit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "duration_minutes": {"type": "integer", "default": 30},
                    "sweep_cols": {"type": "array", "items": {"type": "string"}},
                    "n_bins": {"type": "integer", "default": 12},
                    "max_results": {"type": "integer", "default": 50},
                    "enforcement": {"type": "object"},
                    "row_filters": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["pl_column"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "bracket_sweep",
            "description": "Run a bracket sweep (alias for start_bracket_sweep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "duration_minutes": {"type": "integer", "default": 30},
                    "sweep_cols": {"type": "array", "items": {"type": "string"}},
                    "n_bins": {"type": "integer", "default": 12},
                    "max_results": {"type": "integer", "default": 50},
                    "enforcement": {"type": "object"},
                    "row_filters": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["pl_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_hyperopt_pl_lab",
            "description": "Queue an Optuna-driven hyperopt job (val-only tuning) then distill to rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "duration_minutes": {"type": "integer", "default": 120},
                    "hyperopt_trials": {"type": "integer", "default": 30},
                    "top_fracs": {"type": "array", "items": {"type": "number"}},
                    "enforcement": {"type": "object"},
                    "row_filters": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["pl_column"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "hyperopt_pl_lab",
            "description": "Run hyperopt PL lab (alias for start_hyperopt_pl_lab).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "duration_minutes": {"type": "integer", "default": 120},
                    "hyperopt_trials": {"type": "integer", "default": 30},
                    "top_fracs": {"type": "array", "items": {"type": "number"}},
                    "enforcement": {"type": "object"},
                    "row_filters": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["pl_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_job",
            "description": "Fetch a job row from Supabase by job_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"}
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_job",
            "description": "Poll Supabase until a job is done/failed, then return the final job row (and optionally download the result).",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "timeout_s": {"type": "integer", "default": 600},
                    "poll_s": {"type": "integer", "default": 5},
                    "auto_download": {"type": "boolean", "default": True}
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "download_result",
            "description": "Download a raw result file (JSON) from Supabase Storage given a bucket and result_path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket": {"type": "string"},
                    "result_path": {"type": "string"}
                },
                "required": ["bucket", "result_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_result_json",
            "description": "Download and parse the result JSON for a given job_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"}
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_events",
            "description": "Fetch latest job events for a job_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 100}
                },
                "required": ["job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_chat",
            "description": "Save the current chat session to Supabase (or local store) for persistence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"},
                    "messages": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["chat_id", "messages"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_chat",
            "description": "Load a chat session by chat_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string"}
                },
                "required": ["chat_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_chats",
            "description": "List recent chat sessions.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


SYSTEM_PROMPT = """You are FootballResearcher - an autonomous strategy R&D agent.

NON-NEGOTIABLE SOURCE OF TRUTH
- You MUST use Google Sheet tabs via get_research_context():
  dataset_overview, research_rules, column_definitions, evaluation_framework, research_state, research_memory.
- Treat research_rules as ENFORCEMENT rules, not suggestions.

MISSION / PURPOSE
- Discover strategies that are REPEATABLE on future matches.
- Avoid overfitting and leakage. You are judged on out-of-sample stability and risk:
  - P&L, ROI
  - max drawdown and longest losing streak (in POINTS)
- Output strategies as explicit filters:
  - Numeric ranges + categorical constraints
  - With minimum sample sizes on train/val/test (and unique IDs where possible)
- You MUST separate discovery (train/val) from final confirmation (test).
- Narrate what you are doing, and keep the user updated with job status and interpretation.
"""

# ============================================================
# Streamlit UI setup
# ============================================================

st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Football Researcher")

# ============================================================
# Session management + persistence
# ============================================================

def _sid() -> str:
    return st.session_state.session_id

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
    if isinstance(loaded, dict) and loaded.get("ok") and loaded.get("data", {}).get("messages"):
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

# Agent modes
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "autopilot"  # default to autopilot for your use-case
if "autopilot_narrate" not in st.session_state:
    st.session_state.autopilot_narrate = True

# Autopilot job tracking
for k, default in [
    ("active_job_id", ""),
    ("active_job_last_status", ""),
    ("active_job_last_update_ts", 0.0),
    ("active_job_bucket", ""),
    ("active_job_pl_col", ""),
    ("active_job_last_event_ts", ""),
    ("agent_session_active", False),
    ("agent_session_steps_done", 0),
    ("agent_session_max_steps", 8),
    ("agent_session_budget_minutes", 30),
    ("agent_session_minutes_per_job", 10),
    ("agent_session_pl_column", ""),
    ("agent_session_filters", {}),
    ("agent_session_last_decision", ""),
]:
    if k not in st.session_state:
        st.session_state[k] = default

# ============================================================
# Sanitise history for OpenAI (prevents tool-choice / tool-call mismatches)
# ============================================================

def _sanitize_history_for_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    out: List[Dict[str, Any]] = []
    first = messages[0]
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
# LLM call
# ============================================================

def _call_llm(messages: List[Dict[str, Any]]):
    mode = st.session_state.agent_mode
    safe_messages = _sanitize_history_for_llm(messages)

    if mode == "chat":
        # Chat-only mode: do not send tools/tool_choice
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

# ============================================================
# Helpers
# ============================================================

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)

def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\(\)\[\]\{\}\,\;\:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # common typo: "1o" (letter o) -> "10"
    s = s.replace("1o ", "10 ").replace("1o", "10")
    return s

def _minutes_from_text(t: str, default_minutes: int = 30) -> int:
    s = _normalize(t)
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)\b", s)
    if m:
        mins = int(round(float(m.group(1)) * 60))
        return max(5, min(mins, 360))

    m2 = re.search(r"\b(\d+)\s*(m|min|mins|minute|minutes)\b", s)
    if m2:
        mins = int(m2.group(1))
        return max(5, min(mins, 360))

    m3 = re.search(r"\b(for|in)\s+(\d{1,3})\b", s)
    if m3:
        mins = int(m3.group(2))
        return max(5, min(mins, 360))

    nums = re.findall(r"\b(\d{1,3})\b", s)
    if len(nums) == 1:
        mins = int(nums[0])
        if 5 <= mins <= 360:
            return mins

    return max(5, min(int(default_minutes), 360))

def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

def _append(role: str, content: str, persist: bool = True):
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.messages = _trim_messages(st.session_state.messages)
    if persist:
        _persist_chat()

def _context_snapshot_text(ctx: Dict[str, Any]) -> str:
    ov = (ctx.get("dataset_overview") or {})
    derived = (ctx.get("derived") or {})
    ignored = derived.get("ignored_columns") or []
    outcomes = derived.get("outcome_columns") or []
    primary_goal = ov.get("primary_goal", "")
    fmt = ov.get("strategy_output_format", "")
    return (
        "Context loaded (Google Sheets is the bible):\n"
        f"- primary_goal: {primary_goal}\n"
        f"- strategy_output_format: {fmt}\n"
        f"- ignored_columns: {ignored}\n"
        f"- outcome_columns: {outcomes}\n"
    )

def _resolve_pl_column(user_text: str, ctx: Dict[str, Any]) -> str:
    t = _normalize(user_text)
    outcomes: List[str] = ((ctx.get("derived") or {}).get("outcome_columns") or [])

    aliases = {
        "btts": "BTTS PL",
        "both teams to score": "BTTS PL",
        "over 2.5": "BO 2.5 PL",
        "o2.5": "BO 2.5 PL",
        "bo 2.5": "BO 2.5 PL",
        "bo2.5": "BO 2.5 PL",
        "fh over 1.5": "BO1.5 FHG PL",
        "shg": "SHG PL",
    }
    for k, v in aliases.items():
        if k in t:
            return v

    # match against outcome columns text if present
    def nn(x: str) -> str:
        return _normalize(x).replace(" ", "")

    t_compact = t.replace(" ", "")

    # IMPORTANT: outcome_columns often contains the generic "PL" first.
    # When the user asks for e.g. "LFGHU0.5 PL", we must prefer the more specific column.
    # Sort by: (is_generic_PL, -len(col)) so longer/specific columns match first and "PL" matches last.
    outcomes_sorted = sorted(
        outcomes,
        key=lambda c: ((c or "").strip().lower() == "pl", -len((c or "").strip())),
    )

    for col in outcomes_sorted:
        if nn(col) in t_compact:
            return col

    # default
    return "BO 2.5 PL"

def _render_rule_spec(spec: Dict[str, Any]) -> str:
    parts: List[str] = []
    for c in (spec.get("categorical") or []):
        col = c.get("col")
        if not col:
            continue
        if c.get("in"):
            parts.append(f"**{col} IN** {c['in']}")
        if c.get("not_in"):
            parts.append(f"**{col} NOT IN** {c['not_in']}")
    for n in (spec.get("numeric") or []):
        col = n.get("col")
        if not col:
            continue
        mn = n.get("min")
        mx = n.get("max")
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
        if isinstance(distilled, dict) and distilled.get("error"):
            return f"Distillation error: {distilled.get('error')}"
        return "No distilled strategies found in this result."

    rules = (distilled.get("top_distilled_rules") or [])[: max(1, int(top_n))]

    out: List[str] = []
    out.append(f"### ✅ Distilled strategies (top {len(rules)})")
    out.append(f"- Picked: `{picked}`")
    out.append(f"- Base model: `{distilled.get('best_base_model')}`")
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
        out.append(f"- Train: rows={tr.get('rows')} roi={_fmt_num(tr.get('roi'))} total_pl={_fmt_num(tr.get('total_pl'))}")
        out.append(f"- Val: rows={va.get('rows')} roi={_fmt_num(va.get('roi'))} total_pl={_fmt_num(va.get('total_pl'))}")
        out.append(f"- Test: rows={te.get('rows')} roi={_fmt_num(te.get('roi'))} total_pl={_fmt_num(te.get('total_pl'))}")
        out.append(f"- Stability: gap(train-val)={_fmt_num(gap)}")
        out.append(f"- Test risk (by ID): unique_ids={gl.get('unique_ids')} max_dd={_fmt_num(gl.get('max_dd'))} losing_streak={gl.get('losing_streak')}")
        tf = rr.get("trade_frequency") or {}
        if tf:
            out.append(f"- Test trade frequency: bets/week={_fmt_num(tf.get('bets_per_week'))} games/week={_fmt_num(tf.get('games_per_week'))} (span_weeks={_fmt_num(tf.get('weeks'))})")
        ms = rr.get("test_monthly_stats") or {}
        if ms and ms.get("months"):
            out.append(
                f"- Test regime check: months={ms.get('months')} monthly_roi_std={_fmt_num(ms.get('roi_std'))} neg_month_frac={_fmt_num(ms.get('negative_month_frac'))}"
            )
            try:
                if float(ms.get('roi_std') or 0.0) > 0.25:
                    out.append("  - ⚠️ High month-to-month variance; consider regime/subgroup filters.")
            except Exception:
                pass
        out.append("")
    return "\n".join(out)

def _log_lab_to_sheet(job_id: str, result_obj: Dict[str, Any], tags: str):
    # Best-effort only; should never crash the UI
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
            "enforcement": payload.get("enforcement"),
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
# Enforcement defaults (+ explanation)
# ============================================================

DEFAULT_ENFORCEMENT = {
    "min_train_rows": 300,
    "min_val_rows": 60,
    "min_test_rows": 60,
    "max_train_val_gap_roi": 0.4,
    "max_test_drawdown": -50.0,
    "max_test_losing_streak_bets": 50.0,
}



def _render_result_analysis(payload: dict) -> str:
    """User-facing analysis of a job result (compact, Bible-aligned)."""
    picked = payload.get("picked") or {}
    baseline = payload.get("baseline") or {}
    splits = payload.get("splits") or {}
    features = payload.get("features") or {}
    monthly_stats = (baseline.get("test_monthly_stats") or {})
    regime_warn = (baseline.get("test_regime_warning") or {})

    def _fmt(x, nd=3):
        try:
            if x is None:
                return "n/a"
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    b_train = baseline.get("train") or {}
    b_val = baseline.get("val") or {}
    b_test = baseline.get("test") or {}

    lines: List[str] = []
    lines.append("### 📊 Quick analysis (what we learned)")
    lines.append(f"- Market picked: `{picked}`")
    lines.append(
        f"- Splits (rows): train **{splits.get('train_rows','?')}**, val **{splits.get('val_rows','?')}**, test **{splits.get('test_rows','?')}**"
    )
    lines.append(
        f"- Features used: total **{features.get('n_features_total','?')}** (numeric **{features.get('n_numeric','?')}**, categorical **{features.get('n_categorical','?')}**)"
    )
    lines.append("")
    lines.append("**Baseline performance (strict time-split):**")
    lines.append(
        f"- Train ROI: **{_fmt(b_train.get('roi'))}** | Val ROI: **{_fmt(b_val.get('roi'))}** | Test ROI: **{_fmt(b_test.get('roi'))}**"
    )
    if b_test.get("max_drawdown") is not None:
        lines.append(f"- Test max drawdown: **{_fmt(b_test.get('max_drawdown'))}**")
    if b_test.get("losing_streak_bets") is not None:
        lines.append(f"- Test losing streak (bets): **{b_test.get('losing_streak_bets')}**")
    if b_test.get("n_bets") is not None:
        lines.append(f"- Test bets: **{b_test.get('n_bets')}**")

    lines.append("")
    lines.append("**Regime / monthly stability check (test):**")
    if monthly_stats:
        lines.append(
            f"- Monthly ROI mean: **{_fmt(monthly_stats.get('roi_mean'))}**, std: **{_fmt(monthly_stats.get('roi_std'))}**, best: **{_fmt(monthly_stats.get('roi_max'))}**, worst: **{_fmt(monthly_stats.get('roi_min'))}**"
        )
    if regime_warn:
        lines.append(
            f"- Regime warning: **{bool(regime_warn.get('flag'))}** (ROI std {_fmt(regime_warn.get('roi_std'))} vs threshold {_fmt(regime_warn.get('roi_std_warn_threshold'))})"
        )

    return "\n".join(lines)


def _render_next_job_commentary(next_task: str, why: str, merged_params: dict) -> str:
    """Explain to the user what the next job will do and why we're doing it."""
    picked = {
        "pl_column": merged_params.get("pl_column"),
        "side": merged_params.get("side"),
        "odds_col": merged_params.get("odds_col"),
    }

    lines: List[str] = []
    lines.append("### 🔜 Next job (autopilot) — what & why")
    lines.append(f"- Next task: **`{next_task}`**")
    lines.append(f"- Why: {why}")
    lines.append(f"- Context: `{picked}`")

    if next_task == "bracket_sweep":
        cols = merged_params.get("sweep_cols") or merged_params.get("focus_numeric_cols") or []
        lines.append(f"- It will search value ranges (bins) for: `{cols}` and score each range on train/val/test with gates.")
        lines.append("- Success looks like: a range that stays positive on **val and test** without big drawdown / losing streak.")
    elif next_task == "subgroup_scan":
        cols = merged_params.get("group_cols") or merged_params.get("group_by_cols") or []
        lines.append(f"- It will scan grouped segments for: `{cols}` and find buckets with stable ROI in val/test.")
        lines.append("- Success looks like: a segment that holds up out-of-sample (not just one league/month).")
    elif next_task == "hyperopt_pl_lab":
        trials = merged_params.get("hyperopt_trials") or merged_params.get("trials")
        lines.append(f"- It will tune model/distillation knobs on **validation only** (trials={trials}) to find a more generalisable rule.")
        lines.append("- Success looks like: distilled rules that pass the gates, then re-validated on test.")
    else:
        lines.append("- It will run the selected diagnostic to locate robust pockets of signal and convert them into explicit filter rules.")

    return "\n".join(lines)


def _choose_diagnostic_task(payload: dict, ctx: dict, recent_actions: list) -> tuple[str, dict]:
    """Fallback when the agent tries to stop without producing passing rules.

    Preference order:
      1) bracket_sweep on top numeric features
      2) subgroup_scan on top categorical features
      3) hyperopt_pl_lab (short)
    """
    dataset = (ctx.get("dataset") or {})
    storage_bucket = dataset.get("storage_bucket")
    storage_path = dataset.get("storage_path")

    picked = payload.get("picked") or {}
    pl_col = picked.get("pl_column")
    side = picked.get("side") or "back"
    odds_col = picked.get("odds_col")
    row_filters = picked.get("row_filters") or {}

    ftypes = payload.get("feature_types") or {}
    numeric = set(ftypes.get("numeric") or [])
    categorical = set(ftypes.get("categorical") or [])

    imp = payload.get("feature_importance") or []
    ranked = [d.get("feature") for d in imp if isinstance(d, dict) and d.get("feature")]

    top_num = [c for c in ranked if c in numeric][:4] or list(numeric)[:4]
    top_cat = [c for c in ranked if c in categorical][:3] or list(categorical)[:3]

    # avoid repeating the exact same task_type consecutively
    last_task = None
    if recent_actions:
        last_task = recent_actions[-1].get("task_type")

    if last_task != "bracket_sweep" and top_num:
        return "bracket_sweep", {
            "storage_bucket": storage_bucket,
            "storage_path": storage_path,
            "pl_column": pl_col,
            "side": side,
            "odds_col": odds_col,
            "row_filters": row_filters,
            "focus_numeric_cols": top_num,
            "bins": 8,
            "max_rules": 40,
        }

    if last_task != "subgroup_scan" and (top_cat or top_num):
        return "subgroup_scan", {
            "storage_bucket": storage_bucket,
            "storage_path": storage_path,
            "pl_column": pl_col,
            "side": side,
            "odds_col": odds_col,
            "row_filters": row_filters,
            "group_by_cols": top_cat,
            "focus_numeric_cols": top_num[:3],
            "min_rows": int((ctx.get("enforcement_gates") or {}).get("min_val_rows", 60)),
            "top_k": 30,
        }

    return "hyperopt_pl_lab", {
        "storage_bucket": storage_bucket,
        "storage_path": storage_path,
        "pl_column": pl_col,
        "side": side,
        "odds_col": odds_col,
        "row_filters": row_filters,
        "trials": 15,
        "time_budget_sec": 600,
    }

ENFORCEMENT_EXPLANATION = {
    "min_train_rows": "Minimum number of scan-rows needed in TRAIN for a rule to be considered (avoid tiny-sample mirages).",
    "min_val_rows": "Minimum number of scan-rows needed in VALIDATION (we tune on val, so it must be meaningful).",
    "min_test_rows": "Minimum number of scan-rows needed in TEST (final proof; if too small, it's not trusted).",
    "max_train_val_gap_roi": "Max allowed drop from train ROI to val ROI (penalises overfitting). Example 0.10 means train ROI can be at most 0.10 higher than validation ROI.",
    "max_test_drawdown": "Worst allowed peak-to-trough drawdown on TEST, measured in points (negative). Example -25 means we reject strategies that at any point fall more than 25 points from peak.",
    "max_test_losing_streak_bets": "Maximum allowed consecutive losing bets (by unique match ID) on TEST.",
}

def _enforcement_from_state(ctx: Dict[str, Any]) -> Dict[str, Any]:
    stt = (ctx.get("research_state") or {}) if isinstance(ctx, dict) else {}
    out = dict(DEFAULT_ENFORCEMENT)
    for k in list(out.keys()):
        if k in stt and str(stt[k]).strip() != "":
            try:
                out[k] = float(stt[k]) if "max_" in k else int(float(stt[k]))
            except Exception:
                pass
    return out

# ============================================================
# Autopilot: submit-and-follow (non-blocking, uses autorefresh)
# ============================================================

def _maybe_start_narrated_pl_research(user_text: str) -> bool:
    if st.session_state.agent_mode != "autopilot":
        return False

    t = _normalize(user_text)
    wants_strategy = ("strategy" in t) or ("build" in t and ("pl" in t or "btts" in t or "over" in t or "shg" in t))
    if not wants_strategy:
        return False

    # If a job is already active, don't start a new one
    if st.session_state.get("active_job_id"):
        _append("assistant", f"⚠️ You already have an active job running: `{st.session_state.active_job_id}`. Say **Show results** or wait for it to finish.")
        return True

    ctx = _run_tool("get_research_context", {"limit_notes": 20})
    st.session_state["cached_research_context"] = ctx
    st.session_state.last_context = ctx

    minutes = _minutes_from_text(user_text, default_minutes=10)
    do_hyperopt = any(k in t for k in ["hyperopt", "grid", "cv", "bayes", "optuna"])
    pl_col = _resolve_pl_column(user_text, ctx)
    enforcement = _enforcement_from_state(ctx)

    _append("user", user_text, persist=False)
    _append("assistant", _context_snapshot_text(ctx), persist=False)

    _append(
        "assistant",
        (
            f"✅ **Yes - I'll build a `{pl_col}` strategy.**\n\n"
            f"**What I'm going to do (Bible-aligned):**\n"
            f"1) Start an ML lab using only leakage-safe features (never use PL columns as inputs).\n"
            f"2) Use strict time split: train -> val -> test.\n"
            f"3) Distill the signal into **explicit rule filters**.\n"
            f"4) Enforce stability + risk gates.\n\n"
            f"**Time budget:** {minutes} minutes\n"
            f"**Enforcement gates:** `{enforcement}`\n\n"
            f"If you want, I can explain these gates: type **Explain enforcement**."
        ),
        persist=False,
    )
    _persist_chat()

    submitted = _run_tool(
        "start_pl_lab",
        {
            "duration_minutes": minutes,
            "pl_column": pl_col,
            "do_hyperopt": do_hyperopt,
            "hyperopt_iter": 16,
            "enforcement": enforcement,
        },
    )
    job_id = (submitted or {}).get("job_id") if isinstance(submitted, dict) else None
    if not job_id:
        _append("assistant", f"❌ Failed to start job. Response:\n```json\n{json.dumps(submitted, indent=2)}\n```")
        return True

    # Track job in session state (so the UI can keep narrating without blocking)
    st.session_state.active_job_id = job_id
    st.session_state.active_job_pl_col = pl_col
    st.session_state.active_job_bucket = DEFAULT_RESULTS_BUCKET
    st.session_state["active_job_last_event_ts"] = ""
    st.session_state["agent_session_active"] = True
    st.session_state["agent_session_steps_done"] = 0
    st.session_state["agent_session_pl_column"] = pl_col
    st.session_state["agent_session_minutes_per_job"] = int(minutes)

    st.session_state.active_job_last_status = ""
    st.session_state.active_job_last_update_ts = 0.0

    # Save state to sheet (best-effort)
    try:
        _run_tool("set_research_state", {"key": "last_pl_lab_job_id", "value": job_id})
        _run_tool("set_research_state", {"key": "last_pl_lab_pl_column", "value": pl_col})
        _run_tool("set_research_state", {"key": "last_pl_lab_started_at", "value": datetime.utcnow().isoformat()})
    except Exception:
        pass

    _append("assistant", f"🚀 Started **PL Lab** for **{pl_col}**. Job ID: `{job_id}`\n\nI'll keep checking it automatically and will post results when they're ready.")
    return True


def _summarise_job_result(task_type: str, result: dict) -> str:
    """Create a compact summary for the LLM 'brain'."""
    try:
        if task_type == "pl_lab":
            picked = result.get("picked") or {}
            base = result.get("baseline") or {}
            lb = result.get("leaderboard") or {}
            best = (result.get("best_candidates") or [{}])[0]
            return (
                f"PL_LAB picked={picked}. "
                f"baseline_val_roi={base.get('val', {}).get('roi'):.4f} baseline_test_roi={base.get('test', {}).get('roi'):.4f}. "
                f"best_candidates={best}. "
                f"distilled_rules_count={len((result.get('distilled') or {}).get('top_distilled_rules') or [])}."
            )
        if task_type == "categorical_scan":
            scan = result.get("scan") or {}
            parts = []
            for col, rows in scan.items():
                if rows:
                    top = rows[0]
                    parts.append(f"{col} top={top.get('value')} val_roi={top.get('val', {}).get('roi'):.4f} test_roi={top.get('test', {}).get('roi'):.4f} n_test={top.get('samples', {}).get('test')}")
            return "CATEGORICAL_SCAN " + " | ".join(parts[:6])
        if task_type == "bracket_sweep":
            sweep = result.get("sweep") or {}
            parts = []
            for col, rows in sweep.items():
                if rows:
                    top = rows[0]
                    rg = top.get("range") or {}
                    parts.append(f"{col} best_range=[{rg.get('min'):.3g},{rg.get('max'):.3g}] val_roi={top.get('val', {}).get('roi'):.4f} test_roi={top.get('test', {}).get('roi'):.4f} n_test={top.get('samples', {}).get('test')}")
            return "BRACKET_SWEEP " + " | ".join(parts[:6])
        if task_type == "rule_search":
            best = result.get("best_greedy_path")
            if best:
                return f"RULE_SEARCH best_score={best.get('score'):.4f} rule={best.get('rule')} val_roi={best.get('val', {}).get('roi'):.4f} test_roi={best.get('test', {}).get('roi'):.4f}"
            return "RULE_SEARCH no_rule_found"
        if task_type == "feature_audit":
            return f"FEATURE_AUDIT n_features={len(result.get('feature_cols') or [])} top_missing={result.get('top_missing', [])[:3]}"
        if task_type == "explain_best_model":
            return f"EXPLAIN top_features={[x.get('feature') for x in (result.get('top_features') or [])[:8]]}"
        if task_type == "hyperopt":
            if result.get("error"):
                return f"HYPEROPT error={result.get('error')}"
            return f"HYPEROPT best_value={result.get('best_value'):.4f} best_params={result.get('best_params')}"
    except Exception:
        pass
    return f"{task_type} (summary unavailable)"

def _agent_choose_next_action(*, ctx: dict, last_task_type: str, last_result: dict) -> dict:
    """
    Uses the LLM to choose the next tool/job to run.
    Returns dict with keys: narration, stop(bool), next_task(optional), next_params(optional), note(optional).
    """
    # Guardrails: user dictates budgets; agent decides next action within allowed set.
    allowed = [
        "pl_lab",
        "feature_audit",
        "categorical_scan",
        "bracket_sweep",
        "rule_search",
        "explain_best_model",
        "hyperopt",
    ]
    summary = _summarise_job_result(last_task_type, last_result)
    enforcement = ctx.get("enforcement") or {}
    ignored_cols = ctx.get("ignored_columns") or []
    outcome_cols = ctx.get("outcome_columns") or []
    ignored_feat = ctx.get("ignored_feature_columns") or []
    filters = st.session_state.get("agent_session_filters") or {}
    recent_actions = []
    try:
        ra_raw = (ctx.get('research_state') or {}).get('recent_actions')
        if isinstance(ra_raw, str) and ra_raw.strip():
            recent_actions = json.loads(ra_raw)
        elif isinstance(ra_raw, list):
            recent_actions = ra_raw
    except Exception:
        recent_actions = []

    near_misses = ((last_result or {}).get('distilled') or {}).get('near_misses') or []
    near_summary = _summarize_near_misses(near_misses, max_items=8)

    system = """You are Jarvis Football Research Agent.
You must follow the Bible rules:
- Each row is a scan outcome, not a match.
- PL columns are outcomes only; NEVER use them as predictive features.
- Always split by time (train/val/test). Do not tune on test.
- Strategies must become explicit filters (ranges + categorical constraints) with minimum samples.
- Prefer simple stable rules; penalize overfit/regime dependence.
You can choose what job to run next, within the available job types.

Behaviour requirements (very important):
- If the latest job produced NO passing rules, do NOT stop. First summarise why (use near_misses_summary, feature_importance, drift_report, monthly stats), then run a diagnostic/refinement job.
- Prefer these fallbacks when stuck: bracket_sweep -> categorical_scan -> rule_search -> hyperopt -> explain_best_model.
- Avoid repeating recently-tried experiments (recent_actions) unless you explicitly explain what is changing.
- Always keep strategies leakage-safe and tune ONLY on train/val."""

    user = {
        "goal": "Build a BO2.5 strategy with explicit filters that generalise to future matches.",
        "last_summary": summary,
        "last_task_type": last_task_type,
        "budget": {
            "steps_done": int(st.session_state.get("agent_session_steps_done", 0)),
            "max_steps": int(st.session_state.get("agent_session_max_steps", 8)),
        },
        "current_filters": filters,
        "enforcement_gates": enforcement,
        "ignored_columns": ignored_cols,
        "outcome_columns": outcome_cols,
        "ignored_feature_columns": ignored_feat,
        "available_job_types": allowed,
        "near_misses_summary": near_summary,
        "recent_actions": recent_actions,
        "instruction": "Choose the next job to run. If there are no passing rules yet, you MUST continue with a diagnostic/refinement job (do not stop). Keep params small and focused. Only use train/val for decision-making; test is for final reporting only."
    }

    prompt = f"""Return STRICT JSON with this schema:
{{
  "narration": "what you're doing next and why",
  "stop": false,
  "next_task_type": "one of {allowed}",
  "next_params": {{ ... }}
}}
If you decide to stop, set stop=true and omit next_task_type/next_params.

Context JSON:
{json.dumps(user, ensure_ascii=False)}
"""
    # Call LLM (no tools here)
    resp = _call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ])
    txt = (resp or "").strip()
    # Extract JSON
    try:
        j = json.loads(txt)
        return j
    except Exception:
        # fallback: stop with explanation
        return {"narration": f"Agent could not parse JSON. Raw: {txt[:400]}", "stop": True}


def _autopilot_tick():
    # Cached Bible context (loaded when strategy run starts).
    ctx_local = st.session_state.get('cached_research_context')
    if not isinstance(ctx_local, dict):
        ctx_local = {}

    job_id = st.session_state.get("active_job_id") or ""

    # Prevent duplicate processing of the same completed job across Streamlit reruns.
    # Without this, when a job finishes and st_autorefresh keeps firing, the UI can
    # repeatedly download/interpret the same result and spam the chat.
    if "_handled_job_ids" not in st.session_state:
        st.session_state["_handled_job_ids"] = set()

    # Stream worker events (narrated progress)
    try:
        last_ts = st.session_state.get("active_job_last_event_ts") or None
        ev = _run_tool("get_job_events", {"job_id": job_id, "since_ts": last_ts, "limit": 200})
        events = (ev or {}).get("events", []) if isinstance(ev, dict) else []
        if events:
            for e in events:
                ts = e.get("ts")
                lvl = (e.get("level") or "info").upper()
                msg = e.get("message") or ""
                _append_chat("assistant", f"🧩 {lvl} {msg}")
                if ts:
                    st.session_state["active_job_last_event_ts"] = ts
    except Exception:
        pass
    if not job_id:
        return

    job = _run_tool("get_job", {"job_id": job_id})
    status = (job.get("status") or "").lower().strip()

    # If the job is terminal (done/error) and we've already processed it once,
    # don't process it again on subsequent Streamlit reruns.
    if status in ("done", "error") and job_id in st.session_state.get("_handled_job_ids", set()):
        st.session_state.active_job_id = ""
        return

    now = time.time()
    last_status = st.session_state.get("active_job_last_status") or ""
    last_update = float(st.session_state.get("active_job_last_update_ts") or 0.0)

    should_update = (status != last_status) or ((now - last_update) > 30)
    if should_update:
        _append("assistant", f"⏳ Job `{job_id}` status: **{status or 'unknown'}** (UTC {datetime.utcnow().isoformat()}Z)")
        st.session_state.active_job_last_status = status
        st.session_state.active_job_last_update_ts = now

    if status == "error":
        _append("assistant", f"❌ Job failed.\n```json\n{json.dumps(job, indent=2)}\n```")
        st.session_state.get("_handled_job_ids", set()).add(job_id)
        st.session_state.active_job_id = ""
        return

    if status != "done":
        return

    rp = job.get("result_path")
    params = job.get("params") or {}
    bucket = (params.get("_results_bucket") or DEFAULT_RESULTS_BUCKET).strip()

    if not rp:
        _append("assistant", f"⚠️ Job finished but no result_path was set.\n```json\n{json.dumps(job, indent=2)}\n```")
        st.session_state.active_job_id = ""
        return

    _append("assistant", f"📥 Job done. Downloading results `{rp}` from `{bucket}`...")
    res = _run_tool("download_result", {"bucket": bucket, "result_path": rp})

    # `download_result` returns a wrapper: {"ok": True, "result": <payload>...}
    # Keep the wrapper for helpers that expect it, and also pull out the payload dict.
    result_wrap = res if isinstance(res, dict) else {}
    payload = (result_wrap.get("result") or {}) if isinstance(result_wrap, dict) else {}

    if not payload:
        _append("assistant", f"⚠️ Could not load result payload.\n```json\n{json.dumps(res, indent=2)[:12000]}\n```")
        st.session_state.active_job_id = ""
        return

    # Mark as handled so we don't process it repeatedly on reruns.
    st.session_state.get("_handled_job_ids", set()).add(job_id)

    # Render distilled rules + analysis
    _append("assistant", _render_distilled_top(result_wrap, top_n=3))

    if st.session_state.get("verbose_result_analysis", True):
        _append("assistant", _render_result_analysis(payload))

    # Interpretation (still short, but diagnostic even on failure)
    distilled = (payload.get("distilled") or {})
    rules = distilled.get("top_distilled_rules") or []
    near_misses = distilled.get("near_misses") or []
    nm = _summarize_near_misses(near_misses)

    interp: List[str] = []
    interp.append("### 🧠 Interpretation + next step")
    if rules:
        interp.append(f"- Found **{len(rules)}** rule candidates that passed the current gates (showing top {min(3, len(rules))}).")
        interp.append("- Next: refine brackets/odds bands and verify regime stability (monthly/seasonal), then re-validate on the strict time-split.")
    else:
        interp.append("- **No rules passed the gates** in this run. That doesn't mean there's no signal; it means nothing met the current stability/risk requirements.")
        if nm.get("top_gates"):
            interp.append("- Most common failure gates: " + ", ".join([f"{k} ({v})" for k, v in nm["top_gates"]]))
        if nm.get("top_candidates"):
            interp.append("- Example near-miss candidates:")
            for c in nm["top_candidates"]:
                parts = []
                if c.get("filter_summary"):
                    parts.append(c["filter_summary"])
                if c.get("failed_gates"):
                    parts.append("failed: " + ", ".join(c["failed_gates"]))
                if c.get("val_roi") is not None:
                    parts.append(f"val ROI {c['val_roi']:+.3f}")
                if c.get("test_roi") is not None:
                    parts.append(f"test ROI {c['test_roi']:+.3f}")
                interp.append("  - " + " | ".join(parts))
        b_warn = ((payload.get("baseline") or {}).get("test_regime_warning") or {})
        if b_warn.get("warn"):
            interp.append(
                f"- ⚠️ Regime instability warning: test monthly ROI std {float(b_warn.get('roi_std', 0.0)):.3f} > {float(b_warn.get('roi_std_warn_threshold', 0.25)):.2f}. Consider subgroup/regime filters."
            )
        interp.append("- Next: run a **diagnostic** job (bracket_sweep/subgroup_scan/hyperopt_pl_lab) using the strongest features and near-miss patterns.")
    _append("assistant", "\n".join(interp))

    # Log to sheet
    _log_lab_to_sheet(job_id, result_wrap, tags="pl_lab,bo2.5,narrated_autopilot")

    # Persist a compact action/result memory to research_state to avoid repetition
    try:
        ra = []
        ra_raw = (ctx_local.get("research_state") or {}).get("recent_actions")
        if isinstance(ra_raw, str) and ra_raw.strip():
            ra = json.loads(ra_raw)
        elif isinstance(ra_raw, list):
            ra = ra_raw
        # derive a compact summary
        baseline_test = ((payload.get("baseline") or {}).get("test") or {})
        top_rule = None
        if rules:
            top_rule = rules[0].get("filter_summary") or rules[0].get("filter")
        ra.append({
            "ts_utc": datetime.utcnow().isoformat(),
            "job_id": str(job_id),
            "task_type": str(job.get("task_type") or ""),
            "pl_column": str(payload.get("picked") or payload.get("pl_column") or ""),
            "passed_rules": bool(rules),
            "baseline_test_roi": float(baseline_test.get("roi", 0.0)),
            "top_rule": top_rule,
        })
        ra = ra[-60:]
        _run_tool("set_research_state", {"key": "recent_actions", "value": json.dumps(ra)})
    except Exception:
        pass

    # Decide next step if agent session is active and budget remains
    if st.session_state.get("agent_session_active") and st.session_state.get("agent_session_steps_done", 0) < st.session_state.get("agent_session_max_steps", 8):
        st.session_state["agent_session_steps_done"] = int(st.session_state.get("agent_session_steps_done", 0)) + 1

        # LLM decides what to try next based on results (Bible-safe)
        decision = _agent_choose_next_action(
            ctx=ctx_local,
            last_task_type=str(job.get("task_type") or ""),
            last_result=payload if isinstance(payload, dict) else {}
        )
        st.session_state["agent_session_last_decision"] = json.dumps(decision, ensure_ascii=False)[:2000]
        _append_chat("assistant", "🧠 Agent decision: " + (decision.get("narration") or ""))

        if decision.get("stop"):
            if not rules:
                # Do not stop on a failure-to-distill / failure-to-pass-gates. Run a diagnostic tool next.
                recent_actions = []
                try:
                    ra_raw = (ctx_local.get("research_state") or {}).get("recent_actions")
                    if ra_raw:
                        recent_actions = json.loads(ra_raw) if isinstance(ra_raw, str) else list(ra_raw)
                except Exception:
                    recent_actions = []
                task_type, auto_params, why = _choose_diagnostic_task(payload, ctx_local, recent_actions)
                _append_chat(
                    "assistant",
                    f"🟡 No passing rules yet, so I won't stop. Next I will run **{task_type}**. {why}",
                )
                decision = {
                    "stop": False,
                    "next_task_type": task_type,
                    "next_params": auto_params,
                    "narration": f"Auto-diagnostic because no rules passed. {why}",
                }
            else:
                _append_chat("assistant", "🛑 Agent stopped (no further actions).")
                st.session_state["agent_session_active"] = False
                st.session_state.active_job_id = ""
                return

        next_task = decision.get("next_task_type")
        next_params = decision.get("next_params") or {}

        # Safety net: if the agent didn't select a tool, and we still have no passing rules,
        # pick a diagnostic task deterministically (do not stop).
        if (not next_task) and (not rules):
            recent_actions = []
            try:
                ra_raw = (ctx_local.get("research_state") or {}).get("recent_actions")
                if ra_raw:
                    recent_actions = json.loads(ra_raw) if isinstance(ra_raw, str) else list(ra_raw)
            except Exception:
                recent_actions = []
            task_type, auto_params, why = _choose_diagnostic_task(payload, ctx_local, recent_actions)
            _append_chat("assistant", f"🟡 No passing rules yet and no next task selected; running diagnostic: {task_type}. {why}")
            next_task = task_type
            next_params = {**auto_params, **(next_params or {})}

        if not next_task:
            _append_chat("assistant", "🛑 Agent did not select a next task. Stopping agent session.")
            st.session_state["agent_session_active"] = False
            st.session_state.active_job_id = ""
            return

        # Enforce Bible columns + gates on every job (and keep carry-over filters)
        base_params = {
            "pl_column": (st.session_state.get("agent_session_pl_column") or (job.get("params") or {}).get("pl_column") or "BO 2.5 PL"),
            "storage_path": (job.get("params") or {}).get("storage_path", "football_ai_NNIA.csv"),
            "storage_bucket": (job.get("params") or {}).get("storage_bucket", "football-data"),
            "ignored_columns": (ctx_local.get("derived") or {}).get("ignored_columns") or (job.get("params") or {}).get("ignored_columns") or [],
            "outcome_columns": (ctx_local.get("derived") or {}).get("outcome_columns") or (job.get("params") or {}).get("outcome_columns") or [],
            "ignored_feature_columns": (ctx_local.get("derived") or {}).get("ignored_feature_columns") or (job.get("params") or {}).get("ignored_feature_columns") or [],
            "enforcement": (job.get("params") or {}).get("enforcement") or (ctx_local.get("enforcement") if isinstance(ctx, dict) else {}) or {},
            "duration_minutes": int(st.session_state.get("agent_session_minutes_per_job", 10)),
            "filters": st.session_state.get("agent_session_filters") or {},
            "row_filters": (job.get("params") or {}).get("row_filters") or _coerce_row_filters(st.session_state.get("agent_session_filters") or {}),
            "top_n": int((job.get("params") or {}).get("top_n", 12)),
        }
        merged = {**base_params, **next_params}
        _append("assistant", _render_next_job_commentary(next_task, locals().get("why") or "Diagnostic continuation after gates failed.", merged))

        submitted = _run_tool("submit_job", {"task_type": next_task, "params": merged})
        new_job_id = (submitted or {}).get("job_id") if isinstance(submitted, dict) else None
        if new_job_id:
            st.session_state.active_job_id = new_job_id
            st.session_state.active_job_pl_col = merged.get("pl_column")
            st.session_state["active_job_last_event_ts"] = ""
            _append_chat("assistant", f"🚀 Started next job: {next_task}. Job ID: {new_job_id}")
        else:
            _append_chat("assistant", "⚠️ Could not submit next job. Stopping agent.")
            st.session_state["agent_session_active"] = False
            st.session_state.active_job_id = ""
        return

    # Clear active job (no agent continuation)
    st.session_state.active_job_id = ""

# ============================================================
# Manual job commands (robust UUID extraction)
# ============================================================

def _maybe_handle_job_queries(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    uuids = _UUID_RE.findall(t)
    if not uuids:
        return False

    wants_check = "check job" in t
    wants_show = "show results" in t or "results for" in t

    if not (wants_check or wants_show):
        return False

    job_id = uuids[0]
    _append("user", user_text, persist=False)

    if wants_check:
        job = _run_tool("get_job", {"job_id": job_id})
        _append("assistant", f"```json\n{json.dumps(job, indent=2)}\n```", persist=False)

    if wants_show:
        waited = _run_tool("wait_for_job", {"job_id": job_id, "timeout_s": 2, "poll_s": 1, "auto_download": False})
        job = waited.get("job") or {}
        rp = job.get("result_path")
        params = job.get("params") or {}
        bucket = (params.get("_results_bucket") or DEFAULT_RESULTS_BUCKET).strip()

        if not rp:
            _append("assistant", f"No result_path yet.\n```json\n{json.dumps(job, indent=2)}\n```", persist=False)
            _persist_chat()
            return True

        res = _run_tool("download_result", {"bucket": bucket, "result_path": rp})
        wrap = res if isinstance(res, dict) else {}
        payload = (wrap.get("result") or {}) if isinstance(wrap, dict) else {}

        if payload:
            _append("assistant", _render_distilled_top(wrap, top_n=3), persist=False)
            _append("assistant", f"```json\n{json.dumps(payload, indent=2)[:12000]}\n```", persist=False)
            _log_lab_to_sheet(job_id, wrap, tags="pl_lab,manual")
        else:
            _append("assistant", f"⚠️ Could not download/parse results.\n```json\n{json.dumps(res, indent=2)[:12000]}\n```", persist=False)

    _persist_chat()
    return True

# ============================================================
# Chat handler
# ============================================================

def _chat_with_tools(user_text: str, max_rounds: int = 6):
    # Autopilot "native" behaviours first
    nt = _normalize(user_text)
    if st.session_state.agent_mode == "autopilot":
        if nt in ("explain enforcement", "explain gates", "gates"):
            lines = ["### Enforcement gates (what they mean)"]
            for k, v in DEFAULT_ENFORCEMENT.items():
                lines.append(f"- **{k}={v}**: {ENFORCEMENT_EXPLANATION.get(k, '')}")
            _append("user", user_text, persist=False)
            _append("assistant", "\n".join(lines))
            return

        if _maybe_handle_job_queries(user_text):
            return
        if _maybe_start_narrated_pl_research(user_text):
            return

    # Otherwise, normal LLM chat/tools loop
    _append("user", user_text, persist=False)

    st.session_state.messages = _trim_messages(st.session_state.messages)

    for _ in range(max_rounds):
        try:
            resp = _call_llm(st.session_state.messages)
        except BadRequestError as e:
            st.error("OpenAI BadRequestError (see details).")
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
# Sidebar
# ============================================================

with st.sidebar:
    st.caption(f"Model: `{MODEL}`")
    st.caption(f"Data: `{DEFAULT_STORAGE_BUCKET}/{DEFAULT_STORAGE_PATH}`")
    st.caption(f"Results: `{DEFAULT_RESULTS_BUCKET}`")

    st.radio("Agent mode", ["chat", "autopilot"], key="agent_mode")
    st.checkbox("Narrate autopilot runs", key="autopilot_narrate", value=True)
    st.checkbox("Verbose result analysis", key="verbose_result_analysis", value=True)

    st.divider()
    if st.button("➕ New chat"):
        new_id = _new_session_id()
        _set_session(new_id)
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.loaded_for_sid = new_id
        st.session_state.active_job_id = ""
        _persist_chat(title=f"Session {new_id[:8]}")
        st.rerun()

    st.subheader("Enforcement gates (current defaults)")
    for k, v in DEFAULT_ENFORCEMENT.items():
        st.write(f"- {k}={v}")
        with st.expander(f"Why {k}?"):
            st.write(ENFORCEMENT_EXPLANATION.get(k, ""))

    st.divider()
    if st.session_state.get("last_chat_save_error"):
        st.warning(f"Chat persistence warning: {st.session_state.last_chat_save_error}")
        if st.button("Clear chat save warning"):
            st.session_state.last_chat_save_error = None

    st.subheader("Chat sessions")
    sessions = _run_tool("list_chats", {"limit": 200}).get("sessions") or []
    sessions_display = list(reversed(sessions))
    options = [{"session_id": _sid(), "title": f"(current) {_sid()[:8]}"}] + [
        {"session_id": s.get("session_id"), "title": s.get("title") or s.get("session_id")[:8]}
        for s in sessions_display
        if s.get("session_id") and s.get("session_id") != _sid()
    ]
    labels = [f"{o['title']} - {o['session_id'][:8]}" for o in options]
    chosen = st.selectbox("Select session", options=list(range(len(options))), format_func=lambda i: labels[i], index=0)
    if options[chosen]["session_id"] != _sid():
        _set_session(options[chosen]["session_id"])
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        _try_load_chat(st.session_state.session_id)
        st.session_state.active_job_id = ""
        st.rerun()

# ============================================================
# Main content
# ============================================================

# If a job is active, auto-refresh so you get status updates without manual input.
if st.session_state.get("active_job_id") and st.session_state.get("autopilot_narrate", True):
    st.caption("Autopilot is monitoring a running job. This page will auto-refresh for live updates.")
    if st_autorefresh is None:
        st.info("Live auto-refresh needs the 'streamlit-autorefresh' package. Add it to requirements.txt. For now, use the button below.")
        if st.button("Refresh now"):
            st.rerun()
    (
        st_autorefresh(interval=5000, key="autopilot_refresh")
        if st_autorefresh is not None
        else None
    )

# Tick once per run
_autopilot_tick()

st.subheader("💬 Chat")
for m in st.session_state.messages:
    if m.get("role") == "system":
        continue
    with st.chat_message(m.get("role", "assistant")):
        st.markdown(m.get("content", ""))

user_msg = st.chat_input("Ask the researcher...")
if user_msg:
    _chat_with_tools(user_msg)
    st.rerun()
