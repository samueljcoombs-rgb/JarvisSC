# pages/football_researcher.py
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI

import modules.football_tools as football_tools


# ----------------------------
# Config / Persistence
# ----------------------------
CHAT_LOG_PATH = Path("football_research_chat.json")  # persists across refreshes (same deployment)
AUTOPILOT_MIN_SECONDS = 30  # if autopilot is enabled, run next step at most once per N seconds


def _safe_read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _safe_write_json(path: Path, data) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # On Streamlit Cloud this normally works; if it fails, we still have Google Sheets memory/state.
        pass


# ----------------------------
# Model selection (GPT-5.2 by default)
# ----------------------------
def _init_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)


client = _init_client()


def _select_model() -> str:
    # Prefer explicit env/secret override
    preferred = (st.secrets.get("PREFERRED_OPENAI_MODEL") or "").strip()
    if preferred:
        return preferred

    # Sensible default for modern agentic reasoning
    # (If your account doesn’t have it, OpenAI will error; then set to an allowed model.)
    return "gpt-5.2"


MODEL_NAME = _select_model()


# ----------------------------
# System prompt (solid + autonomous + guardrails)
# ----------------------------
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an autonomous Football Research Agent inside a Streamlit app.\n"
        "\n"
        "PRIMARY GOAL\n"
        "- Build strategy criteria that can be applied to FUTURE matches to generate profit.\n"
        "- Output strategies as explicit filters (ranges/thresholds), with minimum sample sizes and validation.\n"
        "\n"
        "DATA TRUTH (do not guess)\n"
        "- Each row is a scan output (aggregated scan result), not a raw match log.\n"
        "- ROI/exposure: treat each row as a bet instance (duplicate IDs imply multiple bets).\n"
        "- Losing streak / drawdown: aggregate by ID (game-level) to avoid double counting a game.\n"
        "- PL columns are outcomes only, never predictive features.\n"
        "- Do NOT use NO GAMES as an ROI denominator unless explicitly validated.\n"
        "\n"
        "ROI DEFINITIONS (points, not £)\n"
        "- Back ROI = total PL / number of bets (1pt each).\n"
        "- Lay ROI = total PL / total liability, liability per bet = (odds - 1) using mapped odds columns.\n"
        "- All PL, streak and drawdown values are in POINTS.\n"
        "\n"
        "AUTONOMY LOOP (must follow)\n"
        "On every turn (including autopilot):\n"
        "1) Load dataset_overview, column_definitions, research_rules, evaluation_framework, research_state.\n"
        "2) Decide the single most valuable next research action.\n"
        "3) If a tool exists, call it immediately (no permission asking).\n"
        "4) Summarise findings concisely and log meaningful results via append_research_note.\n"
        "5) Update research_state when you start/finish phases.\n"
        "6) If an error occurs, inspect it, propose a fix, implement it using tools, and re-test.\n"
        "\n"
        "TOOL DISCIPLINE\n"
        "- Never claim you computed something unless tool output is present.\n"
        "- Run tools sequentially (no fake parallelism).\n"
        "\n"
        "PERMISSIONS / GUARDRails\n"
        "- You MAY create/edit code via write_module, but only for owned modules (protected files are blocked).\n"
        "- Changes must be permanent (written to repo files via write_module), not temporary.\n"
        "\n"
        "MODEL\n"
        f"- You are running via the OpenAI API model id: {MODEL_NAME}.\n"
    ),
}


# ----------------------------
# Tool schema
# ----------------------------
TOOLS = [
    # Knowledge base
    {"type": "function", "function": {"name": "get_dataset_overview", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_column_definitions", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_research_rules", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_evaluation_framework", "parameters": {"type": "object", "properties": {}}}},

    # Research state (Google Sheets)
    {
        "type": "function",
        "function": {
            "name": "get_research_state",
            "description": "Read research_state tab (key/value) from Google Sheets.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_research_state",
            "description": "Set a key/value in research_state tab in Google Sheets.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        },
    },

    # Permanent memory
    {
        "type": "function",
        "function": {
            "name": "append_research_note",
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
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
    },

    # Data inspection
    {"type": "function", "function": {"name": "load_data_basic", "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "list_columns", "parameters": {"type": "object", "properties": {}}}},

    # Strategy evaluation
    {
        "type": "function",
        "function": {
            "name": "strategy_performance_summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "side": {"type": "string"},
                    "odds_column": {"type": "string"},
                    "time_split_ratio": {"type": "number"},
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
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_columns": {"type": "array", "items": {"type": "string"}},
                    "time_split_ratio": {"type": "number"},
                    "compute_streaks": {"type": "boolean"},
                },
                "required": ["pl_columns"],
            },
        },
    },

    # Code autonomy tools
    {"type": "function", "function": {"name": "list_modules", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_module", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_module", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "code": {"type": "string"}}, "required": ["path", "code"]}}},
    {"type": "function", "function": {"name": "run_module", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "function_name": {"type": "string"}, "args": {"type": "object"}}, "required": ["path", "function_name"]}}},
]


def _call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    # KB
    if name == "get_dataset_overview":
        return football_tools.get_dataset_overview()
    if name == "get_column_definitions":
        return football_tools.get_column_definitions()
    if name == "get_research_rules":
        return football_tools.get_research_rules()
    if name == "get_evaluation_framework":
        return football_tools.get_evaluation_framework()

    # Research state (we implement wrappers using existing Google Sheets doc)
    if name == "get_research_state":
        return football_tools.get_research_state()  # requires you added these functions in football_tools.py
    if name == "set_research_state":
        return football_tools.set_research_state(args["key"], args["value"])

    # Memory
    if name == "append_research_note":
        return football_tools.append_research_note(args["note"], args.get("tags") or [])
    if name == "get_recent_research_notes":
        return football_tools.get_recent_research_notes(args.get("limit", 20))

    # Data
    if name == "load_data_basic":
        return football_tools.load_data_basic(limit=args.get("limit", 200))
    if name == "list_columns":
        return football_tools.list_columns()

    # Strategy
    if name == "strategy_performance_summary":
        return football_tools.strategy_performance_summary(
            pl_column=args["pl_column"],
            side=args.get("side"),
            odds_column=args.get("odds_column"),
            time_split_ratio=float(args.get("time_split_ratio", 0.7)),
            compute_streaks=bool(args.get("compute_streaks", True)),
        )
    if name == "strategy_performance_batch":
        return football_tools.strategy_performance_batch(
            pl_columns=args["pl_columns"],
            time_split_ratio=float(args.get("time_split_ratio", 0.7)),
            compute_streaks=bool(args.get("compute_streaks", True)),
        )

    # Code autonomy
    if name == "list_modules":
        return football_tools.list_modules()
    if name == "read_module":
        return football_tools.read_module(args["path"])
    if name == "write_module":
        return football_tools.write_module(args["path"], args["code"])
    if name == "run_module":
        return football_tools.run_module(args["path"], args["function_name"], args.get("args", {}))

    return {"error": f"Unknown tool: {name}"}


def _save_chat():
    _safe_write_json(CHAT_LOG_PATH, st.session_state.football_research_chat)


def _render_chat():
    # show user/assistant only (hide tool payloads in main timeline)
    for m in st.session_state.football_research_chat:
        if m.get("role") in ("user", "assistant"):
            with st.chat_message(m["role"]):
                st.markdown(m.get("content", ""))


def _run_agent_turn(user_text: str):
    st.session_state.football_research_chat.append({"role": "user", "content": user_text})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.football_research_chat,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            # If tool calls are present, execute sequentially
            if getattr(msg, "tool_calls", None):
                # record assistant message that contains tool_calls
                st.session_state.football_research_chat.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments or "{}")
                    out = _call_tool(tool_name, tool_args)

                    st.session_state.football_research_chat.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tool_name,
                            "content": json.dumps(out),
                        }
                    )
                    with st.expander(f"🛠 Tool call: {tool_name}", expanded=False):
                        st.json(out)

                # Follow-up final response
                follow = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.football_research_chat,
                )
                final_text = follow.choices[0].message.content or ""
                st.markdown(final_text)
                st.session_state.football_research_chat.append({"role": "assistant", "content": final_text})
            else:
                # Normal assistant response
                text = msg.content or ""
                st.markdown(text)
                st.session_state.football_research_chat.append({"role": "assistant", "content": text})

    _save_chat()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Autonomous Football Researcher")
st.caption(f"Model: `{MODEL_NAME}` | Chat persists on refresh | State + memory in Google Sheets")

# Load chat on startup (persists across refresh)
if "football_research_chat" not in st.session_state:
    loaded = _safe_read_json(CHAT_LOG_PATH, default=None)
    if isinstance(loaded, list) and len(loaded) > 0:
        st.session_state.football_research_chat = loaded
    else:
        st.session_state.football_research_chat = [SYSTEM_PROMPT]
        _save_chat()

if "autopilot_enabled" not in st.session_state:
    st.session_state.autopilot_enabled = False

if "last_autopilot_ts" not in st.session_state:
    st.session_state.last_autopilot_ts = 0.0

with st.sidebar:
    st.subheader("Controls")
    st.write(f"**Model in use:** `{MODEL_NAME}`")

    if st.button("🗑️ Clear chat"):
        st.session_state.football_research_chat = [SYSTEM_PROMPT]
        _save_chat()
        st.rerun()

    st.session_state.autopilot_enabled = st.toggle("Autopilot (agent decides next step)", value=st.session_state.autopilot_enabled)

    if st.button("▶ Run next step now"):
        st.session_state.last_autopilot_ts = 0.0  # force run
        st.rerun()

    st.divider()
    st.caption("Autopilot runs at most once every ~30 seconds (on page reruns).")

_render_chat()

user_msg = st.chat_input("Ask the Football Researcher…")

# Manual user message
if user_msg:
    _run_agent_turn(user_msg)
    st.rerun()

# Autopilot: run a self-directed turn (but not in a tight loop)
if st.session_state.autopilot_enabled:
    now = time.time()
    if (now - st.session_state.last_autopilot_ts) >= AUTOPILOT_MIN_SECONDS:
        st.session_state.last_autopilot_ts = now
        _save_chat()
        # The agent is instructed to read research_state and continue autonomously.
        _run_agent_turn(
            "AUTOPILOT: Continue autonomously. "
            "Load dataset_overview, column_definitions, research_rules, evaluation_framework, research_state. "
            "Then perform the single highest value next research action. "
            "Use tools as needed. Log findings and update research_state."
        )
        st.rerun()
