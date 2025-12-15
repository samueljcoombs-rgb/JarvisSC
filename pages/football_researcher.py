import json
import streamlit as st
from openai import OpenAI

import modules.football_tools as football_tools


# =========================
# Model setup
# =========================

API_KEY = st.secrets.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=API_KEY)

MODEL_NAME = st.secrets.get("PREFERRED_OPENAI_MODEL", "").strip() or "gpt-4o"


# =========================
# Tool schema
# =========================

TOOLS = [
    # --- Knowledge base ---
    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": "Load high-level dataset/scans semantics from Google Sheets.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_definitions",
            "description": "Load detailed column dictionary from Google Sheets.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_research_rules",
            "description": "Load research guardrails from Google Sheets.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_evaluation_framework",
            "description": "Load success metrics / evaluation framework from Google Sheets.",
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # --- Permanent memory ---
    {
        "type": "function",
        "function": {
            "name": "append_research_note",
            "description": "Append a research note to permanent Google Sheets memory.",
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
            "description": "Fetch last N research notes from permanent Google Sheets memory.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
        },
    },

    # --- Data inspection ---
    {
        "type": "function",
        "function": {
            "name": "load_data_basic",
            "description": "Load dataset preview (rows, cols, sample).",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List dataset columns.",
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # --- Strategy evaluation (core) ---
    {
        "type": "function",
        "function": {
            "name": "strategy_performance_summary",
            "description": (
                "Evaluate one PL column with correct Back/Lay ROI, time-split train/test, "
                "and game-level streak/drawdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string"},
                    "side": {"type": "string", "description": "back or lay (optional; inferred if omitted)"},
                    "odds_column": {"type": "string", "description": "Odds column (optional; inferred if omitted)"},
                    "time_split_ratio": {"type": "number", "description": "e.g. 0.7 train / 0.3 test"},
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
            "description": "Evaluate a list of PL columns in one call.",
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

    # --- Module tools (optional autonomy) ---
    {
        "type": "function",
        "function": {
            "name": "list_modules",
            "description": "List /modules files and ownership info.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_module",
            "description": "Read a module under /modules.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_module",
            "description": "Create/update an owned module under /modules (protected files blocked).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "code": {"type": "string"}},
                "required": ["path", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_module",
            "description": "Run a function in a /modules module.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "function_name": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["path", "function_name"],
            },
        },
    },
]


def _call_tool(name: str, args: dict):
    try:
        # Knowledge base
        if name == "get_dataset_overview":
            return football_tools.get_dataset_overview()
        if name == "get_column_definitions":
            return football_tools.get_column_definitions()
        if name == "get_research_rules":
            return football_tools.get_research_rules()
        if name == "get_evaluation_framework":
            return football_tools.get_evaluation_framework()

        # Memory
        if name == "append_research_note":
            return football_tools.append_research_note(args["note"], args.get("tags") or [])
        if name == "get_recent_research_notes":
            return football_tools.get_recent_research_notes(args.get("limit", 20))

        # Data inspection
        if name == "load_data_basic":
            return football_tools.load_data_basic(limit=args.get("limit", 200))
        if name == "list_columns":
            return football_tools.list_columns()

        # Strategy eval
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

        # Module tools
        if name == "list_modules":
            return football_tools.list_modules()
        if name == "read_module":
            return football_tools.read_module(args["path"])
        if name == "write_module":
            return football_tools.write_module(args["path"], args["code"])
        if name == "run_module":
            return football_tools.run_module(args["path"], args["function_name"], args.get("args", {}))

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": f"Tool error in {name}: {e}"}


def _find_last_user_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _chat_with_tools(user_input: str) -> None:
    st.session_state.football_research_chat.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.football_research_chat,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # Tool calls
            if msg.tool_calls:
                # store assistant tool_calls message
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

                # execute tools sequentially
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                    result = _call_tool(tool_name, tool_args)

                    st.session_state.football_research_chat.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(result),
                        }
                    )

                    with st.expander(f"🛠 Tool call: {tool_name}", expanded=False):
                        st.json(result)

                # follow-up for final response
                follow = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.football_research_chat,
                )
                final_text = follow.choices[0].message.content or ""
                st.markdown(final_text)
                st.session_state.football_research_chat.append({"role": "assistant", "content": final_text})
            else:
                text = msg.content or ""
                st.markdown(text)
                st.session_state.football_research_chat.append({"role": "assistant", "content": text})


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Autonomous Football Researcher")
st.caption("Permanent knowledge base + permanent memory + correct ROI (Back/Lay) + time-split validation.")

# Seed system rules once
if "football_research_chat" not in st.session_state:
    st.session_state.football_research_chat = [
        {
            "role": "system",
            "content": (
                "You are an autonomous football research agent.\n"
                "Hard rules:\n"
                "1) Before analysis, load dataset_overview, column_definitions, research_rules, evaluation_framework.\n"
                "2) If a tool exists for a request, call it immediately.\n"
                "3) Never claim you computed something unless tool output exists.\n"
                "4) Never ask permission to run tools; just run them.\n"
                "5) Use correct ROI: back = PL/bets (1pt each); lay = PL/sum(odds-1) liability.\n"
                "6) Duplicate IDs count as multiple bets for ROI/exposure.\n"
                "7) Aggregate by ID for losing streak/drawdown.\n"
                "8) Validate strategies using time split train/test; prefer stable performance.\n"
                "9) Log significant findings with append_research_note.\n"
            ),
        }
    ]

# Render chat history (skip tool messages)
for m in st.session_state.football_research_chat:
    if m["role"] in ("user", "assistant"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

user_msg = st.chat_input("Ask the Football Researcher…")
if user_msg:
    _chat_with_tools(user_msg)
