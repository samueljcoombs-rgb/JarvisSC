import json
import streamlit as st
from openai import OpenAI

import modules.football_tools as football_tools


# =========================
#  Model Setup
# =========================

PREFERRED_MODEL = st.secrets.get("PREFERRED_OPENAI_MODEL", "").strip()
API_KEY = st.secrets.get("OPENAI_API_KEY", None)

if not API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=API_KEY)


def _select_model() -> str:
    """
    Try to choose the best available model automatically.
    """
    if PREFERRED_MODEL:
        return PREFERRED_MODEL
    try:
        models = {m.id for m in client.models.list().data}
        for cand in ["gpt-5", "gpt-4.1", "gpt-4o", "gpt-4.1-mini"]:
            if cand in models:
                return cand
    except Exception:
        pass
    return "gpt-4o"


MODEL_NAME = _select_model()


# =========================
#  Tools Schema
# =========================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_modules",
            "description": "Return all modules in /modules with ownership metadata.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_module",
            "description": "Read the contents of a module file inside /modules.",
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
            "description": (
                "Create or update a Python module under /modules with full autonomy, "
                "except protected core modules. Returns error if syntax invalid or modification blocked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["path", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_module",
            "description": (
                "Import a module from /modules and run a function with kwargs. "
                "Use for backtests, feature scans, etc."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "load_data_basic",
            "description": "Load a preview of the football dataset (rows, cols, sample).",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "Return dataset column names.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "basic_roi_for_pl_column",
            "description": "Compute PL + ROI stats for a given PL column.",
            "parameters": {
                "type": "object",
                "properties": {"pl_column": {"type": "string"}},
                "required": ["pl_column"],
            },
        },
    },
    # ----- NEW: permanent research memory tools -----
    {
        "type": "function",
        "function": {
            "name": "append_research_note",
            "description": "Store a persistent research insight in Google Sheets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_research_notes",
            "description": "Retrieve recent research notes from Google Sheets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
]


# =========================
#  Tool dispatcher
# =========================

def _call_tool(name: str, args: dict):
    try:
        if name == "list_modules":
            return football_tools.list_modules()
        if name == "read_module":
            return football_tools.read_module(args["path"])
        if name == "write_module":
            return football_tools.write_module(args["path"], args["code"])
        if name == "run_module":
            return football_tools.run_module(
                args["path"],
                args["function_name"],
                args.get("args", {}),
            )
        if name == "load_data_basic":
            return football_tools.load_data_basic(limit=args.get("limit", 200))
        if name == "list_columns":
            return football_tools.list_columns()
        if name == "basic_roi_for_pl_column":
            return football_tools.basic_roi_for_pl_column(args["pl_column"])
        if name == "append_research_note":
            return football_tools.append_research_note(
                args["note"],
                args.get("tags") or [],
            )
        if name == "get_recent_research_notes":
            return football_tools.get_recent_research_notes(
                args.get("limit", 20)
            )

        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": f"Exception in tool '{name}': {str(e)}"}


# =========================
#  Chat + Tool Handling
# =========================

def _chat_with_tools(user_input: str) -> None:
    """
    Handles:
      1. User message → model call
      2. Tool calls (if any)
      3. Follow-up call → final assistant message
    """
    # Add the user message
    st.session_state.football_research_chat.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # First call (model decides whether to call tools)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.football_research_chat,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # If tools requested
            if msg.tool_calls:
                # Record assistant tool_call message
                st.session_state.football_research_chat.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute each tool
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                    result = _call_tool(tool_name, tool_args)

                    # Append tool result, with tool_call_id
                    st.session_state.football_research_chat.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    })

                    with st.expander(f"🛠 Tool call: {tool_name}", expanded=False):
                        st.json(result)

                # Follow-up call with tool outputs in history
                follow = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.football_research_chat,
                )
                final_msg = follow.choices[0].message
                final_text = final_msg.content or ""

                st.markdown(final_text)
                st.session_state.football_research_chat.append({
                    "role": "assistant",
                    "content": final_text,
                })
            else:
                # No tools, just answer
                reply = msg.content or ""
                st.markdown(reply)
                st.session_state.football_research_chat.append({
                    "role": "assistant",
                    "content": reply,
                })


# =========================
#  Streamlit Page UI
# =========================

st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Autonomous Football Researcher")

st.caption(
    "Isolated quant-style agent. It can read your dataset, create and run tools, "
    "and store permanent research notes in Google Sheets."
)

if "football_research_chat" not in st.session_state:
    # Initial system-like memory: we could add principles here in future
    st.session_state.football_research_chat = []

# Show chat history
for msg in st.session_state.football_research_chat:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Input
user_msg = st.chat_input("Ask the Football Researcher…")
if user_msg:
    _chat_with_tools(user_msg)
