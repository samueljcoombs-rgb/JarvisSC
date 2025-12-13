# pages/football_researcher.py

import os
import json
from pathlib import Path
from typing import Dict, Any

import streamlit as st
from openai import OpenAI

from modules import football_tools  # our tool layer

st.set_page_config(page_title="⚽ Football Researcher", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]

# ---- OpenAI client ----
def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        st.error("Missing OPENAI_API_KEY.")
        st.stop()
    return OpenAI(api_key=api_key)

client = _init_client()
MODEL_NAME = os.getenv("PREFERRED_OPENAI_MODEL", "").strip() or "gpt-4o"

st.title("⚽ Autonomous Football Trading Researcher")

st.markdown(
    "This is a **specialised quant agent** with its own rules and tools. "
    "It can read the football dataset, design tools, create and edit its own modules, "
    "run them, read errors, and attempt to fix them."
)

# ---- Tool definitions for OpenAI ----

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_data_basic",
            "description": "Load a preview of the football dataset (rows, columns, sample).",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "Return the list of available dataset columns.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "basic_roi_for_pl_column",
            "description": "Compute total PL, total games, and ROI per game for a given PL column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pl_column": {"type": "string", "description": "Name of the PL column, e.g. 'BO 2.5 PL'."}
                },
                "required": ["pl_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_modules",
            "description": "List all modules in /modules with ownership/protection flags.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_module",
            "description": "Read the contents of a module file in /modules (bot uses this to debug and iterate).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Module file name, e.g. 'my_tool.py'."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_module",
            "description": (
                "Create or update a module in /modules. "
                "Bot may use this to create new tools or strategies. "
                "Protected core modules are blocked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target module file name, e.g. 'my_tool.py'."},
                    "code": {"type": "string", "description": "Full Python source code for the module."},
                },
                "required": ["path", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_module",
            "description": "Run a function from a module in /modules with keyword arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Module file name, e.g. 'my_tool.py'."},
                    "function_name": {"type": "string", "description": "Function to call in that module."},
                    "args": {
                        "type": "object",
                        "description": "Keyword arguments as a JSON object.",
                    },
                },
                "required": ["path", "function_name"],
            },
        },
    },
]


# ---- Chat state ----

if "football_research_chat" not in st.session_state:
    st.session_state.football_research_chat = [
        {
            "role": "system",
            "content": (
                "You are an autonomous football-trading quant agent inside a Streamlit app.\n"
                "You are completely isolated from Jarvis's general chat and memory.\n"
                "You can:\n"
                "- Inspect a scanned football dataset with odds, xG, form, and PL columns like "
                "  SHG PL, SHG 2+ PL, BO 2.5 PL, LU1.5 PL, LFGHU0.5 PL, BO1.5 FHG PL, BTTS PL.\n"
                "- Call tools to load data, list columns, compute ROI, and manage Python modules.\n"
                "- Create NEW Python modules in /modules (your own tools and strategies).\n"
                "- Edit only the modules that you created or that are marked as owned in bot_registry.json.\n"
                "- Run functions from modules and read their errors.\n"
                "- Fix code when errors occur and iterate until it works.\n\n"
                "Rules:\n"
                "- Never attempt to modify protected core modules like layout_manager.py or chat_ui.py.\n"
                "- Always validate and reason about tool outputs before taking further actions.\n"
                "- Prefer simple, robust strategies with good sample sizes and stable ROI.\n"
                "- Explain your reasoning and what tools you're creating or using.\n"
            ),
        }
    ]

# UI: show chat messages
for msg in st.session_state.football_research_chat:
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
    elif role == "tool":
        # Optional: show tool outputs more compactly
        with st.expander(f"🛠 Tool: {msg.get('name', 'tool')}"):
            st.json(msg["content"])


user_msg = st.chat_input("Ask the Football Researcher about strategies, PL, ROI, xG, odds, etc...")

def _call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Routes tool calls to our python functions."""
    if name == "load_data_basic":
        limit = arguments.get("limit") or 200
        return football_tools.load_data_basic(limit=limit)
    if name == "list_columns":
        return football_tools.list_columns()
    if name == "basic_roi_for_pl_column":
        return football_tools.basic_roi_for_pl_column(arguments["pl_column"])
    if name == "list_modules":
        return football_tools.list_modules()
    if name == "read_module":
        return football_tools.read_module(arguments["path"])
    if name == "write_module":
        return football_tools.write_module(arguments["path"], arguments["code"])
    if name == "run_module":
        return football_tools.run_module(
            arguments["path"],
            arguments["function_name"],
            arguments.get("args") or {},
        )
    return {"error": f"Unknown tool: {name}"}


def _chat_with_tools(user_input: str) -> None:
    """
    Send a user message to the model, handle one or more tool calls,
    then get a final answer.
    """
    # 1) Add user message
    st.session_state.football_research_chat.append({"role": "user", "content": user_input})

    # 2) First call – model may decide to call tools
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.football_research_chat,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # If there are tool calls, handle them
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments or "{}")

                    result = _call_tool(tool_name, tool_args)

                    # Store tool result in history
                    st.session_state.football_research_chat.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(result),
                        }
                    )

                    # Show result in UI
                    with st.expander(f"🛠 Tool call: {tool_name}", expanded=False):
                        st.json(result)

                # 3) After tools, ask model again with updated history for final answer
                followup = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.football_research_chat,
                )
                final_msg = followup.choices[0].message
                final_text = final_msg.content or ""
                st.markdown(final_text)
                st.session_state.football_research_chat.append(
                    {"role": "assistant", "content": final_text}
                )
            else:
                # No tool call – just a direct answer
                text = msg.content or ""
                st.markdown(text)
                st.session_state.football_research_chat.append(
                    {"role": "assistant", "content": text}
                )


if user_msg:
    _chat_with_tools(user_msg)
