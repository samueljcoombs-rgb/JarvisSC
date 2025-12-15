import json
import streamlit as st
from openai import OpenAI
import modules.football_tools as tools

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = st.secrets.get("PREFERRED_OPENAI_MODEL", "gpt-4o")

TOOLS = [
    {"type": "function", "function": {"name": "get_dataset_overview", "parameters": {}}},
    {"type": "function", "function": {"name": "get_column_definitions", "parameters": {}}},
    {"type": "function", "function": {"name": "get_research_rules", "parameters": {}}},
    {"type": "function", "function": {"name": "append_research_note", "parameters": {"type":"object","properties":{"note":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}},"required":["note"]}}},
    {"type": "function", "function": {"name": "get_recent_research_notes", "parameters": {}}},
    {"type": "function", "function": {"name": "roi_summary_for_pl_columns", "parameters": {"type":"object","properties":{"pl_columns":{"type":"array","items":{"type":"string"}}},"required":["pl_columns"]}}},
]

def call_tool(name, args):
    return getattr(tools, name)(**args) if args else getattr(tools, name)()

st.set_page_config(page_title="Football Researcher", layout="wide")
st.title("⚽ Autonomous Football Researcher")

if "chat" not in st.session_state:
    st.session_state.chat = [{
        "role": "system",
        "content": (
            "You are an autonomous football research agent. "
            "Before analysis, load dataset_overview, column_definitions, and research_rules. "
            "Use tools immediately. Never ask permission."
        )
    }]

for m in st.session_state.chat:
    if m["role"] != "system":
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

prompt = st.chat_input("Ask the Football Researcher…")

if prompt:
    st.session_state.chat.append({"role":"user","content":prompt})

    res = client.chat.completions.create(
        model=MODEL,
        messages=st.session_state.chat,
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = res.choices[0].message

    if msg.tool_calls:
        st.session_state.chat.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            out = call_tool(tc.function.name, args)
            st.session_state.chat.append({
                "role":"tool",
                "tool_call_id":tc.id,
                "name":tc.function.name,
                "content":json.dumps(out),
            })
            with st.expander(f"🛠 {tc.function.name}"):
                st.json(out)

        follow = client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.chat,
        )
        final = follow.choices[0].message.content
        st.session_state.chat.append({"role":"assistant","content":final})
        with st.chat_message("assistant"):
            st.markdown(final)
    else:
        st.session_state.chat.append({"role":"assistant","content":msg.content})
        with st.chat_message("assistant"):
            st.markdown(msg.content)
