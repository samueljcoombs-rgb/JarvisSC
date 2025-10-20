# app.py — JARVIS Agent (with persistent memory, code self-awareness, and GitHub PR automation)

import json, os, time, datetime as dt, requests, feedparser
import streamlit as st
from openai import OpenAI
import memory  # persistent memory (memory.py)
import gspread
from google.oauth2.service_account import Credentials

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="JARVIS", page_icon="🤖", layout="centered")

# === 🔐 API KEYS & CONSTANTS (pre-filled) ===
OPENAI_API_KEY = "sk-provided-by-you"
OWM_API_KEY = "openweather-provided-by-you"
TMDB_API_KEY = "tmdb-provided-by-you"
DEFAULT_CITY = "Basingstoke,GB"
PODCAST_FEEDS = ["https://feeds.simplecast.com/54nAGcIl"]
YOUTUBE_CHANNEL_IDS = ["UC_x5XG1OV2P6uZZ5FSM9Ttw"]
ATHLETIC_QUERY = "Manchester United"

GOOGLE_JSON = json.loads("""{your_google_service_account_json_here}""")
MAKE_WEBHOOK_URL = "https://hook.make.com/your-make-webhook"
GITHUB_REPO = "yourgithubusername/yourrepo"
GITHUB_BRANCH = "main"

# ---------------------- SETUP ----------------------
client = OpenAI(api_key=OPENAI_API_KEY)

creds = Credentials.from_service_account_info(
    GOOGLE_JSON,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
gc = gspread.authorize(creds)

# ---------------------- UTILITIES ----------------------
@st.cache_data(ttl=300)
def get_weather(city=DEFAULT_CITY):
    r = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_API_KEY}&units=metric"
    )
    d = r.json()
    t, feels = round(d["main"]["temp"]), round(d["main"]["feels_like"])
    desc = d["weather"][0]["description"].title()
    return f"{city}: {t}°C (feels {feels}°C), {desc}"

def read_code(filename="app.py", max_chars=7000):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()[:max_chars]
        return f"--- Start of {filename} ---\n{code}\n--- End of {filename} ---"
    except Exception as e:
        return f"Error reading {filename}: {e}"

# ---------------------- TOOLS ----------------------
TOOLS = {
    "get_weather": lambda args: get_weather(args.get("city", DEFAULT_CITY)),
    "read_code": lambda args: read_code(args.get("filename", "app.py")),
    "remember": lambda args: memory.add_fact(args.get("text", ""), kind="note") or "Saved to memory.",
    "list_memory": lambda args: memory.recent_summary(),
}

# ---------------------- AI CORE ----------------------
def call_openai(messages, tools=None):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[{"type": "function", "function": {
                "name": k,
                "description": f"Tool: {k}",
                "parameters": {"type": "object", "properties": {}}}} for k in TOOLS.keys()] if tools else None,
            temperature=0.4,
        )
        msg = resp.choices[0].message
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                fn = call.function
                if fn.name in TOOLS:
                    result = TOOLS[fn.name](json.loads(fn.arguments or "{}"))
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
            return call_openai(messages)
        return msg.content
    except Exception as e:
        return f"Error: {e}"

# ---------------------- UI ----------------------
st.title("🤖 JARVIS — AI Agent with Memory + GitHub Updates")

st.write("### Quick Info")
col1, col2 = st.columns(2)
with col1: st.write(get_weather())
with col2: st.write(f"📅 {dt.date.today()}")

# ------------- Chat Section -------------
if "chat" not in st.session_state: st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]): st.markdown(m["content"])

prompt = st.chat_input("Ask JARVIS anything...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        mem_summary = memory.recent_summary()
        msgs = [
            {"role": "system", "content": f"You are JARVIS, the user's integrated life assistant.\n\nMemory:\n{mem_summary}"},
            *st.session_state.chat,
        ]
        answer = call_openai(msgs, tools=True)
    st.session_state.chat.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"): st.markdown(answer)

# ------------- Memory Section -------------
st.markdown("---")
st.subheader("🧠 Memory")
if st.button("View Memory"): st.info(memory.recent_summary() or "No saved memories yet.")
fact = st.text_input("Add something to remember:")
if fact: memory.add_fact(fact); st.success("Remembered ✅")

# ------------- Developer Mode -------------
st.markdown("---")
st.subheader("🧩 Developer Mode — Auto PR Updates via Make.com")

desc = st.text_area("Describe the code change you want JARVIS to make:")
if st.button("🚀 Propose PR"):
    try:
        with open("app.py", "r", encoding="utf-8") as f: code_now = f.read()
        payload = {
            "repo": GITHUB_REPO,
            "branch": GITHUB_BRANCH,
            "filename": "app.py",
            "change_request": desc or "(no description)",
            "code_now": code_now,
        }
        r = requests.post(MAKE_WEBHOOK_URL, json=payload, timeout=20)
        if r.status_code == 200:
            st.success("✅ Sent to Make.com. It will create a Pull Request automatically.")
        else:
            st.error(f"❌ Webhook error: {r.status_code} {r.text[:150]}")
    except Exception as e:
        st.error(f"Error sending PR request: {e}")