import os
import json
import time
import traceback
from datetime import datetime

import streamlit as st
import requests

try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("You need to install openai: pip install openai")

import memory  # your memory.py


# ----------------- CONFIG -----------------
st.set_page_config(page_title="Jarvis AI Dashboard", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("No OPENAI_API_KEY found. Add it in Streamlit secrets or env.")
    st.stop()

st.sidebar.write(f"🔍 OpenAI key loaded • prefix: {OPENAI_API_KEY[:10]}")

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------- WEATHER -----------------
def get_weather(city: str = "Basingstoke") -> str:
    """Fetch current weather from OpenWeatherMap safely."""
    owm_key = (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY")
        or st.secrets.get("weather_api_key")
        or "e5084c56702e0e7de0de917e0e7edbe3"  # fallback
    )
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={owm_key}&units=metric"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        return f"{desc}, {temp}°C"
    except Exception:
        return "Weather data not available."


# ----------------- MEMORY & CHAT STORAGE -----------------
TEMP_CHAT_FILE = "temp_chat.json"
CHAT_SESSIONS_FILE = "chat_sessions.json"


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Could not save data to {path}: {e}")


def archive_current_chat():
    chat = st.session_state.get("chat", [])
    if not chat:
        return
    sessions = load_json(CHAT_SESSIONS_FILE, [])
    ts = int(time.time())
    new_id = max([s.get("id", 0) for s in sessions], default=0) + 1
    title = next((m["content"] for m in chat if m["role"] == "user"), f"Session {new_id}")[:60]
    sessions.append({"id": new_id, "ts": ts, "title": title, "messages": chat})
    save_json(CHAT_SESSIONS_FILE, sessions)


# ----------------- OPENAI -----------------
def call_jarvis(chat_history, mem_text: str) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are Jarvis, an AI assistant inside a Streamlit app (app.py).\n"
            "You can modify layout, visuals, or logic but must NEVER change:\n"
            "  - API key handling\n"
            "  - get_weather()\n"
            "  - memory or chat-saving logic\n"
            "If you make visual/layout updates, use get_weather() safely for weather data.\n"
            f"Current memory:\n{mem_text}\n"
        ),
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_msg] + chat_history,
        temperature=0.4,
    )
    return resp.choices[0].message.content


# ----------------- SESSION STATE -----------------
if "chat" not in st.session_state:
    st.session_state.chat = load_json(TEMP_CHAT_FILE, [])
if "last_ai" not in st.session_state:
    st.session_state.last_ai = ""


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("🧠 Memory & Sessions")

    mem_text = memory.recent_summary()
    st.caption(f"💬 Chat messages: {len(st.session_state.chat)}")
    st.caption(f"🧠 Long-term memory: {len(memory._load())}")
    st.caption(f"📚 Saved sessions: {len(load_json(CHAT_SESSIONS_FILE, []))}")

    st.divider()
    st.subheader("Long-term memory (summary)")
    st.write(mem_text if mem_text else "No memories yet.")

    new_mem = st.text_input("Add to memory manually:")
    if new_mem:
        memory.add_fact(new_mem, kind="manual")
        st.success("Saved to memory.")
        st.rerun()

    st.divider()
    if st.button("💾 Save current chat"):
        archive_current_chat()
        save_json(TEMP_CHAT_FILE, [])
        st.success("Chat saved.")
        st.rerun()

    if st.button("🗑️ Start New Chat"):
        archive_current_chat()
        st.session_state.chat = []
        save_json(TEMP_CHAT_FILE, [])
        st.success("New chat started.")
        st.rerun()


# ----------------- MAIN UI -----------------
st.title("🤖 Jarvis AI Dashboard (with memory)")

today = datetime.now().strftime("%A, %B %d, %Y")
weather = get_weather("Basingstoke")
st.subheader(f"Today: {today}  |  Weather: {weather}")

st.write("You can say things like:")
st.write("- **remember** that I prefer evening workouts")
st.write("- **show weather in right panel** (Jarvis will redesign UI)")
st.write("- **save chat** or **load previous session**")
st.divider()

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Ask / tell Jarvis something...")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    save_json(TEMP_CHAT_FILE, st.session_state.chat)
    lower = user_msg.lower().strip()

    if lower.startswith("remember "):
        to_store = user_msg[len("remember "):].strip()
        if to_store:
            memory.add_fact(to_store, kind="user")
            ai_reply = f"Got it. I will remember: **{to_store}**"
        else:
            ai_reply = "You said 'remember' but didn’t tell me what to remember."
        st.session_state.chat.append({"role": "assistant", "content": ai_reply})
        save_json(TEMP_CHAT_FILE, st.session_state.chat)
        with st.chat_message("assistant"):
            st.markdown(ai_reply)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Jarvis thinking..."):
                try:
                    ai_reply = call_jarvis(st.session_state.chat, memory.recent_summary())
                    st.markdown(ai_reply)
                    st.session_state.chat.append({"role": "assistant", "content": ai_reply})
                    save_json(TEMP_CHAT_FILE, st.session_state.chat)

                    if "```python" in ai_reply:
                        start = ai_reply.find("```python") + len("```python")
                        end = ai_reply.find("```", start)
                        if end != -1:
                            code = ai_reply[start:end].strip()
                            # SECURITY: prevent overwriting protected functions
                            if "weather_api_key" not in code and "st.secrets" not in code:
                                with open("app.py", "w", encoding="utf-8") as f:
                                    f.write(code)
                                st.success("✅ Code updated — reloading app...")
                                st.stop()
                            else:
                                st.warning("❌ Update rejected: unsafe API key changes detected.")

                except Exception:
                    st.error("Jarvis error.")
                    st.code(traceback.format_exc())
