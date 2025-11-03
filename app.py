import os
import json
import time
import traceback
from datetime import datetime
import requests
import streamlit as st

# we use the new openai client (openai>=1.x)
try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("You need to install openai: pip install openai")

import memory  # your memory.py

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Jarvis AI Dashboard", layout="wide")

# 1) get OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    st.error("No OPENAI_API_KEY found. Add it in Streamlit secrets or env.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- WEATHER -----------------
def get_weather(city: str = "Basingstoke") -> str:
    """Fetch current weather from OpenWeatherMap."""
    owm_key = (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY")
        or st.secrets.get("weather_api_key")
        or "e5084c56702e0e7de0de917e0e7edbe3"
    )
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={owm_key}&units=metric"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        return f"{desc}, {temp}°C"
    except Exception:
        return "Weather data not available."


# ----------------- SAFE FILE LOADER -----------------
def safe_read_file(filepath: str, max_chars: int = 15000) -> str:
    """Safely read and sanitize a file before sending to the model."""
    if not os.path.exists(filepath):
        return f"[File not found: {filepath}]"
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # redact secrets or keys
        filtered = "\n".join(
            line for line in content.splitlines()
            if not any(k in line for k in ["OPENAI_API_KEY", "OWM_API_KEY", "st.secrets"])
        )
        if len(filtered) > max_chars:
            return f"[Truncated content — last {max_chars} chars]\n" + filtered[-max_chars:]
        return filtered
    except Exception as e:
        return f"[Error reading {filepath}: {e}]"


# ----------------- FILE CONTEXT PROVIDER -----------------
def get_system_context() -> str:
    """Combine code from app.py, memory.py, and requirements.txt into one reference."""
    app_code = safe_read_file("app.py")
    mem_code = safe_read_file("memory.py")
    reqs = safe_read_file("requirements.txt")

    return (
        "Below are the current code files for your system:\n\n"
        "### app.py\n" + app_code +
        "\n\n### memory.py\n" + mem_code +
        "\n\n### requirements.txt\n" + reqs +
        "\n\nYou may use this information to make informed and consistent updates."
    )


# ----------------- TEMPORARY CHAT STORAGE -----------------
TEMP_CHAT_FILE = "temp_chat.json"
def load_temp_chat():
    if os.path.exists(TEMP_CHAT_FILE):
        try:
            with open(TEMP_CHAT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_temp_chat(chat_data):
    try:
        with open(TEMP_CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save temporary chat: {e}")


# ----------------- CHAT SESSION ARCHIVE -----------------
CHAT_SESSIONS_FILE = "chat_sessions.json"
def load_chat_sessions():
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_sessions(sessions):
    try:
        with open(CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save chat sessions: {e}")

def archive_current_chat(label: str | None = None):
    chat = st.session_state.get("chat", [])
    if not chat:
        return
    sessions = load_chat_sessions()
    new_id = max([s.get("id", 0) for s in sessions], default=0) + 1
    ts = int(time.time())
    first_user_msg = next((m["content"] for m in chat if m["role"] == "user"), "")
    title = label or (first_user_msg[:60] if first_user_msg else f"Session {new_id}")
    sessions.append({"id": new_id, "ts": ts, "title": title, "messages": chat})
    save_chat_sessions(sessions)


# ----------------- HELPER: call OpenAI -----------------
def call_jarvis(chat_history, mem_text: str) -> str:
    """Handles all communication with the OpenAI model."""
    file_context = get_system_context()

    system_msg = {
        "role": "system",
        "content": (
            "You are Jarvis, an AI assistant living inside a Streamlit app (app.py).\n"
            "You may be asked to modify code. Only respond with FULL valid Python when explicitly asked.\n"
            "Here is your current system context:\n\n"
            f"{file_context}\n\n"
            "Do not modify secrets or API keys. Use existing logic and memory safely."
        ),
    }

    resp = client.chat.completions.create(
        model="gpt-4o",  # GPT-5 successor tier model
        messages=[system_msg] + chat_history,
        # temperature removed for compliance with model rules
    )
    return resp.choices[0].message.content


# ----------------- SESSION STATE -----------------
if "chat" not in st.session_state:
    st.session_state.chat = load_temp_chat()
if "last_ai" not in st.session_state:
    st.session_state.last_ai = ""

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("🧠 Memory & Sessions")
    mem_text = memory.recent_summary()
    st.caption(f"🧠 Long-term memories: {len(memory._load())}")
    st.caption(f"💬 Current chat messages: {len(st.session_state.chat)}")
    st.divider()

    if mem_text:
        st.write(mem_text)
    else:
        st.write("No memories yet.")

    new_mem = st.text_input("Add to memory (manual):")
    if new_mem:
        memory.add_fact(new_mem, kind="manual")
        st.success("Saved to memory.")
        st.rerun()

    if st.button("💾 Save current chat"):
        archive_current_chat()
        st.success("Current chat archived.")
        st.rerun()

    if st.button("🗑️ Start New Chat"):
        archive_current_chat()
        st.session_state.chat = []
        save_temp_chat([])
        st.success("Chat cleared and archived.")
        st.rerun()

# ----------------- MAIN UI -----------------
st.title("🤖 Jarvis AI Dashboard (with file awareness & memory)")
today = datetime.now().strftime("%A, %B %d, %Y")
st.subheader(f"Today: {today} | Weather: {get_weather()}")

st.divider()
st.write("Talk to Jarvis below. You can say things like:")
st.write("- *remember that I prefer evening workouts*")
st.write("- *update the weather UI*")
st.write("- *show the memory summary better*")

st.divider()
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Ask / tell Jarvis something...")

if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    save_temp_chat(st.session_state.chat)

    with st.chat_message("assistant"):
        with st.spinner("Jarvis thinking..."):
            try:
                mem_now = memory.recent_summary()
                ai_reply = call_jarvis(st.session_state.chat, mem_now)
                st.markdown(ai_reply)
                st.session_state.chat.append({"role": "assistant", "content": ai_reply})
                save_temp_chat(st.session_state.chat)

                if "```python" in ai_reply:
                    start = ai_reply.find("```python") + len("```python")
                    end = ai_reply.find("```", start)
                    if end != -1:
                        new_code = ai_reply[start:end].strip()
                        backup_file = f"backup_{int(time.time())}.py"
                        with open(backup_file, "w", encoding="utf-8") as f:
                            f.write(safe_read_file("app.py"))
                        with open("app.py", "w", encoding="utf-8") as f:
                            f.write(new_code)
                        st.success(f"✅ Code updated and backed up ({backup_file}) — rerunning app...")
                        st.stop()
            except Exception:
                st.error("Jarvis error.")
                st.code(traceback.format_exc())
