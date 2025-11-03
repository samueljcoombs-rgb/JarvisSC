import os
import json
import time
import traceback
from datetime import datetime

import streamlit as st
import requests

# we use the new openai client (openai>=1.x)
try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("You need to install openai: pip install openai")

import memory  # your memory.py


# ----------------- CONFIG -----------------
st.set_page_config(page_title="Jarvis AI Dashboard", layout="wide")

# 1) get OpenAI key from env or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    st.error("No OPENAI_API_KEY found. Add it in Streamlit secrets or env.")
    st.stop()

# show where we loaded the key from (for debugging on streamlit cloud)
src = "env" if os.getenv("OPENAI_API_KEY") else "st.secrets"
st.sidebar.write(f"🔍 Using OpenAI key from: {src} • prefix: {OPENAI_API_KEY[:10]}")

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


# ----------------- TEMPORARY CHAT STORAGE -----------------
TEMP_CHAT_FILE = "temp_chat.json"


def load_temp_chat():
    """Load temporary chat history from disk (if exists)."""
    if os.path.exists(TEMP_CHAT_FILE):
        try:
            with open(TEMP_CHAT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_temp_chat(chat_data):
    """Save chat history to disk."""
    try:
        with open(TEMP_CHAT_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save temporary chat: {e}")


# ----------------- CHAT SESSION ARCHIVE -----------------
CHAT_SESSIONS_FILE = "chat_sessions.json"


def load_chat_sessions():
    """Load all archived chat sessions."""
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_chat_sessions(sessions):
    """Save all archived chat sessions."""
    try:
        with open(CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save chat sessions: {e}")


def archive_current_chat(label: str | None = None):
    """Archive the current chat into the sessions file, if there is any content."""
    chat = st.session_state.get("chat", [])
    if not chat:
        return  # nothing to save

    sessions = load_chat_sessions()
    new_id = max([s.get("id", 0) for s in sessions], default=0) + 1
    ts = int(time.time())

    # Try to infer a simple title from the first user message
    first_user_msg = next((m["content"] for m in chat if m["role"] == "user"), "")
    if label:
        title = label
    elif first_user_msg:
        title = first_user_msg[:60]
    else:
        title = f"Session {new_id}"

    new_session = {
        "id": new_id,
        "ts": ts,
        "title": title,
        "messages": chat,
    }
    sessions.append(new_session)
    save_chat_sessions(sessions)


# ----------------- HELPER: call OpenAI -----------------
def call_jarvis(chat_history, mem_text: str) -> str:
    """Handles all communication with the OpenAI model."""
    system_msg = {
        "role": "system",
        "content": (
            "You are Jarvis, an AI assistant living inside a Streamlit app called app.py.\n"
            "You MAY be asked to update the code of app.py. When (and only when) the user asks "
            "for a code change, respond with FULL Python code for the ENTIRE app.py inside a "
            "```python ... ``` block.\n"
            "You also have ACCESS to a persistent memory via memory.py; do not remove it.\n"
            "Here is what you currently remember:\n"
            f"{mem_text}\n"
            "If the user says 'remember ...', summarise and save it.\n"
            "If the user just chats, answer normally.\n"
            "Do not invent API keys. Use the ones already in the code/secrets."
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
    st.session_state.chat = load_temp_chat()

if "last_ai" not in st.session_state:
    st.session_state.last_ai = ""


# Pre-load sessions + long-term memory counts for sidebar
_chat_sessions = load_chat_sessions()
_long_term_data = memory._load()  # list of all memory entries


# ----------------- SIDEBAR (memory + controls + session archive) -----------------
with st.sidebar:
    st.header("🧠 Memory & Sessions")

    # tracker
    short_term_count = len(st.session_state.chat)
    long_term_count = len(_long_term_data)
    sessions_count = len(_chat_sessions)

    st.caption(f"💬 Current chat messages: {short_term_count}")
    st.caption(f"🧠 Long-term memories: {long_term_count}")
    st.caption(f"📚 Saved chat sessions: {sessions_count}")

    st.divider()
    st.subheader("Long-term memory (summary)")
    mem_text = memory.recent_summary()
    if mem_text:
        st.write(mem_text)
    else:
        st.write("No memories yet.")

    new_mem = st.text_input("Add to memory (manual):", key="sidebar_mem_input")
    if new_mem:
        memory.add_fact(new_mem, kind="manual")
        st.success("Saved to memory.")
        st.rerun()

    st.divider()
    st.subheader("Chat controls")

    if st.button("💾 Save current chat"):
        archive_current_chat()
        st.success("Current chat archived.")
        # Reload sessions and re-render
        _chat_sessions = load_chat_sessions()
        st.rerun()

    if st.button("🗑️ Start New Chat"):
        # archive before clearing
        archive_current_chat()
        st.session_state.chat = []
        save_temp_chat([])
        st.success("Chat archived & cleared. Start fresh!")
        st.rerun()

    st.divider()
    st.subheader("📚 Load a saved chat")

    if _chat_sessions:
        # Build labels for selection
        options = []
        for s in _chat_sessions:
            dt = datetime.fromtimestamp(s["ts"]).strftime("%Y-%m-%d %H:%M")
            msg_count = len(s.get("messages", []))
            title = s.get("title", f"Session {s.get('id')}")
            label = f"{s['id']} | {dt} | {msg_count} msgs | {title}"
            options.append(label)

        selected = st.selectbox(
            "Pick a session to load into the current chat:",
            options,
            index=len(options) - 1,  # by default show the most recent
        )

        if st.button("🔁 Load selected session"):
            # find the matching session by id
            try:
                sel_id_str = selected.split("|", 1)[0].strip()
                sel_id = int(sel_id_str)
                sess = next(s for s in _chat_sessions if s["id"] == sel_id)
                st.session_state.chat = sess.get("messages", [])
                save_temp_chat(st.session_state.chat)
                st.success(f"Loaded session {sel_id}.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load session: {e}")
    else:
        st.caption("No archived chats yet.")


# ----------------- MAIN UI -----------------
st.title("🤖 Jarvis AI Dashboard (with memory)")

# header / hero
today = datetime.now().strftime("%A, %B %d, %Y")
weather = get_weather("Basingstoke")
st.subheader(f"Today: {today}  |  Basingstoke weather: {weather}")

st.write("Talk to Jarvis below. You can say things like:")
st.write("- **remember** that I prefer evening workouts")
st.write("- **what do you remember about me?**")
st.write("- **save this chat** so we can reload it later")
st.write("- **add a feeds panel**")
st.write("- **change the layout** to put weather at the top (Jarvis will rewrite app.py)")
st.divider()

# show chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------- CHAT INPUT -----------------
user_msg = st.chat_input("Ask / tell Jarvis something...")

if user_msg:
    # add user msg to history
    st.session_state.chat.append({"role": "user", "content": user_msg})
    save_temp_chat(st.session_state.chat)
    lower = user_msg.lower().strip()

    # explicit memory command
    if lower.startswith("remember "):
        to_store = user_msg[len("remember "):].strip()
        if to_store:
            memory.add_fact(to_store, kind="user")
            ai_reply = f"Got it. I will remember: **{to_store}**"
        else:
            ai_reply = "You said 'remember' but didn't tell me what to remember."
        st.session_state.chat.append({"role": "assistant", "content": ai_reply})
        save_temp_chat(st.session_state.chat)
        with st.chat_message("assistant"):
            st.markdown(ai_reply)

    else:
        # normal AI flow
        with st.chat_message("assistant"):
            with st.spinner("Jarvis thinking..."):
                try:
                    mem_now = memory.recent_summary()
                    ai_reply = call_jarvis(st.session_state.chat, mem_now)

                    # show reply
                    st.markdown(ai_reply)
                    st.session_state.chat.append({"role": "assistant", "content": ai_reply})
                    save_temp_chat(st.session_state.chat)

                    # 🧠 Auto-save memory when Jarvis confirms remembering something
                    if any(
                        kw in ai_reply.lower()
                        for kw in [
                            "i will remember",
                            "stored",
                            "saved to memory",
                            "noted",
                            "core directive",
                        ]
                    ):
                        memory.add_fact(ai_reply, kind="assistant")
                        st.sidebar.success("Jarvis memory updated.")

                    # 🧩 Self-update app.py if code was returned
                    if "```python" in ai_reply:
                        start = ai_reply.find("```python") + len("```python")
                        end = ai_reply.find("```", start)
                        if end != -1:
                            new_code = ai_reply[start:end].strip()
                            with open("app.py", "w", encoding="utf-8") as f:
                                f.write(new_code)
                            st.success("✅ Code updated — rerunning app...")
                            st.stop()

                except Exception:
                    st.error("Jarvis error.")
                    st.code(traceback.format_exc())
