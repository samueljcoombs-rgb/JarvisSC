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

# Use the same top-tier model family you're chatting with here
JARVIS_MODEL = "gpt-5"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("No OPENAI_API_KEY found. Add it in Streamlit secrets or env.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------- WEATHER -----------------
def get_weather(city: str = "Basingstoke"):
    """
    Fetch current weather from OpenWeatherMap safely.
    Returns a dict with fields or None on error.
    """
    owm_key = (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY")
        or st.secrets.get("weather_api_key")
        or "e5084c56702e0e7de0de917e0e7edbe3"  # fallback from earlier
    )
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={owm_key}&units=metric"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return {
            "city": data.get("name", city),
            "temp": data["main"]["temp"],
            "desc": data["weather"][0]["description"].capitalize(),
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"],
            "icon": data["weather"][0]["icon"],
        }
    except Exception:
        return None


def format_weather_summary(w):
    if not w:
        return "Weather data not available."
    return f"{w['desc']}, {w['temp']}°C"


# ----------------- CHAT / SESSION STORAGE -----------------
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
    """Archive the current chat to chat_sessions.json."""
    chat = st.session_state.get("chat", [])
    if not chat:
        return
    sessions = load_json(CHAT_SESSIONS_FILE, [])
    ts = int(time.time())
    new_id = max([s.get("id", 0) for s in sessions], default=0) + 1
    title = next((m["content"] for m in chat if m["role"] == "user"), f"Session {new_id}")[:60]
    sessions.append({"id": new_id, "ts": ts, "title": title, "messages": chat})
    save_json(CHAT_SESSIONS_FILE, sessions)


# ----------------- OPENAI CALL -----------------
def call_jarvis(chat_history, mem_text: str) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are Jarvis, an AI assistant inside a Streamlit app (app.py).\n"
            "You can modify layout, visuals, and UI components.\n"
            "You must NEVER modify or add any API key handling or st.secrets usage.\n"
            "You must NOT change the implementation of get_weather().\n"
            "You must NOT change memory or chat saving logic.\n"
            "If you output code, it must be a FULL, RUNNABLE app.py.\n"
            "Unsafe changes to secrets or keys will be ignored.\n\n"
            f"Current long-term memory:\n{mem_text}\n"
        ),
    }

    resp = client.chat.completions.create(
        model=JARVIS_MODEL,
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

    long_term = memory._load()
    mem_text = memory.recent_summary()
    sessions = load_json(CHAT_SESSIONS_FILE, [])

    st.caption(f"🤖 Jarvis model: {JARVIS_MODEL}")
    st.caption(f"💬 Current chat messages: {len(st.session_state.chat)}")
    st.caption(f"🧠 Long-term memories: {len(long_term)}")
    st.caption(f"📚 Saved sessions: {len(sessions)}")

    st.divider()
    st.subheader("Long-term memory (summary)")
    st.write(mem_text if mem_text else "No memories yet.")

    new_mem = st.text_input("Add to memory manually:")
    if new_mem:
        memory.add_fact(new_mem, kind="manual")
        st.success("Saved to memory.")
        st.rerun()

    st.divider()
    st.subheader("Chat controls")

    if st.button("💾 Save current chat"):
        archive_current_chat()
        st.success("Chat archived.")
        st.rerun()

    if st.button("🗑️ Start New Chat"):
        archive_current_chat()
        st.session_state.chat = []
        save_json(TEMP_CHAT_FILE, [])
        st.success("New chat started.")
        st.rerun()

    st.divider()
    st.subheader("📚 Load a saved chat")
    if sessions:
        options = []
        for s in sessions:
            dt = datetime.fromtimestamp(s["ts"]).strftime("%Y-%m-%d %H:%M")
            msg_count = len(s.get("messages", []))
            title = s.get("title", f"Session {s.get('id')}")
            label = f"{s['id']} | {dt} | {msg_count} msgs | {title}"
            options.append(label)

        selected = st.selectbox("Pick a session:", options, index=len(options) - 1)
        if st.button("🔁 Load selected session"):
            try:
                sel_id = int(selected.split("|", 1)[0].strip())
                sess = next(s for s in sessions if s["id"] == sel_id)
                st.session_state.chat = sess.get("messages", [])
                save_json(TEMP_CHAT_FILE, st.session_state.chat)
                st.success(f"Loaded session {sel_id}.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load session: {e}")
    else:
        st.caption("No archived chats yet.")

    st.divider()
    st.subheader("Code safety")

    # 🔙 Revert to last code backup (app_backup.py)
    if st.button("🔙 Revert to last code backup"):
        if os.path.exists("app_backup.py"):
            try:
                with open("app_backup.py", "r", encoding="utf-8") as f_backup:
                    prev_code = f_backup.read()
                with open("app.py", "w", encoding="utf-8") as f_app:
                    f_app.write(prev_code)
                st.success("Reverted to previous app.py — reloading...")
                st.stop()
            except Exception as e:
                st.error(f"Failed to revert: {e}")
        else:
            st.warning("No backup file found yet.")


# ----------------- MAIN UI -----------------
st.title("🤖 Jarvis AI Dashboard (with memory)")

today = datetime.now().strftime("%A, %B %d, %Y")
basingstoke_weather = get_weather("Basingstoke")
header_weather = format_weather_summary(basingstoke_weather)
st.subheader(f"Today: {today}  |  Basingstoke: {header_weather}")

st.write("You can say things like:")
st.write("- **remember** that I prefer evening workouts")
st.write("- **save this chat** so we can reload it later")
st.write("- **redesign the layout** (but keep API keys & memory logic untouched)")
st.divider()

# Two-column layout: Chat (left) and Weather panel (right)
col1, col2 = st.columns([2, 1])

# ---- LEFT: CHAT ----
with col1:
    st.header("💬 Chat")

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
                        ai_reply = call_jarvis(
                            st.session_state.chat,
                            memory.recent_summary()
                        )
                        st.markdown(ai_reply)
                        st.session_state.chat.append(
                            {"role": "assistant", "content": ai_reply}
                        )
                        save_json(TEMP_CHAT_FILE, st.session_state.chat)

                        # 🔧 Self-update: ONLY if safe (no secrets / key handling)
                        if "```python" in ai_reply:
                            start = ai_reply.find("```python") + len("```python")
                            end = ai_reply.find("```", start)
                            if end != -1:
                                code = ai_reply[start:end].strip()

                                # Reject if code tries to touch secrets or key vars
                                unsafe_markers = [
                                    "st.secrets",
                                    "OPENAI_API_KEY",
                                    "OWM_API_KEY",
                                    "weather_api_key",
                                ]
                                if any(m in code for m in unsafe_markers):
                                    st.warning(
                                        "❌ Update rejected: unsafe API key changes detected."
                                    )
                                else:
                                    # First, back up current app.py
                                    try:
                                        with open("app.py", "r", encoding="utf-8") as f_old:
                                            old_code = f_old.read()
                                        with open("app_backup.py", "w", encoding="utf-8") as f_backup:
                                            f_backup.write(old_code)
                                    except Exception as e:
                                        st.warning(
                                            f"⚠️ Could not create backup before update: {e}"
                                        )

                                    # Now write the new code
                                    with open("app.py", "w", encoding="utf-8") as f:
                                        f.write(code)

                                    st.success("✅ Code updated — reloading app...")
                                    st.stop()

                    except Exception:
                        st.error("Jarvis error.")
                        st.code(traceback.format_exc())

# ---- RIGHT: WEATHER PANEL ----
with col2:
    st.header("🌦️ Weather Panel")

    city = st.text_input("City:", "Basingstoke")
    w = get_weather(city)

    if not w:
        st.write("Weather data not available.")
    else:
        # Apple Watch-ish card
        st.markdown(
            """
            <div style="
                border-radius: 16px;
                padding: 16px;
                border: 1px solid #333;
                background: linear-gradient(135deg, #1c1c1e, #2c2c2e);
                color: white;
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
            ">
            """,
            unsafe_allow_html=True,
        )

        emoji = "☀️"
        d = w["desc"].lower()
        if "cloud" in d:
            emoji = "☁️"
        elif "rain" in d or "drizzle" in d:
            emoji = "🌧️"
        elif "storm" in d or "thunder" in d:
            emoji = "⛈️"
        elif "snow" in d:
            emoji = "❄️"
        elif "fog" in d or "mist" in d:
            emoji = "🌫️"

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 14px; opacity: 0.8;">{w['city']}</div>
                    <div style="font-size: 32px; font-weight: 600;">{w['temp']}°C</div>
                    <div style="font-size: 14px; opacity: 0.8;">{w['desc']}</div>
                </div>
                <div style="font-size: 40px;">{emoji}</div>
            </div>
            <div style="margin-top: 8px; font-size: 12px; opacity: 0.8;">
                Humidity: {w['humidity']}% &nbsp;•&nbsp; Wind: {w['wind']} m/s
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
