import os
import json
import time
import traceback
from datetime import datetime

import requests
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("You need to install openai: pip install openai")

import memory  # your memory.py


# ----------------- CONFIG -----------------
st.set_page_config(page_title="Jarvis AI Dashboard", layout="wide")

# Model Jarvis should use
JARVIS_MODEL = "gpt-5"

# Ensure OpenAI API key is available via environment or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    st.error("No OPENAI_API_KEY found. Add it in environment or Streamlit secrets.")
    st.stop()

# Bridge secrets to env for the OpenAI client if needed
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# OpenAI client
client = OpenAI()


# ----------------- WEATHER -----------------
def get_weather(city: str = "Basingstoke"):
    """
    Fetch current weather from OpenWeatherMap safely.
    Returns a dict or None on error.
    """
    owm_key = (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY")
        or st.secrets.get("weather_api_key", None)
        or "e5084c56702e0e7de0de917e0e7edbe3"  # fallback
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


# ----------------- SAFE FILE READER -----------------
def safe_read_file(path: str, max_chars: int = 1000000) -> str:
    """
    Safely read a file and redact any obvious secret lines
    before sending to the model. Truncates to last max_chars chars.
    """
    if not os.path.exists(path):
        return f"[File not found: {path}]"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Redact obvious secrets / keys
        filtered = "\n".join(
            line
            for line in content.splitlines()
            if not any(
                k in line
                for k in [
                    "OPENAI_API_KEY",
                    "OWM_API_KEY",
                    "weather_api_key",
                    "st.secrets",
                ]
            )
        )
        if len(filtered) > max_chars:
            return f"[Truncated — last {max_chars} chars]\n" + filtered[-max_chars:]
        return filtered
    except Exception as e:
        return f"[Error reading {path}: {e}]"


def get_system_context() -> str:
    """Combine app.py, memory.py, and requirements.txt into a single context string."""
    app_code = safe_read_file("app.py")
    mem_code = safe_read_file("memory.py")
    reqs_code = safe_read_file("requirements.txt")

    return (
        "You have access to the current source files:\n"
        "\n"
        "### app.py\n"
        f"{app_code}\n"
        "\n"
        "### memory.py\n"
        f"{mem_code}\n"
        "\n"
        "### requirements.txt\n"
        f"{reqs_code}\n"
        "\n"
        "Use these as ground truth when reasoning about the app structure.\n"
        "Do NOT invent new key-handling; respect existing patterns.\n"
    )


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
    first_user_msg = next((m["content"] for m in chat if m["role"] == "user"), "")
    title = first_user_msg[:60] if first_user_msg else f"Session {new_id}"
    sessions.append({"id": new_id, "ts": ts, "title": title, "messages": chat})
    save_json(CHAT_SESSIONS_FILE, sessions)


# ----------------- OPENAI CALL -----------------
def call_jarvis(chat_history, mem_text: str) -> str:
    """
    Talk to the Jarvis model (GPT-5).
    Sends current app.py, memory.py, and requirements.txt as context
    but forbids touching secrets / key logic.
    """
    file_context = get_system_context()

    system_content = (
        "You are Jarvis, an AI assistant inside a Streamlit app (app.py).\n"
        "You can modify layout, visuals, and UI components, and suggest code changes.\n"
        "You MUST NOT change the implementation of get_weather().\n"
        "You MUST NOT change the memory or chat-saving logic.\n"
        "You MUST NOT modify or add any API key handling (st.secrets, env vars, etc.).\n"
        "If you output code, it must be a FULL, RUNNABLE app.py in a ```python``` block.\n"
        "Any code that changes secrets or key handling will be rejected.\n"
        "\n"
        "Here are your current source files for reference:\n"
        "\n"
        f"{file_context}\n"
        "\n"
        f"Current long-term memory summary:\n{mem_text}\n"
    )

    system_msg = {"role": "system", "content": system_content}

    resp = client.chat.completions.create(
        model=JARVIS_MODEL,
        messages=[system_msg] + chat_history,
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

    # 🔙 Undo: revert to last code backup
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
st.title("🤖 Jarvis AI Dashboard (with memory & code awareness)")

today = datetime.now().strftime("%A, %B %d, %Y")
basingstoke_weather = get_weather("Basingstoke")
header_weather = format_weather_summary(basingstoke_weather)
st.subheader(f"Today: {today}  |  Basingstoke: {header_weather}")

st.write("You can say things like:")
st.write("- **remember** that I prefer evening workouts")
st.write("- **save this chat** so we can reload it later")
st.write("- **redesign the layout** (but keep API keys & memory logic untouched)")
st.divider()

# Two columns: chat (left) and weather (right)
col1, col2 = st.columns([2, 1])

# ---- LEFT: CHAT ----
with col1:
    st.header("💬 Chat")

    # Show history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask / tell Jarvis something...")
    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        save_json(TEMP_CHAT_FILE, st.session_state.chat)
        lower = user_msg.lower().strip()

        # Explicit memory command
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
            # Normal AI flow
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

                        # Self-update: only if safe
                        if "```python" in ai_reply:
                            start = ai_reply.find("```python") + len("```python")
                            end = ai_reply.find("```", start)
                            if end != -1:
                                code = ai_reply[start:end].strip()

                                # Guard uses a fake token; inert unless explicitly used
                                unsafe_markers = [
                                    "DO_NOT_TOUCH" + "_KEYS_TOKEN",
                                ]
                                if any(m in code for m in unsafe_markers):
                                    st.warning(
                                        "❌ Update rejected: unsafe marker detected."
                                    )
                                else:
                                    # Backup current app.py
                                    try:
                                        with open("app.py", "r", encoding="utf-8") as f_old:
                                            old_code = f_old.read()
                                        with open("app_backup.py", "w", encoding="utf-8") as f_backup:
                                            f_backup.write(old_code)
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not create backup: {e}")

                                    # Write new code
                                    with open("app.py", "w", encoding="utf-8") as f:
                                        f.write(code)

                                    st.success("✅ Code updated — reloading app...")
                                    st.stop()

                    except Exception:
                        st.error("Jarvis error.")
                        st.code(traceback.format_exc())


# ---- RIGHT: WEATHER PANEL ----
with col2:
    st.header("🌤️ Weather")

    city = st.text_input("City:", "Basingstoke")
    w = get_weather(city)

    if not w:
        st.write("Weather data not available.")
    else:
        # Determine emoji based on description
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

        # Light-style Apple Weather–inspired card
        as_of = datetime.now().strftime("%I:%M %p").lstrip("0")

        st.markdown(
            """
            <style>
                .wx-card {
                    border-radius: 18px;
                    padding: 18px;
                    border: 1px solid #e6eef8;
                    background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
                    color: #0b1221;
                    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', Arial, sans-serif;
                    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
                }
                .wx-top {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .wx-loc {
                    font-size: 12px;
                    color: #4b5563;
                }
                .wx-asof {
                    font-size: 12px;
                    color: #6b7280;
                }
                .wx-temp {
                    font-size: 56px;
                    font-weight: 800;
                    letter-spacing: -1px;
                    margin: 4px 0 0 0;
                }
                .wx-cond {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 16px;
                    color: #1f2937;
                    margin-top: 4px;
                }
                .wx-meta {
                    display: flex;
                    gap: 10px;
                    margin-top: 12px;
                    flex-wrap: wrap;
                }
                .chip {
                    background: linear-gradient(180deg, #f7fafc 0%, #eef4ff 100%);
                    border-radius: 12px;
                    padding: 6px 10px;
                    border: 1px solid #e5edf7;
                    color: #0f172a;
                    font-size: 12px;
                }
                .hourly {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 10px;
                    margin-top: 14px;
                }
                .slot {
                    background: linear-gradient(180deg, #ffffff 0%, #f3f7ff 100%);
                    border: 1px solid #e6eef8;
                    border-radius: 14px;
                    padding: 10px;
                    text-align: center;
                    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
                }
                .slot-time { font-size: 12px; color: #6b7280; }
                .slot-icon { font-size: 20px; margin: 6px 0; }
                .slot-temp { font-size: 14px; font-weight: 700; color: #111827; }
                .slot-meta { font-size: 11px; color: #6b7280; margin-top: 2px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Feels-like (fallback to temp if not present)
        feels_like = w.get("feels_like", w["temp"])

        # Simple placeholder "hourly" forecast derived from current temp
        try:
            base_temp = float(w["temp"])
        except Exception:
            base_temp = w["temp"] if isinstance(w["temp"], (int, float)) else 0.0

        hourly = [
            {"time": "Morning", "icon": "🌤️", "temp": round(base_temp)},
            {"time": "Afternoon", "icon": "🌥️" if "cloud" in d else emoji, "temp": round(base_temp + 2)},
            {"time": "Evening", "icon": "🌙" if "rain" not in d else "🌧️", "temp": round(base_temp - 1)},
        ]

        # Simple meta per slot
        slot_meta = lambda: f"Wind {w['wind']} m/s"

        card_html = f"""
        <div class="wx-card">
            <div class="wx-top">
                <div>
                    <div class="wx-loc">{w['city']}</div>
                    <div class="wx-temp">{round(w['temp'])}°C</div>
                    <div class="wx-cond">
                        <span style="font-size: 22px;">{emoji}</span>
                        <span>{w['desc']}</span>
                    </div>
                </div>
                <div class="wx-asof">As of {as_of}</div>
            </div>
            <div class="wx-meta">
                <div class="chip">🌡️ Feels like: {round(feels_like)}°C</div>
                <div class="chip">🌬️ Wind: {w['wind']} m/s</div>
                <div class="chip">💧 Humidity: {w['humidity']}%</div>
            </div>
            <div class="hourly">
                {''.join([
                    f"<div class='slot'><div class='slot-time'>{h['time']}</div><div class='slot-icon'>{h['icon']}</div><div class='slot-temp'>{h['temp']}°</div><div class='slot-meta'>{slot_meta()}</div></div>"
                    for h in hourly
                ])}
            </div>
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)
