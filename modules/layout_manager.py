# modules/layout_manager.py

import os
import time
from datetime import datetime

import requests
import streamlit as st
import traceback


# ------------ WEATHER (local to this module) ------------

OWM_FALLBACK_KEY = "e5084c56702e0e7de0de917e0e7edbe3"  # your existing key


def get_weather(city: str = "Basingstoke"):
    """
    Simple OpenWeatherMap fetcher.
    Returns a dict or None on error.
    """
    key = (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY", OWM_FALLBACK_KEY)
    )
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={key}&units=metric"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return {
            "city": data.get("name", city),
            "temp": data["main"]["temp"],
            "desc": data["weather"][0]["description"].capitalize(),
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"],
        }
    except Exception:
        return None


def render_weather_panel():
    """Right-hand Apple-style weather card for Basingstoke."""
    st.header("🌦️ Weather")

    w = get_weather("Basingstoke")
    if not w:
        st.write("Weather data not available.")
        return

    # pick an emoji from description
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
                font-size: 48px;
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        base_temp = float(w["temp"])
    except Exception:
        base_temp = 0.0

    hourly = [
        {"time": "Morning", "icon": "🌤️", "temp": round(base_temp)},
        {"time": "Afternoon", "icon": "🌥️" if "cloud" in d else emoji, "temp": round(base_temp + 2)},
        {"time": "Evening", "icon": "🌙" if "rain" not in d else "🌧️", "temp": round(base_temp - 1)},
    ]

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
            <div class="chip">🌬️ Wind: {w['wind']} m/s</div>
            <div class="chip">💧 Humidity: {w['humidity']}%</div>
        </div>
        <div class="hourly">
            {''.join([
                f"<div class='slot'><div class='slot-time'>{h['time']}</div><div class='slot-icon'>{h['icon']}</div><div class='slot-temp'>{h['temp']}°</div></div>"
                for h in hourly
            ])}
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)


# ------------ CHAT + SELF-EDIT (left side) ------------

def _try_self_update(ai_reply: str, safe_write_module):
    """
    Look for a ```python``` block and, if present,
    treat it as a new version of layout_manager.py.
    """
    if "```python" not in ai_reply:
        return

    start = ai_reply.find("```python") + len("```python")
    end = ai_reply.find("```", start)
    if end == -1:
        return

    new_code = ai_reply[start:end].strip()
    # Compile + backup + write is handled by safe_write_module in app.py
    safe_write_module("layout_manager", new_code)


def render(
    chat,
    mem_text,
    call_jarvis,
    safe_write_module,
    safe_save_json,
    temp_chat_file,
    memory_module,
):
    """
    Main layout: chat on the left, weather on the right.
    This is what app.py calls.
    """
    # Header row
    today = datetime.now().strftime("%A, %d %B %Y")
    st.subheader(f"Today: {today}")

    col1, col2 = st.columns([2, 1])

    # ----- LEFT: CHAT -----
    with col1:
        st.header("💬 Chat")

        # Show history
        for msg in chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input("Ask / tell Jarvis something...")
        if user_msg:
            chat.append({"role": "user", "content": user_msg})
            safe_save_json(temp_chat_file, chat)

            lower = user_msg.lower().strip()

            # "remember ..." => push to long-term memory
            if lower.startswith("remember "):
                to_store = user_msg[len("remember "):].strip()
                if to_store:
                    memory_module.add_fact(to_store, kind="user")
                    ai_reply = f"Got it. I will remember: **{to_store}**"
                else:
                    ai_reply = "You said 'remember' but didn't tell me what to remember."

                chat.append({"role": "assistant", "content": ai_reply})
                safe_save_json(temp_chat_file, chat)
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)

            else:
                # Normal Jarvis response
                with st.chat_message("assistant"):
                    with st.spinner("Jarvis thinking..."):
                        try:
                            ai_reply = call_jarvis(chat, mem_text)
                            st.markdown(ai_reply)
                            chat.append({"role": "assistant", "content": ai_reply})
                            safe_save_json(temp_chat_file, chat)

                            # Let Jarvis update THIS module only, safely
                            _try_self_update(ai_reply, safe_write_module)

                        except Exception:
                            st.error("Jarvis error.")
                            st.code(traceback.format_exc())

    # ----- RIGHT: WEATHER -----
    with col2:
        render_weather_panel()
