import os
from datetime import datetime

import requests
import streamlit as st


def _get_weather(city: str = "Basingstoke"):
    """
    Uses the same OpenWeather API style you've been using:
    - env OWM_API_KEY
    - Streamlit secret OWM_API_KEY
    - Streamlit secret weather_api_key
    - hard-coded fallback (your original key)
    Returns dict or None.
    """
    owm_key = (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY")
        or st.secrets.get("weather_api_key")
        or "e5084c56702e0e7de0de917e0e7edbe3"
    )
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather?"
            f"q={city}&appid={owm_key}&units=metric"
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
            "icon": data["weather"][0]["icon"],
        }
    except Exception:
        return None


def render(default_city: str = "Basingstoke"):
    st.header("🌤️ Weather")

    city = st.text_input("City:", default_city)
    w = _get_weather(city)

    if not w:
        st.write("Weather data not available.")
        return

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

    slot_meta = lambda: f"Wind {w['wind']} m/s"

    card_html = f"""
    <div class="wx-card">
        <div class="wx-top">
            <div>
                <div class="wx-loc">{w['city']}</div>
                <div class="wx-temp">{round(w['temp'])}°C</div>
                <div class="wx-cond"><span style="font-size: 22px;">{emoji}</span><span>{w['desc']}</span></div>
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
