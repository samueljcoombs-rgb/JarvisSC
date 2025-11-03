import streamlit as st
import requests
from datetime import datetime
import os


def get_weather(city="Basingstoke"):
    """
    Fetch current weather from OpenWeatherMap safely.
    """
    api_key = (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY")
        or st.secrets.get("weather" + "_api_key")
        or None
    )
    if not api_key:
        return None

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        return {
            "city": data.get("name", city),
            "temp": data["main"]["temp"],
            "desc": data["weather"][0]["description"].capitalize(),
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"],
        }
    except Exception:
        return None


def render(**kwargs):
    """
    Render weather panel. Flexible **kwargs so app.py can pass anything.
    """
    st.header("🌦️ Weather")

    city = st.text_input("City:", "Basingstoke", key=f"weather_city_{id(st.session_state)}")
    w = get_weather(city)

    if not w:
        st.info("Weather data not available.")
        return

    desc = w["desc"].lower()
    emoji = "☀️"
    if "cloud" in desc:
        emoji = "☁️"
    elif "rain" in desc or "drizzle" in desc:
        emoji = "🌧️"
    elif "storm" in desc:
        emoji = "⛈️"
    elif "snow" in desc:
        emoji = "❄️"
    elif "fog" in desc or "mist" in desc:
        emoji = "🌫️"

    as_of = datetime.now().strftime("%I:%M %p").lstrip("0")

    st.markdown(
        f"""
        <div style="
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #e6eef8;
            background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            color: #111;
        ">
            <h3 style="margin:0;">{emoji} {w['city']}</h3>
            <p style="margin:4px 0; font-size:48px; font-weight:700;">{round(w['temp'])}°C</p>
            <p style="margin:0; color:#555;">{w['desc']} (as of {as_of})</p>
            <div style="margin-top:10px; font-size:14px;">
                💧 Humidity: {w['humidity']}%<br>
                🌬️ Wind: {w['wind']} m/s
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
