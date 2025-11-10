# modules/weather_panel.py
from __future__ import annotations
import os
import requests
from datetime import datetime
import streamlit as st

# -------------------------------------------------------------------
# Apple-style Weather Panel for Jarvis
# Shows current weather + morning/afternoon/evening forecast (side-by-side)
# -------------------------------------------------------------------

API_TIMEOUT = 8

def _api_key():
    # Uses env or Streamlit secrets; last fallback is your previous sample’s key.
    return (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY")
        or st.secrets.get("weather_api_key", "e5084c56702e0e7de0de917e0e7edbe3")
    )

def get_weather_data(city: str = "Basingstoke"):
    """
    Fetch current + forecast weather data from OpenWeatherMap.
    """
    api_key = _api_key()
    if not api_key:
        return None

    base = "https://api.openweathermap.org/data/2.5"
    try:
        current = requests.get(
            f"{base}/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=API_TIMEOUT,
        ).json()

        forecast = requests.get(
            f"{base}/forecast",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=API_TIMEOUT,
        ).json()

        # Minimal shape checks
        if not isinstance(current, dict) or "main" not in current or "weather" not in current:
            return None
        if not isinstance(forecast, dict) or "list" not in forecast or not isinstance(forecast["list"], list):
            return {"current": current, "forecast": {"list": []}}

        return {"current": current, "forecast": forecast}
    except Exception:
        return None


def _emoji_for(desc: str) -> str:
    d = (desc or "").lower()
    if "storm" in d or "thunder" in d:
        return "⛈️"
    if "rain" in d or "drizzle" in d:
        return "🌧️"
    if "snow" in d or "sleet" in d:
        return "❄️"
    if "fog" in d or "mist" in d or "haze" in d:
        return "🌫️"
    if "cloud" in d or "overcast" in d:
        return "☁️"
    return "☀️"


def _closest_for_hour_today(forecast_list, target_hour: int):
    """
    Pick the forecast entry closest to target_hour for *today*.
    Falls back to the nearest entry overall if none are today.
    """
    if not forecast_list:
        return None

    today = datetime.now().date()
    today_entries = [x for x in forecast_list if datetime.fromtimestamp(x["dt"]).date() == today]
    pool = today_entries if today_entries else forecast_list

    def score(x):
        return abs(datetime.fromtimestamp(x["dt"]).hour - target_hour)

    return min(pool, key=score) if pool else None


def render(default_city: str = "Basingstoke"):
    """
    Render the Apple-style weather card with side-by-side dayparts.
    """
    st.header("🌤️ Weather Forecast")

    city = st.text_input("City:", default_city, key="weather_city_input")
    data = get_weather_data(city)
    if not data:
        st.warning("Weather data unavailable.")
        return

    current = data["current"]
    forecast = data["forecast"]

    # Guard against incomplete payloads
    if "main" not in current or "weather" not in current:
        st.warning("Weather information incomplete.")
        return

    name = current.get("name", city)
    temp = round(current["main"]["temp"])
    desc = current["weather"][0]["description"].capitalize()
    humidity = current["main"].get("humidity", "?")
    wind = round(current.get("wind", {}).get("speed", 0))
    emoji = _emoji_for(desc)
    asof = datetime.now().strftime("%I:%M %p").lstrip("0")

    # ---- Current conditions card (bright gradient, clear contrast) ----
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #8EC5FC 0%, #3B82F6 100%);
            padding: 1.2rem 1.6rem; border-radius: 1.2rem;
            color: #ffffff; box-shadow: 0 8px 20px rgba(0,0,0,0.25);
            border: 1px solid rgba(255,255,255,0.25);
        ">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <h2 style="margin:0; font-weight:700; letter-spacing:.2px;">{emoji} {name}</h2>
                <div style="opacity:.9; font-size:.95rem;">as of {asof}</div>
            </div>
            <div style="display:flex; align-items:center; gap:12px; margin-top:.25rem;">
                <div style="font-size:3rem; font-weight:800; line-height:1;">{temp}°C</div>
                <div style="font-size:1rem; opacity:.95;">{desc} · 💧 {humidity}% · 🌬️ {wind} m/s</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Forecast section ----
    st.markdown("### 🕒 Today")

    # Side-by-side Morning / Afternoon / Evening, aligned to typical local hours
    slots = {"Morning": 9, "Afternoon": 15, "Evening": 21}
    f_list = forecast.get("list", []) if isinstance(forecast, dict) else []
    items = []
    for label, target_hour in slots.items():
        entry = _closest_for_hour_today(f_list, target_hour)
        if entry:
            t = round(entry["main"]["temp"])
            d = entry["weather"][0]["description"].capitalize()
            items.append((label, t, d))
        else:
            items.append((label, None, None))

    cols = st.columns(len(items))
    for i, (label, t, d) in enumerate(items):
        with cols[i]:
            # Card styling: subtle glass on white/gray Streamlit background
            if t is not None:
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255,255,255,0.60);
                        backdrop-filter: blur(6px);
                        -webkit-backdrop-filter: blur(6px);
                        border: 1px solid rgba(0,0,0,0.08);
                        border-radius: 1rem;
                        padding: 0.7rem 0.8rem;
                        text-align: center;
                        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
                    ">
                        <p style="margin:0; font-weight:700; color:#0f172a;">{label}</p>
                        <p style="font-size:1.6rem; margin:.15rem 0 .1rem 0;">{_emoji_for(d)}</p>
                        <p style="margin:0; font-size:1rem; font-weight:700; color:#0f172a;">{t}°C</p>
                        <p style="margin:0; font-size:.85rem; opacity:.85; color:#0f172a;">{d}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255,255,255,0.50);
                        backdrop-filter: blur(5px);
                        border: 1px solid rgba(0,0,0,0.06);
                        border-radius: 1rem;
                        padding: 0.7rem 0.8rem;
                        text-align: center;
                    ">
                        <p style="margin:0; font-weight:700;">{label}</p>
                        <p style="margin:.2rem 0 0 0; font-size:.9rem; opacity:.75;">No data</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
