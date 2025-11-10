# modules/weather_panel.py
from __future__ import annotations
import os
import requests
from datetime import datetime, timedelta
import streamlit as st
from statistics import mean
from collections import Counter

# -------------------------------------------------------------------
# Apple-style Weather Panel for Jarvis
# Current + Morning/Afternoon/Evening forecast (side-by-side)
# Distinct windows; past windows use *tomorrow*; style advice card.
# - No city input above the widget (uses default_city)
# - Tiny orange "T" badge on any pill that corresponds to tomorrow
# -------------------------------------------------------------------

API_TIMEOUT = 8

def _api_key():
    return (
        os.getenv("OWM_API_KEY")
        or st.secrets.get("OWM_API_KEY")
        or st.secrets.get("weather_api_key", "e5084c56702e0e7de0de917e0e7edbe3")
    )

def get_weather_data(city: str = "Basingstoke"):
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

        if not isinstance(current, dict) or "main" not in current or "weather" not in current:
            return None
        if not isinstance(forecast, dict) or "list" not in forecast:
            forecast = {"list": []}
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

def _window_for_label(now_hour: int, label: str):
    windows = {
        "Morning": (6, 12),
        "Afternoon": (12, 18),
        "Evening": (18, 24),
    }
    return windows[label]

def _pick_day_for_window(now: datetime, start_h: int, end_h: int):
    if now.hour >= end_h:
        return (now + timedelta(days=1)).date()
    return now.date()

def _summarize_window(forecast_list, target_date, start_h, end_h):
    if not forecast_list:
        return None
    window_items = []
    for x in forecast_list:
        try:
            dt = datetime.fromtimestamp(x["dt"])
        except Exception:
            continue
        if dt.date() != target_date:
            continue
        if start_h <= dt.hour < end_h:
            window_items.append(x)

    if not window_items:
        for widen in (1, 2):
            widened = []
            for x in forecast_list:
                try:
                    dt = datetime.fromtimestamp(x["dt"])
                except Exception:
                    continue
                if dt.date() != target_date:
                    continue
                if (start_h - widen) <= dt.hour < (end_h + widen):
                    widened.append(x)
            if widened:
                window_items = widened
                break

    if not window_items:
        return None

    temps = [i.get("main", {}).get("temp") for i in window_items]
    temps = [t for t in temps if isinstance(t, (int, float))]
    descs = []
    for i in window_items:
        w = i.get("weather") or []
        if w and isinstance(w, list) and w[0].get("description"):
            descs.append(w[0]["description"])
    pops  = [i.get("pop", 0) for i in window_items]
    winds = [i.get("wind", {}).get("speed", 0) for i in window_items]

    if not temps or not descs:
        return None

    avg_temp = round(mean(temps))
    desc = Counter(descs).most_common(1)[0][0].capitalize()
    pop = max(pops) if pops else 0
    wind_vals = [w for w in winds if isinstance(w, (int, float))]
    wind = round(mean(wind_vals)) if wind_vals else 0

    return {"temp": avg_temp, "desc": desc, "pop": pop, "wind": wind}

def _style_advice(current_desc: str, current_temp: int, current_wind: int, daypart_pop_max: float):
    tip = []
    d = (current_desc or "").lower()
    rainish = any(k in d for k in ["rain", "drizzle", "shower", "storm"])
    pop = max(daypart_pop_max, 1.0 if rainish else 0.0)

    if pop >= 0.6:
        tip.append("🌂 High chance of rain — take an umbrella and a waterproof coat.")
    elif pop >= 0.3:
        tip.append("☔ Possible showers — a light jacket or compact umbrella is smart.")
    else:
        tip.append("😎 Low rain risk — dress for comfort.")

    if current_temp <= 5:
        tip.append("🧥 It’s cold — warm layers recommended.")
    elif current_temp <= 12:
        tip.append("🧣 Chilly — a light jacket or sweater helps.")
    elif current_temp >= 24:
        tip.append("🧴 Warm — consider sunscreen and breathable fabrics.")

    if current_wind >= 8:
        tip.append("💨 Windy — a windbreaker will help.")

    return " ".join(tip) if tip else "Dress comfortably for the day."

# ---- NEW: robust HTML renderer that strips indentation on every line ----
def _html(s: str) -> str:
    if not s:
        return s
    lines = s.splitlines()
    # strip leading spaces/tabs on every line to avoid markdown code blocks
    lines = [ln.lstrip() for ln in lines]
    out = "\n".join(lines).strip()
    # ensure the very first char is '<' so markdown doesn't treat it as text
    return out

def render(default_city: str = "Basingstoke"):
    st.header("🌤️ Weather Forecast")

    city = default_city
    data = get_weather_data(city)
    if not data:
        st.warning("Weather data unavailable.")
        return

    current = data["current"]
    forecast = data["forecast"]

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

    # ---- Current conditions card ----
    st.markdown(_html(f"""
    <div style="background: linear-gradient(135deg, #8EC5FC 0%, #3B82F6 100%);
                padding: 1.2rem 1.6rem; border-radius: 1.2rem;
                color: #ffffff; box-shadow: 0 8px 20px rgba(0,0,0,0.25);
                border: 1px solid rgba(255,255,255,0.25);">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <h2 style="margin:0; font-weight:700; letter-spacing:.2px;">{emoji} {name}</h2>
            <div style="opacity:.9; font-size:.95rem;">as of {asof}</div>
        </div>
        <div style="display:flex; align-items:center; gap:12px; margin-top:.25rem;">
            <div style="font-size:3rem; font-weight:800; line-height:1;">{temp}°C</div>
            <div style="font-size:1rem; opacity:.95;">{desc} · 💧 {humidity}% · 🌬️ {wind} m/s</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # ---- Build distinct dayparts with windowing logic ----
    f_list = forecast.get("list", []) if isinstance(forecast, dict) else []
    now = datetime.now()
    parts = {}
    tomorrow_flags = {}
    daypart_order = ["Morning", "Afternoon", "Evening"]
    for label in daypart_order:
        start_h, end_h = _window_for_label(now.hour, label)
        target_date = _pick_day_for_window(now, start_h, end_h)
        parts[label] = _summarize_window(f_list, target_date, start_h, end_h)
        tomorrow_flags[label] = (target_date != now.date())

    # ---- Show side-by-side pills (with tiny orange "T" for tomorrow) ----
    st.markdown("### 🕒 Today & Next")
    cols = st.columns(3)
    daypart_pops = []
    for i, label in enumerate(daypart_order):
        info = parts.get(label)
        is_tomorrow = tomorrow_flags.get(label, False)
        with cols[i]:
            if info:
                daypart_pops.append(info.get("pop", 0))
                badge = ("""
                <div style="position:absolute; top:8px; right:8px;
                            width:18px; height:18px; border-radius:999px;
                            background:#f97316; color:#fff; font-weight:800;
                            font-size:0.7rem; display:flex; align-items:center;
                            justify-content:center; box-shadow:0 2px 6px rgba(0,0,0,0.18);">T</div>
                """ if is_tomorrow else "")
                st.markdown(_html(f"""
                <div style="position: relative;
                            background: rgba(255,255,255,0.60);
                            backdrop-filter: blur(6px);
                            -webkit-backdrop-filter: blur(6px);
                            border: 1px solid rgba(0,0,0,0.08);
                            border-radius: 1rem;
                            padding: 0.7rem 0.8rem;
                            text-align: center;
                            box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
                    {badge}
                    <p style="margin:0; font-weight:700; color:#0f172a;">{label}</p>
                    <p style="font-size:1.6rem; margin:.15rem 0 .1rem 0;">{_emoji_for(info['desc'])}</p>
                    <p style="margin:0; font-size:1rem; font-weight:700; color:#0f172a;">{info['temp']}°C</p>
                    <p style="margin:0; font-size:.85rem; opacity:.85; color:#0f172a;">{info['desc']}</p>
                </div>
                """), unsafe_allow_html=True)
            else:
                daypart_pops.append(0)
                badge = ("""
                <div style="position:absolute; top:8px; right:8px;
                            width:18px; height:18px; border-radius:999px;
                            background:#f97316; color:#fff; font-weight:800;
                            font-size:0.7rem; display:flex; align-items:center;
                            justify-content:center; box-shadow: 0 2px 6px rgba(0,0,0,0.18);">T</div>
                """ if is_tomorrow else "")
                st.markdown(_html(f"""
                <div style="position: relative;
                            background: rgba(255,255,255,0.50);
                            backdrop-filter: blur(5px);
                            border: 1px solid rgba(0,0,0,0.06);
                            border-radius: 1rem;
                            padding: 0.7rem 0.8rem;
                            text-align: center;">
                    {badge}
                    <p style="margin:0; font-weight:700;">{label}</p>
                    <p style="margin:.2rem 0 0 0; font-size:.9rem; opacity:.75;">No data</p>
                </div>
                """), unsafe_allow_html=True)

    # ---- Style advice card (depends on rain/temp/wind) ----
    advice = _style_advice(desc, temp, wind, max(daypart_pops) if daypart_pops else 0)
    st.markdown(_html(f"""
    <div style="margin-top: .8rem;
                background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
                border: 1px solid rgba(0,0,0,0.06);
                border-radius: 1rem;
                padding: 0.9rem 1rem;
                color: #0f172a;
                box-shadow: 0 6px 16px rgba(0,0,0,0.12);">
        <div style="font-weight:700; margin-bottom:.25rem;">👕 What to wear</div>
        <div style="opacity:.95;">{advice}</div>
    </div>
    """), unsafe_allow_html=True)
