# modules/weather_panel.py
from __future__ import annotations
import os
import requests
from datetime import datetime, timedelta, date
import streamlit as st
from statistics import mean
from collections import Counter

# -------------------------------------------------------------------
# Apple-style Weather Panel for Jarvis
# Current + adaptive Morning/Afternoon/Evening (side-by-side)
# - Windows use 06–12 / 12–18 / 18–24
# - Remaining windows for TODAY first (chronological), then TOMORROW
# - Each pill shows a small "Tomorrow" badge when applicable
# - Style advice card under the row
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
    if "rain" in d or "drizzle" in d or "shower" in d:
        return "🌧️"
    if "snow" in d or "sleet" in d:
        return "❄️"
    if "fog" in d or "mist" in d or "haze" in d:
        return "🌫️"
    if "cloud" in d or "overcast" in d:
        return "☁️"
    return "☀️"

# Windows: label -> (start_hour, end_hour)
WINDOWS = [
    ("Morning", 6, 12),
    ("Afternoon", 12, 18),
    ("Evening", 18, 24),
]

def _target_date_for_window(now: datetime, start_h: int, end_h: int) -> date:
    """
    If the window already passed today (now.hour >= end_h), show it for tomorrow.
    Otherwise, show today.
    """
    return (now + timedelta(days=1)).date() if now.hour >= end_h else now.date()

def _summarize_window(forecast_list, target_date: date, start_h: int, end_h: int):
    """Summarize temps/desc/pop/wind for entries within the window, with gentle fallback."""
    if not forecast_list:
        return None

    window_items = []
    for x in forecast_list:
        dt = datetime.fromtimestamp(x["dt"])
        if dt.date() != target_date:
            continue
        if start_h <= dt.hour < end_h:
            window_items.append(x)

    # Fallback widening if the exact 3h slots don't align
    if not window_items:
        for widen in (1, 2):
            widened = []
            for x in forecast_list:
                dt = datetime.fromtimestamp(x["dt"])
                if dt.date() != target_date:
                    continue
                if (start_h - widen) <= dt.hour < (end_h + widen):
                    widened.append(x)
            if widened:
                window_items = widened
                break

    if not window_items:
        return None

    temps = [i["main"]["temp"] for i in window_items if "main" in i and "temp" in i["main"]]
    descs = [i["weather"][0]["description"] for i in window_items if "weather" in i and i["weather"]]
    pops  = [i.get("pop", 0) for i in window_items]  # precipitation probability (0..1)
    winds = [i.get("wind", {}).get("speed", 0) for i in window_items]

    if not temps or not descs:
        return None

    avg_temp = round(mean(temps))
    desc = Counter(descs).most_common(1)[0][0].capitalize()
    pop = max(pops) if pops else 0
    wind = round(mean(winds)) if winds else 0

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

def _inject_pill_css_once():
    if st.session_state.get("_wx_css_loaded"):
        return
    st.session_state["_wx_css_loaded"] = True
    st.markdown(
        """
<style>
.wx-pill {
  background: rgba(255,255,255,0.60);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 1rem;
  padding: 0.7rem 0.8rem;
  text-align: center;
  box-shadow: 0 4px 14px rgba(0,0,0,0.12);
  position: relative;
}
.wx-pill .wx-badge {
  position: absolute; top: 8px; right: 8px;
  background: linear-gradient(135deg, #f59e0b, #f97316);
  color: #fff; font-weight: 700; font-size: .65rem;
  padding: 2px 8px; border-radius: 999px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.18);
}
.wx-pill .wx-title { margin: 0; font-weight: 700; color: #0f172a; }
.wx-pill .wx-emoji { font-size: 1.6rem; margin: .15rem 0 .1rem 0; }
.wx-pill .wx-temp { margin: 0; font-size: 1rem; font-weight: 700; color: #0f172a; }
.wx-pill .wx-desc { margin: 0; font-size: .85rem; opacity: .85; color: #0f172a; }
</style>
        """,
        unsafe_allow_html=True,
    )

def render(default_city: str = "Basingstoke"):
    st.header("🌤️ Weather Forecast")

    # Hide the "City:" label, keep the input
    city = st.text_input(
        label="",
        value=default_city,
        key="weather_city_input",
        placeholder="Search city…",
        label_visibility="collapsed",
    )

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

    # ---- Build adaptive daypart order: today's remaining windows, then tomorrow's ----
    f_list = forecast.get("list", []) if isinstance(forecast, dict) else []
    now = datetime.now()

    # Build [(label, start, end, target_date, info_dict, is_tomorrow), ...]
    items = []
    for label, start_h, end_h in WINDOWS:
        tdate = _target_date_for_window(now, start_h, end_h)
        info = _summarize_window(f_list, tdate, start_h, end_h)
        items.append({
            "label": label,
            "start": start_h,
            "end": end_h,
            "date": tdate,
            "info": info,
            "tomorrow": (tdate != now.date())
        })

    # Sort: today first (date==today), then tomorrow; within each, by start hour
    items.sort(key=lambda x: (x["date"] != now.date(), x["start"]))

    # Explain what’s shown
    any_tomorrow = any(x["tomorrow"] for x in items)
    caption = "Next forecast windows"
    if any_tomorrow:
        caption += " — today first, then tomorrow"
    st.markdown(f"### 🕒 {caption}")

    _inject_pill_css_once()
    cols = st.columns(3)
    daypart_pops = []
    for i, data_item in enumerate(items):
        label = data_item["label"]
        info  = data_item["info"]
        is_tomorrow = data_item["tomorrow"]
        with cols[i]:
            if info:
                daypart_pops.append(info.get("pop", 0))
                badge = '<div class="wx-badge">Tomorrow</div>' if is_tomorrow else ""
                st.markdown(
                    f"""
                    <div class="wx-pill">
                        {badge}
                        <p class="wx-title">{label}</p>
                        <p class="wx-emoji">{_emoji_for(info['desc'])}</p>
                        <p class="wx-temp">{info['temp']}°C</p>
                        <p class="wx-desc">{info['desc']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                badge = '<div class="wx-badge">Tomorrow</div>' if is_tomorrow else ""
                st.markdown(
                    f"""
                    <div class="wx-pill">
                        {badge}
                        <p class="wx-title">{label}</p>
                        <p class="wx-desc" style="margin-top:.2rem; opacity:.75;">No data</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---- Style advice card (depends on rain/temp/wind across parts) ----
    advice = _style_advice(desc, temp, wind, max(daypart_pops) if daypart_pops else 0)
    st.markdown(
        f"""
        <div style="
            margin-top: .8rem;
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 1rem;
            padding: 0.9rem 1rem;
            color: #0f172a;
            box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        ">
            <div style="font-weight:700; margin-bottom:.25rem;">👕 What to wear</div>
            <div style="opacity:.95;">{advice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
