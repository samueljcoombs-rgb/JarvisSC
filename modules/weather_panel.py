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
# - Windows 06–12 / 12–18 / 18–24
# - Today’s remaining windows first (chronological), then tomorrow’s
# - Tiny orange "T" circle badge for tomorrow
# - No city input above: optional expander *below* to change city
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

WINDOWS = [
    ("Morning", 6, 12),
    ("Afternoon", 12, 18),
    ("Evening", 18, 24),
]

def _target_date_for_window(now: datetime, start_h: int, end_h: int) -> date:
    return (now + timedelta(days=1)).date() if now.hour >= end_h else now.date()

def _summarize_window(forecast_list, target_date: date, start_h: int, end_h: int):
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

    # Gentle widening if exact slots don't align
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
    wind = round(mean([w for w in winds if isinstance(w, (int, float))])) if winds else 0

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

def _inject_css_once():
    if st.session_state.get("_wx_css_loaded"):
        return
    st.session_state["_wx_css_loaded"] = True
    st.markdown(
        """
<style>
/* Card + pills */
.wx-card {
  background: linear-gradient(135deg, #8EC5FC 0%, #3B82F6 100%);
  padding: 1.2rem 1.6rem; border-radius: 1.2rem;
  color: #ffffff; box-shadow: 0 8px 20px rgba(0,0,0,0.25);
  border: 1px solid rgba(255,255,255,0.25);
}
.wx-main { display:flex; align-items:center; gap:12px; margin-top:.25rem; }
.wx-temp { font-size:3rem; font-weight:800; line-height:1; }
.wx-sub  { font-size:1rem; opacity:.95; }

/* Pills row */
.wx-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px; }
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
  overflow: hidden;
}
.wx-title { margin: 0; font-weight: 700; color: #0f172a; }
.wx-emoji { font-size: 1.6rem; margin: .15rem 0 .1rem 0; }
.wx-temp-sm { margin: 0; font-size: 1rem; font-weight: 700; color: #0f172a; }
.wx-desc { margin: 0; font-size: .85rem; opacity: .85; color: #0f172a; }

/* Tiny orange circle "T" badge (tomorrow) */
.wx-badge-t {
  position: absolute; top: 8px; right: 8px;
  width: 18px; height: 18px; border-radius: 999px;
  background: #f97316;
  color: #fff; font-weight: 800; font-size: 0.7rem;
  display:flex; align-items:center; justify-content:center;
  box-shadow: 0 2px 6px rgba(0,0,0,0.18);
}

/* Advice card */
.wx-advice {
  margin-top: .8rem;
  background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 1rem;
  padding: 0.9rem 1rem;
  color: #0f172a;
  box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}
</style>
        """,
        unsafe_allow_html=True,
    )

def render(default_city: str = "Basingstoke"):
    st.header("🌤️ Weather Forecast")

    # No city input above. We use a stored city or default.
    city = st.session_state.get("_wx_city", default_city)

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

    _inject_css_once()

    # ---- Current conditions card ----
    st.markdown(
        f"""
        <div class="wx-card">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <h2 style="margin:0; font-weight:700; letter-spacing:.2px;">{emoji} {name}</h2>
                <div style="opacity:.9; font-size:.95rem;">as of {asof}</div>
            </div>
            <div class="wx-main">
                <div class="wx-temp">{temp}°C</div>
                <div class="wx-sub">{desc} · 💧 {humidity}% · 🌬️ {wind} m/s</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Build adaptive daypart order: today's remaining, then tomorrow ----
    f_list = forecast.get("list", []) if isinstance(forecast, dict) else []
    now = datetime.now()

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

    items.sort(key=lambda x: (x["date"] != now.date(), x["start"]))

    # ---- Pills row (no caption text) ----
    st.markdown('<div class="wx-row">', unsafe_allow_html=True)
    for data_item in items:
        label = data_item["label"]
        info  = data_item["info"]
        is_tomorrow = data_item["tomorrow"]

        if info:
            badge_html = '<div class="wx-badge-t">T</div>' if is_tomorrow else ''
            st.markdown(
                f"""
                <div class="wx-pill">
                    {badge_html}
                    <p class="wx-title">{label}</p>
                    <p class="wx-emoji">{_emoji_for(info['desc'])}</p>
                    <p class="wx-temp-sm">{info['temp']}°C</p>
                    <p class="wx-desc">{info['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            badge_html = '<div class="wx-badge-t">T</div>' if is_tomorrow else ''
            st.markdown(
                f"""
                <div class="wx-pill">
                    {badge_html}
                    <p class="wx-title">{label}</p>
                    <p class="wx-desc" style="margin-top:.2rem; opacity:.75;">No data</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)  # /wx-row

    # ---- Style advice card ----
    # Collate max precipitation from dayparts to inform advice
    daypart_pop_max = 0.0
    for it in items:
        if it["info"]:
            daypart_pop_max = max(daypart_pop_max, it["info"].get("pop", 0) or 0.0)

    advice = _style_advice(desc, temp, wind, daypart_pop_max)
    st.markdown(
        f"""
        <div class="wx-advice">
            <div style="font-weight:700; margin-bottom:.25rem;">👕 What to wear</div>
            <div style="opacity:.95;">{advice}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Optional: small expander below to change city (no box above) ----
    with st.expander("Change city", expanded=False):
        new_city = st.text_input(
            "City",
            value=city,
            key="weather_city_input_hidden",
            placeholder="Type a city…",
        )
        if new_city and new_city.strip() != city:
            st.session_state["_wx_city"] = new_city.strip()
            st.rerun()
