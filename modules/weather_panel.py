import streamlit as st, requests, os
from datetime import datetime, timezone
from math import floor

API_TIMEOUT = 10

def _api_key():
    # Supports env var and Streamlit secrets
    return os.getenv("OWM_API_KEY") or (
        st.secrets.get("OWM_API_KEY", st.secrets.get("weather_api_key", None))
    )

def _geocode(city, key):
    try:
        r = requests.get(
            "http://api.openweathermap.org/geo/1.0/direct",
            params={"q": city, "limit": 1, "appid": key},
            timeout=API_TIMEOUT,
        )
        j = r.json()
        if isinstance(j, list) and j:
            return {"name": j[0].get("name", city), "lat": j[0]["lat"], "lon": j[0]["lon"], "country": j[0].get("country", "")}
    except:
        pass
    return None

def _current(lat, lon, key):
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": key, "units": "metric"},
            timeout=API_TIMEOUT,
        )
        j = r.json()
        return {
            "temp": j["main"]["temp"],
            "desc": j["weather"][0]["description"],
            "humidity": j["main"]["humidity"],
            "wind": j["wind"]["speed"],
            "dt": j.get("dt"),
        }
    except:
        return None

def _forecast(lat, lon, key):
    """
    Returns list of 3-hour forecast entries with dt (unix), main/temp, weather[0]/description
    """
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": key, "units": "metric"},
            timeout=API_TIMEOUT,
        )
        j = r.json()
        return j.get("list", [])
    except:
        return []

def _select_dayparts(forecast_list):
    """
    Build Morning (06-12), Afternoon (12-18), Evening (18-24) from 3h forecast blocks for 'today'.
    """
    if not forecast_list:
        return None

    def bucket(hour):
        if 6 <= hour < 12: return "Morning"
        if 12 <= hour < 18: return "Afternoon"
        if 18 <= hour <= 23: return "Evening"
        return None

    now = datetime.now()
    today = now.date()

    dayparts = {"Morning": [], "Afternoon": [], "Evening": []}
    for item in forecast_list:
        # item["dt"] is unix seconds (UTC)
        dt = datetime.fromtimestamp(item["dt"])
        if dt.date() != today:
            continue
        hr = dt.hour
        b = bucket(hr)
        if b:
            dayparts[b].append(item)

    def summarize(items):
        if not items:
            return None
        temps = [x["main"]["temp"] for x in items if "main" in x and "temp" in x["main"]]
        descs = [x["weather"][0]["description"] for x in items if "weather" in x and x["weather"]]
        if not temps or not descs:
            return None
        avg_temp = sum(temps) / len(temps)
        # pick the most frequent description
        desc = max(set(descs), key=descs.count)
        return {"temp": round(avg_temp), "desc": desc}

    return {k: summarize(v) for k, v in dayparts.items()}

def _emoji_for(desc):
    d = (desc or "").lower()
    if "thunder" in d: return "⛈️"
    if "rain" in d or "drizzle" in d: return "🌧️"
    if "snow" in d: return "❄️"
    if "cloud" in d: return "☁️"
    if "mist" in d or "fog" in d or "haze" in d: return "🌫️"
    return "☀️"

def _card_css():
    st.markdown("""
<style>
.weather-card {
  background: linear-gradient(135deg, #0d0f14 0%, #1b2a3a 100%);
  border-radius: 20px;
  padding: 16px 18px;
  color: #eaf2ff;
  box-shadow: 0 10px 24px rgba(0,0,0,0.25);
  border: 1px solid rgba(255,255,255,0.08);
}
.weather-header {
  display: flex; justify-content: space-between; align-items: baseline;
}
.weather-title {
  font-size: 1.15rem; font-weight: 600; letter-spacing: .2px; margin: 0;
}
.weather-asof {
  opacity: .75; font-size: .9rem;
}
.weather-main {
  display: flex; align-items: center; gap: 10px; margin-top: 8px;
}
.weather-temp {
  font-size: 2.2rem; font-weight: 700; line-height: 1;
}
.weather-desc {
  opacity: .9; font-size: 1rem;
}
.dayparts {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px;
}
.part {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 14px;
  padding: 10px;
  text-align: center;
}
.part h4 {
  margin: 0 0 6px 0; font-size: .95rem; font-weight: 600; color: #ffffff;
}
.part .t {
  font-size: 1.25rem; font-weight: 700; color: #ffffff;
}
.part .d {
  font-size: .9rem; opacity: .85; color: #eaf2ff;
}
</style>
""", unsafe_allow_html=True)

def get_weather_bundle(city="Basingstoke"):
    key = _api_key()
    if not key:
        return {"error": "Missing OWM_API_KEY in env or Streamlit secrets."}
    loc = _geocode(city, key)
    if not loc:
        return {"error": "Unable to geocode city."}
    cur = _current(loc["lat"], loc["lon"], key)
    fc = _forecast(loc["lat"], loc["lon"], key)
    parts = _select_dayparts(fc)
    return {"loc": loc, "current": cur, "parts": parts}

def render(default_city="Basingstoke"):
    st.header("🌦️ Weather")
    city = st.text_input("City:", default_city, key="weather_city_input")

    bundle = get_weather_bundle(city)
    if bundle.get("error"):
        st.write("Weather unavailable. " + bundle["error"])
        return

    loc = bundle["loc"]
    cur = bundle["current"]
    parts = bundle["parts"]

    _card_css()
    asof = datetime.now().strftime("%I:%M %p").lstrip("0")

    with st.container():
        st.markdown('<div class="weather-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="weather-header">'
            f'<div class="weather-title">{loc["name"]}</div>'
            f'<div class="weather-asof">as of {asof}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if cur:
            emoji = _emoji_for(cur.get("desc"))
            st.markdown(
                f'<div class="weather-main">'
                f'<div class="weather-temp">{round(cur["temp"])}°C</div>'
                f'<div class="weather-desc">{emoji} {cur.get("desc","").title()} · 💧{cur.get("humidity","?")}% · 🌬️{cur.get("wind","?")} m/s</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="weather-main"><div class="weather-desc">Current conditions unavailable.</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="dayparts">', unsafe_allow_html=True)
        if parts:
            for label in ["Morning", "Afternoon", "Evening"]:
                info = parts.get(label)
                if info:
                    emoji = _emoji_for(info["desc"])
                    st.markdown(
                        f'<div class="part"><h4>{label}</h4>'
                        f'<div class="t">{info["temp"]}°C</div>'
                        f'<div class="d">{emoji} {info["desc"].title()}</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="part"><h4>{label}</h4>'
                        f'<div class="d">No data</div></div>',
                        unsafe_allow_html=True
                    )
        else:
            for label in ["Morning", "Afternoon", "Evening"]:
                st.markdown(f'<div class="part"><h4>{label}</h4><div class="d">No data</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # dayparts
        st.markdown('</div>', unsafe_allow_html=True)  # weather-card
