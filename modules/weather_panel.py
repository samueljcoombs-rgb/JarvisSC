import streamlit as st, requests, os
from datetime import datetime

API_TIMEOUT = 10

def _api_key():
    return os.getenv("OWM_API_KEY") or st.secrets.get("OWM_API_KEY", None)

def _geocode(city, key):
    try:
        r = requests.get(
            "http://api.openweathermap.org/geo/1.0/direct",
            params={"q": city, "limit": 1, "appid": key},
            timeout=API_TIMEOUT,
        )
        j = r.json()
        if isinstance(j, list) and j:
            return {
                "name": j[0].get("name", city),
                "lat": j[0]["lat"],
                "lon": j[0]["lon"],
                "country": j[0].get("country", ""),
            }
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
        }
    except:
        return None

def _forecast(lat, lon, key):
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": key, "units": "metric"},
            timeout=API_TIMEOUT,
        )
        return r.json().get("list", [])
    except:
        return []

def _emoji_for(desc):
    d = (desc or "").lower()
    if "thunder" in d: return "⛈️"
    if "rain" in d or "drizzle" in d: return "🌧️"
    if "snow" in d: return "❄️"
    if "cloud" in d: return "☁️"
    if "mist" in d or "fog" in d or "haze" in d: return "🌫️"
    return "☀️"

def _select_dayparts_today(forecast):
    """
    Select representative Morning (06-12), Afternoon (12-18), Evening (18-24) for today.
    """
    if not forecast:
        return None
    today = datetime.now().date()
    buckets = {"Morning": [], "Afternoon": [], "Evening": []}

    def bucket(dt):
        h = dt.hour
        if 6 <= h < 12: return "Morning"
        if 12 <= h < 18: return "Afternoon"
        if 18 <= h <= 23: return "Evening"
        return None

    for item in forecast:
        dt = datetime.fromtimestamp(item["dt"])
        if dt.date() != today:
            continue
        b = bucket(dt)
        if b:
            buckets[b].append(item)

    def summarize(items):
        if not items:
            return None
        temps = [x["main"]["temp"] for x in items if "main" in x]
        descs = [x["weather"][0]["description"] for x in items if "weather" in x and x["weather"]]
        if not temps or not descs:
            return None
        avg = round(sum(temps) / len(temps))
        # Pick median-ish entry for description for stability
        mid = descs[len(descs) // 2]
        return {"temp": avg, "desc": mid}

    return {k: summarize(v) for k, v in buckets.items()}

def _inject_css():
    st.markdown("""
<style>
.weather-wrap {
  margin-top: .2rem;
}
.weather-card {
  background: radial-gradient(1200px 400px at 10% -10%, rgba(255,255,255,0.10), rgba(255,255,255,0.02) 40%),
              linear-gradient(135deg, #0b0e14 0%, #142030 40%, #1e2f46 100%);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 20px;
  padding: 16px 18px;
  color: #f2f7ff;
  box-shadow: 0 14px 32px rgba(0,0,0,0.35);
  backdrop-filter: blur(8px);
}
.w-header { display: flex; justify-content: space-between; align-items: baseline; }
.w-city { font-weight: 700; font-size: 1.15rem; letter-spacing: .2px; }
.w-asof { opacity: .8; font-size: .9rem; }
.w-main { display: flex; align-items: center; gap: 12px; margin-top: 8px; }
.w-temp { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.w-desc { opacity: .95; font-size: 1rem; }
.w-pills { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
@media (min-width: 560px) {
  .w-pills { display: grid; grid-template-columns: 1fr 1fr 1fr; }
}
.pill {
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 10px;
  text-align: center;
}
.pill h4 {
  margin: 0 0 6px 0; font-size: .95rem; font-weight: 700; color: #ffffff;
  letter-spacing: .2px;
}
.pill .t { font-size: 1.25rem; font-weight: 800; color: #ffffff; }
.pill .d { font-size: .9rem; opacity: .9; color: #eaf2ff; }
</style>
""", unsafe_allow_html=True)

def render(default_city="Basingstoke"):
    st.header("🌦️ Weather")

    key = _api_key()
    city = st.text_input("City:", default_city, key="weather_city_input")

    if not key:
        st.write("Weather unavailable. Missing OWM_API_KEY in environment or Streamlit secrets.")
        return

    loc = _geocode(city, key)
    if not loc:
        st.write("Weather unavailable. Could not resolve city.")
        return

    cur = _current(loc["lat"], loc["lon"], key)
    fc = _forecast(loc["lat"], loc["lon"], key)
    parts = _select_dayparts_today(fc)

    _inject_css()
    asof = datetime.now().strftime("%I:%M %p").lstrip("0")

    st.markdown('<div class="weather-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="weather-card">', unsafe_allow_html=True)

    # Header
    st.markdown(
        f'<div class="w-header"><div class="w-city">{loc["name"]}</div>'
        f'<div class="w-asof">as of {asof}</div></div>',
        unsafe_allow_html=True
    )

    # Main line
    if cur:
        st.markdown(
            f'<div class="w-main"><div class="w-temp">{round(cur["temp"])}°C</div>'
            f'<div class="w-desc">{_emoji_for(cur.get("desc"))} {cur.get("desc","").title()} · '
            f'💧{cur.get("humidity","?")}% · 🌬️{cur.get("wind","?")} m/s</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="w-main"><div class="w-desc">Current conditions unavailable.</div></div>', unsafe_allow_html=True)

    # Dayparts row (true side-by-side)
    st.markdown('<div class="w-pills">', unsafe_allow_html=True)
    for label in ["Morning", "Afternoon", "Evening"]:
        info = parts.get(label) if parts else None
        if info:
            st.markdown(
                f'<div class="pill"><h4>{label}</h4>'
                f'<div class="t">{info["temp"]}°C</div>'
                f'<div class="d">{_emoji_for(info["desc"])} {info["desc"].title()}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="pill"><h4>{label}</h4><div class="d">No data</div></div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)  # w-pills

    st.markdown('</div>', unsafe_allow_html=True)  # weather-card
    st.markdown('</div>', unsafe_allow_html=True)  # weather-wrap
