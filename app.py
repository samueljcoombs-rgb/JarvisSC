# app.py — JARVIS (Agent + Dashboard + Memory + Code Reading + Auto-PR via Make.com)

import json, os, time, datetime as dt
from typing import List, Dict, Any

import requests, feedparser, streamlit as st
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials

import memory  # local persistent memory (memory.py)

# ---------- Page ----------
st.set_page_config(page_title="JARVIS", page_icon="🤖", layout="centered")

# ---------- Secrets / Config ----------
def _secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return default

# Read keys from Secrets (you already added them in Streamlit → Settings → Secrets)
OPENAI_API_KEY       = _secret("OPENAI_API_KEY")
OWM_KEY              = _secret("OWM_API_KEY")
TMDB_KEY             = _secret("TMDB_API_KEY")
DEFAULT_CITY         = _secret("DEFAULT_CITY", "Basingstoke,GB")
PODCAST_FEEDS        = [s.strip() for s in _secret("PODCAST_FEEDS","").split(",") if s.strip()]
YOUTUBE_CHANNEL_IDS  = [s.strip() for s in _secret("YOUTUBE_CHANNEL_IDS","").split(",") if s.strip()]
ATHLETIC_QUERY       = _secret("ATHLETIC_QUERY","Manchester United")
SA_INFO              = _secret("GOOGLE_SERVICE_ACCOUNT_JSON")

# --- Your Make.com webhook + your GitHub repo/branch (hard-wired as requested) ---
MAKE_WEBHOOK_URL     = "https://hook.eu2.make.com/sd6mavmdqbyxx0mi5cvbv7ndimyfzh2u"
GITHUB_REPO          = "samueljcoombs-rgb/JarvisSC"
GITHUB_BRANCH        = "main"

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in Streamlit Secrets."); st.stop()

# ---------- OpenAI client with lazy init + retry ----------
_client_cache = None
def get_openai_client():
    global _client_cache
    if _client_cache is None:
        _client_cache = OpenAI(api_key=OPENAI_API_KEY)
    return _client_cache

def call_openai_with_retry(kwargs: Dict[str, Any], max_retries: int = 5):
    delay = 1.0
    for _ in range(max_retries):
        try:
            return get_openai_client().chat.completions.create(**kwargs)
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 8)
    st.warning("The AI is rate-limited. Try again in ~10s.")
    return None

# ---------- Styling ----------
st.markdown("""
<style>
.block-container { padding-top: 1rem; max-width: 780px; }
.card { border-radius:14px; padding:16px 18px; border:1px solid rgba(0,0,0,.06);
        box-shadow:0 4px 20px rgba(0,0,0,.04); margin-bottom:14px; background:#ffffffaa; backdrop-filter: blur(4px);}
.small { color:#666; }
</style>
""", unsafe_allow_html=True)

# ---------- Google Sheets ----------
gc = None
if SA_INFO:
    try:
        creds = Credentials.from_service_account_info(
            json.loads(SA_INFO),
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",  # open by name
            ],
        )
        gc = gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google Sheets auth failed: {e}")
else:
    st.info("Add GOOGLE_SERVICE_ACCOUNT_JSON in Secrets for To-Do & Workouts.")

TODO_SHEET_NAME, WORKOUT_SHEET_NAME = "ToDo", "Workouts"

def open_sheet(name: str):
    if not gc:
        st.error("Sheets not connected: add GOOGLE_SERVICE_ACCOUNT_JSON and share your sheets with the service account (Editor).")
        return None
    try:
        return gc.open(name).sheet1
    except Exception as e:
        st.error(f"Could not open sheet '{name}': {e}")
        return None

def todo_list(limit=10):
    sh = open_sheet(TODO_SHEET_NAME)
    if not sh: return []
    rows = sh.get_all_records()
    rows.sort(key=lambda r: (str(r.get("Done","")).upper()=="TRUE", r.get("Priority", 99)))
    return rows[:limit]

def todo_toggle(task_text: str, done: bool):
    sh = open_sheet(TODO_SHEET_NAME)
    if not sh: return "Sheet unavailable."
    data = sh.get_all_records()
    for idx, row in enumerate(data, start=2):
        if str(row.get("Task","")).strip() == task_text.strip():
            sh.update_acell(f"A{idx}", "TRUE" if done else "FALSE")  # Column A = Done
            return f"Set '{task_text}' to {'Done' if done else 'Not Done'}."
    return "Task not found."

def workout_today():
    sh = open_sheet(WORKOUT_SHEET_NAME)
    if not sh: return "No workout sheet."
    today = dt.date.today().isoformat()
    for r in sh.get_all_records():
        if str(r.get("Date","")) == today:
            return r.get("Plan","(No plan)")
    return "(No plan for today)"

# ---------- Data fetchers (cached) ----------
@st.cache_data(ttl=300)
def get_weather(city: str = DEFAULT_CITY):
    if not OWM_KEY: return "Weather key missing."
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": OWM_KEY, "units": "metric"},
            timeout=10
        )
        d = r.json()
        t = round(d["main"]["temp"])
        feels = round(d["main"]["feels_like"])
        desc = d["weather"][0]["description"].title()
        return f"{city}: {t}°C (feels {feels}°C), {desc}"
    except Exception as e:
        return f"Weather error: {e}"

def parse_rss(url: str, since_hours: int = 36):
    feed = feedparser.parse(url)
    cutoff = dt.datetime.utcnow() - dt.timedelta(hours=since_hours)
    items = []
    for it in feed.entries[:12]:
        date_parsed = None
        for k in ("published_parsed","updated_parsed"):
            if getattr(it, k, None):
                date_parsed = dt.datetime(*getattr(it,k)[:6])
                break
        if date_parsed and date_parsed > cutoff:
            items.append({"title": it.get("title","(untitled)"),
                          "link": it.get("link","#")})
    return items

@st.cache_data(ttl=300)
def get_podcasts(feeds=PODCAST_FEEDS):
    out = []
    for f in feeds:
        try: out += parse_rss(f)
        except Exception: pass
    return out[:8