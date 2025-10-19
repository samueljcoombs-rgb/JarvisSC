# app.py
# My JARVIS — Streamlit dashboard with an in-app AI agent
# Includes rate-limit handling, rerun guards, and caching.

import json
import time
import datetime as dt
from typing import List, Dict, Any

import requests
import feedparser
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="My JARVIS", page_icon="🤖", layout="centered")

# ---------- Secrets & Config ----------
def _secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return default

OPENAI_API_KEY       = _secret("OPENAI_API_KEY")
OWM_KEY              = _secret("OWM_API_KEY")
TMDB_KEY             = _secret("TMDB_API_KEY")
DEFAULT_CITY         = _secret("DEFAULT_CITY", "Basingstoke,GB")
PODCAST_FEEDS        = [s.strip() for s in _secret("PODCAST_FEEDS","").split(",") if s.strip()]
YOUTUBE_CHANNEL_IDS  = [s.strip() for s in _secret("YOUTUBE_CHANNEL_IDS","").split(",") if s.strip()]
ATHLETIC_QUERY       = _secret("ATHLETIC_QUERY", "Manchester United")
SA_INFO              = _secret("GOOGLE_SERVICE_ACCOUNT_JSON")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not set. Add it in **Settings → Secrets** as `OPENAI_API_KEY`.")
    st.stop()

if not OWM_KEY:
    st.warning("OWM_API_KEY (weather) is missing — Weather card will show an error.")

if not TMDB_KEY:
    st.warning("TMDB_API_KEY (cinema) is missing — Cinema card will show an error.")

# ---------- Google Sheets auth ----------
import gspread
from google.oauth2.service_account import Credentials

gc = None
if SA_INFO:
    try:
        creds = Credentials.from_service_account_info(
            json.loads(SA_INFO),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google Sheets auth failed. Check GOOGLE_SERVICE_ACCOUNT_JSON in Secrets. Details: {e}")
else:
    st.warning("GOOGLE_SERVICE_ACCOUNT_JSON missing — To-Do & Workouts won’t load until you add it in **Secrets**.")

# ---- CHANGE THESE TO MATCH YOUR SHEETS ----
TODO_SHEET_NAME    = "ToDo"
WORKOUT_SHEET_NAME = "Workouts"
# -------------------------------------------

# ---------- OpenAI client + retry/backoff ----------
from openai import OpenAI
_client_cache = None

def get_openai_client():
    global _client_cache
    if _client_cache is None:
        _client_cache = OpenAI(api_key=OPENAI_API_KEY)
    return _client_cache

def call_openai_with_retry(create_kwargs: Dict[str, Any], max_retries: int = 5):
    """Retry on rate limits/transient errors with exponential backoff."""
    delay = 1.0
    last_err = None
    for _ in range(max_retries):
        try:
            return get_openai_client().chat.completions.create(**create_kwargs)
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 8)
    st.warning("The AI is currently rate-limited. Please try again in ~10 seconds.")
    if last_err:
        st.caption("Retried and still blocked; details are in Streamlit logs.")
    return None

# ---------- Styling ----------
st.markdown("""
<style>
.block-container { padding-top: 1rem; max-width: 780px; }
.card {
  border-radius: 14px; padding: 16px 18px;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 4px 20px rgba(0,0,0,0.04);
  margin-bottom: 14px; background: #ffffffaa; backdrop-filter: blur(4px);
}
.small { font-size: 0.9rem; color: #666; }
</style>
""", unsafe_allow_html=True)

# ---------- Google Sheets helpers ----------
def open_sheet(name: str):
    if not gc:
        st.error("Google Sheets not connected. Ensure Secrets include GOOGLE_SERVICE_ACCOUNT_JSON and the sheet is shared with that service account email (Editor).")
        return None
    try:
        return gc.open(name).sheet1
    except Exception as e:
        st.error(f"Could not open sheet '{name}': {e}")
        return None

def todo_list(limit=10):
    sh = open_sheet(TODO_SHEET_NAME)
    if not sh: return []
    values = sh.get_all_records()
    # Expect headers: Done | Task | Notes | Priority | CreatedAt | Due
    values.sort(key=lambda r: (str(r.get("Done","")).upper()=="TRUE", r.get("Priority", 99)))
    return values[:limit]

def todo_toggle(task_text: str, done: bool):
    sh = open_sheet(TODO_SHEET_NAME)
    if not sh: return "Sheet unavailable."
    data = sh.get_all_records()
    for idx, row in enumerate(data, start=2):  # row 1 is headers
        if str(row.get("Task","")).strip() == task_text.strip():
            sh.update_acell(f"A{idx}", "TRUE" if done else "FALSE")  # Column A = Done
            return f"Set '{task_text}' to {'Done' if done else 'Not Done'}."
    return "Task not found."

def workout_today():
    sh = open_sheet(WORKOUT_SHEET_NAME)
    if not sh: return "No workout sheet."
    today = dt.date.today().isoformat()
    rows = sh.get_all_records()
    for r in rows:
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
    return out[:8]

@st.cache_data(ttl=300)
def get_youtube_uploads(channel_ids=YOUTUBE_CHANNEL_IDS):
    out = []
    for cid in channel_ids:
        rss = f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
        try: out += parse_rss(rss)
        except Exception: pass
    return out[:6]

@st.cache_data(ttl=300)
def get_athletic_mu(query=ATHLETIC_QUERY):
    # Google News RSS scoped to The Athletic
    q = f"https://news.google.com/rss/search?q=site:theathletic.com+{requests.utils.quote(query)}&hl=en-GB&gl=GB&ceid=GB:en"
    try:
        return parse_rss(q)
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_cinema_now_playing(region="GB"):
    if not TMDB_KEY: return ["TMDB key missing."]
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/movie/now_playing",
            params={"api_key": TMDB_KEY, "region": region},
            timeout=10
        )
        res = r.json().get("results", [])[:8]
        return [f"{m.get('title','?')} ({m.get('release_date','')})" for m in res]
    except Exception as e:
        return [f"TMDB error: {e}"]

# ---------- Agent tool-calling ----------
TOOL_SPECS = [
    {"name":"get_weather", "description":"Get weather for a city",
     "parameters":{"type":"object","properties":{"city":{"type":"string"}}}},
    {"name":"get_podcasts", "description":"Fetch recent podcast episodes",
     "parameters":{"type":"object","properties":{}} },
    {"name":"get_youtube_uploads", "description":"Fetch recent YouTube uploads",
     "parameters":{"type":"object","properties":{}} },
    {"name":"get_athletic_mu", "description":"Recent The Athletic MUFC items",
     "parameters":{"type":"object","properties":{}} },
    {"name":"get_cinema_now_playing", "description":"Now playing films in UK",
     "parameters":{"type":"object","properties":{"region":{"type":"string"}}}},
    {"name":"todo_list", "description":"Read To-Do items from Google Sheet",
     "parameters":{"type":"object","properties":{"limit":{"type":"number"}}}},
    {"name":"todo_toggle", "description":"Toggle a To-Do by exact Task text",
     "parameters":{"type":"object","properties":{"task_text":{"type":"string"},"done":{"type":"boolean"}}}},
    {"name":"workout_today", "description":"Read today's workout plan from sheet",
     "parameters":{"type":"object","properties":{}}}
]

def run_tool(name, args):
    try:
        if name == "get_weather":
            return get_weather(args.get("city", DEFAULT_CITY))
        if name == "get_podcasts":
            return get_podcasts()
        if name == "get_youtube_uploads":
            return get_youtube_uploads()
        if name == "get_athletic_mu":
            return get_athletic_mu()
        if name == "get_cinema_now_playing":
            return get_cinema_now_playing(args.get("region","GB"))
        if name == "todo_list":
            return todo_list(limit=int(args.get("limit", 10)))
        if name == "todo_toggle":
            return todo_toggle(args.get("task_text",""), bool(args.get("done", True)))
        if name == "workout_today":
            return workout_today()
    except Exception as e:
        return f"[{name}] error: {e}"
    return "Unknown tool."

def agent_chat(history: List[Dict[str, Any]]):
    tools = [{"type":"function","function":{
        "name": t["name"], "description": t["description"], "parameters": t["parameters"]
    }} for t in TOOL_SPECS]

    # First request (with retry)
    resp = call_openai_with_retry({
        "model": "gpt-4o-mini",
        "messages": history,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 400,
    })
    if resp is None:
        return "Sorry — I’m temporarily rate-limited. Please try again in ~10s."

    msg = resp.choices[0].message
    all_messages = history + [msg]

    # Tool-call loop with cap to avoid many API hits
    MAX_TOOL_LOOPS = 2
    loops = 0
    while getattr(msg, "tool_calls", None) and loops < MAX_TOOL_LOOPS:
        loops += 1
        tool_msgs = []
        for call in msg.tool_calls:
            fn = call.function
            args = json.loads(fn.arguments or "{}")
            result = run_tool(fn.name, args)
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fn.name,
                "content": json.dumps(result)
            })

        resp = call_openai_with_retry({
            "model": "gpt-4o-mini",
            "messages": all_messages + tool_msgs,
            "temperature": 0.2,
            "max_tokens": 500,
        })
        if resp is None:
            return "Got rate-limited while finishing the answer. Try again in a few seconds."
        msg = resp.choices[0].message
        all_messages = all_messages + tool_msgs + [msg]

    return msg.content or "Done."

# ---------- UI ----------
st.title("🤖 My JARVIS")
st.caption("Agent + life dashboard")

# Quick cards
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Weather")
    st.write(get_weather())
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Today’s Workout")
    st.write(workout_today())
    st.markdown('</div>', unsafe_allow_html=True)

# Feeds
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Latest Feeds")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Podcasts**")
    for it in get_podcasts():
        st.markdown(f"• [{it['title']}]({it['link']})")
with c2:
    st.markdown("**YouTube**")
    for it in get_youtube_uploads():
        st.markdown(f"• [{it['title']}]({it['link']})")
st.markdown("**The Athletic — MUFC**")
for it in get_athletic_mu():
    st.markdown(f"• [{it['title']}]({it['link']})")
st.markdown("**Cinema (UK Now Playing)**")
for line in get_cinema_now_playing():
    st.markdown(f"• {line}")
st.markdown('</div>', unsafe_allow_html=True)

# To-Do
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("To-Do")
todos = todo_list(limit=20)
if todos:
    for r in todos:
        key = f"todo_{r.get('Task','')}"
        checked = str(r.get("Done","")).upper()=="TRUE"
        new_val = st.checkbox(r.get("Task","(untitled)"), checked, key=key)
        if new_val != checked:
            msg = todo_toggle(r.get("Task",""), new_val)
            st.toast(msg)
else:
    st.write("(No items)")
st.markdown('</div>', unsafe_allow_html=True)

# --- Rerun guard for the agent ---
if "chat" not in st.session_state: st.session_state.chat = []
if "last_prompt" not in st.session_state: st.session_state.last_prompt = None

# Agent chat
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Agent")

prompt = st.chat_input("Ask me to update sections, tick tasks, summarize news, etc.")
if prompt:
    if prompt != st.session_state.last_prompt:
        st.session_state.chat.append({"role":"user","content":prompt})
        with st.spinner("Thinking..."):
            out = agent_chat([*st.session_state.chat])
        st.session_state.chat.append({"role":"assistant","content":out})
        st.session_state.last_prompt = prompt
    else:
        st.info("That prompt was just answered. Type something new.")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.markdown('</div>', unsafe_allow_html=True)
