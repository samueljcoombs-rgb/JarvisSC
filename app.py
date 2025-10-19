import os, json, datetime as dt, requests, feedparser, streamlit as st, gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ---------- Page ----------
st.set_page_config(page_title="My JARVIS", page_icon="🤖", layout="centered")

# ---------- Secrets & Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OWM_KEY = os.environ.get("OWM_API_KEY")
TMDB_KEY = os.environ.get("TMDB_API_KEY")
DEFAULT_CITY = os.environ.get("DEFAULT_CITY", "Basingstoke,GB")  # <- your default
PODCAST_FEEDS = [s.strip() for s in os.environ.get("PODCAST_FEEDS", "").split(",") if s.strip()]
YOUTUBE_CHANNEL_IDS = [s.strip() for s in os.environ.get("YOUTUBE_CHANNEL_IDS", "").split(",") if s.strip()]
ATHLETIC_QUERY = os.environ.get("ATHLETIC_QUERY", "Manchester United")

SA_INFO = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
gc = None
if SA_INFO:
    creds = Credentials.from_service_account_info(json.loads(SA_INFO), scopes=SCOPES)
    gc = gspread.authorize(creds)

TODO_SHEET_NAME = "ToDo"
WORKOUT_SHEET_NAME = "Workouts"

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container { padding-top: 1rem; max-width: 780px; }
.card { border-radius: 14px; padding: 16px 18px; border: 1px solid rgba(0,0,0,0.06); 
        box-shadow: 0 4px 20px rgba(0,0,0,0.04); margin-bottom: 14px; background: #ffffffaa; backdrop-filter: blur(4px); }
.small { font-size: 0.9rem; color: #666; }
</style>
""", unsafe_allow_html=True)

# ---------- Google Sheets helpers ----------
def open_sheet(name: str):
    if not gc:
        st.error("Google Sheets not connected. Add GOOGLE_SERVICE_ACCOUNT_JSON in secrets and share your sheet with that service account email.")
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
    values.sort(key=lambda r: (str(r.get("Done","")).upper()=="TRUE", r.get("Priority", 99)))
    return values[:limit]

def todo_toggle(task_text: str, done: bool):
    sh = open_sheet(TODO_SHEET_NAME)
    if not sh: return "Sheet unavailable."
    data = sh.get_all_records()
    for idx, row in enumerate(data, start=2):
        if str(row.get("Task","")).strip() == task_text.strip():
            sh.update_acell(f"A{idx}", "TRUE" if done else "FALSE")
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

# ---------- Data fetchers ----------
def get_weather(city: str = DEFAULT_CITY):
    if not OWM_KEY: return "Weather key missing."
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                         params={"q": city, "appid": OWM_KEY, "units":"metric"}, timeout=10)
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
    for it in feed.entries[:10]:
        date_parsed = None
        for k in ("published_parsed","updated_parsed"):
            if getattr(it, k, None):
                date_parsed = dt.datetime(*getattr(it,k)[:6])
                break
        if date_parsed and date_parsed > cutoff:
            items.append({"title": it.get("title","(untitled)"), "link": it.get("link","#")})
    return items

def get_podcasts(feeds=PODCAST_FEEDS):
    all_items = []
    for f in feeds: all_items += parse_rss(f)
    return all_items[:8]

def get_youtube_uploads(channel_ids=YOUTUBE_CHANNEL_IDS):
    items = []
    for cid in channel_ids:
        rss = f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
        items += parse_rss(rss)
    return items[:6]

def get_athletic_mu(query=ATHLETIC_QUERY):
    q = f"https://news.google.com/rss/search?q=site:theathletic.com+{requests.utils.quote(query)}&hl=en-GB&gl=GB&ceid=GB:en"
    return parse_rss(q)

def get_cinema_now_playing(region="GB"):
    if not TMDB_KEY: return []
    try:
        r = requests.get("https://api.themoviedb.org/3/movie/now_playing",
                         params={"api_key": TMDB_KEY, "region": region}, timeout=10)
        res = r.json().get("results", [])[:8]
        return [f"{m.get('title','?')} ({m.get('release_date','')})" for m in res]
    except Exception as e:
        return [f"TMDB error: {e}"]

# ---------- Agent tool-calling ----------
TOOL_SPECS = [
    {"name":"get_weather","description":"Get weather for a city",
     "parameters":{"type":"object","properties":{"city":{"type":"string"}}}},
    {"name":"get_podcasts","description":"Fetch recent podcast episodes","parameters":{"type":"object","properties":{}}},
    {"name":"get_youtube_uploads","description":"Fetch recent YouTube uploads","parameters":{"type":"object","properties":{}}},
    {"name":"get_athletic_mu","description":"Recent The Athletic MUFC","parameters":{"type":"object","properties":{}}},
    {"name":"get_cinema_now_playing","description":"Now playing films in UK","parameters":{"type":"object","properties":{"region":{"type":"string"}}}},
    {"name":"todo_list","description":"Read To-Do items","parameters":{"type":"object","properties":{"limit":{"type":"number"}}}},
    {"name":"todo_toggle","description":"Toggle a To-Do by exact Task","parameters":{"type":"object","properties":{"task_text":{"type":"string"},"done":{"type":"boolean"}}}},
    {"name":"workout_today","description":"Today’s workout","parameters":{"type":"object","properties":{}}}
]

def run_tool(name, args):
    if name == "get_weather": return get_weather(args.get("city", DEFAULT_CITY))
    if name == "get_podcasts": return get_podcasts()
    if name == "get_youtube_uploads": return get_youtube_uploads()
    if name == "get_athletic_mu": return get_athletic_mu()
    if name == "get_cinema_now_playing": return get_cinema_now_playing(args.get("region","GB"))
    if name == "todo_list": return todo_list(limit=int(args.get("limit", 10)))
    if name == "todo_toggle": return todo_toggle(args.get("task_text",""), bool(args.get("done", True)))
    if name == "workout_today": return workout_today()
    return "Unknown tool."

def agent_chat(history):
    tools = [{"type":"function","function":{
        "name": t["name"], "description": t["description"], "parameters": t["parameters"]
    }} for t in TOOL_SPECS]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        tools=tools,
        tool_choice="auto",
        temperature=0.2
    )

    msg = resp.choices[0].message
    all_messages = history + [msg]
    while getattr(msg, "tool_calls", None):
        tool_msgs = []
        for call in msg.tool_calls:
            fn = call.function
            args = json.loads(fn.arguments or "{}")
            result = run_tool(fn.name, args)
            tool_msgs.append({"role":"tool","tool_call_id":call.id,"name":fn.name,"content":json.dumps(result)})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=all_messages + tool_msgs,
            temperature=0.2
        )
        msg = resp.choices[0].message
        all_messages = all_messages + tool_msgs + [msg]
    return msg.content

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
    for it in get_podcasts(): st.markdown(f"• [{it['title']}]({it['link']})")
with c2:
    st.markdown("**YouTube**")
    for it in get_youtube_uploads(): st.markdown(f"• [{it['title']}]({it['link']})")
st.markdown("**The Athletic — MUFC**")
for it in get_athletic_mu(): st.markdown(f"• [{it['title']}]({it['link']})")
st.markdown("**Cinema (UK Now Playing)**")
for line in get_cinema_now_playing(): st.markdown(f"• {line}")
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

# Agent chat
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Agent")
if "chat" not in st.session_state: st.session_state.chat = []
prompt = st.chat_input("Ask me to update sections, tick tasks, summarize news, etc.")
if prompt:
    st.session_state.chat.append({"role":"user","content":prompt})
    with st.spinner("Thinking..."):
        out = agent_chat([*st.session_state.chat])
    st.session_state.chat.append({"role":"assistant","content":out})
for m in st.session_state.chat:
    with st.chat_message(m["role"]): st.markdown(m["content"])
st.markdown('</div>', unsafe_allow_html=True)