# app.py
"""
Jarvis AI Dashboard - Main Application
Multi-page Streamlit app with Google Sheets persistence
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path
import importlib.util
import streamlit as st
from openai import OpenAI

# Import sheets memory for persistent storage
try:
    from modules import sheets_memory as sm
    from modules import global_styles as gs
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

# Legacy memory import for backwards compatibility
import memory

BASE_DIR = Path(__file__).parent
MODULES_DIR = BASE_DIR / "modules"
TEMP_CHAT_FILE = BASE_DIR / "temp_chat.json"
CHAT_SESSIONS_FILE = BASE_DIR / "chat_sessions.json"

# ----- Model selection -----
PREFERRED_ENV = os.getenv("PREFERRED_OPENAI_MODEL", "").strip()

def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        st.error("Missing OPENAI_API_KEY.")
        st.stop()
    return OpenAI(api_key=api_key)

client = _init_client()

def _select_best_model(client: OpenAI) -> str:
    if PREFERRED_ENV:
        return PREFERRED_ENV
    try:
        names = {m.id for m in client.models.list().data}
        for candidate in ["gpt-5", "gpt-latest", "gpt-4.1", "gpt-4o", "gpt-4.1-mini"]:
            if candidate in names:
                return candidate
    except Exception:
        pass
    return "gpt-4o"

JARVIS_MODEL = _select_best_model(client)

# ----- JSON helpers -----
def safe_load_json(path: Path, default):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def safe_save_json(path: Path, data):
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except Exception as e:
        st.warning(f"⚠️ Could not save {path.name}: {e}")

# ----- Module hot-loader -----
def backup_module(name: str):
    src = MODULES_DIR / f"{name}.py"
    dst = MODULES_DIR / f"{name}_backup.py"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

def safe_write_module(name: str, code: str) -> bool:
    path = MODULES_DIR / f"{name}.py"
    if not path.exists():
        st.error(f"Missing module {name}.py")
        return False
    try:
        compile(code, str(path), "exec")
    except SyntaxError as e:
        st.error(f"❌ Syntax error in {name}.py: {e}")
        return False
    backup_module(name)
    tmp = path.with_suffix(".py.tmp")
    tmp.write_text(code, encoding="utf-8")
    tmp.replace(path)
    st.success(f"✅ {name}.py updated.")
    return True

def load_module(name: str):
    try:
        path = MODULES_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        st.error(f"⚠️ Failed to load {name}: {e}")
        return None

# ----- Memory Functions -----
def get_memory_text():
    """Get memory summary from sheets or local."""
    if SHEETS_AVAILABLE:
        try:
            return sm.get_memory_summary()
        except Exception:
            pass
    return memory.recent_summary()

def save_chat_to_sheets(role: str, content: str):
    """Save chat message to Google Sheets."""
    if SHEETS_AVAILABLE:
        try:
            session_id = st.session_state.get("session_id", str(int(time.time())))
            sm.save_chat_message(role, content, session_id)
        except Exception as e:
            pass  # Silently fail, local backup exists

# ----- OpenAI call -----
def call_jarvis(chat_history, mem_text):
    sys = (
        "You are Jarvis, a sophisticated AI assistant inside a Streamlit dashboard. "
        "You help with daily planning, health tracking, goal setting, entertainment, travel, and more. "
        "You have access to the user's data stored in Google Sheets including their health logs, "
        "workout history, bucket list, yearly goals, and travel plans. "
        "Be helpful, proactive, and personalized. "
        f"\n\nMemory summary:\n{mem_text}"
    )
    msgs = [{"role": "system", "content": sys}, *chat_history]
    resp = client.chat.completions.create(model=JARVIS_MODEL, messages=msgs)
    return resp.choices[0].message.content

# ----- Page Config -----
st.set_page_config(
    page_title="Jarvis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject global styles
if SHEETS_AVAILABLE:
    gs.inject_global_styles()

# ----- Session State -----
if "chat" not in st.session_state:
    st.session_state.chat = safe_load_json(TEMP_CHAT_FILE, [])
if "last_processed_index" not in st.session_state:
    st.session_state.last_processed_index = -1
if "session_id" not in st.session_state:
    st.session_state.session_id = str(int(time.time()))

# ----- Custom CSS for Main Page - PREMIUM DESIGN -----
st.markdown("""
<style>
/* Animated gradient background for header */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4); }
    50% { box-shadow: 0 8px 48px rgba(139, 92, 246, 0.6); }
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

.main-header {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    padding: 2rem 2.5rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(139, 92, 246, 0.3);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 
                inset 0 1px 0 rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 200%;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 60%);
    animation: float 6s ease-in-out infinite;
}

.main-header::after {
    content: '';
    position: absolute;
    bottom: -50%;
    left: -50%;
    width: 100%;
    height: 200%;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 60%);
    animation: float 8s ease-in-out infinite reverse;
}

.main-header h1 {
    color: white;
    margin: 0;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 40px rgba(139, 92, 246, 0.5);
}

.main-header p {
    color: rgba(255,255,255,0.8);
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    position: relative;
    z-index: 1;
}

/* Navigation Cards - Glassmorphism Premium */
.nav-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.02) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    cursor: pointer;
    text-decoration: none;
    position: relative;
    overflow: hidden;
}

.nav-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: left 0.5s;
}

.nav-card:hover::before {
    left: 100%;
}

.nav-card:hover {
    background: linear-gradient(145deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
    border-color: rgba(139, 92, 246, 0.5);
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 40px rgba(139, 92, 246, 0.3),
                0 0 0 1px rgba(139, 92, 246, 0.2);
}

.nav-card .icon {
    font-size: 2.8rem;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
    transition: transform 0.3s ease;
}

.nav-card:hover .icon {
    transform: scale(1.15);
}

.nav-card .title {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.25rem;
    letter-spacing: 0.3px;
}

.nav-card .desc {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
    font-weight: 400;
}

/* Section styling */
section[data-testid="stVerticalBlock"] > div:has(.nav-card) {
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# ----- Header -----
current_time = datetime.now().strftime("%A, %d %B %Y")
st.markdown(f"""
<div class="main-header">
    <h1>🤖 Jarvis Dashboard</h1>
    <p>✨ {current_time} • Your AI-powered life assistant</p>
</div>
""", unsafe_allow_html=True)

# ----- Flight Deals Banner -----
try:
    from modules import travel_tools as tt
    
    serpapi_key = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
    if serpapi_key:
        # Check for deals (silent - no notifications on page load)
        deals = tt.check_flight_alerts_silent()
        
        if deals:
            deals_html = ""
            for deal in deals[:3]:  # Show max 3 deals
                route = deal.get("route", "Unknown")
                price = deal.get("current_price", "?")
                target = deal.get("max_price", "?")
                savings = deal.get("savings", 0)
                
                deals_html += f"""
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1)); 
                            border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 1rem; 
                            margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #34d399;">✈️ {route}</strong><br>
                        <span style="color: rgba(255,255,255,0.7);">£{price} (Target: £{target}) • Save £{savings}</span>
                    </div>
                    <a href="https://www.google.com/flights" target="_blank" 
                       style="background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 8px; 
                              text-decoration: none; font-weight: 600;">Book Now →</a>
                </div>
                """
            
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
                        border: 2px solid rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 1.25rem;
                        margin-bottom: 1.5rem; animation: pulse-glow 2s ease-in-out infinite;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                    <span style="font-size: 1.5rem;">🎉</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #34d399;">
                        Flight Deal{'' if len(deals) == 1 else 's'} Found!
                    </span>
                    <span style="background: #10b981; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; 
                                 font-size: 0.75rem; font-weight: 700;">{len(deals)} ALERT{'S' if len(deals) > 1 else ''}</span>
                </div>
                {deals_html}
            </div>
            """, unsafe_allow_html=True)
except Exception as e:
    pass  # Silently fail if travel_tools not available

# ----- Navigation Cards -----
st.markdown("### Quick Navigation")

# Create navigation using columns and buttons
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">🏋️</div>
        <div class="title">Health & Fitness</div>
        <div class="desc">Track workouts, weight, nutrition</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Health", key="nav_health", use_container_width=True):
        st.switch_page("pages/1_🏋️_Health_Fitness.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">🎬</div>
        <div class="title">Entertainment</div>
        <div class="desc">Movies, TV, watchlist</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Entertainment", key="nav_ent", use_container_width=True):
        st.switch_page("pages/2_🎬_Entertainment.py")

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">🎯</div>
        <div class="title">Goals</div>
        <div class="desc">Bucket list & yearly goals</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Goals", key="nav_goals", use_container_width=True):
        st.switch_page("pages/3_🎯_Goals.py")

with col4:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">✈️</div>
        <div class="title">Travel</div>
        <div class="desc">Trips, flights, alerts</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Travel", key="nav_travel", use_container_width=True):
        st.switch_page("pages/4_✈️_Travel.py")

with col5:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">📰</div>
        <div class="title">News</div>
        <div class="desc">Personalized news feed</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open News", key="nav_news", use_container_width=True):
        st.switch_page("pages/5_📰_News.py")

st.markdown("---")

# ----- Main Dashboard Content -----
# Load modules
layout_mod = load_module("layout_manager")
chat_mod = load_module("chat_ui")
weather_mod = load_module("weather_panel")
podcasts_mod = load_module("podcasts_panel")
athletic_mod = load_module("athletic_feed")
todos_mod = load_module("todos_panel")

# Sidebar with memory and sessions
with st.sidebar:
    st.markdown("### 🤖 Jarvis")
    st.caption(f"Model: {JARVIS_MODEL}")
    
    with st.expander("🧠 Memory & Sessions", expanded=False):
        mem_text = get_memory_text()
        sessions = safe_load_json(CHAT_SESSIONS_FILE, [])
        
        st.caption(f"Messages: {len(st.session_state.chat)}")
        st.caption(f"Sessions: {len(sessions)}")
        
        st.divider()
        st.subheader("Memory Preview")
        preview = (mem_text or "").strip()
        if preview:
            st.write(preview[:300] + "..." if len(preview) > 300 else preview)
        else:
            st.write("No memories yet.")
        
        new_mem = st.text_input("Add to memory:")
        if new_mem:
            if SHEETS_AVAILABLE:
                sm.add_memory(new_mem, "fact", "manual")
            memory.add_fact(new_mem, "manual")
            st.success("Saved.")
            st.rerun()
        
        st.divider()
        if st.button("💾 Save chat"):
            sessions.append({
                "id": int(time.time()),
                "ts": int(time.time()),
                "messages": st.session_state.chat
            })
            safe_save_json(CHAT_SESSIONS_FILE, sessions)
            st.success("Saved.")
            st.rerun()
        
        if st.button("🗑️ New chat"):
            st.session_state.chat = []
            st.session_state.session_id = str(int(time.time()))
            safe_save_json(TEMP_CHAT_FILE, [])
            st.session_state.last_processed_index = -1
            st.success("Cleared.")
            st.rerun()

# Render the main layout
if layout_mod:
    layout_mod.render(
        chat=st.session_state.chat,
        mem_text=get_memory_text(),
        call_jarvis=call_jarvis,
        safe_write_module=safe_write_module,
        safe_save_json=safe_save_json,
        temp_chat_file=TEMP_CHAT_FILE,
        memory_module=memory,
        chat_module=chat_mod,
        weather_module=weather_mod,
        podcasts_module=podcasts_mod,
        athletic_module=athletic_mod,
        todos_module=todos_mod,
    )
