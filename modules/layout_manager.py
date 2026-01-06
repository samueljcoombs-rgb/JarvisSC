# modules/layout_manager.py
"""
Layout Manager for Jarvis Dashboard
Handles the main 3-column layout with integrated widgets and navigation.
"""
import streamlit as st
from datetime import datetime

try:
    from modules import global_styles as gs
    from modules import sheets_memory as sm
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

def _find_last_user_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1

def _top_bar():
    """Render the compact top bar with date and quick links."""
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin:0 0 0.5rem 0;
                    padding: 0.75rem 1rem; background: rgba(255,255,255,0.03); border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-weight:700;font-size:1.1rem;color:#60a5fa;">🤖 Jarvis Assistant</div>
            <div style="color:rgba(255,255,255,0.6);font-size:0.9rem;">{today_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _quick_links():
    """Render quick navigation links to other pages."""
    st.markdown("""
    <style>
    .quick-links {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .quick-link {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .quick-link:hover {
        background: rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.4);
        color: #93c5fd;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("🏋️ Health", key="ql_health", use_container_width=True):
            st.switch_page("pages/1_🏋️_Health_Fitness.py")
    with col2:
        if st.button("🎬 Entertainment", key="ql_ent", use_container_width=True):
            st.switch_page("pages/2_🎬_Entertainment.py")
    with col3:
        if st.button("🎯 Goals", key="ql_goals", use_container_width=True):
            st.switch_page("pages/3_🎯_Goals.py")
    with col4:
        if st.button("✈️ Travel", key="ql_travel", use_container_width=True):
            st.switch_page("pages/4_✈️_Travel.py")
    with col5:
        if st.button("📰 News", key="ql_news", use_container_width=True):
            st.switch_page("pages/5_📰_News.py")

def render(
    chat,
    mem_text,
    call_jarvis,
    safe_write_module,
    safe_save_json,
    temp_chat_file,
    memory_module,
    chat_module=None,
    weather_module=None,
    podcasts_module=None,
    athletic_module=None,
    todos_module=None,
):
    """Render the main dashboard layout."""
    
    _top_bar()
    _quick_links()
    
    st.markdown("---")

    # 3-column layout: Left (To-Do/Gym/Health + Man Utd) | Mid (Chat) | Right (Weather + Podcasts)
    left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

    with left_col:
        # Football Research Link
        st.markdown("### ⚽ Football Research")
        try:
            st.page_link("pages/football_researcher.py", label="Open Football Researcher")
        except Exception:
            st.caption("Football Researcher page not available")
        
        st.divider()
        
        # To-Do / Gym / Health Panel
        if todos_module:
            try:
                todos_module.render(
                    show_header=True,
                    show_tasks_title=False,
                    show_gym_title=True,
                    show_health_title=True,
                )
            except TypeError:
                todos_module.render()

        st.divider()
        
        # Man United News
        st.markdown("### ⚽ Man United News")
        if athletic_module:
            try:
                athletic_module.render(show_header=False)
            except TypeError:
                athletic_module.render()

    with mid_col:
        st.markdown("### 💬 Chat with Jarvis")
        
        # Chat container with custom styling
        st.markdown("""
        <style>
        .chat-container {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if chat_module:
            chat_module.render()

        # Process any pending user messages
        lst = st.session_state.get("chat", [])
        last_processed = st.session_state.get("last_processed_index", -1)
        last_user_idx = _find_last_user_index(lst)

        if last_user_idx > last_processed:
            try:
                with st.spinner("Jarvis is thinking..."):
                    reply = call_jarvis(lst, mem_text)
                lst.append({"role": "assistant", "content": reply})
                st.session_state.last_processed_index = len(lst) - 1
                st.session_state.chat = lst
                safe_save_json(temp_chat_file, lst)
                
                # Also save to Google Sheets if available
                if SHEETS_AVAILABLE:
                    try:
                        session_id = st.session_state.get("session_id", "default")
                        sm.save_chat_message("assistant", reply, session_id)
                    except Exception:
                        pass
                
                st.rerun()
            except Exception as e:
                st.error(f"Jarvis error: {e}")

        st.session_state.chat = lst
        safe_save_json(temp_chat_file, lst)

    with right_col:
        # Weather Widget
        if weather_module:
            st.markdown("### 🌤️ Weather")
            weather_module.render()
            st.divider()
        
        # Podcasts Widget
        if podcasts_module:
            st.markdown("### 🎧 Podcasts")
            podcasts_module.render()
        
        # Quick Actions
        st.divider()
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Today's Stats", use_container_width=True):
                if SHEETS_AVAILABLE:
                    try:
                        health = sm.read_all_rows("health_logs")
                        workouts = sm.read_all_rows("workout_logs")
                        today = datetime.now().strftime("%Y-%m-%d")
                        today_health = [h for h in health if h.get("date") == today]
                        today_workouts = [w for w in workouts if w.get("date") == today]
                        
                        if today_health or today_workouts:
                            st.success(f"✅ {len(today_health)} health log(s), {len(today_workouts)} workout(s) today")
                        else:
                            st.info("No logs yet today")
                    except Exception as e:
                        st.warning(f"Could not load stats: {e}")
                else:
                    st.info("Connect Google Sheets for stats")
        
        with col2:
            if st.button("🎯 My Goals", use_container_width=True):
                st.switch_page("pages/3_🎯_Goals.py")
