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

    # 3-column layout: Left (To-Do + Man Utd) | Mid (Chat) | Right (Weather + Podcasts)
    left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

    with left_col:
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
        
        # Chat styling
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
        
        # Buttons at TOP (like Health page)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat", key="layout_clear_btn", use_container_width=True):
                st.session_state["chat"] = []
                st.session_state["last_processed_index"] = -1
                safe_save_json(temp_chat_file, [])
                st.rerun()
        
        with col2:
            msg_count = len(st.session_state.get("chat", []))
            st.caption(f"💬 {msg_count} messages")
        
        with col3:
            if SHEETS_AVAILABLE:
                st.caption("☁️ Synced")
            else:
                st.caption("💾 Local only")
        
        # Chat input at TOP (below buttons)
        user_input = st.chat_input("Talk to Jarvis...", key="layout_chat_input")
        
        if user_input:
            # Add user message
            st.session_state["chat"].append({"role": "user", "content": user_input})
            
            # Save to sheets if available
            if SHEETS_AVAILABLE:
                try:
                    session_id = st.session_state.get("session_id", "default")
                    sm.save_chat_message("user", user_input, session_id)
                except Exception:
                    pass
            
            st.rerun()
        
        st.markdown("---")
        
        # Chat history in scrollable container (longer than health page)
        messages = st.session_state.get("chat", [])
        if messages:
            chat_container = st.container(height=550)
            with chat_container:
                # Show newest first
                for msg in reversed(messages):
                    with st.chat_message(msg.get("role", "assistant")):
                        st.write(msg.get("content", ""))
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.5);">
                <p>👋 Hi! I'm Jarvis, your AI assistant.</p>
                <p style="font-size: 0.9rem;">Ask me anything about your health goals, travel plans, entertainment, or just chat!</p>
            </div>
            """, unsafe_allow_html=True)

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
        # Weather Widget (module has its own header, so don't add another)
        if weather_module:
            weather_module.render()
            st.divider()
        
        # Podcasts Widget (module has its own header, so don't add another)
        if podcasts_module:
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
