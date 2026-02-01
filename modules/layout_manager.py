# modules/layout_manager.py
"""
Layout Manager for Jarvis Dashboard
Handles the main 3-column layout with integrated widgets and navigation.
"""
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/London")
TODAY = datetime.now(TZ)
DAY_NAME = TODAY.strftime("%A")

# Day configuration
GYM_DAYS = ["Monday", "Wednesday", "Friday", "Saturday"]
RUN_DAYS = ["Tuesday", "Thursday", "Sunday"]
IS_GYM_DAY = DAY_NAME in GYM_DAYS
IS_RUN_DAY = DAY_NAME in RUN_DAYS

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

    # 3-column layout: Left (To-Do + Training + Man Utd) | Mid (Chat) | Right (Weather + Podcasts)
    left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

    with left_col:
        # To-Do Panel (just tasks, not gym routine)
        if todos_module:
            try:
                todos_module.render(
                    show_header=True,
                    show_tasks_title=False,
                    show_gym_title=False,  # Don't show gym from todo sheet
                    show_health_title=True,
                )
            except TypeError:
                todos_module.render()

        st.divider()
        
        # Training Day Card - Premium animated design
        st.markdown("""
        <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        @keyframes glow {
            0%, 100% { box-shadow: 0 8px 32px rgba(16, 185, 129, 0.4); }
            50% { box-shadow: 0 12px 48px rgba(16, 185, 129, 0.6); }
        }
        @keyframes glow-blue {
            0%, 100% { box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4); }
            50% { box-shadow: 0 12px 48px rgba(59, 130, 246, 0.6); }
        }
        .training-card {
            padding: 1.5rem;
            border-radius: 20px;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        .training-card:hover {
            transform: translateY(-4px);
        }
        .training-card.gym {
            background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
            animation: glow 3s ease-in-out infinite;
        }
        .training-card.run {
            background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 50%, #60a5fa 100%);
            animation: glow-blue 3s ease-in-out infinite;
        }
        .training-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
            pointer-events: none;
        }
        .training-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
        }
        .training-title {
            font-weight: 900;
            font-size: 1.3rem;
            color: white;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin-bottom: 0.25rem;
        }
        .training-subtitle {
            font-size: 0.95rem;
            color: rgba(255,255,255,0.9);
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if IS_GYM_DAY:
            st.markdown(f"""
            <div class="training-card gym">
                <div class="training-icon">💪</div>
                <div class="training-title">Gym Day</div>
                <div class="training-subtitle">{DAY_NAME}'s Workout Ready</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🏋️ Open Workout →", key="goto_gym", use_container_width=True, type="primary"):
                st.switch_page("pages/1_🏋️_Health_Fitness.py")
        else:
            st.markdown(f"""
            <div class="training-card run">
                <div class="training-icon">🏃</div>
                <div class="training-title">Running Day</div>
                <div class="training-subtitle">Time to hit the road!</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("👟 Log Run →", key="goto_run", use_container_width=True, type="primary"):
                st.switch_page("pages/1_🏋️_Health_Fitness.py")

        st.divider()
        
        # Man United News with premium header
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
            <span style="font-size: 1.5rem;">⚽</span>
            <span style="font-weight: 800; font-size: 1.1rem; 
                         background: linear-gradient(135deg, #da291c 0%, #ffd700 100%);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                         background-clip: text;">Man United News</span>
        </div>
        """, unsafe_allow_html=True)
        if athletic_module:
            try:
                athletic_module.render(show_header=False)
            except TypeError:
                athletic_module.render()

    with mid_col:
        st.markdown("### 💬 Chat with Jarvis")
        
        # Premium chat styling
        st.markdown("""
        <style>
        /* Premium chat container */
        .stChatMessage {
            background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 16px !important;
            backdrop-filter: blur(10px) !important;
            margin-bottom: 0.75rem !important;
        }
        
        .stChatMessage:hover {
            border-color: rgba(139, 92, 246, 0.3) !important;
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.15) !important;
        }
        
        /* User messages */
        .stChatMessage[data-testid="user-message"] {
            background: linear-gradient(145deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%) !important;
            border-color: rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Assistant messages */
        .stChatMessage[data-testid="assistant-message"] {
            background: linear-gradient(145deg, rgba(139, 92, 246, 0.1) 0%, rgba(139, 92, 246, 0.02) 100%) !important;
            border-color: rgba(139, 92, 246, 0.15) !important;
        }
        
        /* Chat input styling */
        .stChatInput > div {
            border-radius: 16px !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            background: rgba(255,255,255,0.05) !important;
        }
        
        .stChatInput > div:focus-within {
            border-color: rgba(139, 92, 246, 0.5) !important;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.2) !important;
        }
        
        /* Welcome message styling */
        .welcome-msg {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            margin: 1rem 0;
        }
        .welcome-msg .wave {
            font-size: 3rem;
            margin-bottom: 1rem;
            animation: wave 2s ease-in-out infinite;
        }
        @keyframes wave {
            0%, 100% { transform: rotate(0deg); }
            25% { transform: rotate(20deg); }
            75% { transform: rotate(-20deg); }
        }
        .welcome-msg h3 {
            color: rgba(255,255,255,0.9);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .welcome-msg p {
            color: rgba(255,255,255,0.6);
            font-size: 0.95rem;
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
            <div class="welcome-msg">
                <div class="wave">👋</div>
                <h3>Hi! I'm Jarvis</h3>
                <p>Ask me anything about your health goals, travel plans, entertainment, or just chat!</p>
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
        # Weather Widget
        if weather_module:
            weather_module.render()
            st.divider()
        
        # Podcasts Widget
        if podcasts_module:
            podcasts_module.render()
