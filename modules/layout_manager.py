import streamlit as st
from datetime import datetime

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
):
    # Page header
    st.subheader(f"Today: {datetime.now().strftime('%A, %B %d, %Y')}")

    # Layout: Athletic feed LEFT (narrow), Chat + widgets RIGHT (wide)
    # Adjust ratios if you want the Athletic list wider: e.g., [3, 5]
    left_col, right_col = st.columns([3, 5], gap="large")

    # LEFT: Athletic feed (runs down the left)
    with left_col:
        st.header("⚽ The Athletic Feed")
        if athletic_module:
            athletic_module.render()

    # RIGHT: Weather at the very top, then Chat, then Podcasts
    with right_col:
        # Weather module first (so it appears at the top, not halfway down)
        if weather_module:
            st.header("🌤️ Weather")
            weather_module.render()
            st.divider()

        # Chat
        st.header("💬 Chat with Jarvis")
        if chat_module:
            chat_module.render()

        lst = st.session_state.get("chat", [])
        last_processed = st.session_state.get("last_processed_index", -1)
        last_user_idx = _find_last_user_index(lst)

        # Process only new user messages
        if last_user_idx > last_processed:
            try:
                reply = call_jarvis(lst, mem_text)
                lst.append({"role": "assistant", "content": reply})
                st.session_state.last_processed_index = len(lst) - 1
                st.session_state.chat = lst
                safe_save_json(temp_chat_file, lst)
                st.rerun()
            except Exception as e:
                st.error(f"Jarvis error: {e}")

        st.session_state.chat = lst
        safe_save_json(temp_chat_file, lst)

        # Podcasts (below chat)
        if podcasts_module:
            st.divider()
            st.header("🎧 Podcasts")
            podcasts_module.render()
