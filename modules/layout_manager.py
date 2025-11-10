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
    # Top date header
    st.subheader(f"Today: {datetime.now().strftime('%A, %B %d, %Y')}")

    # Three fixed columns: LEFT (Athletic), MIDDLE (Chat), RIGHT (Weather/Podcasts)
    c_left, c_mid, c_right = st.columns([1.2, 1.8, 1.2], gap="large")

    # --- LEFT: Athletic feed only ---
    with c_left:
        st.header("⚽ The Athletic Feed")
        if athletic_module:
            athletic_module.render()

    # --- MIDDLE: Chat only ---
    with c_mid:
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

        # Persist chat history
        st.session_state.chat = lst
        safe_save_json(temp_chat_file, lst)

    # --- RIGHT: Weather at the top, then Podcasts ---
    with c_right:
        if weather_module:
            st.header("🌤️ Weather")
            weather_module.render()

        if podcasts_module:
            st.divider()
            st.header("🎧 Podcasts")
            podcasts_module.render()
