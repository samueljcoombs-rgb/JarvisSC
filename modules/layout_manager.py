import streamlit as st
from datetime import datetime

def _find_last_user_index(messages):
    # Return last index where role == 'user', else -1
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
    weather_module=None
):
    st.subheader(f"Today: {datetime.now().strftime('%A, %B %d, %Y')}")

    # Two columns: chat left (2x), weather right (1x)
    c1, c2 = st.columns([2, 1])

    with c1:
        st.header("💬 Chat with Jarvis")
        if chat_module:
            chat_module.render()

        # Robust "new user message" detection — only respond once per new input
        lst = st.session_state.get("chat", [])
        last_processed = st.session_state.get("last_processed_index", -1)
        last_user_idx = _find_last_user_index(lst)

        if last_user_idx > last_processed:
            try:
                reply = call_jarvis(lst, mem_text)
                lst.append({"role": "assistant", "content": reply})
                # Mark the assistant message index as processed
                st.session_state.last_processed_index = len(lst) - 1
                st.session_state.chat = lst
                safe_save_json(temp_chat_file, lst)
                # Force a rerun so the new assistant message appears at the top immediately
                st.rerun()
            except Exception as e:
                st.error(f"Jarvis error: {e}")

        # Keep a save in case nothing changed this run
        st.session_state.chat = lst
        safe_save_json(temp_chat_file, lst)

    with c2:
        if weather_module:
            weather_module.render()
