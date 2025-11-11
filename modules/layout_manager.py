import streamlit as st
from datetime import datetime

def _find_last_user_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1

def _top_bar():
    # Slim top bar with small app name (left) and subtle date (right)
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin:0 0 0.35rem 0;">
            <div style="font-weight:600;font-size:1.05rem;">🤖 Jarvis</div>
            <div style="color:#6b7280;font-size:0.92rem;">{today_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin-top:0.35rem;margin-bottom:0.75rem;'>", unsafe_allow_html=True)

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
    todos_module=None,          # NEW: To-Do panel
):
    # Compact top bar
    _top_bar()

    # 3-column layout: To-Do + Athletic (left) | Chat (middle) | Weather + Podcasts (right)
    left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

    # LEFT: To-Do (top), then Man United news
    with left_col:
        if todos_module:
            try:
                st.subheader("📝 To-Do")
                todos_module.render()
            except Exception as e:
                st.warning(f"To-Do module error: {e}")
            st.divider()

        st.subheader("⚽ Man United News")
        if athletic_module:
            # Prefer the module's ability to suppress its own header if available
            try:
                athletic_module.render(show_header=False)
            except TypeError:
                athletic_module.render()
            except Exception as e:
                st.warning(f"Athletic module error: {e}")

    # MIDDLE: Chat
    with mid_col:
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

    # RIGHT: Weather (keep its own internal "Weather Forecast" title), then Podcasts
    with right_col:
        if weather_module:
            try:
                weather_module.render()
            except Exception as e:
                st.warning(f"Weather module error: {e}")
            st.divider()
        if podcasts_module:
            try:
                podcasts_module.render()
            except Exception as e:
                st.warning(f"Podcasts module error: {e}")
