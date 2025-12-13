# modules/layout_manager.py
import streamlit as st
from datetime import datetime

def _find_last_user_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1

def _top_bar():
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
    todos_module=None,   # To-Do / Gym / Health panel
):
    _top_bar()

    # 3-column layout: Left (To-Do/Gym/Health + Man Utd) | Mid (Chat) | Right (Weather + Podcasts)
    left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

    with left_col:

        # ⭐⭐⭐ ADDED: Football Research Page Link ⭐⭐⭐
        st.markdown("### ⚽ Football Research")
        try:
            st.page_link("pages/football_researcher.py", label="Open Football Researcher")
        except Exception:
            st.markdown("[Open Football Researcher](/football_researcher)")
        st.divider()
        # ⭐⭐⭐ END OF INSERT ⭐⭐⭐

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

        st.subheader("⚽ Man United News")
        if athletic_module:
            try:
                athletic_module.render(show_header=False)
            except TypeError:
                athletic_module.render()

    with mid_col:
        if chat_module:
            chat_module.render()

        lst = st.session_state.get("chat", [])
        last_processed = st.session_state.get("last_processed_index", -1)
        last_user_idx = _find_last_user_index(lst)

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

    with right_col:
        if weather_module:
            weather_module.render()
            st.divider()
        if podcasts_module:
            podcasts_module.render()
