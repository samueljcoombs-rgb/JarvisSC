import streamlit as st

from . import chat_ui, weather_panel


def render(
    chat,
    mem_text,
    call_jarvis,
    safe_write_module,
    safe_save_json,
    temp_chat_file,
    memory_module,
):
    """
    Layout manager: decides how the dashboard is arranged.
    Right now:
      - Left: chat
      - Right: weather panel
    Jarvis can later modify THIS file to change the layout.
    """
    today = st.session_state.get("today_str")
    if not today:
        from datetime import datetime

        today = datetime.now().strftime("%A, %B %d, %Y")
        st.session_state.today_str = today

    st.subheader(f"Today: {today}")

    st.write("You can say things like:")
    st.write("- **remember** that I prefer evening workouts")
    st.write("- **save this chat** so we can reload it later")
    st.write("- **redesign the layout** (I will update layout_manager.py)")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat")
        chat_ui.render(
            chat=chat,
            mem_text=mem_text,
            call_jarvis=call_jarvis,
            safe_write_module=safe_write_module,
            safe_save_json=safe_save_json,
            temp_chat_file=temp_chat_file,
            memory_module=memory_module,
        )

    with col2:
        weather_panel.render(default_city="Basingstoke")
