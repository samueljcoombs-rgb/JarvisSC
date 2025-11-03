import streamlit as st
from datetime import datetime
from importlib import import_module


def render(chat, mem_text, call_jarvis, safe_write_module, safe_save_json, temp_chat_file, memory_module):
    """
    Handles the overall layout — pulls in chat_ui and weather_panel modules dynamically.
    """

    st.markdown("## 🤖 Jarvis Modular Dashboard")
    today = datetime.now().strftime("%A, %B %d, %Y")
    st.caption(f"Today: {today}")

    # Create layout columns
    col1, col2 = st.columns([2, 1])

    # --- LEFT: Chat Module ---
    with col1:
        chat_ui = import_module("modules.chat_ui")
        if chat_ui:
            chat_ui.render(
                chat=chat,
                mem_text=mem_text,
                call_jarvis=call_jarvis,
                safe_write_module=safe_write_module,
                safe_save_json=safe_save_json,
                temp_chat_file=temp_chat_file,
                memory_module=memory_module,
            )
        else:
            st.error("❌ Failed to load chat_ui module.")

    # --- RIGHT: Weather Module ---
    with col2:
        weather_panel = import_module("modules.weather_panel")
        if weather_panel:
            weather_panel.render(mem_text=mem_text)
        else:
            st.error("❌ Failed to load weather_panel module.")

