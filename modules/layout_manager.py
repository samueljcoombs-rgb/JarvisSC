import streamlit as st
from datetime import datetime
from importlib import import_module


def render(chat, mem_text, call_jarvis, safe_write_module, safe_save_json, temp_chat_file, memory_module):
    """
    Handles overall layout — loads and renders chat_ui + weather_panel dynamically.
    """

    st.markdown("## 🤖 Jarvis Modular Dashboard")
    today = datetime.now().strftime("%A, %B %d, %Y")
    st.caption(f"Today: {today}")

    # Create layout columns
    col1, col2 = st.columns([2, 1])

    # --- LEFT: Chat Module ---
    with col1:
        try:
            chat_ui = import_module("modules.chat_ui")
            with st.container():
                chat_ui.render(
                    chat=chat,
                    mem_text=mem_text,
                    call_jarvis=call_jarvis,
                    safe_write_module=safe_write_module,
                    safe_save_json=safe_save_json,
                    temp_chat_file=temp_chat_file,
                    memory_module=memory_module,
                )
        except Exception as e:
            st.error(f"⚠️ Failed to load chat_ui module: {e}")

    # --- RIGHT: Weather Module ---
    with col2:
        try:
            weather_panel = import_module("modules.weather_panel")
            weather_panel.render(mem_text=mem_text)
        except Exception as e:
            st.error(f"⚠️ Failed to load weather_panel module: {e}")

