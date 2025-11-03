# modules/layout_manager.py
import streamlit as st
from modules import chat_ui, weather_panel


def render_dashboard():
    """Main page layout for Jarvis Modular Dashboard."""
    st.title("🤖 Jarvis Modular Dashboard")

    col1, col2 = st.columns([2, 1])

    with col1:
        chat_ui.render_chat()

    with col2:
        weather_panel.render_weather()
