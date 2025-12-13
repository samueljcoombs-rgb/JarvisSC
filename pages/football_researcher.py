# pages/football_researcher.py

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---- Basic config ----
st.set_page_config(page_title="⚽ Football Researcher", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]

# 🔁 ADJUST THIS PATH IF NEEDED
DATA_PATH = Path("/Users/SamECee/football_ai/data/raw/football_ai_NNIA.csv")

# ---- OpenAI client ----
def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        st.error("Missing OPENAI_API_KEY.")
        st.stop()
    return OpenAI(api_key=api_key)

client = _init_client()
MODEL_NAME = os.getenv("PREFERRED_OPENAI_MODEL", "").strip() or "gpt-4o"

# ---- Data loader ----
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"CSV not found at {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df

st.title("⚽ Football Trading AI Researcher")

with st.expander("Dataset preview", expanded=False):
    df = load_data()
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    st.dataframe(df.head(20))

st.markdown("---")

# ---- Chat state ----
if "research_chat" not in st.session_state:
    st.session_state.research_chat = [
        {
            "role": "system",
            "content": (
                "You are an autonomous football trading research assistant.\n"
                "You are working with a scanned dataset containing odds, xG, form, and PL columns "
                "like SHG PL, BO 2.5 PL, LU1.5 PL, LFGHU0.5 PL, BO1.5 FHG PL, BTTS PL.\n"
                "Your job is to help the user find robust, ROI-positive strategies and criteria "
                "they can apply in future games.\n"
                "Ask for any needed details (like column meanings) and propose concrete next steps, "
                "including code snippets if helpful. You do NOT edit code here directly – the main Jarvis "
                "editor on the home page does that."
            ),
        }
    ]

user_msg = st.chat_input("Ask the AI researcher about your football data...")

# ---- Display history ----
for msg in st.session_state.research_chat:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# ---- Handle new question ----
if user_msg:
    st.session_state.research_chat.append({"role": "user", "content": user_msg})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.research_chat,
            )
            reply = resp.choices[0].message.content
            st.markdown(reply)
            st.session_state.research_chat.append(
                {"role": "assistant", "content": reply}
            )
