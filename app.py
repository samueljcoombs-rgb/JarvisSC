import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import importlib.util

import streamlit as st
from openai import OpenAI

import memory  # local memory module


# ----------------- PATHS & CONSTANTS -----------------
BASE_DIR = Path(__file__).parent
MODULES_DIR = BASE_DIR / "modules"
TEMP_CHAT_FILE = BASE_DIR / "temp_chat.json"
CHAT_SESSIONS_FILE = BASE_DIR / "chat_sessions.json"

JARVIS_MODEL = "gpt-5"  # model used inside the app
client = OpenAI()  # uses OPENAI_API_KEY from env / Streamlit secrets


# ----------------- JSON HELPERS -----------------
def safe_load_json(path: Path, default):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def safe_save_json(path: Path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Could not save {path.name}: {e}")


# ----------------- MODULE EDIT / BACKUP -----------------
def backup_module(module_name: str):
    src = MODULES_DIR / f"{module_name}.py"
    dst = MODULES_DIR / f"{module_name}_backup.py"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def safe_write_module(module_name: str, new_code: str) -> bool:
    """
    Safely overwrite one module file (e.g. weather_panel.py).
    We compile first to avoid syntax errors, and keep a backup.
    """
    path = MODULES_DIR / f"{module_name}.py"
    if not path.exists():
        st.error(f"Module {module_name}.py not found.")
        return False

    try:
        compile(new_code, str(path), "exec")  # syntax check only
    except SyntaxError as e:
        st.error(f"❌ Syntax error in new {module_name}.py: {e}")
        return False

    backup_module(module_name)
    path.write_text(new_code, encoding="utf-8")
    st.success(f"✅ {module_name}.py updated successfully.")
    return True


def load_module(name: str):
    try:
        spec = importlib.util.spec_from_file_location(
            name, MODULES_DIR / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        st.error(f"⚠️ Failed to load module '{name}': {e}")
        return None


# ----------------- OPENAI / JARVIS CALL -----------------
def call_jarvis(chat_history, mem_text: str) -> str:
    """
    Send the conversation + memory summary to the OpenAI model
    and get Jarvis's reply.
    """
    system_prompt = (
        "You are Jarvis, a modular AI assistant living inside a Streamlit app.\n"
        "You can modify layout, visuals, and *module* code inside the /modules folder only.\n"
        "You MUST NOT change app.py, memory.py, or any API key handling.\n"
        "Modules you are allowed to edit include: chat_ui.py, weather_panel.py, layout_manager.py.\n"
        "Each module defines a render(...) function that Streamlit calls.\n"
        "When you output code, it must be FULL code for a single module inside one ```python``` block.\n"
        "If you want to change layout, edit layout_manager.py.\n"
        "If you want to change the weather widget UI, edit weather_panel.py.\n"
        "Always keep your code syntactically valid Python.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *chat_history,
        {
            "role": "user",
            "content": f"Here is a summary of your long-term memory:\n{mem_text}",
        },
    ]

    resp = client.chat.completions.create(
        model=JARVIS_MODEL,
        messages=messages,
    )
    return resp.choices[0].message.content


# ----------------- STREAMLIT PAGE CONFIG -----------------
st.set_page_config(page_title="Jarvis Modular AI", layout="wide")


# ----------------- SESSION STATE -----------------
if "chat" not in st.session_state:
    st.session_state.chat = safe_load_json(TEMP_CHAT_FILE, [])


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("🧠 Memory & Sessions")

    long_term = memory._load()
    mem_text = memory.recent_summary()
    sessions = safe_load_json(CHAT_SESSIONS_FILE, [])

    st.caption(f"🤖 Model: {JARVIS_MODEL}")
    st.caption(f"💬 Chat messages: {len(st.session_state.chat)}")
    st.caption(f"🧠 Long-term memories: {len(long_term)}")
    st.caption(f"📚 Saved sessions: {len(sessions)}")

    st.divider()
    st.subheader("Long-term memory (summary)")
    st.write(mem_text or "No memories yet.")

    new_mem = st.text_input("Add to memory:")
    if new_mem:
        memory.add_fact(new_mem, kind="manual")
        st.success("Saved to memory.")
        st.rerun()

    st.divider()
    if st.button("💾 Save current chat"):
        sessions.append(
            {
                "id": int(time.time()),
                "ts": int(time.time()),
                "messages": st.session_state.chat,
            }
        )
        safe_save_json(CHAT_SESSIONS_FILE, sessions)
        st.success("Chat saved.")
        st.rerun()

    if st.button("🗑️ Start new chat"):
        st.session_state.chat = []
        safe_save_json(TEMP_CHAT_FILE, [])
        st.success("Cleared current chat.")
        st.rerun()

    st.divider()
    if st.button("🔙 Restore module backups"):
        any_restored = False
        for backup in MODULES_DIR.glob("*_backup.py"):
            target = backup.with_name(
                backup.stem.replace("_backup", "") + ".py"
            )
            target.write_text(
                backup.read_text(encoding="utf-8"), encoding="utf-8"
            )
            any_restored = True
        if any_restored:
            st.success("Restored module backups. Reloading...")
            st.experimental_rerun()
        else:
            st.info("No module backups found.")


# ----------------- MAIN LAYOUT -----------------
st.title("🤖 Jarvis Modular Dashboard")

layout_mod = load_module("layout_manager")
if layout_mod is None:
    st.error("layout_manager.py is missing or broken in /modules.")
else:
    # layout_manager is responsible for placing chat/weather/etc on the page
    layout_mod.render(
        chat=st.session_state.chat,
        mem_text=memory.recent_summary(),
        call_jarvis=call_jarvis,
        safe_write_module=safe_write_module,
        safe_save_json=safe_save_json,
        temp_chat_file=TEMP_CHAT_FILE,
        memory_module=memory,
    )
