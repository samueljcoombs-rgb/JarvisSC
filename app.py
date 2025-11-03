import os
import json
import time
import traceback
import importlib.util
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

import memory

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Jarvis Modular AI", layout="wide")

BASE_DIR = Path(__file__).parent
MODULES_DIR = BASE_DIR / "modules"
TEMP_CHAT_FILE = BASE_DIR / "temp_chat.json"
CHAT_SESSIONS_FILE = BASE_DIR / "chat_sessions.json"

JARVIS_MODEL = "gpt-5"
client = OpenAI()


# ----------------- UTILITIES -----------------
def safe_load_json(path, default):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def safe_save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Could not save {path}: {e}")


def backup_module(module_name):
    src = MODULES_DIR / f"{module_name}.py"
    dst = MODULES_DIR / f"{module_name}_backup.py"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def safe_write_module(module_name, new_code):
    """Safely rewrite one module without killing Jarvis."""
    path = MODULES_DIR / f"{module_name}.py"
    if not path.exists():
        st.error("Module not found.")
        return False
    try:
        compile(new_code, str(path), "exec")  # test syntax
        backup_module(module_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_code)
        st.success(f"✅ Updated {module_name}.py successfully.")
        return True
    except SyntaxError as e:
        st.error(f"❌ Syntax error: {e}")
        return False


def load_module(name):
    try:
        spec = importlib.util.spec_from_file_location(name, MODULES_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        st.error(f"⚠️ Failed to load {name}: {e}")
        return None


# ----------------- OPENAI CALL -----------------
def call_jarvis(chat_history, mem_text):
    system_prompt = (
        "You are Jarvis, a modular AI assistant living inside a Streamlit app.\n"
        "You can modify layout, visuals, and module code (inside /modules only).\n"
        "You CANNOT modify app.py, memory.py, or any API key logic.\n"
        "When updating a module, output only full code for that module in a ```python block.\n"
        "Modules always define a `render()` function.\n"
        "If you want to add or rearrange panels, modify layout_manager.py only.\n"
        "Always test that your code compiles syntactically.\n"
        "Never include secrets or credentials.\n"
    )

    resp = client.chat.completions.create(
        model=JARVIS_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *chat_history,
            {"role": "user", "content": f"Current memory summary:\n{mem_text}"},
        ],
    )
    return resp.choices[0].message.content


# ----------------- STATE -----------------
if "chat" not in st.session_state:
    st.session_state.chat = safe_load_json(TEMP_CHAT_FILE, [])


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("🧠 Memory & Sessions")

    long_term = memory._load()
    mem_text = memory.recent_summary()
    sessions = safe_load_json(CHAT_SESSIONS_FILE, [])

    st.caption(f"🤖 Model: {JARVIS_MODEL}")
    st.caption(f"💬 Chat msgs: {len(st.session_state.chat)}")
    st.caption(f"🧠 Memories: {len(long_term)}")
    st.caption(f"📚 Sessions: {len(sessions)}")

    st.divider()
    st.subheader("Long-term memory (summary)")
    st.write(mem_text or "No memories yet.")

    new_mem = st.text_input("Add to memory:")
    if new_mem:
        memory.add_fact(new_mem, kind="manual")
        st.success("Saved.")
        st.rerun()

    st.divider()
    if st.button("💾 Save chat"):
        sessions.append({"id": int(time.time()), "messages": st.session_state.chat})
        safe_save_json(CHAT_SESSIONS_FILE, sessions)
        st.success("Chat saved!")

    if st.button("🗑️ New chat"):
        st.session_state.chat = []
        safe_save_json(TEMP_CHAT_FILE, [])
        st.rerun()

    st.divider()
    if st.button("🔙 Undo last module change"):
        for f in MODULES_DIR.glob("*_backup.py"):
            original = f.with_name(f.stem.replace("_backup", "") + ".py")
            if f.exists():
                original.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        st.success("Reverted to backups.")
        st.rerun()


# ----------------- MAIN INTERFACE -----------------
layout = load_module("layout_manager")
if layout:
    layout.render(st.session_state.chat, mem_text, call_jarvis, safe_write_module, safe_save_json)
else:
    st.error("Layout module missing.")
