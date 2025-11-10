import os, json, time, traceback
from datetime import datetime
from pathlib import Path
import importlib.util
import streamlit as st
from openai import OpenAI
import memory

BASE_DIR = Path(__file__).parent
MODULES_DIR = BASE_DIR / "modules"
TEMP_CHAT_FILE = BASE_DIR / "temp_chat.json"
CHAT_SESSIONS_FILE = BASE_DIR / "chat_sessions.json"
JARVIS_MODEL = "gpt-4o-mini"  # can be switched to GPT-5 later

def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        st.error("Missing OPENAI_API_KEY."); st.stop()
    return OpenAI(api_key=api_key)

client = _init_client()

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
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"⚠️ Could not save {path.name}: {e}")

def backup_module(name: str):
    src = MODULES_DIR / f"{name}.py"
    dst = MODULES_DIR / f"{name}_backup.py"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

def safe_write_module(name: str, code: str) -> bool:
    path = MODULES_DIR / f"{name}.py"
    if not path.exists():
        st.error(f"Missing module {name}.py")
        return False
    try:
        compile(code, str(path), "exec")
    except SyntaxError as e:
        st.error(f"❌ Syntax error: {e}")
        return False
    backup_module(name)
    path.write_text(code, encoding="utf-8")
    st.success(f"✅ {name}.py updated.")
    return True

def load_module(name: str):
    try:
        path = MODULES_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        st.error(f"⚠️ Failed to load {name}: {e}")
        return None

def call_jarvis(chat_history, mem_text):
    sys = (
        "You are Jarvis inside a Streamlit app. "
        "Edit only /modules/*.py. Do not touch app.py or memory.py. "
        "Always output full valid Python code inside one ```python block when editing. "
        f"Memory summary:\n{mem_text}"
    )
    msgs = [{"role": "system", "content": sys}, *chat_history]
    resp = client.chat.completions.create(model=JARVIS_MODEL, messages=msgs)
    return resp.choices[0].message.content

st.set_page_config(page_title="Jarvis Modular AI", layout="wide")

if "chat" not in st.session_state:
    st.session_state.chat = safe_load_json(TEMP_CHAT_FILE, [])
if "last_processed_index" not in st.session_state:
    st.session_state.last_processed_index = -1

with st.sidebar:
    st.header("🧠 Memory & Sessions")
    long_term = memory._load()
    mem_text = memory.recent_summary()
    sessions = safe_load_json(CHAT_SESSIONS_FILE, [])
    st.caption(f"Model: {JARVIS_MODEL}")
    st.caption(f"Messages: {len(st.session_state.chat)}")
    st.caption(f"Memories: {len(long_term)}")
    st.caption(f"Sessions: {len(sessions)}")

    st.divider()
    st.subheader("Memory (preview)")
    preview = (mem_text or "").strip()
    short = (preview[:200] + "…") if preview and len(preview) > 200 else (preview or "No memories yet.")
    st.write(short)
    with st.expander("Show full recent memory"):
        st.write(preview or "No memories yet.")

    new_mem = st.text_input("Add to memory:")
    if new_mem:
        memory.add_fact(new_mem, "manual")
        st.success("Saved.")
        st.rerun()

    st.divider()
    if st.button("💾 Save chat"):
        sessions.append({"id": int(time.time()), "ts": int(time.time()), "messages": st.session_state.chat})
        safe_save_json(CHAT_SESSIONS_FILE, sessions)
        st.success("Saved.")
        st.rerun()
    if st.button("🗑️ New chat"):
        st.session_state.chat = []
        safe_save_json(TEMP_CHAT_FILE, [])
        st.session_state.last_processed_index = -1
        st.success("Cleared.")
        st.rerun()

st.title("🤖 Jarvis Modular Dashboard")
layout_mod = load_module("layout_manager")
chat_mod = load_module("chat_ui")
weather_mod = load_module("weather_panel")

if layout_mod:
    layout_mod.render(
        chat=st.session_state.chat,
        mem_text=memory.recent_summary(),
        call_jarvis=call_jarvis,
        safe_write_module=safe_write_module,
        safe_save_json=safe_save_json,
        temp_chat_file=TEMP_CHAT_FILE,
        memory_module=memory,
        chat_module=chat_mod,
        weather_module=weather_mod,
    )
