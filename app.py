import os, json, time
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

# ----- Model selection: always try to use the best available GPT automatically -----
PREFERRED_ENV = os.getenv("PREFERRED_OPENAI_MODEL", "").strip()

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

def _select_best_model(client: OpenAI) -> str:
    """
    Picks the best available chat model without hardcoding to 4o-mini.
    Preference order:
    1) PREFERRED_OPENAI_MODEL (env)
    2) gpt-5 (if available)
    3) gpt-latest
    4) gpt-4.1, gpt-4o, gpt-4.1-mini
    5) gpt-4o-mini (last resort)
    """
    if PREFERRED_ENV:
        return PREFERRED_ENV

    try:
        # List models once and pick by preference.
        # If listing fails (permissions), fall back to a safe default.
        names = {m.id for m in client.models.list().data}  # type: ignore[attr-defined]
        for candidate in [
            "gpt-5",
            "gpt-latest",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4o-mini",
        ]:
            if candidate in names:
                return candidate
    except Exception:
        pass
    return "gpt-4o"  # sensible default if listing not permitted

JARVIS_MODEL = _select_best_model(client)

# ----- JSON helpers (with atomic saves) -----
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
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except Exception as e:
        st.warning(f"⚠️ Could not save {path.name}: {e}")

# ----- Module hot-loader + guarded writer -----
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
        st.error(f"❌ Syntax error in {name}.py: {e}")
        return False
    backup_module(name)
    # atomic write
    tmp = path.with_suffix(".py.tmp")
    tmp.write_text(code, encoding="utf-8")
    tmp.replace(path)
    st.success(f"✅ {name}.py updated.")
    return True

def load_module(name: str):
    try:
        path = MODULES_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod
    except Exception as e:
        st.error(f"⚠️ Failed to load {name}: {e}")
        return None

# ----- OpenAI call -----
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

# ----- App -----
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
    if preview and len(preview) > 220:
        st.write(preview[:220] + "…")
        with st.expander("Show full recent memory"):
            st.write(preview)
    else:
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
