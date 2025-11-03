import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from importlib import import_module

import streamlit as st
from openai import OpenAI

import memory  # your memory.py in the same folder as app.py

# ----------------- PATHS & CONSTANTS -----------------
BASE_DIR = Path(__file__).parent
MODULES_DIR = BASE_DIR / "modules"
TEMP_CHAT_FILE = BASE_DIR / "temp_chat.json"
CHAT_SESSIONS_FILE = BASE_DIR / "chat_sessions.json"

JARVIS_MODEL = "gpt-5"  # Jarvis runs on GPT-4.1
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
        st.error(f"Module {module_name}.py not found in /modules.")
        return False

    # Syntax check
    try:
        compile(new_code, str(path), "exec")
    except SyntaxError as e:
        st.error(f"❌ Syntax error in new {module_name}.py: {e}")
        return False

    # Backup + write
    backup_module(module_name)
    path.write_text(new_code, encoding="utf-8")
    st.success(f"✅ {module_name}.py updated successfully.")
    return True


# ----------------- JARVIS CALL -----------------
def call_jarvis(chat_history, mem_text: str) -> str:
    """
    Send the conversation + memory summary to the OpenAI model
    and get Jarvis's reply.
    """
    system_prompt = (
        "You are Jarvis, a modular AI assistant living inside a Streamlit app.\n"
        "You live in app.py but you are only allowed to directly change the small modules "
        "inside the /modules folder: chat_ui.py, weather_panel.py, layout_manager.py.\n"
        "You MUST NOT change app.py, memory.py, or any API key handling.\n"
        "Each module exposes a render(...) function that the app calls.\n"
        "When you output code, it must be the FULL contents of exactly one module file, "
        "inside a single ```python``` code block.\n"
        "Never output partial fragments; always full, syntactically valid Python modules.\n"
        "If you want to change layout, modify layout_manager.py. If you want to change the "
        "chat UI, modify chat_ui.py. If you want to change the weather UI, modify weather_panel.py.\n"
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
    st.caption(f"💬 Chat messages (current): {len(st.session_state.chat)}")
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
    st.subheader("Chat controls")

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
    st.subheader("Load a saved chat")

    if sessions:
        labels = []
        for s in sessions:
            dt = datetime.fromtimestamp(s["ts"]).strftime("%Y-%m-%d %H:%M")
            msg_count = len(s.get("messages", []))
            label = f"{s['id']} | {dt} | {msg_count} msgs"
            labels.append(label)

        selected = st.selectbox("Select session:", labels)
        if st.button("🔁 Load selected session"):
            try:
                sel_id = int(selected.split("|", 1)[0].strip())
                sess = next(s for s in sessions if s["id"] == sel_id)
                st.session_state.chat = sess.get("messages", [])
                safe_save_json(TEMP_CHAT_FILE, st.session_state.chat)
                st.success(f"Loaded session {sel_id}.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load session: {e}")
    else:
        st.caption("No saved chat sessions yet.")

    st.divider()
    st.subheader("Module backups")

    if st.button("🔙 Restore module backups"):
        any_restored = False
        for backup in MODULES_DIR.glob("*_backup.py"):
            target = backup.with_name(backup.stem.replace("_backup", "") + ".py")
            target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            any_restored = True
        if any_restored:
            st.success("Restored module backups. Reloading...")
            st.rerun()
        else:
            st.info("No module backups found.")


# ----------------- MAIN LAYOUT -----------------
st.title("🤖 Jarvis Modular Dashboard")
today = datetime.now().strftime("%A, %B %d, %Y")
st.caption(f"Today: {today}")
st.divider()

try:
    layout_mod = import_module("modules.layout_manager")
except Exception as e:
    st.error(f"⚠️ Could not import layout_manager module: {e}")
else:
    try:
        layout_mod.render(
            chat=st.session_state.chat,
            mem_text=memory.recent_summary(),
            call_jarvis=call_jarvis,
            safe_write_module=safe_write_module,
            safe_save_json=safe_save_json,
            temp_chat_file=TEMP_CHAT_FILE,
            memory_module=memory,
        )
    except Exception:
        st.error("⚠️ Error in layout_manager.render():")
        st.code(traceback.format_exc())
