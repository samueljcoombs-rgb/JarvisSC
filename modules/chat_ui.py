import streamlit as st, datetime, os, shutil

CHAT_KEY = "chat"
BK_FLAG = "chat_ui_backup_done"

def _backup():
    try:
        if st.session_state.get(BK_FLAG):
            return
        src = __file__
        dst = os.path.join(os.path.dirname(src), "chat_ui_backup.py")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        st.session_state[BK_FLAG] = True
    except:
        pass

def _ensure():
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = []

def _render_input_top():
    return st.chat_input("Talk to Jarvis...", key="chat_ui_input")

def _render_msgs_newest_first():
    # Render newest directly beneath input
    for m in reversed(st.session_state[CHAT_KEY]):
        with st.chat_message(m.get("role", "assistant")):
            st.write(m.get("content", ""))

def render(*_, **__):
    _backup()
    _ensure()

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Clear chat", key="chat_ui_clear_btn"):
            st.session_state[CHAT_KEY] = []
            st.session_state["last_processed_index"] = -1
            st.rerun()

    txt = _render_input_top()
    if txt:
        st.session_state[CHAT_KEY].append({
            "role": "user",
            "content": txt,
            "ts": datetime.datetime.utcnow().isoformat() + "Z"
        })

    _render_msgs_newest_first()
