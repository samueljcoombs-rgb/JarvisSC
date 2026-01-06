# modules/chat_ui.py
"""
Chat UI Module for Jarvis
Handles chat input, message display, and persistence to Google Sheets.
"""
import streamlit as st
import datetime
import os
import shutil
import time

# Try to import sheets_memory for Google Sheets persistence
try:
    from modules import sheets_memory as sm
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

CHAT_KEY = "chat"
BK_FLAG = "chat_ui_backup_done"
SESSION_KEY = "session_id"

def _backup():
    """Create a backup of this file (safety measure)."""
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
    """Ensure chat session state exists."""
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = []
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = str(int(time.time()))

def _save_to_sheets(role: str, content: str):
    """Save a chat message to Google Sheets."""
    if not SHEETS_AVAILABLE:
        return
    try:
        session_id = st.session_state.get(SESSION_KEY, "default")
        sm.save_chat_message(role, content, session_id)
    except Exception as e:
        pass  # Silently fail, local storage is primary

def _render_input_top():
    """Render the chat input box."""
    return st.chat_input("Talk to Jarvis...", key="chat_ui_input")

def _render_msgs_newest_first():
    """Render chat messages with newest at top."""
    messages = st.session_state.get(CHAT_KEY, [])
    
    if not messages:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.5);">
            <p>👋 Hi! I'm Jarvis, your AI assistant.</p>
            <p style="font-size: 0.9rem;">Ask me anything about your health goals, travel plans, entertainment, or just chat!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Render newest first
    for m in reversed(messages):
        role = m.get("role", "assistant")
        content = m.get("content", "")
        
        with st.chat_message(role):
            st.write(content)

def render(*_, **__):
    """Main render function for the chat UI."""
    _backup()
    _ensure()
    
    # Chat controls in columns
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🗑️ Clear Chat", key="chat_ui_clear_btn", use_container_width=True):
            st.session_state[CHAT_KEY] = []
            st.session_state["last_processed_index"] = -1
            st.session_state[SESSION_KEY] = str(int(time.time()))  # New session
            st.rerun()
    
    with col2:
        msg_count = len(st.session_state.get(CHAT_KEY, []))
        st.caption(f"💬 {msg_count} messages")
    
    with col3:
        if SHEETS_AVAILABLE:
            st.caption("☁️ Synced")
        else:
            st.caption("💾 Local only")
    
    # Chat input
    txt = _render_input_top()
    
    if txt:
        # Add user message to session state
        user_msg = {
            "role": "user",
            "content": txt,
            "ts": datetime.datetime.utcnow().isoformat() + "Z"
        }
        st.session_state[CHAT_KEY].append(user_msg)
        
        # Save to Google Sheets
        _save_to_sheets("user", txt)
        
        # Rerun to process the message
        st.rerun()
    
    # Render messages
    _render_msgs_newest_first()

def get_chat_history(limit: int = 50):
    """
    Get chat history from session state or Google Sheets.
    
    Args:
        limit: Maximum number of messages to return
    
    Returns:
        List of message dicts
    """
    # Try session state first (current session)
    session_messages = st.session_state.get(CHAT_KEY, [])
    if session_messages:
        return session_messages[-limit:]
    
    # Try Google Sheets for historical messages
    if SHEETS_AVAILABLE:
        try:
            return sm.get_chat_history(limit=limit)
        except Exception:
            pass
    
    return []

def add_message(role: str, content: str):
    """
    Programmatically add a message to the chat.
    
    Args:
        role: 'user' or 'assistant'
        content: Message content
    """
    _ensure()
    
    msg = {
        "role": role,
        "content": content,
        "ts": datetime.datetime.utcnow().isoformat() + "Z"
    }
    st.session_state[CHAT_KEY].append(msg)
    
    # Save to Google Sheets
    _save_to_sheets(role, content)
