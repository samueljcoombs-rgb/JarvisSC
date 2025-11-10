# chat_ui.py
# Chat UI module for Jarvis in Streamlit.
# - Exposes render(*args, **kwargs)
# - Displays messages from st.session_state.chat
# - Creates at most one st.chat_input, and only if no other module owns it.
# - Performs a one-time self-backup to chat_ui_backup.py for safety.

from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st


_BACKUP_FLAG_KEY = "chat_ui_backup_done"
_CHAT_OWNER_KEY = "chat_input_owner"    # global owner lock name used app-wide
_CHAT_LIST_KEY = "chat"                 # session state key used by main app
_NEW_MSG_KEY = "chat_new_user_message"
_NEEDS_RESPONSE_KEY = "chat_needs_response"


def _safe_rerun() -> None:
    """Use st.rerun() if available; fallback to deprecated st.experimental_rerun()."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        # Last resort: refresh hint
        st.info("Please manually refresh the app to continue.")


def _backup_self_once() -> None:
    """One-time backup of this module to chat_ui_backup.py in the same directory."""
    try:
        if st.session_state.get(_BACKUP_FLAG_KEY):
            return
        src = __file__
        dst = os.path.join(os.path.dirname(src), "chat_ui_backup.py")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        st.session_state[_BACKUP_FLAG_KEY] = True
    except Exception:
        pass  # Never break UI


def _ensure_chat_state() -> None:
    """Ensure chat container exists and is a list of dicts with role/content."""
    if _CHAT_LIST_KEY not in st.session_state or not isinstance(st.session_state.get(_CHAT_LIST_KEY), list):
        st.session_state[_CHAT_LIST_KEY] = []

    # Coerce any non-dict entries into dict form (best-effort)
    coerced: List[Dict[str, Any]] = []
    for item in list(st.session_state[_CHAT_LIST_KEY]):
        if isinstance(item, dict) and "role" in item and "content" in item:
            coerced.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            role, content = item[0], item[1]
            coerced.append({"role": str(role), "content": str(content)})
    st.session_state[_CHAT_LIST_KEY] = coerced


def _render_messages() -> None:
    """Render chat messages from session state."""
    for msg in st.session_state[_CHAT_LIST_KEY]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def _render_controls() -> None:
    """Render minimal controls beneath the chat history."""
    cols = st.columns(2)
    with cols[0]:
        if st.button("Clear chat", key="chat_ui_clear_btn"):
            st.session_state[_CHAT_LIST_KEY] = []
            st.session_state.pop(_NEW_MSG_KEY, None)
            st.session_state.pop(_NEEDS_RESPONSE_KEY, None)
            _safe_rerun()
    with cols[1]:
        st.write("")  # Placeholder for expansion


def _maybe_render_chat_input() -> None:
    """
    Render a single chat input if this module owns it or if no owner is set.
    Respects the global 'chat_input_owner' lock to avoid duplicates.
    """
    owner = st.session_state.get(_CHAT_OWNER_KEY)
    can_own = owner in (None, "chat_ui")

    if can_own:
        st.session_state[_CHAT_OWNER_KEY] = "chat_ui"

        user_text = st.chat_input("Type a message", key="chat_ui_input")
        if user_text:
            st.session_state[_CHAT_LIST_KEY].append(
                {
                    "role": "user",
                    "content": user_text,
                    "ts": datetime.utcnow().isoformat() + "Z",
                }
            )
            st.session_state[_NEW_MSG_KEY] = user_text
            st.session_state[_NEEDS_RESPONSE_KEY] = True
            _safe_rerun()


def render(*args: Any, **kwargs: Any) -> None:
    """Entry point called by the main app."""
    _backup_self_once()
    _ensure_chat_state()
    _render_messages()
    _render_controls()
    _maybe_render_chat_input()
