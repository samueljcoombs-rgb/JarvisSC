# modules/todos_panel.py
from __future__ import annotations
import json
from typing import List, Optional, Tuple

import streamlit as st

# Requires gspread in requirements.txt
try:
    import gspread
except Exception:
    gspread = None

# ---- Your sheet specifics ----
# Reads tasks from column B (starting at B2), writes check state to column C (starting at C2)
COLUMN_TASK = "B"
COLUMN_DONE = "C"
START_ROW = 2

# Default to your provided URL; can be overridden by st.secrets or the UI
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ALDZIzwaYe9mQH1q068L5jNIWe7o_BFer1UanXKmdCI/edit?gid=1475116413#gid=1475116413"

# ----------------- Auth helpers -----------------

def _get_creds_from_secrets() -> Optional[dict]:
    """Read GOOGLE_SERVICE_ACCOUNT_JSON from st.secrets (string or dict)."""
    raw = None
    try:
        raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    except Exception:
        raw = None
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None

def _open_sheet() -> Optional[Tuple[object, object]]:
    """
    Returns (gc, worksheet) or None.
    Uses:
      - GOOGLE_SHEETS_TODO_SHEET_ID (ID or full URL) [optional]
      - GOOGLE_SHEETS_TODO_TAB (worksheet/tab name) [optional]
    Falls back to DEFAULT_SHEET_URL and its active tab.
    """
    if gspread is None:
        st.warning("`gspread` is not installed. Add `gspread` to requirements.txt and restart.")
        return None

    creds = _get_creds_from_secrets()
    if not creds:
        st.info("Add **GOOGLE_SERVICE_ACCOUNT_JSON** to `st.secrets` to enable the To-Do panel.")
        return None

    # From secrets (optional)
    sheet_id_or_url = None
    tab_name = None
    try:
        sheet_id_or_url = st.secrets.get("GOOGLE_SHEETS_TODO_SHEET_ID")
        tab_name = st.secrets.get("GOOGLE_SHEETS_TODO_TAB", None)
    except Exception:
        pass

    # Inline override (pre-filled with your URL)
    with st.expander("Configure Google Sheet (optional)", expanded=False):
        sheet_id_or_url = st.text_input(
            "Google Sheet URL or ID",
            value=sheet_id_or_url or DEFAULT_SHEET_URL,
            help="Paste the full URL or just the Sheet ID.",
            key="todos_sheet_url",
        )
        tab_name = st.text_input(
            "Worksheet (tab) name (leave blank for active tab)",
            value=tab_name or "",
            key="todos_tab_name",
        )

    if not sheet_id_or_url:
        st.info("Provide a Google **Sheet URL/ID** to load tasks.")
        return None

    try:
        gc = gspread.service_account_from_dict(creds)
        # Accept full URL or bare ID
        if sheet_id_or_url.startswith("http"):
            sh = gc.open_by_url(sheet_id_or_url)
        else:
            sh = gc.open_by_key(sheet_id_or_url)
        ws = sh.worksheet(tab_name) if tab_name else sh.sheet1
        return gc, ws
    except Exception as e:
        st.error(f"Could not open Google Sheet/Worksheet: {e}")
        return None

# ----------------- Sheet I/O -----------------

def _col_values(ws, col_letter: str, start_row: int) -> List[str]:
    """
    Read a single column from start_row downward (e.g., B2:B).
    Uses ws.col_values; trims list to start_row offset.
    """
    try:
        col_idx = ord(col_letter.upper()) - ord("A") + 1
        vals = ws.col_values(col_idx)  # full column
        # Pad to avoid IndexError and slice from start_row-1
        if len(vals) < start_row - 1:
            return []
        return vals[start_row - 1 :]  # 1-based -> 0-based
    except Exception:
        return []

def _set_cell(ws, a1: str, value: str) -> None:
    try:
        ws.update(a1, [[value]])
    except Exception as e:
        st.error(f"Failed to update {a1}: {e}")

def _batch_update(ws, updates: List[Tuple[str, str]]) -> None:
    """updates: list of (A1, value)"""
    if not updates:
        return
    try:
        ws.batch_update([{"range": a1, "values": [[val]]} for a1, val in updates])
    except Exception as e:
        st.error(f"Batch update failed: {e}")

# ----------------- UI helpers -----------------

def _truthy(s: str) -> bool:
    s = (s or "").strip().lower()
    return s in {"true", "1", "yes", "y", "x", "checked"}

def _to_sheet_bool(v: bool) -> str:
    return "TRUE" if v else "FALSE"

def _inject_css_once():
    if st.session_state.get("_todos_css_loaded"):
        return
    st.session_state["_todos_css_loaded"] = True
    st.markdown(
        """
<style>
.todo-card {
  background: rgba(255,255,255,0.66);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 14px;
  padding: 10px 12px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.08);
  margin-bottom: 8px;
}
</style>
        """,
        unsafe_allow_html=True,
    )

# ----------------- Render -----------------

def render():
    st.header("📝 To-Do")
    _inject_css_once()

    opened = _open_sheet()
    if not opened:
        return
    _, ws = opened

    # Read tasks and their 'done' flags
    tasks = _col_values(ws, COLUMN_TASK, START_ROW)  # B2:B
    dones = _col_values(ws, COLUMN_DONE, START_ROW)  # C2:C (may be shorter)

    # Build list of (row_number, task_text, done_bool), skipping blank tasks
    items: List[Tuple[int, str, bool]] = []
    max_len = max(len(tasks), len(dones))
    for i in range(max_len):
        task = (tasks[i] if i < len(tasks) else "").strip()
        if not task:
            continue  # skip blanks in column B
        done_str = (dones[i] if i < len(dones) else "").strip()
        done_bool = _truthy(done_str)
        row_num = START_ROW + i  # absolute sheet row
        items.append((row_num, task, done_bool))

    if not items:
        st.write("No tasks in column B (from B2). Add one below!")

    # Keep originals to compute diffs
    orig_map_key = "_todos_orig_map"
    if orig_map_key not in st.session_state:
        st.session_state[orig_map_key] = {}

    # Render tasks as checkboxes
    for row_num, task, done in items:
        key = f"todo_{row_num}"
        if key not in st.session_state:
            st.session_state[key] = done
        if row_num not in st.session_state[orig_map_key]:
            st.session_state[orig_map_key][row_num] = done
        st.checkbox(task, key=key)

    # Actions
    col_add, col_apply, col_refresh = st.columns([2, 1, 1])
    with col_add:
        new_task = st.text_input("Add a task", placeholder="Type and press Enter…")
        if new_task:
            # Find next free row in column B (after the last non-empty task)
            end_row = START_ROW + len(tasks)
            # If trailing blanks exist, we still append at end_row+1
            a1_task = f"{COLUMN_TASK}{end_row + 1}"
            a1_done = f"{COLUMN_DONE}{end_row + 1}"
            _set_cell(ws, a1_task, new_task.strip())
            _set_cell(ws, a1_done, "FALSE")
            st.success("Task added.")
            st.rerun()

    with col_apply:
        if st.button("Apply changes"):
            updates: List[Tuple[str, str]] = []
            for row_num, task, _ in items:
                key = f"todo_{row_num}"
                new_val = bool(st.session_state.get(key, False))
                old_val = bool(st.session_state[orig_map_key].get(row_num, False))
                if new_val != old_val:
                    updates.append((f"{COLUMN_DONE}{row_num}", _to_sheet_bool(new_val)))
            if updates:
                _batch_update(ws, updates)
                for a1, val in updates:
                    # Update original map for rows we changed
                    try:
                        row_num = int(a1[len(COLUMN_DONE):])
                        st.session_state[orig_map_key][row_num] = (val == "TRUE")
                    except Exception:
                        pass
                st.success(f"Updated {len(updates)} task(s).")
                st.rerun()
            else:
                st.info("No changes to apply.")

    with col_refresh:
        if st.button("Refresh"):
            st.rerun()
