# modules/todos_panel.py
from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

# gspread + service account
import gspread
from google.oauth2.service_account import Credentials


# ---------- CONFIG ----------
SHEET_URL = "https://docs.google.com/spreadsheets/d/1ALDZIzwaYe9mQH1q068L5jNIWe7o_BFer1UanXKmdCI/edit?gid=1475116413"

# Columns to read (1-indexed)
COL_TODO = 2         # B
COL_HEALTH = 4       # D
COL_GYM = 6          # F

# Local file for workout history (so Jarvis can analyze later if asked)
_HISTORY_FILE = Path(__file__).resolve().parents[1] / "workout_history.json"


# ---------- CSS (glassy, high-contrast, compact like your podcast/weather panels) ----------
def _inject_css_once():
    if st.session_state.get("_todo_css_loaded"):
        return
    st.session_state["_todo_css_loaded"] = True
    st.markdown(
        """
<style>
.todo-panel {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.08);
  margin-bottom: 10px;
}
.todo-title {
  font-weight: 900;
  font-size: 1.05rem;
  color: #0f172a;  /* slate-900 for clear contrast */
  letter-spacing: .2px;
  margin: 0 0 8px 0;
  display: flex; align-items: center; gap: 8px;
}
.todo-list { margin: 0; padding: 0; }
.todo-item { list-style: none; margin: 2px 0; }
.todo-note { font-size: 0.85rem; color: #475569; margin-top: 6px; }

.mini-col-label{
  font-size: 0.75rem;
  color: rgba(15,23,42,0.75); /* readable label */
  font-weight: 700;
  letter-spacing: .2px;
  margin: 0 0 2px 0;
  text-transform: uppercase;
}

/* Inline rows for gym inputs */
.inline-row {
  display: grid;
  grid-template-columns: 1fr 90px 90px;
  gap: 10px;
  align-items: center;
  margin: 4px 0;
}
.inline-title {
  color: #0f172a;
  font-weight: 700;
}
.save-wrap {
  display: flex; justify-content: flex-end; margin-top: 8px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Google Sheets helpers ----------
@st.cache_data(show_spinner=False, ttl=60)
def _fetch_column_values(sheet_url: str, col: int) -> List[str]:
    """Return non-empty values from a column (starting row 1)."""
    try:
        raw_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not raw_json:
            return []
        info = json.loads(raw_json)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)

        # Prefer worksheet by gid if present in URL; else default to first sheet.
        m = re.search(r"gid=(\d+)", sheet_url)
        ws = None
        if m:
            gid = int(m.group(1))
            try:
                ws = sh.get_worksheet_by_id(gid)  # type: ignore[attr-defined]
            except Exception:
                ws = None
        if ws is None:
            ws = sh.sheet1

        values = ws.col_values(col)  # from row 1
        clean = [v.strip() for v in values if v and str(v).strip()]
        return clean
    except Exception:
        return []


def _load_lists() -> Tuple[List[str], List[str], List[str]]:
    todos = _fetch_column_values(SHEET_URL, COL_TODO)
    health = _fetch_column_values(SHEET_URL, COL_HEALTH)
    gym = _fetch_column_values(SHEET_URL, COL_GYM)
    return todos, health, gym


# ---------- Local history helpers ----------
def _load_history() -> dict:
    try:
        if _HISTORY_FILE.exists():
            return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_history(data: dict) -> None:
    try:
        _HISTORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------- UI blocks ----------
def _panel_title(title: str, icon: str = ""):
    st.markdown(
        f"""<div class="todo-title">{icon}{title}</div>""",
        unsafe_allow_html=True,
    )

def _checkbox_list(items: List[str], prefix_key: str) -> None:
    """Shared renderer for To-Do and Health — simple, tight checkbox list."""
    if not items:
        st.write("Nothing here right now.")
        return

    for i, label in enumerate(items):
        key = f"{prefix_key}_{i}"
        # Keep checkbox state in session only; never reflect to Sheets
        _ = st.checkbox(label, key=key)


def _gym_block(gym_lines: List[str]) -> None:
    if not gym_lines:
        st.write("No gym items found.")
        return

    # If the sole line is exactly "Run via Runna" (case-insensitive), show a single checkbox
    single = [g for g in gym_lines if g]
    if len(single) == 1 and single[0].strip().lower() == "run via runna":
        _ = st.checkbox("Run via Runna", key="gym_runna_only")
        # No Save button in this mode
        return

    # Else: per-exercise line with inline KG / Reps inputs
    # Build a minimal inline grid row for each exercise
    # Keep user inputs in session (so it doesn't jitter when interacting elsewhere)
    for idx, line in enumerate(gym_lines):
        ex_name = line.strip()
        if not ex_name:
            continue

        # Inline row
        st.markdown(
            f"""
<div class="inline-row">
  <div class="inline-title">{ex_name}</div>
  <div>
    <div class="mini-col-label">KG</div>
  </div>
  <div>
    <div class="mini-col-label">Reps</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([6, 2, 2])
        with c1:
            # placeholder to align (already rendered title)
            st.write("")
        with c2:
            st.number_input(
                "Weight (kg)",
                min_value=0.0,
                step=0.5,
                key=f"_gym_w_{idx}",
                label_visibility="collapsed",
            )
        with c3:
            st.number_input(
                "Reps",
                min_value=0,
                step=1,
                key=f"_gym_r_{idx}",
                label_visibility="collapsed",
            )

    # Show Save button (only in this branch)
    with st.container():
        st.markdown('<div class="save-wrap">', unsafe_allow_html=True)
        if st.button("Save Today’s Workout", use_container_width=False):
            today = datetime.now().strftime("%Y-%m-%d")
            entry = []
            for idx, line in enumerate(gym_lines):
                name = line.strip()
                if not name:
                    continue
                w = st.session_state.get(f"_gym_w_{idx}", 0.0)
                r = st.session_state.get(f"_gym_r_{idx}", 0)
                entry.append({"exercise": name, "kg": w, "reps": r})

            hist = _load_history()
            day_list = hist.get(today, [])
            day_list.append({"saved_at": datetime.now().isoformat(), "items": entry})
            hist[today] = day_list
            _save_history(hist)
            st.success("Saved.")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Public render ----------
def render():
    _inject_css_once()

    todos, health, gym = _load_lists()

    # To-Do
    with st.container():
        st.markdown('<div class="todo-panel">', unsafe_allow_html=True)
        _panel_title("To-Do", "📝")
        _checkbox_list(todos, "todo_item")
        st.markdown("</div>", unsafe_allow_html=True)

    # Gym
    with st.container():
        st.markdown('<div class="todo-panel">', unsafe_allow_html=True)
        _panel_title("Gym Routine", "🏋️")
        _gym_block(gym)
        st.markdown("</div>", unsafe_allow_html=True)

    # Health
    with st.container():
        st.markdown('<div class="todo-panel">', unsafe_allow_html=True)
        _panel_title("Health Goals", "💚")
        _checkbox_list(health, "health_item")
        st.markdown("</div>", unsafe_allow_html=True)
