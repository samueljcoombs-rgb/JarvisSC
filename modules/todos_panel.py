# modules/todos_panel.py
from __future__ import annotations
import os
import json
from urllib.parse import urlparse, parse_qs
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

# --------- Config ---------
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ALDZIzwaYe9mQH1q068L5jNIWe7o_BFer1UanXKmdCI/edit?gid=1475116413#gid=1475116413"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
TZ = ZoneInfo("Europe/London")

# Local persistent log file (so Jarvis can analyse later)
# Saved in the app root as workout_logs.json
_STORAGE_PATH = (Path(__file__).resolve().parent.parent / "workout_logs.json")

# ---------------- Secrets / Sheet helpers ----------------

def _get_sheet_url() -> str:
    return (
        st.secrets.get("TODO_SHEET_URL")
        or os.getenv("TODO_SHEET_URL")
        or DEFAULT_SHEET_URL
    )

def _creds() -> Optional[Credentials]:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return Credentials.from_service_account_info(data, scopes=SCOPES)
    except Exception:
        if isinstance(raw, dict):
            try:
                return Credentials.from_service_account_info(raw, scopes=SCOPES)
            except Exception:
                return None
        return None

def _parse_gid(sheet_url: str) -> Optional[int]:
    try:
        q = parse_qs(urlparse(sheet_url).query)
        gid = q.get("gid", [None])[0]
        return int(gid) if gid is not None else None
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_columns(sheet_url: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (todos_colB, health_colD, gym_colF), each a list of non-empty strings
    starting from row 2.
    """
    creds = _creds()
    if not creds:
        return [], [], []

    gc = gspread.authorize(creds)
    sh = gc.open_by_url(sheet_url)

    ws = None
    gid = _parse_gid(sheet_url)
    if gid is not None:
        try:
            ws = sh.get_worksheet_by_id(gid)
        except Exception:
            ws = None
    if ws is None:
        ws = sh.sheet1

    def _col(n: int) -> List[str]:
        try:
            return [v.strip() for v in ws.col_values(n)[1:] if isinstance(v, str) and v.strip()]
        except Exception:
            return []

    col_b = _col(2)  # To-Do
    col_d = _col(4)  # Health goals
    col_f = _col(6)  # Gym routine
    return col_b, col_d, col_f

# ---------------- Local log helpers ----------------

def _load_logs() -> List[Dict[str, Any]]:
    if _STORAGE_PATH.exists():
        try:
            return json.loads(_STORAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_logs(logs: List[Dict[str, Any]]) -> None:
    try:
        _STORAGE_PATH.write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        st.warning(f"Could not save workout logs: {e}")

# ---------------- Styling ----------------

def _inject_css_once():
    if st.session_state.get("_todo_css_loaded"):
        return
    st.session_state["_todo_css_loaded"] = True
    st.markdown(
        """
<style>
/* Card wrapper */
.todo-card {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 8px 22px rgba(0,0,0,0.08);
  margin-bottom: 10px;
}

/* Section headers */
.todo-title {
  font-weight: 800;
  letter-spacing: .2px;
  margin: 0 0 6px 0;
  color: #0f172a;
}

/* Health goal "chips" */
.goal-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 6px;
}
.goal-chip {
  background: linear-gradient(180deg, #f8fafc, #eef2ff);
  border: 1px solid rgba(99,102,241,0.25);
  color: #111827;
  font-weight: 700;
  font-size: 0.88rem;
  padding: 6px 10px;
  border-radius: 999px;
  box-shadow: 0 4px 12px rgba(99,102,241,0.12);
}

/* Subtle divider */
.subtle-div {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,0,0,0.08), transparent);
  margin: 8px 0 10px 0;
}

/* Gym row layout */
.gym-row {
  display: grid;
  grid-template-columns: 1fr 120px 120px;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
}
.gym-label {
  font-weight: 700;
}
.gym-note {
  font-size: 0.85rem;
  color: #6b7280;
  margin-top: 4px;
}
</style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- UI ----------------

def render(show_header: bool = True):
    _inject_css_once()

    if show_header:
        st.subheader("📝 To-Do")

    sheet_url = _get_sheet_url()
    todos, health, gym = _fetch_columns(sheet_url)

    with st.container():
        st.markdown('<div class="todo-card">', unsafe_allow_html=True)

        # --- Tasks ---
        st.markdown('<div class="todo-title">Tasks</div>', unsafe_allow_html=True)
        if not todos:
            st.caption("No tasks found (Column B is empty).")
        else:
            for i, item in enumerate(todos, start=1):
                key = f"_todo_local_{i}_{hash(item)}"
                default = st.session_state.get(key, False)
                st.checkbox(item, key=key, value=default)

        st.markdown('<div class="subtle-div"></div>', unsafe_allow_html=True)

        # --- Gym Routine (Column F) ---
        st.markdown('<div class="todo-title">🏋️ Gym Routine</div>', unsafe_allow_html=True)

        if not gym:
            st.caption("No gym items found (Column F is empty).")
        else:
            # Build dynamic inputs
            # Use deterministic keys so values persist if you switch tabs/re-run
            run_keys = []
            lift_keys = []  # (name, weight_key, reps_key)

            for idx, name in enumerate(gym, start=1):
                norm = name.strip().lower()
                # "Run via Runna" special case (checkbox only)
                if norm == "run via runna":
                    key_run = f"_gym_run_{idx}"
                    run_keys.append((name, key_run))
                    checked = st.session_state.get(key_run, False)
                    st.checkbox("Run via Runna", key=key_run, value=checked)
                else:
                    # Weight + Reps inputs
                    weight_key = f"_gym_w_{idx}"
                    reps_key   = f"_gym_r_{idx}"
                    col1, col2, col3 = st.columns([3, 1.2, 1.2])
                    with col1:
                        st.markdown(f'<div class="gym-label">{name}</div>', unsafe_allow_html=True)
                    with col2:
                        st.number_input("Weight (kg)", min_value=0.0, step=0.5, key=weight_key, label_visibility="collapsed")
                    with col3:
                        st.number_input("Reps", min_value=0, step=1, key=reps_key, label_visibility="collapsed")
                    lift_keys.append((name, weight_key, reps_key))

            # Save button
            if st.button("Save today’s workout"):
                today = datetime.now(TZ).date().isoformat()
                payload = {
                    "date": today,
                    "entries": []
                }

                # Collect runs (store only if checked)
                for name, key_run in run_keys:
                    if st.session_state.get(key_run, False):
                        payload["entries"].append({
                            "name": "Run via Runna",
                            "type": "run",
                            "completed": True
                        })

                # Collect lifts (store only if both values meaningful)
                for name, w_key, r_key in lift_keys:
                    w = st.session_state.get(w_key, 0.0)
                    r = st.session_state.get(r_key, 0)
                    if (isinstance(w, (int, float)) and w > 0) and (isinstance(r, int) and r > 0):
                        payload["entries"].append({
                            "name": name,
                            "type": "lift",
                            "weight": float(w),
                            "reps": int(r)
                        })

                if payload["entries"]:
                    logs = _load_logs()
                    logs.append(payload)
                    _save_logs(logs)
                    st.success("Workout saved. Jarvis can analyse it later.")
                else:
                    st.info("Nothing to save yet — tick your run or add weight & reps.")

        st.markdown('<div class="subtle-div"></div>', unsafe_allow_html=True)

        # --- Health Goals (Column D) ---
        st.markdown('<div class="todo-title">Health Goals</div>', unsafe_allow_html=True)
        if not health:
            st.caption("No health goals found (Column D is empty).")
        else:
            chips = "".join(f'<span class="goal-chip">{h}</span>' for h in health)
            st.markdown(f'<div class="goal-chips">{chips}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Optional tiny hint
    st.caption("Your workout history is saved locally to `workout_logs.json`.")
