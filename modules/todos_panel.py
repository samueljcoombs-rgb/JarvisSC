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
    Returns (todos_colB, health_colD, gym_colF), each list includes non-empty
    strings starting from ROW 1.
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
            return [v.strip() for v in ws.col_values(n) if isinstance(v, str) and v.strip()]
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
    except Exception:
        pass

# ---------------- Styling ----------------
def _inject_css_once():
    if st.session_state.get("_todo_css_loaded_v2"):
        return
    st.session_state["_todo_css_loaded_v2"] = True
    st.markdown(
        """
<style>
/* Titles + labels – tuned for dark theme */
.panel-title {
  display:flex; align-items:center; gap:.5rem;
  font-weight:900; letter-spacing:.2px;
  color: rgba(255,255,255,0.94); font-size:1.05rem; margin: 0 0 .3rem 0;
}
.section-title {
  font-weight:800; letter-spacing:.2px;
  color: rgba(255,255,255,0.9); font-size:1rem; margin:.25rem 0 .35rem 0;
}
.light-caption { color: rgba(255,255,255,0.6); }

/* Make all labels readable on dark theme */
.stCheckbox label, .stNumberInput label, .gym-label,
[data-testid="stMarkdownContainer"] label {
  color: rgba(255,255,255,0.9) !important;
  font-weight: 600 !important;
}

/* Health chips */
.goal-chips { display:flex; flex-wrap:wrap; gap:8px; margin-top: 2px; }
.goal-chip {
  background: linear-gradient(180deg, #111827, #0b1220);
  border: 1px solid rgba(148,163,184,0.25);
  color: rgba(255,255,255,0.92);
  font-weight: 800; font-size: 0.9rem;
  padding: 6px 11px; border-radius: 999px;
  box-shadow: 0 8px 22px rgba(2,6,23,0.35);
}

/* Subtle divider */
.subtle-div {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(148,163,184,0.25), transparent);
  margin: .35rem 0 .35rem 0;
}
</style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- UI ----------------
def render(
    show_header: bool = True,
    show_tasks_title: bool = False,   # avoid duplicate "To-Do"
    show_gym_title: bool = True,
    show_health_title: bool = True,
):
    _inject_css_once()

    sheet_url = _get_sheet_url()
    todos, health, gym = _fetch_columns(sheet_url)

    with st.container(border=True):
        if show_header:
            st.markdown('<div class="panel-title">✅ Today — Tasks, Gym & Health</div>', unsafe_allow_html=True)

        # --- Tasks (Column B) ---
        if show_tasks_title:
            st.markdown('<div class="section-title">To-Do</div>', unsafe_allow_html=True)
        if not todos:
            st.caption("No tasks found.", help=None)
        else:
            for i, item in enumerate(todos, start=1):
                key = f"_todo_local_{i}_{hash(item)}"
                default = st.session_state.get(key, False)
                st.checkbox(item, key=key, value=default)

        st.markdown('<div class="subtle-div"></div>', unsafe_allow_html=True)

        # --- Gym Routine (Column F) ---
        if show_gym_title:
            st.markdown('<div class="section-title">🏋️ Gym Routine</div>', unsafe_allow_html=True)

        has_lifts = False
        run_keys: List[Tuple[str, str]] = []

        if not gym:
            st.caption("No gym items found.", help=None)
        else:
            for idx, name in enumerate(gym, start=1):
                label = (name or "").strip()
                if label.lower() == "run via runna":
                    key_run = f"_gym_run_{idx}"
                    run_keys.append((label, key_run))
                    st.checkbox("Run via Runna", key=key_run, value=st.session_state.get(key_run, False))
                else:
                    has_lifts = True
                    c1, c2, c3 = st.columns([2.0, 1.0, 1.0], gap="small")
                    with c1:
                        st.markdown(f"<div class='gym-label'>{label}</div>", unsafe_allow_html=True)
                    with c2:
                        st.number_input(
                            "Weight (kg)", min_value=0.0, step=0.5,
                            key=f"_gym_w_{idx}", label_visibility="collapsed"
                        )
                    with c3:
                        st.number_input(
                            "Reps", min_value=0, step=1,
                            key=f"_gym_r_{idx}", label_visibility="collapsed"
                        )

        # Only show Save if there is at least one lift row
        if has_lifts:
            if st.button("Save today’s workout"):
                today = datetime.now(TZ).date().isoformat()
                payload = {"date": today, "entries": []}

                # run completion flags
                for _, key_run in run_keys:
                    if st.session_state.get(key_run, False):
                        payload["entries"].append({"name": "Run via Runna", "type": "run", "completed": True})

                # gather lifts
                for k, v in list(st.session_state.items()):
                    if k.startswith("_gym_w_"):
                        idx = k.split("_")[-1]
                        w = float(v or 0.0)
                        r = int(st.session_state.get(f"_gym_r_{idx}", 0) or 0)
                        try:
                            name = gym[int(idx) - 1]
                        except Exception:
                            name = "Exercise"
                        if w > 0 and r > 0 and name.strip().lower() != "run via runna":
                            payload["entries"].append({"name": name, "type": "lift", "weight": w, "reps": r})

                if payload["entries"]:
                    logs = _load_logs()
                    logs.append(payload)
                    _save_logs(logs)
                    st.success("Workout saved.")
                else:
                    st.info("Nothing to save yet — add weight & reps for at least one lift.")

        st.markdown('<div class="subtle-div"></div>', unsafe_allow_html=True)

        # --- Health Goals (Column D) ---
        if show_health_title:
            st.markdown('<div class="section-title">Health Goals</div>', unsafe_allow_html=True)
        if not health:
            st.caption("No health goals found.", help=None)
        else:
            chips = "".join(f'<span class="goal-chip">{h}</span>' for h in health)
            st.markdown(f'<div class="goal-chips">{chips}</div>', unsafe_allow_html=True)
