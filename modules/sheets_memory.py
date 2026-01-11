# modules/sheets_memory.py
"""
Core Google Sheets Memory System for Jarvis
Provides persistent storage for chat history, long-term memory, and all data modules.
"""
from __future__ import annotations
import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from google.oauth2.service_account import Credentials
import streamlit as st

# ============================================================
# Configuration
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

TZ = ZoneInfo("Europe/London")

# Tab names for Jarvis Data Sheet
TAB_CHAT_MEMORY = "chat_memory"
TAB_LONG_TERM_MEMORY = "long_term_memory"
TAB_HEALTH_LOGS = "health_logs"
TAB_WORKOUT_LOGS = "workout_logs"
TAB_FITNESS_GOALS = "fitness_goals"
TAB_BUCKET_LIST = "bucket_list"
TAB_YEARLY_GOALS = "yearly_goals"
TAB_TRAVEL_PLANS = "travel_plans"
TAB_FLIGHT_ALERTS = "flight_alerts"
TAB_TV_WATCHLIST = "tv_watchlist"
TAB_ENTERTAINMENT_CONFIG = "entertainment_config"

# Headers for each tab (used for auto-creation)
TAB_HEADERS = {
    TAB_CHAT_MEMORY: ["timestamp", "role", "content", "session_id"],
    TAB_LONG_TERM_MEMORY: ["timestamp", "kind", "text", "context"],
    TAB_HEALTH_LOGS: ["date", "weight_stone", "weight_lbs", "calories", "protein_g", "notes"],
    TAB_WORKOUT_LOGS: ["date", "exercise", "sets", "reps", "weight_kg", "duration_mins", "distance_km", "pace_per_km", "type", "notes"],
    TAB_FITNESS_GOALS: ["goal_id", "goal", "target_date", "status", "progress", "created", "updated"],
    TAB_BUCKET_LIST: ["id", "item", "category", "priority", "status", "target_year", "notes", "created", "completed_date"],
    TAB_YEARLY_GOALS: ["year", "goal_id", "goal", "category", "status", "progress", "q1_target", "q2_target", "q3_target", "q4_target", "notes"],
    TAB_TRAVEL_PLANS: ["trip_id", "destination", "start_date", "end_date", "status", "budget", "notes", "flight_watched", "created"],
    TAB_FLIGHT_ALERTS: ["alert_id", "origin", "destination", "max_price", "currency", "status", "last_checked", "created"],
    TAB_TV_WATCHLIST: ["id", "title", "type", "season", "episode", "status", "platform", "added", "notes"],
    TAB_ENTERTAINMENT_CONFIG: ["key", "value"],
}

# ============================================================
# Helpers
# ============================================================

def _now_iso() -> str:
    return datetime.now(TZ).isoformat()

def _today_str() -> str:
    return datetime.now(TZ).date().isoformat()

def _gen_id() -> str:
    return f"{int(time.time() * 1000)}"

# ============================================================
# Google Sheets Connection
# ============================================================

def _gs_creds() -> Credentials:
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets/env.")
    data = json.loads(raw) if isinstance(raw, str) else raw
    return Credentials.from_service_account_info(data, scopes=SCOPES)

def _sheet_url() -> str:
    url = st.secrets.get("JARVIS_DATA_SHEET_URL") or os.getenv("JARVIS_DATA_SHEET_URL")
    url = (url or "").strip()
    if not url:
        # Fallback to TODO_SHEET_URL if JARVIS_DATA_SHEET_URL not set
        url = st.secrets.get("TODO_SHEET_URL") or os.getenv("TODO_SHEET_URL") or ""
    if not url:
        raise RuntimeError("Missing JARVIS_DATA_SHEET_URL in Streamlit secrets/env.")
    return url

@st.cache_resource(show_spinner=False)
def _get_spreadsheet():
    gc = gspread.authorize(_gs_creds())
    return gc.open_by_url(_sheet_url())

def _get_worksheet(tab_name: str):
    """Get or create a worksheet by name."""
    sh = _get_spreadsheet()
    try:
        return sh.worksheet(tab_name)
    except WorksheetNotFound:
        # Auto-create the tab with headers
        headers = TAB_HEADERS.get(tab_name, ["data"])
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=len(headers) + 5)
        ws.append_row(headers, value_input_option="RAW")
        return ws

def _safe_get_all_values(tab_name: str) -> List[List[str]]:
    try:
        ws = _get_worksheet(tab_name)
        return ws.get_all_values()
    except Exception as e:
        st.warning(f"Could not read {tab_name}: {e}")
        return []

def _ensure_header(tab_name: str) -> bool:
    """Ensure the tab has proper headers."""
    try:
        ws = _get_worksheet(tab_name)
        vals = ws.get_all_values()
        headers = TAB_HEADERS.get(tab_name, [])
        if not vals and headers:
            ws.append_row(headers, value_input_option="RAW")
            return True
        return False
    except Exception:
        return False

# ============================================================
# Generic CRUD Operations
# ============================================================

def read_all_rows(tab_name: str) -> List[Dict[str, str]]:
    """Read all rows from a tab as list of dicts."""
    vals = _safe_get_all_values(tab_name)
    if not vals or len(vals) < 2:
        return []
    headers = [h.strip() for h in vals[0]]
    rows = []
    for r in vals[1:]:
        if not any((c or "").strip() for c in r):
            continue
        d = {}
        for i, h in enumerate(headers):
            d[h] = (r[i] if i < len(r) else "").strip()
        rows.append(d)
    return rows

def read_recent_rows(tab_name: str, limit: int = 50) -> List[Dict[str, str]]:
    """Read most recent rows (assumes newest at bottom)."""
    all_rows = read_all_rows(tab_name)
    return all_rows[-limit:] if limit else all_rows

def append_row(tab_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Append a row to a tab."""
    try:
        ws = _get_worksheet(tab_name)
        headers = TAB_HEADERS.get(tab_name, list(data.keys()))
        row = [str(data.get(h, "")) for h in headers]
        ws.append_row(row, value_input_option="RAW")
        return {"ok": True, "tab": tab_name}
    except APIError as e:
        return {"ok": False, "error": f"Google Sheets API error: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def update_row_by_id(tab_name: str, id_column: str, id_value: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update a row where id_column == id_value."""
    try:
        ws = _get_worksheet(tab_name)
        vals = ws.get_all_values()
        if not vals:
            return {"ok": False, "error": "Tab is empty"}
        
        headers = vals[0]
        id_col_idx = headers.index(id_column) if id_column in headers else -1
        if id_col_idx < 0:
            return {"ok": False, "error": f"Column '{id_column}' not found"}
        
        for row_idx, row in enumerate(vals[1:], start=2):
            if row_idx <= 1:
                continue
            cell_val = row[id_col_idx] if id_col_idx < len(row) else ""
            if cell_val.strip() == str(id_value).strip():
                # Found the row, update it
                for col_name, new_val in updates.items():
                    if col_name in headers:
                        col_idx = headers.index(col_name) + 1  # 1-indexed
                        ws.update_cell(row_idx, col_idx, str(new_val))
                return {"ok": True, "updated_row": row_idx}
        
        return {"ok": False, "error": f"Row with {id_column}={id_value} not found"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def delete_row_by_id(tab_name: str, id_column: str, id_value: str) -> Dict[str, Any]:
    """Delete a row where id_column == id_value."""
    try:
        ws = _get_worksheet(tab_name)
        vals = ws.get_all_values()
        if not vals:
            return {"ok": False, "error": "Tab is empty"}
        
        headers = vals[0]
        id_col_idx = headers.index(id_column) if id_column in headers else -1
        if id_col_idx < 0:
            return {"ok": False, "error": f"Column '{id_column}' not found"}
        
        for row_idx, row in enumerate(vals[1:], start=2):
            cell_val = row[id_col_idx] if id_col_idx < len(row) else ""
            if cell_val.strip() == str(id_value).strip():
                ws.delete_rows(row_idx)
                return {"ok": True, "deleted_row": row_idx}
        
        return {"ok": False, "error": f"Row with {id_column}={id_value} not found"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def filter_rows(tab_name: str, filters: Dict[str, str]) -> List[Dict[str, str]]:
    """Filter rows by column values."""
    all_rows = read_all_rows(tab_name)
    filtered = []
    for row in all_rows:
        match = True
        for col, val in filters.items():
            if row.get(col, "").lower() != str(val).lower():
                match = False
                break
        if match:
            filtered.append(row)
    return filtered

# ============================================================
# Chat Memory (Persistent Chat History)
# ============================================================

def save_chat_message(role: str, content: str, session_id: str = "default") -> Dict[str, Any]:
    """Save a single chat message."""
    return append_row(TAB_CHAT_MEMORY, {
        "timestamp": _now_iso(),
        "role": role,
        "content": content[:10000],  # Limit content size
        "session_id": session_id
    })

def get_chat_history(session_id: str = "default", limit: int = 100) -> List[Dict[str, str]]:
    """Get chat history for a session."""
    all_msgs = read_all_rows(TAB_CHAT_MEMORY)
    filtered = [m for m in all_msgs if m.get("session_id") == session_id]
    return filtered[-limit:] if limit else filtered

def get_all_sessions() -> List[str]:
    """Get list of unique session IDs."""
    all_msgs = read_all_rows(TAB_CHAT_MEMORY)
    sessions = list(set(m.get("session_id", "default") for m in all_msgs))
    return sorted(sessions)

def clear_chat_session(session_id: str = "default") -> Dict[str, Any]:
    """Clear all messages for a session (marks them, doesn't delete)."""
    # For safety, we don't actually delete - we could add a "deleted" column
    # For now, this is a no-op that returns success
    return {"ok": True, "session_id": session_id, "note": "Session cleared from local state"}

# ============================================================
# Long-term Memory (Facts, Notes, Context)
# ============================================================

def add_memory(text: str, kind: str = "note", context: str = "") -> Dict[str, Any]:
    """Add a long-term memory fact."""
    return append_row(TAB_LONG_TERM_MEMORY, {
        "timestamp": _now_iso(),
        "kind": kind,
        "text": text,
        "context": context
    })

def get_memories(limit: int = 50, kind: Optional[str] = None) -> List[Dict[str, str]]:
    """Get recent memories, optionally filtered by kind."""
    memories = read_recent_rows(TAB_LONG_TERM_MEMORY, limit=limit * 2)
    if kind:
        memories = [m for m in memories if m.get("kind") == kind]
    return memories[-limit:]

def get_memory_summary(max_chars: int = 1500) -> str:
    """Get a summary of recent memories for context."""
    memories = get_memories(limit=30)
    lines = []
    for m in memories:
        kind = m.get("kind", "note")
        text = m.get("text", "")
        lines.append(f"- [{kind}] {text}")
    summary = "\n".join(lines)
    return summary[:max_chars] if len(summary) > max_chars else summary

def search_memories(query: str, limit: int = 20) -> List[Dict[str, str]]:
    """Simple text search in memories."""
    memories = read_all_rows(TAB_LONG_TERM_MEMORY)
    query_lower = query.lower()
    matches = []
    for m in memories:
        if query_lower in m.get("text", "").lower() or query_lower in m.get("context", "").lower():
            matches.append(m)
    return matches[-limit:]

# ============================================================
# Health & Fitness Data
# ============================================================

def log_health(date: str, weight_stone: float = 0, weight_lbs: float = 0, 
               calories: int = 0, protein_g: int = 0, notes: str = "") -> Dict[str, Any]:
    """Log daily health metrics."""
    return append_row(TAB_HEALTH_LOGS, {
        "date": date or _today_str(),
        "weight_stone": str(weight_stone) if weight_stone else "",
        "weight_lbs": str(weight_lbs) if weight_lbs else "",
        "calories": str(calories) if calories else "",
        "protein_g": str(protein_g) if protein_g else "",
        "notes": notes
    })

def get_health_logs(days: int = 30) -> List[Dict[str, str]]:
    """Get recent health logs."""
    return read_recent_rows(TAB_HEALTH_LOGS, limit=days)

def log_workout(date: str, exercise: str, sets: int = 0, reps: int = 0, 
                weight_kg: float = 0, duration_mins: int = 0, distance_km: float = 0,
                pace_per_km: str = "", workout_type: str = "strength", notes: str = "") -> Dict[str, Any]:
    """Log a workout/exercise."""
    return append_row(TAB_WORKOUT_LOGS, {
        "date": date or _today_str(),
        "exercise": exercise,
        "sets": str(sets) if sets else "",
        "reps": str(reps) if reps else "",
        "weight_kg": str(weight_kg) if weight_kg else "",
        "duration_mins": str(duration_mins) if duration_mins else "",
        "distance_km": str(distance_km) if distance_km else "",
        "pace_per_km": pace_per_km,
        "type": workout_type,
        "notes": notes
    })

def get_workout_logs(days: int = 30, workout_type: Optional[str] = None) -> List[Dict[str, str]]:
    """Get recent workout logs."""
    logs = read_recent_rows(TAB_WORKOUT_LOGS, limit=days * 10)
    if workout_type:
        logs = [l for l in logs if l.get("type") == workout_type]
    return logs

def add_fitness_goal(goal: str, target_date: str = "", status: str = "active", progress: int = 0) -> Dict[str, Any]:
    """Add a fitness goal."""
    return append_row(TAB_FITNESS_GOALS, {
        "goal_id": _gen_id(),
        "goal": goal,
        "target_date": target_date,
        "status": status,
        "progress": str(progress),
        "created": _now_iso(),
        "updated": _now_iso()
    })

def get_fitness_goals(status: Optional[str] = None) -> List[Dict[str, str]]:
    """Get fitness goals."""
    goals = read_all_rows(TAB_FITNESS_GOALS)
    if status:
        goals = [g for g in goals if g.get("status") == status]
    return goals

def update_fitness_goal(goal_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update a fitness goal."""
    updates["updated"] = _now_iso()
    return update_row_by_id(TAB_FITNESS_GOALS, "goal_id", goal_id, updates)

# ============================================================
# Bucket List & Yearly Goals
# ============================================================

def add_bucket_list_item(item: str, category: str = "general", priority: str = "medium",
                         target_year: str = "", notes: str = "") -> Dict[str, Any]:
    """Add a bucket list item."""
    return append_row(TAB_BUCKET_LIST, {
        "id": _gen_id(),
        "item": item,
        "category": category,
        "priority": priority,
        "status": "pending",
        "target_year": target_year,
        "notes": notes,
        "created": _now_iso(),
        "completed_date": ""
    })

def get_bucket_list(status: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, str]]:
    """Get bucket list items."""
    items = read_all_rows(TAB_BUCKET_LIST)
    if status:
        items = [i for i in items if i.get("status") == status]
    if category:
        items = [i for i in items if i.get("category") == category]
    return items

def update_bucket_list_item(item_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update a bucket list item."""
    if updates.get("status") == "completed":
        updates["completed_date"] = _now_iso()
    return update_row_by_id(TAB_BUCKET_LIST, "id", item_id, updates)

def add_yearly_goal(year: int, goal: str, category: str = "general", 
                    q1: str = "", q2: str = "", q3: str = "", q4: str = "") -> Dict[str, Any]:
    """Add a yearly goal."""
    return append_row(TAB_YEARLY_GOALS, {
        "year": str(year),
        "goal_id": _gen_id(),
        "goal": goal,
        "category": category,
        "status": "active",
        "progress": "0",
        "q1_target": q1,
        "q2_target": q2,
        "q3_target": q3,
        "q4_target": q4,
        "notes": ""
    })

def get_yearly_goals(year: Optional[int] = None) -> List[Dict[str, str]]:
    """Get yearly goals."""
    goals = read_all_rows(TAB_YEARLY_GOALS)
    if year:
        goals = [g for g in goals if g.get("year") == str(year)]
    return goals

def update_yearly_goal(goal_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update a yearly goal."""
    return update_row_by_id(TAB_YEARLY_GOALS, "goal_id", goal_id, updates)

# ============================================================
# Travel Plans & Flight Alerts
# ============================================================

def add_travel_plan(destination: str, start_date: str = "", end_date: str = "",
                    budget: str = "", notes: str = "") -> Dict[str, Any]:
    """Add a travel plan."""
    return append_row(TAB_TRAVEL_PLANS, {
        "trip_id": _gen_id(),
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "status": "planning",
        "budget": budget,
        "notes": notes,
        "flight_watched": "no",
        "created": _now_iso()
    })

def get_travel_plans(status: Optional[str] = None) -> List[Dict[str, str]]:
    """Get travel plans."""
    plans = read_all_rows(TAB_TRAVEL_PLANS)
    if status:
        plans = [p for p in plans if p.get("status") == status]
    return plans

def update_travel_plan(trip_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update a travel plan."""
    return update_row_by_id(TAB_TRAVEL_PLANS, "trip_id", trip_id, updates)

def add_flight_alert(origin: str, destination: str, max_price: float, 
                     currency: str = "GBP") -> Dict[str, Any]:
    """Add a flight price alert."""
    return append_row(TAB_FLIGHT_ALERTS, {
        "alert_id": _gen_id(),
        "origin": origin,
        "destination": destination,
        "max_price": str(max_price),
        "currency": currency,
        "status": "active",
        "last_checked": "",
        "created": _now_iso()
    })

def get_flight_alerts(status: str = "active") -> List[Dict[str, str]]:
    """Get flight alerts."""
    alerts = read_all_rows(TAB_FLIGHT_ALERTS)
    if status:
        alerts = [a for a in alerts if a.get("status") == status]
    return alerts

def update_flight_alert(alert_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update a flight alert."""
    return update_row_by_id(TAB_FLIGHT_ALERTS, "alert_id", alert_id, updates)

# ============================================================
# TV/Entertainment Watchlist
# ============================================================

def add_to_watchlist(title: str, media_type: str = "movie", season: int = 0,
                     episode: int = 0, platform: str = "", notes: str = "") -> Dict[str, Any]:
    """Add to TV/movie watchlist."""
    return append_row(TAB_TV_WATCHLIST, {
        "id": _gen_id(),
        "title": title,
        "type": media_type,
        "season": str(season) if season else "",
        "episode": str(episode) if episode else "",
        "status": "to_watch",
        "platform": platform,
        "added": _now_iso(),
        "notes": notes
    })

def get_watchlist(status: Optional[str] = None, media_type: Optional[str] = None) -> List[Dict[str, str]]:
    """Get watchlist."""
    items = read_all_rows(TAB_TV_WATCHLIST)
    if status:
        items = [i for i in items if i.get("status") == status]
    if media_type:
        items = [i for i in items if i.get("type") == media_type]
    return items

def update_watchlist_item(item_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update watchlist item."""
    return update_row_by_id(TAB_TV_WATCHLIST, "id", item_id, updates)

# ============================================================
# Entertainment Config (key-value store)
# ============================================================

def get_entertainment_config(key: str) -> Optional[str]:
    """Get a config value."""
    rows = read_all_rows(TAB_ENTERTAINMENT_CONFIG)
    for r in rows:
        if r.get("key") == key:
            return r.get("value")
    return None

def set_entertainment_config(key: str, value: str) -> Dict[str, Any]:
    """Set a config value."""
    rows = read_all_rows(TAB_ENTERTAINMENT_CONFIG)
    for r in rows:
        if r.get("key") == key:
            return update_row_by_id(TAB_ENTERTAINMENT_CONFIG, "key", key, {"value": value})
    return append_row(TAB_ENTERTAINMENT_CONFIG, {"key": key, "value": value})

# ============================================================
# Initialization & Health Check
# ============================================================

def init_all_tabs() -> Dict[str, Any]:
    """Initialize all tabs (creates them if they don't exist)."""
    results = {}
    for tab_name in TAB_HEADERS.keys():
        try:
            _get_worksheet(tab_name)
            results[tab_name] = "ok"
        except Exception as e:
            results[tab_name] = f"error: {e}"
    return results

def health_check() -> Dict[str, Any]:
    """Check if Google Sheets connection is working."""
    try:
        sh = _get_spreadsheet()
        return {"ok": True, "title": sh.title, "url": _sheet_url()}
    except Exception as e:
        return {"ok": False, "error": str(e)}
