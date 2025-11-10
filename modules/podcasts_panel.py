# modules/podcasts_panel.py
from __future__ import annotations
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

# ------------------------------------------------------------
# Podcasts Panel (Spotify)
# - Gets direct Spotify episode links released TODAY (Europe/London)
# - Works with show names (auto-search) or explicit Spotify Show IDs
# - Requires SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in env or st.secrets
# ------------------------------------------------------------

# ✏️ Configure your shows here.
# You can list names, or tuples of (display_name, spotify_show_id).
FAVOURITE_SHOWS: List[Tuple[str, Optional[str]]] = [
    ("JaackMaates Happy Hour", None),
    # ("The Diary Of A CEO", None),
    # ("The Rest Is Politics", None),
]

MARKET = "GB"                 # For search & episodes
TZ = ZoneInfo("Europe/London")  # Your local day boundary
MAX_SHOW_EPISODES = 20        # How many to fetch per show before filtering "today"


# ---------------- Spotify auth helpers ----------------

def _spotify_creds() -> Tuple[Optional[str], Optional[str]]:
    cid = os.getenv("SPOTIFY_CLIENT_ID") or st.secrets.get("SPOTIFY_CLIENT_ID")
    cs  = os.getenv("SPOTIFY_CLIENT_SECRET") or st.secrets.get("SPOTIFY_CLIENT_SECRET")
    return cid, cs

def _get_token() -> Optional[str]:
    """
    Client Credentials flow. Cache token in session_state until expiry.
    """
    if "_sp_token" in st.session_state and "_sp_token_exp" in st.session_state:
        if time.time() < st.session_state["_sp_token_exp"]:
            return st.session_state["_sp_token"]

    cid, cs = _spotify_creds()
    if not cid or not cs:
        return None

    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(cid, cs),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        token = data["access_token"]
        expires_in = int(data.get("expires_in", 3600))
        st.session_state["_sp_token"] = token
        st.session_state["_sp_token_exp"] = time.time() + expires_in * 0.9
        return token
    except Exception:
        return None

def _sp_get(url: str, params: Dict) -> Optional[dict]:
    token = _get_token()
    if not token:
        return None
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=15)
        if r.status_code == 401:
            # refresh once
            st.session_state.pop("_sp_token", None)
            st.session_state.pop("_sp_token_exp", None)
            token = _get_token()
            if not token:
                return None
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------- Spotify search/fetch ----------------

def _search_show_id(show_name: str) -> Optional[str]:
    data = _sp_get(
        "https://api.spotify.com/v1/search",
        {"q": show_name, "type": "show", "limit": 1, "market": MARKET},
    )
    try:
        items = (data or {}).get("shows", {}).get("items", [])
        if items:
            return items[0]["id"]
    except Exception:
        pass
    return None

def _get_show_episodes(show_id: str, limit: int = MAX_SHOW_EPISODES) -> List[dict]:
    data = _sp_get(
        f"https://api.spotify.com/v1/shows/{show_id}/episodes",
        {"market": MARKET, "limit": limit},
    )
    try:
        return (data or {}).get("items", []) or []
    except Exception:
        return []


# ---------------- UI helpers ----------------

def _today_str() -> str:
    return datetime.now(TZ).date().isoformat()  # 'YYYY-MM-DD'

def _episode_card(show_display: str, ep: dict):
    title = ep.get("name", "Untitled")
    url = ep.get("external_urls", {}).get("spotify", "")
    date = ep.get("release_date", "")
    time_str = ""
    # Some episodes have release_date_precision = 'day' only
    if ep.get("release_date_precision") == "day":
        time_str = ""
    else:
        # If we ever had time precision, format it. Currently Spotify returns day for podcasts.
        time_str = ""

    st.markdown(
        f"""
<div style="
  background: rgba(255,255,255,0.6);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.08);
  margin-bottom: 10px;">
  <div style="font-weight:700; color:#0f172a; margin: 0 0 4px 0;">{title}</div>
  <div style="font-size:0.9rem; opacity:0.8; margin-bottom:8px;">{show_display} — {date}</div>
  <div>
    <a href="{url}" target="_blank" style="
      display:inline-block; text-decoration:none; padding:6px 10px; border-radius:999px;
      border:1px solid rgba(0,0,0,0.10); font-weight:700; font-size:0.9rem;">
      ▶︎ Listen on Spotify
    </a>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- Public render ----------------

def render():
    st.header("🎧 New Podcast Episodes (Today)")

    cid, cs = _spotify_creds()
    if not cid or not cs:
        st.info("Add **SPOTIFY_CLIENT_ID** and **SPOTIFY_CLIENT_SECRET** to your environment or `st.secrets` to enable direct Spotify links.")
        return

    today = _today_str()
    total_found = 0

    for show_display, show_id in FAVOURITE_SHOWS:
        # Resolve ID if not provided
        sid = show_id or _search_show_id(show_display)
        if not sid:
            continue

        episodes = _get_show_episodes(sid, limit=MAX_SHOW_EPISODES)
        # Filter to today's Europe/London date
        todays = [ep for ep in episodes if ep.get("release_date") == today]

        if todays:
            st.subheader(show_display)
            for ep in todays:
                _episode_card(show_display, ep)
            total_found += len(todays)

    if total_found == 0:
        st.write("No new episodes today for your selected shows.")
