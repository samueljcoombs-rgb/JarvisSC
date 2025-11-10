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
# - Direct Spotify episode links released TODAY (Europe/London)
# - Auto-searches show IDs by name (or accept explicit IDs)
# - Requires SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
# - Sleek UI: glassy cards, Spotify-green CTA, tiny "NEW" badge
# - Adds a bottom-right rounded artwork thumbnail (episode > show)
# ------------------------------------------------------------

# ✅ Your favourites (name, optional Spotify Show ID)
FAVOURITE_SHOWS: List[Tuple[str, Optional[str]]] = [
    ("JaackMaates Happy Hour", None),
    ("Wolf & Owl with Romesh Ranganathan and Tom Davis", None),
    ("Wafflin'", None),
    ("That Peter Crouch Podcast", None),
    ("Soccer Cards United", None),
    ("Pitch Side", None),
    ("The Club", None),
    ("Talk of the Devils", None),
    ("Manchester United Podcast by Stretford Paddock", None),
    ("Talk of the Devils: The Athletics FC's Manchester United show", None),
    ("The Rest is Entertainment", None),
    ("Lets Talk FPL", None),
    ("Chatabix", None),
]

MARKET = "GB"
TZ = ZoneInfo("Europe/London")
MAX_SHOW_EPISODES = 20
TIMEOUT = 15

# ---------------- Spotify auth helpers ----------------

def _spotify_creds() -> Tuple[Optional[str], Optional[str]]:
    cid = os.getenv("SPOTIFY_CLIENT_ID") or st.secrets.get("SPOTIFY_CLIENT_ID")
    cs  = os.getenv("SPOTIFY_CLIENT_SECRET") or st.secrets.get("SPOTIFY_CLIENT_SECRET")
    return cid, cs

def _get_token() -> Optional[str]:
    """Client Credentials flow with simple in-session cache."""
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
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        token = data["access_token"]
        expires_in = int(data.get("expires_in", 3600))
        st.session_state["_sp_token"] = token
        st.session_state["_sp_token_exp"] = time.time() + int(expires_in * 0.9)
        return token
    except Exception:
        return None

def _sp_get(url: str, params: Dict | None = None) -> Optional[dict]:
    token = _get_token()
    if not token:
        return None
    try:
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT)
        if r.status_code == 401:
            # refresh once
            st.session_state.pop("_sp_token", None)
            st.session_state.pop("_sp_token_exp", None)
            token = _get_token()
            if not token:
                return None
            headers = {"Authorization": f"Bearer {token}"}
            r = requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT)
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
        items = (data or {}).get("shows", {}).get("items", []) or []
        if items and isinstance(items[0], dict):
            return items[0].get("id")
    except Exception:
        pass
    return None

def _get_show_episodes(show_id: str, limit: int = MAX_SHOW_EPISODES) -> List[dict]:
    data = _sp_get(
        f"https://api.spotify.com/v1/shows/{show_id}/episodes",
        {"market": MARKET, "limit": limit},
    )
    try:
        items = (data or {}).get("items", []) or []
        return [it for it in items if isinstance(it, dict)]
    except Exception:
        return []

def _get_show_image(show_id: str) -> Optional[str]:
    """
    Fetch and cache a show's primary image (medium/small).
    """
    cache = st.session_state.setdefault("_show_img_cache", {})
    if show_id in cache:
        return cache[show_id]

    data = _sp_get(f"https://api.spotify.com/v1/shows/{show_id}")
    url = None
    try:
        imgs = (data or {}).get("images", []) or []
        # Prefer a smaller image for the thumbnail to keep it sharp + light.
        # Spotify images are usually sorted largest->smallest; pick last or middle.
        if imgs:
            # pick the smallest available, else first
            imgs_sorted = sorted(
                [i for i in imgs if isinstance(i, dict) and i.get("url")],
                key=lambda i: (i.get("width", 10**9), i.get("height", 10**9))
            )
            url = imgs_sorted[0]["url"] if imgs_sorted else None
    except Exception:
        url = None

    cache[show_id] = url
    return url

# ---------------- UI helpers ----------------

def _today_str() -> str:
    return datetime.now(TZ).date().isoformat()  # 'YYYY-MM-DD'

def _inject_css_once():
    if st.session_state.get("_pod_css_loaded"):
        return
    st.session_state["_pod_css_loaded"] = True
    st.markdown(
        """
<style>
.pod-card {
  background: rgba(255,255,255,0.66);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 10px 12px;                 /* tight */
  box-shadow: 0 8px 20px rgba(0,0,0,0.10);
  margin-bottom: 8px;                 /* tight */
  position: relative;
  overflow: hidden;
}
.pod-title {                          /* keep fonts/colours as requested */
  font-weight: 800; margin: 0 0 2px 0; color: #0f172a; letter-spacing: .1px;
}
.pod-meta {
  font-size: 0.9rem; opacity: 0.8; margin-bottom: 8px;
}
.pod-actions a {
  display: inline-flex; align-items: center; gap: 6px;
  text-decoration: none !important;
  padding: 6px 10px;                  /* tight */
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.10);
  font-weight: 800; font-size: 0.9rem; line-height: 1;
}
.pod-actions a.spotify {
  background: #1DB954; color: #fff; border: 1px solid #17a84a;
  box-shadow: 0 5px 12px rgba(29,185,84,0.32);
}
.pod-actions a.spotify:hover { filter: brightness(1.05); }
.pod-badge {
  position: absolute; top: 8px; right: 8px;
  background: linear-gradient(135deg, #f59e0b, #f97316);
  color: #fff; font-weight: 900; font-size: .7rem;
  padding: 2px 7px; border-radius: 999px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.18);
}
.pod-spot {
  width: 14px; height: 14px; display: inline-block; vertical-align: middle;
}

/* Thumbnail in bottom-right */
.pod-thumb {
  position: absolute; right: 10px; bottom: 10px;
  width: 44px; height: 44px; object-fit: cover;
  border-radius: 10px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.25);
  border: 1px solid rgba(0,0,0,0.08);
}
</style>
        """,
        unsafe_allow_html=True,
    )

_SPOTIFY_SVG = """
<svg class="pod-spot" viewBox="0 0 168 168" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
  <circle fill="#191414" cx="84" cy="84" r="84"/>
  <path fill="#1ED760" d="M119.3 115.1c-1.5 2.5-4.8 3.3-7.3 1.8-20-12.2-45.2-15-74.8-8.4-2.8.6-5.6-1.2-6.2-4-0.6-2.8 1.2-5.6 4-6.2 32.6-7.2 61.1-4.1 83.1 9.1 2.5 1.6 3.3 4.9 1.9 7.7zm10.3-22.9c-1.9 3.1-5.9 4.1-9 2.2-22.9-14-57.7-18.1-84.7-10.1-3.5 1-7.2-1-8.1-4.5-1-3.5 1-7.2 4.5-8.1 30.9-8.9 69.4-4.5 95.6 11.3 3.1 1.9 4.1 5.9 2.2 9.1zM130 68.5c-25.7-15.3-68.3-16.7-92.9-9.4-4.2 1.3-8.7-1.1-10-5.3-1.3-4.2 1.1-8.7 5.3-10 28.9-8.9 76-7.4 106.2 10.6 3.8 2.2 5 7 2.8 11.1-1.3 4.2-7.1 5.6-11.6 3z"/>
</svg>
""".strip()

def _episode_image_url(ep: dict, fallback_show_img: Optional[str]) -> Optional[str]:
    """
    Prefer an episode image if present; otherwise fallback to the show's image.
    Spotify episode objects sometimes include an 'images' array; if not, use show image.
    """
    # Episode-level images
    try:
        imgs = ep.get("images", []) or []
        if imgs and isinstance(imgs, list):
            imgs_sorted = sorted(
                [i for i in imgs if isinstance(i, dict) and i.get("url")],
                key=lambda i: (i.get("width", 10**9), i.get("height", 10**9))
            )
            if imgs_sorted:
                return imgs_sorted[0]["url"]
    except Exception:
        pass
    return fallback_show_img

def _episode_card(show_display: str, ep: dict, thumb_url: Optional[str]):
    title = ep.get("name", "Untitled")
    url = ep.get("external_urls", {}).get("spotify", "")
    date = ep.get("release_date", "")

    thumb_html = f'<img class="pod-thumb" src="{thumb_url}" alt="artwork"/>' if thumb_url else ""

    st.markdown(
        f"""
<div class="pod-card">
  <div class="pod-badge">NEW</div>
  <div class="pod-title">{title}</div>
  <div class="pod-meta">{show_display} — {date}</div>
  <div class="pod-actions">
    <a class="spotify" href="{url}" target="_blank">
      {_SPOTIFY_SVG}
      <span>▶︎ Listen on Spotify</span>
    </a>
  </div>
  {thumb_html}
</div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Public render ----------------

def render():
    st.header("🎧 New Podcast Episodes (Today)")

    _inject_css_once()

    cid, cs = _spotify_creds()
    if not cid or not cs:
        st.info("Add **SPOTIFY_CLIENT_ID** and **SPOTIFY_CLIENT_SECRET** to your environment or `st.secrets` to enable direct Spotify links.")
        return

    today = _today_str()
    total_found = 0

    for show_display, maybe_id in FAVOURITE_SHOWS:
        sid = maybe_id or _search_show_id(show_display)
        if not sid:
            continue

        # Fetch show image once per show (cached)
        show_img = _get_show_image(sid)

        episodes = _get_show_episodes(sid, limit=MAX_SHOW_EPISODES)
        todays = [
            ep for ep in episodes
            if isinstance(ep, dict) and ep.get("release_date") == today
        ]

        for ep in todays:
            thumb = _episode_image_url(ep, show_img)
            _episode_card(show_display, ep, thumb)
        total_found += len(todays)

    if total_found == 0:
        st.write("No new episodes today for your selected shows.")
