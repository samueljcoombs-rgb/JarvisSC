# modules/podcasts_panel.py
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import streamlit as st

# --- Optional: use Spotify Web API via client credentials ---
# Requires env vars:
#   SPOTIFY_CLIENT_ID
#   SPOTIFY_CLIENT_SECRET
#   SPOTIFY_SHOW_IDS  (comma-separated, e.g. "5lEszZz...,3E8m...,2Q6...")
#
# The module will gracefully handle missing creds by showing a note.

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    _SPOTIFY_AVAILABLE = True
except Exception:
    _SPOTIFY_AVAILABLE = False


def _spotify_client():
    cid = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    if not (_SPOTIFY_AVAILABLE and cid and secret):
        return None
    try:
        auth = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        return spotipy.Spotify(auth_manager=auth)
    except Exception:
        return None


def _date_label(d: datetime, today: datetime) -> str:
    if d.date() == today.date():
        return "Today"
    if d.date() == (today.date() - timedelta(days=1)):
        return "Yesterday"
    return d.strftime("%a %d %b")


def _make_date_choices(n_days: int = 7) -> List[Dict[str, str]]:
    now = datetime.now()
    days = []
    for i in range(n_days):
        day = now - timedelta(days=i)
        label = _date_label(day, now)
        value = day.strftime("%Y-%m-%d")
        days.append({"label": label, "value": value})
    return days


def _load_show_ids() -> List[str]:
    raw = os.getenv("SPOTIFY_SHOW_IDS", "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _fetch_episodes_for_shows(sp: "spotipy.Spotify", show_ids: List[str], limit_per_show: int = 50) -> List[Dict]:
    """
    Fetch recent episodes for the configured shows.
    Returns list of dicts with keys: show_name, episode_name, release_date, external_url, image_url, description
    """
    episodes: List[Dict] = []

    for sid in show_ids:
        try:
            show = sp.show(sid)
            show_name = show.get("name", "Unknown Show")
            images = show.get("images") or []
            show_image = images[0]["url"] if images else None
        except Exception:
            # Skip shows that fail to load
            continue

        # Spotify API returns episodes newest-first by default
        try:
            eps = sp.show_episodes(sid, limit=limit_per_show)
        except Exception:
            continue

        items = eps.get("items", []) if isinstance(eps, dict) else []
        for it in items:
            release_date = it.get("release_date") or ""
            # Sometimes precision can be "day" or "hour"; we only filter by YYYY-MM-DD
            if len(release_date) >= 10:
                release_date = release_date[:10]

            ext = (it.get("external_urls") or {}).get("spotify") or it.get("uri") or ""
            images = it.get("images") or []
            image_url = images[0]["url"] if images else show_image
            episodes.append(
                {
                    "show_name": show_name,
                    "episode_name": it.get("name", "Untitled"),
                    "release_date": release_date,
                    "external_url": ext,
                    "image_url": image_url,
                    "description": it.get("description", ""),
                }
            )

    return episodes


def _filter_episodes_by_date(episodes: List[Dict], ymd: str) -> List[Dict]:
    return [e for e in episodes if (e.get("release_date") or "").startswith(ymd)]


def render(selected_date: Optional[str] = None, n_days_window: int = 7):
    """
    Render Spotify episodes with a day filter (Today + previous 6 days).
    - selected_date: optional "YYYY-MM-DD" string to preselect a day
    - n_days_window: number of days to offer in the dropdown (default 7)
    """
    # Header kept concise as requested
    st.subheader("🎧 New Podcast Episodes")

    # Date selector
    date_choices = _make_date_choices(max(1, n_days_window))
    labels = [c["label"] for c in date_choices]
    values = [c["value"] for c in date_choices]

    # Default to 'selected_date' if provided, else 'Today'
    if selected_date and selected_date in values:
        default_index = values.index(selected_date)
    else:
        default_index = 0  # Today

    sel_label = st.selectbox(
        "Show episodes released on:",
        labels,
        index=default_index,
        key="podcasts_day_selector",
    )
    # Map back to YYYY-MM-DD
    selected_value = values[labels.index(sel_label)]

    # Fetch data
    show_ids = _load_show_ids()
    sp = _spotify_client() if show_ids else None

    if not show_ids:
        st.info(
            "Add env var **SPOTIFY_SHOW_IDS** (comma-separated show IDs) to enable episodes here.\n\n"
            "Tip: right-click a show in Spotify → Share → Copy link. The show ID is the long string in the URL.",
            icon="ℹ️",
        )
        return

    if not sp:
        if not _SPOTIFY_AVAILABLE:
            st.error("`spotipy` not installed. `pip install spotipy` inside your environment.", icon="⚠️")
        else:
            st.error("Spotify credentials missing/invalid. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.", icon="⚠️")
        return

    with st.spinner("Loading episodes…"):
        all_eps = _fetch_episodes_for_shows(sp, show_ids, limit_per_show=50)

    todays_eps = _filter_episodes_by_date(all_eps, selected_value)

    # Results
    if not todays_eps:
        st.write("No new episodes for this day.")
        return

    # Grid-like cards (2 columns for readability)
    cols = st.columns(2)
    for i, ep in enumerate(sorted(todays_eps, key=lambda x: x.get("show_name", "").lower())):
        with cols[i % 2]:
            with st.container(border=True):
                if ep.get("image_url"):
                    st.image(ep["image_url"])
                st.markdown(f"**{ep['episode_name']}**")
                st.caption(f"{ep['show_name']} • {ep.get('release_date', '')}")
                if ep.get("description"):
                    # Keep it compact
                    trimmed = (ep["description"][:220] + "…") if len(ep["description"]) > 220 else ep["description"]
                    st.write(trimmed)
                if ep.get("external_url"):
                    st.link_button("Open in Spotify", ep["external_url"], use_container_width=False)


if __name__ == "__main__":
    render()
