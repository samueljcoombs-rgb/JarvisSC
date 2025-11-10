# modules/athletic_feed.py
import re
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import feedparser
import streamlit as st

FEED_URL = "https://www.nytimes.com/athletic/rss/football/manchester-united/"

# --- Safe image helper + compatibility ----------------------------------------
try:
    from PIL import Image as _PILImage  # optional
except Exception:
    _PILImage = None  # type: ignore


def _image_compat(img: Any, *, container_width: bool = True) -> None:
    """
    Call st.image with compatibility across Streamlit versions:
    - Prefer use_container_width (newer)
    - Fallback to use_column_width=True (older)
    Accepts URL string, bytes, PIL image, requests-like response, or file-like .read().
    """
    if img is None:
        return

    def _call_image(obj):
        try:
            st.image(obj, use_container_width=container_width)
        except TypeError:
            # Older Streamlit (<1.26) doesn't support use_container_width
            st.image(obj, use_column_width=True)

    try:
        # URL string
        if isinstance(img, str):
            if img.strip():
                _call_image(img)
            return

        # Raw bytes
        if isinstance(img, (bytes, bytearray)):
            _call_image(img)
            return

        # PIL Image
        if _PILImage is not None and isinstance(img, _PILImage.Image):  # type: ignore[attr-defined]
            _call_image(img)
            return

        # requests.Response
        content = getattr(img, "content", None)
        if isinstance(content, (bytes, bytearray)):
            _call_image(content)
            return

        # File-like with .read()
        reader = getattr(img, "read", None)
        if callable(reader):
            data = reader()
            if isinstance(data, (bytes, bytearray)):
                _call_image(data)
            return
    except Exception as e:
        st.caption(f"Image unavailable ({e})")


# --- Helpers ------------------------------------------------------------------
def _to_dt(entry) -> Optional[datetime]:
    """
    Convert an RSS entry's published date to datetime.
    Tries entry.published_parsed first, then feedparser.parse_date, then None.
    """
    try:
        if getattr(entry, "published_parsed", None):
            tup = entry.published_parsed
            return datetime(*tup[:6])
        published = entry.get("published") or entry.get("updated") or ""
        if published:
            d = feedparser.parse_date(published)
            if d:
                return datetime(*d[:6])
    except Exception:
        pass
    return None


_IMG_TAG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)


def _first_image(entry) -> Optional[str]:
    """
    Try several common RSS fields to find an image URL.
    """
    # media_content (The Athletic often supplies this)
    media_content = entry.get("media_content") or []
    for m in media_content:
        url = (m or {}).get("url")
        if url:
            return url

    # media_thumbnail
    thumbs = entry.get("media_thumbnail") or []
    for m in thumbs:
        url = (m or {}).get("url")
        if url:
            return url

    # content/summary HTML
    for key in ("content", "summary", "description"):
        blob = entry.get(key)
        if isinstance(blob, list) and blob:
            blob = blob[0].get("value", "")
        if isinstance(blob, str) and blob:
            m = _IMG_TAG_RE.search(blob)
            if m:
                return m.group(1)

    # links with rel="enclosure"
    links = entry.get("links") or []
    for l in links:
        if l.get("rel") == "enclosure" and l.get("type", "").startswith("image/"):
            return l.get("href")

    return None


# --- UI -----------------------------------------------------------------------
def render():
    st.header("The Athletic — Manchester United (RSS)")

    with st.spinner("Fetching latest…"):
        feed = feedparser.parse(FEED_URL)

    # Graceful empty-feed handling
    try:
        feed_title = feed.feed.get("title", "Manchester United - The Athletic")
    except Exception:
        feed_title = "Manchester United - The Athletic"

    st.caption(f"Source: {urlparse(FEED_URL).netloc} • {feed_title}")

    # Controls
    c1, c2 = st.columns([3, 1])
    with c1:
        q = st.text_input("Search title/summary", "", placeholder="e.g. Ten Hag, transfer, injury")
    with c2:
        max_items = st.number_input("Max items", min_value=1, max_value=20, value=5, step=1)

    # Filters
    entries_raw = getattr(feed, "entries", []) or []
    authors = sorted({(e.get("author") or "").strip() for e in entries_raw if e.get("author")})
    author = st.selectbox("Filter by author", ["(All)"] + authors, index=0)

    # Apply filters
    q_lower = q.lower().strip()
    entries = []
    for e in entries_raw:
        title = e.get("title", "")
        summary = e.get("summary", "") or e.get("description", "")
        a = (e.get("author") or "").strip()

        if q_lower and (q_lower not in title.lower() and q_lower not in summary.lower()):
            continue
        if author != "(All)" and a != author:
            continue
        entries.append(e)

    # Sort newest first and cap to max_items
    entries.sort(key=lambda x: _to_dt(x) or datetime.min, reverse=True)
    entries = entries[: int(max_items)]

    if not entries:
        st.info("No matching articles right now. Try clearing filters or increase Max items.")
        return

    # Render cards
    for e in entries:
        with st.container(border=True):
            dt = _to_dt(e)
            top = f"{dt:%a, %d %b %Y %H:%M}" if dt else (e.get("published") or e.get("updated") or "")

            img = _first_image(e)

            cols = st.columns([3, 1])
            with cols[0]:
                st.subheader(e.get("title", "Untitled"))
                st.write(f"**By:** {e.get('author','Unknown')} • {top}")
                if e.get("summary"):
                    st.write(e.get("summary"), unsafe_allow_html=True)
                elif e.get("description"):
                    st.write(e.get("description"), unsafe_allow_html=True)

                link = e.get("link") or e.get("id")
                if link:
                    st.link_button("Read on The Athletic", link, use_container_width=False)

            with cols[1]:
                if img:
                    _image_compat(img, container_width=True)
                else:
                    st.caption("No image")


# Local debug
if __name__ == "__main__":
    render()
