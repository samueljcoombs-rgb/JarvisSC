# modules/athletic_feed.py
import streamlit as st
import feedparser
from datetime import datetime
from urllib.parse import urlparse

FEED_URL = "https://www.nytimes.com/athletic/rss/football/manchester-united/"

def _to_dt(entry) -> datetime | None:
    # Prefer parsed time when available
    try:
        if getattr(entry, "published_parsed", None):
            tup = entry.published_parsed
            return datetime(tup.tm_year, tup.tm_mon, tup.tm_mday, tup.tm_hour, tup.tm_min, tup.tm_sec)
    except Exception:
        pass
    # Fallback: try parsing published string
    try:
        if entry.get("published"):
            tup = feedparser._parse_date(entry["published"])
            return datetime(*tup[:6])
    except Exception:
        pass
    return None

def render(show_header: bool = True, max_items: int = 5):
    """
    Renders Manchester United feed from The Athletic.

    Args:
        show_header: When False, suppresses the module's own title/header.
        max_items:  Number of entries to display (after filters & sort).
    """
    if show_header:
        st.subheader("The Athletic — Manchester United (RSS)")

    with st.spinner("Fetching latest…"):
        feed = feedparser.parse(FEED_URL)

    # Small, unobtrusive source line
    st.caption(f"Source: {urlparse(FEED_URL).netloc} • {feed.feed.get('title', 'Manchester United - The Athletic')}")

    # Optional quick filters
    q = st.text_input("Search title/summary", "", key="athletic_q")
    authors = sorted({e.get("author","").strip() for e in feed.entries if e.get("author")})
    author = st.selectbox("Filter by author", ["(All)"] + authors, index=0, key="athletic_author")

    # Apply filters
    filtered = []
    seen_titles = set()
    for e in feed.entries:
        title = e.get("title", "").strip()
        summary = e.get("summary", "")
        a = e.get("author","").strip()

        if not title or title in seen_titles:
            continue
        if q and (q.lower() not in title.lower() and q.lower() not in summary.lower()):
            continue
        if author != "(All)" and a != author:
            continue

        seen_titles.add(title)
        filtered.append(e)

    # Sort newest first
    filtered.sort(key=lambda x: _to_dt(x) or datetime.min, reverse=True)

    # Limit to latest N
    entries = filtered[: max(1, int(max_items))]

    # Render cards
    for e in entries:
        dt = _to_dt(e)
        top = f"{dt:%a, %d %b %Y %H:%M}" if dt else e.get("published","")

        # Try to find an image
        img = None
        try:
            for m in e.get("media_content", []) or []:
                img = m.get("url") or img
        except Exception:
            pass
        # Fallback from media_thumbnail if present
        if not img:
            try:
                for m in e.get("media_thumbnail", []) or []:
                    img = m.get("url") or img
            except Exception:
                pass

        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{e.get('title','(No title)')}**")
                st.write(f"**By:** {e.get('author','Unknown')} • {top}")
                if e.get("summary"):
                    st.write(e.get("summary"), unsafe_allow_html=True)
                st.link_button("Read on The Athletic", e.get("link",""), use_container_width=False)
            with cols[1]:
                if img:
                    # Avoid use_container_width for broader Streamlit compatibility
                    st.image(img, width=140)

if __name__ == "__main__":
    render()
