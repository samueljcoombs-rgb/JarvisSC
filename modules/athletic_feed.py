# modules/athletic_manu_feed.py
import streamlit as st
import feedparser
from datetime import datetime
from urllib.parse import urlparse

FEED_URL = "https://www.nytimes.com/athletic/rss/football/manchester-united/"

def _to_dt(published: str):
    try:
        return datetime(*feedparser._parse_date(published)[:6])
    except Exception:
        return None

def render():
    st.header("The Athletic — Manchester United (RSS)")
    with st.spinner("Fetching latest…"):
        feed = feedparser.parse(FEED_URL)

    st.caption(f"Source: {urlparse(FEED_URL).netloc} • {feed.feed.get('title', 'Manchester United - The Athletic')}")
    q = st.text_input("Search title/summary", "")
    authors = sorted({e.get("author","").strip() for e in feed.entries if e.get("author")})
    author = st.selectbox("Filter by author", ["(All)"] + authors, index=0)

    entries = []
    for e in feed.entries:
        title = e.title
        summary = e.get("summary", "")
        a = e.get("author","").strip()
        if q and (q.lower() not in title.lower() and q.lower() not in summary.lower()):
            continue
        if author != "(All)" and a != author:
            continue
        entries.append(e)

    # Sort newest first
    entries.sort(key=lambda x: _to_dt(x.get("published","")) or datetime.min, reverse=True)

    for e in entries:
        with st.container(border=True):
            dt = _to_dt(e.get("published",""))
            top = f"{dt:%a, %d %b %Y %H:%M}" if dt else e.get("published","")
            img = None
            for m in e.get("media_content", []):
                img = m.get("url") or img
            cols = st.columns([3,1])
            with cols[0]:
                st.subheader(e.title)
                st.write(f"**By:** {e.get('author','Unknown')} • {top}")
                if e.get("summary"):
                    st.write(e.get("summary"), unsafe_allow_html=True)
                st.link_button("Read on The Athletic", e.link, use_container_width=False)
            with cols[1]:
                if img:
                    st.image(img, use_container_width=True)

if __name__ == "__main__":
    render()
