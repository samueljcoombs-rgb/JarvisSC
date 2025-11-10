# modules/athletic_feed.py
from __future__ import annotations
import streamlit as st
import feedparser
from urllib.parse import quote_plus

# ------------------------------------------------------------
# The Athletic — Manchester United (Latest 3 via Google News RSS)
# - Headlines + links only (opens in new tab; your browser handles login)
# - No credentials stored; no scraping of full text
# ------------------------------------------------------------

# Google News RSS query: The Athletic + "Manchester United"
# Localised to GB English results.
QUERY = 'site:theathletic.com "Manchester United"'
RSS_URL = (
    "https://news.google.com/rss/search?"
    f"q={quote_plus(QUERY)}&hl=en-GB&gl=GB&ceid=GB:en"
)

MAX_ITEMS = 3

def _inject_css_once():
    if st.session_state.get("_ath_css_loaded"):
        return
    st.session_state["_ath_css_loaded"] = True
    st.markdown(
        """
<style>
.ath-card {
  background: rgba(255,255,255,0.60);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 12px;
  padding: 10px 12px;
  box-shadow: 0 6px 14px rgba(0,0,0,0.08);
  margin-bottom: 8px;
}
.ath-title {
  font-weight: 800; color: #0f172a; margin: 0 0 2px 0;
}
.ath-link {
  text-decoration: none !important;
}
.ath-meta {
  font-size: .85rem; opacity: .75; margin-top: 2px;
}
.ath-ext {
  display:inline-block; margin-left: 6px; font-weight: 800;
  opacity: .85; text-decoration: none !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

def render():
    st.header("📰 Man United — The Athletic (Latest 3)")
    _inject_css_once()

    try:
        feed = feedparser.parse(RSS_URL)
        entries = feed.entries[:MAX_ITEMS] if getattr(feed, "entries", None) else []
    except Exception:
        entries = []

    if not entries:
        st.write("No recent headlines found right now.")
        return

    for e in entries:
        title = e.get("title") or "Untitled"
        link = e.get("link") or "#"

        # Some Google News links are redirect URLs; still fine — your browser session will handle login.
        st.markdown(
            f"""
<div class="ath-card">
  <div class="ath-title"><a class="ath-link" href="{link}" target="_blank">{title}</a></div>
  <div class="ath-meta">The Athletic · <a class="ath-ext" href="{link}" target="_blank">↗︎ Open</a></div>
</div>
            """,
            unsafe_allow_html=True,
        )

