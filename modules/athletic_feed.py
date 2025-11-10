# modules/athletic_feed.py
from __future__ import annotations
import streamlit as st
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from typing import List, Tuple

# ------------------------------------------------------------
# The Athletic — Manchester United (Latest 3)
# Primary: Google News RSS (manual fetch + headers)
# Fallback: Scrape The Athletic team page for headlines (titles+links only)
# ------------------------------------------------------------

MAX_ITEMS = 3

# Google News query: The Athletic + Manchester United, last 7 days
QUERY = 'site:theathletic.com Manchester United when:7d'
RSS_URL = (
    "https://news.google.com/rss/search?"
    f"q={quote_plus(QUERY)}&hl=en-GB&gl=GB&ceid=GB:en"
)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

TEAM_PAGE = "https://theathletic.com/team/manchester-united/"

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
.ath-title { font-weight: 800; color: #0f172a; margin: 0 0 2px 0; }
.ath-link { text-decoration: none !important; }
.ath-meta { font-size: .85rem; opacity: .75; margin-top: 2px; }
.ath-ext { display:inline-block; margin-left: 6px; font-weight: 800;
           opacity: .85; text-decoration: none !important; }
</style>
        """,
        unsafe_allow_html=True,
    )

def _fetch_google_news_entries() -> List[Tuple[str, str]]:
    """Return list of (title, link) from Google News RSS."""
    try:
        r = requests.get(RSS_URL, headers={"User-Agent": UA}, timeout=12)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        entries = getattr(feed, "entries", []) or []
        out: List[Tuple[str, str]] = []
        for e in entries:
            title = e.get("title")
            link = e.get("link")
            if title and link:
                out.append((title, link))
            if len(out) >= MAX_ITEMS:
                break
        return out
    except Exception:
        return []

def _scrape_team_page() -> List[Tuple[str, str]]:
    """
    Fallback: scrape The Athletic's Man United team page for top headlines.
    We only grab title + canonical link (no paywall bypass).
    """
    try:
        r = requests.get(TEAM_PAGE, headers={"User-Agent": UA}, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        # Collect likely article anchors
        items: List[Tuple[str, str]] = []
        seen = set()

        # Heuristic: find anchors pointing to theathletic.com article URLs.
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                href = "https://theathletic.com" + href
            if not href.startswith("https://theathletic.com/"):
                continue
            # Avoid non-article links (e.g., author pages, subscriptions)
            if any(x in href for x in ("/author/", "/subscribe", "/team/", "/tag/")):
                continue

            # Use inner text or aria-label as title
            title = (a.get_text(strip=True) or a.get("aria-label") or "").strip()
            if not title:
                continue

            key = (title, href)
            if key in seen:
                continue
            seen.add(key)
            items.append((title, href))
            if len(items) >= MAX_ITEMS:
                break

        return items
    except Exception:
        return []

def render():
    st.header("📰 Man United — The Athletic (Latest 3)")
    _inject_css_once()

    # Try Google News first
    entries = _fetch_google_news_entries()

    # Fallback to direct team page if GN returns nothing
    if not entries:
        entries = _scrape_team_page()

    if not entries:
        st.write("No recent headlines found right now.")
        return

    for title, link in entries[:MAX_ITEMS]:
        st.markdown(
            f"""
<div class="ath-card">
  <div class="ath-title"><a class="ath-link" href="{link}" target="_blank">{title}</a></div>
  <div class="ath-meta">The Athletic · <a class="ath-ext" href="{link}" target="_blank">↗︎ Open</a></div>
</div>
            """,
            unsafe_allow_html=True,
        )
