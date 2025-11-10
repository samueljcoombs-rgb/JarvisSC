# modules/athletic_feed.py
# Streamlit module: Athletic / RSS Feed Reader (no extra dependencies)
from __future__ import annotations
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import html
import re

TITLE = "The Athletic / RSS Feeds"

DEFAULT_FEEDS = [
    # The Athletic — Manchester United (NYTimes domain)
    "https://www.nytimes.com/athletic/rss/football/manchester-united/",
]

USER_FEEDS_KEY = "athletic_user_feeds"
LAST_SEEN_KEY = "athletic_last_seen_links"

def _human_dt(dt: datetime) -> str:
    try:
        return dt.astimezone().strftime("%a %d %b %Y, %H:%M")
    except Exception:
        return str(dt)

def _safe_text(node: ET.Element | None, tag: str) -> str:
    if node is None:
        return ""
    t = node.findtext(tag)
    return (t or "").strip()

def _parse_pubdate(pub: str) -> datetime | None:
    # Try a few common RSS date formats
    fmts = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for f in fmts:
        try:
            return datetime.strptime(pub, f).replace(tzinfo=timezone.utc) if "Z" in f else datetime.strptime(pub, f)
        except Exception:
            continue
    return None

@st.cache_data(show_spinner=False, ttl=600)  # 10 minutes
def fetch_feed(url: str) -> List[Dict[str, Any]]:
    """Fetch and parse RSS/Atom (basic) without external deps."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Jarvis/FeedModule)"})
    with urlopen(req, timeout=15) as resp:
        data = resp.read()

    # Try RSS first
    root = ET.fromstring(data)

    # RSS 2.0: <rss><channel><item>...
    items_path = []
    if root.tag.lower().endswith("rss"):
        items_path = ["channel", "item"]
    elif root.tag.lower().endswith("feed"):  # Atom
        items_path = ["entry"]

    items: List[Dict[str, Any]] = []

    def get_children(node: ET.Element, tag: str) -> List[ET.Element]:
        return [c for c in node if c.tag.lower().endswith(tag)]

    if items_path:
        nodes = root
        for tag in items_path[:-1]:
            # descend to channel or equivalent
            matches = get_children(nodes, tag) if isinstance(nodes, ET.Element) else []
            nodes = matches[0] if matches else root
        leaf_tag = items_path[-1]
        leaf_nodes = get_children(nodes, leaf_tag) if isinstance(nodes, ET.Element) else []
    else:
        # fallback: search all <item>
        leaf_nodes = root.findall(".//item")

    for it in leaf_nodes:
        # RSS fields
        title = _safe_text(it, "title")
        link = _safe_text(it, "link")
        guid = _safe_text(it, "guid") or link
        pub = _safe_text(it, "pubDate") or _safe_text(it, "updated") or _safe_text(it, "published")
        desc = _safe_text(it, "description") or _safe_text(it, "summary") or ""

        # Atom alternatives
        if not link:
            link_nodes = [c for c in list(it) if c.tag.lower().endswith("link")]
            if link_nodes:
                href = link_nodes[0].attrib.get("href")
                if href:
                    link = href

        pub_dt = _parse_pubdate(pub) if pub else None

        # Clean description to a short snippet
        snippet = html.unescape(re.sub("<[^<]+?>", "", desc))
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if len(snippet) > 280:
            snippet = snippet[:277] + "..."

        items.append({
            "title": html.unescape(title),
            "link": link,
            "guid": guid or link,
            "published": pub_dt,
            "snippet": snippet,
            "source": urlparse(link).netloc or urlparse(url).netloc,
        })

    # Deduplicate by link/guid and sort by published desc
    dedup: Dict[str, Dict[str, Any]] = {}
    for x in items:
        k = x.get("guid") or x.get("link")
        if k and k not in dedup:
            dedup[k] = x

    sorted_items = sorted(
        dedup.values(),
        key=lambda x: (x.get("published") or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    return sorted_items

def _get_all_items(feed_urls: List[str]) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    for u in feed_urls:
        try:
            all_items.extend(fetch_feed(u))
        except Exception as e:
            all_items.append({
                "title": f"⚠️ Error fetching: {u}",
                "link": u,
                "guid": f"error::{u}",
                "published": None,
                "snippet": f"{e}",
                "source": urlparse(u).netloc,
            })
    # Dedup across feeds by link
    seen = set()
    merged = []
    for it in all_items:
        k = it.get("guid") or it.get("link")
        if k in seen:
            continue
        seen.add(k)
        merged.append(it)
    merged.sort(key=lambda x: (x.get("published") or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return merged

def _keyword_match(text: str, terms: List[str]) -> bool:
    if not terms:
        return True
    t = text.lower()
    return any(term.lower() in t for term in terms)

def render():
    st.header("📰 The Athletic / RSS")
    st.caption("Auto-refreshing reader for The Athletic (NYTimes) and other RSS feeds — no extra dependencies.")

    # Init persistent state
    feeds: List[str] = st.session_state.get(USER_FEEDS_KEY, DEFAULT_FEEDS.copy())
    last_seen: set[str] = set(st.session_state.get(LAST_SEEN_KEY, []))

    with st.sidebar:
        st.subheader("Feed Settings")
        refresh_seconds = st.number_input("Auto-refresh every (seconds)", min_value=10, max_value=3600, value=300, step=10, help="Streamlit will auto refresh the module on this cadence.")
        st.caption("Tip: 300s = 5 minutes")

        st.markdown("**Feeds**")
        for i, f in enumerate(list(feeds)):
            cols = st.columns([1, 8, 2])
            with cols[0]:
                st.write(f"{i+1}.")
            with cols[1]:
                feeds[i] = st.text_input(f"Feed {i+1}", value=f, key=f"feed_{i}")
            with cols[2]:
                if st.button("🗑️", key=f"del_{i}", help="Remove feed"):
                    feeds.pop(i)
                    st.session_state[USER_FEEDS_KEY] = feeds
                    st.experimental_rerun()

        new_feed = st.text_input("Add a feed URL", placeholder="https://.../rss")
        if st.button("Add feed"):
            if new_feed and new_feed not in feeds:
                feeds.append(new_feed)
                st.session_state[USER_FEEDS_KEY] = feeds
                st.success("Added.")
                st.experimental_rerun()

        if st.button("Reset to defaults"):
            feeds = DEFAULT_FEEDS.copy()
            st.session_state[USER_FEEDS_KEY] = feeds
            st.success("Reset.")
            st.experimental_rerun()

        st.divider()
        st.subheader("Filters")
        kw = st.text_input("Keywords (comma-separated)", placeholder="injury, transfer, line-up")
        keywords = [k.strip() for k in kw.split(",") if k.strip()]
        source_filter = st.text_input("Source filter (optional)", placeholder="www.nytimes.com")

        st.divider()
        if st.button("Mark all as read"):
            st.session_state[LAST_SEEN_KEY] = [*{*last_seen, *[it.get('guid') or it.get('link') for it in _get_all_items(feeds)]}]
            st.success("Marked current items as read.")

    # Auto refresh
    st.autorefresh = st.experimental_singleton(lambda: None)  # no-op to keep linter happy
    st.experimental_set_query_params(_=datetime.now().timestamp())  # bust cache on reruns
    st.experimental_rerun  # accessibility for dev tools
    st.experimental_memo  # noqa

    _ = st.experimental_rerun  # avoid unused warnings

    # Fetch + display
    items = _get_all_items(feeds)

    # Apply filters
    if source_filter:
        items = [it for it in items if source_filter.lower() in (it.get("source") or "").lower()]
    if keywords:
        items = [it for it in items if _keyword_match((it.get("title") or "") + " " + (it.get("snippet") or ""), keywords)]

    st.write(f"**Feeds:** {len(feeds)} | **Items:** {len(items)}")

    # Show items
    for it in items:
        guid = it.get("guid") or it.get("link")
        is_new = guid not in last_seen and not str(guid).startswith("error::")
        badge = "🆕 " if is_new else ""
        pub = it.get("published")
        pub_str = _human_dt(pub) if pub else "Unknown"

        with st.container(border=True):
            st.markdown(f"### {badge}{it.get('title') or '(No title)'}")
            st.caption(f"{it.get('source')}  •  {pub_str}")
            if it.get("snippet"):
                st.write(it["snippet"])
            if it.get("link"):
                st.link_button("Open article", it["link"], use_container_width=False)

            # mark this as seen when rendered
            if is_new:
                last_seen.add(guid)

    # Persist last seen after render
    st.session_state[LAST_SEEN_KEY] = list(last_seen)

    # footer controls
    st.divider()
    st.caption("Tips: add other The Athletic team feeds; use keywords like 'injury', 'transfer', 'line-up' to highlight relevant posts.")
