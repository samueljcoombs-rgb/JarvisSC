# modules/news_tools.py
"""
News Tools for Jarvis
Personalized news feeds from various sources.
"""
from __future__ import annotations
import os
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
import feedparser
import streamlit as st

TZ = ZoneInfo("Europe/London")

# ============================================================
# API Configuration
# ============================================================

def _newsapi_key() -> Optional[str]:
    return st.secrets.get("NEWSAPI_KEY") or os.getenv("NEWSAPI_KEY")

NEWSAPI_BASE = "https://newsapi.org/v2"

# ============================================================
# News Categories & Sources
# ============================================================

NEWS_CATEGORIES = [
    "general",
    "technology",
    "business",
    "sports",
    "science",
    "entertainment",
    "health",
]

# Curated RSS feeds by category
RSS_FEEDS = {
    "tech": [
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
        ("Wired", "https://www.wired.com/feed/rss"),
    ],
    "uk_news": [
        ("BBC News", "http://feeds.bbci.co.uk/news/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/uk/rss"),
        ("Sky News", "https://feeds.skynews.com/feeds/rss/uk.xml"),
    ],
    "business": [
        ("Financial Times", "https://www.ft.com/rss/home/uk"),
        ("Bloomberg", "https://feeds.bloomberg.com/markets/news.rss"),
        ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ],
    "sports": [
        ("BBC Sport", "http://feeds.bbci.co.uk/sport/rss.xml"),
        ("ESPN", "https://www.espn.com/espn/rss/news"),
        ("Sky Sports", "https://www.skysports.com/rss/12040"),
    ],
    "football": [
        ("BBC Football", "http://feeds.bbci.co.uk/sport/football/rss.xml"),
        ("The Athletic - Football", "https://theathletic.com/uk/feed/"),
        ("Sky Sports Football", "https://www.skysports.com/rss/12040"),
    ],
    "science": [
        ("New Scientist", "https://www.newscientist.com/feed/home/"),
        ("Science Daily", "https://www.sciencedaily.com/rss/all.xml"),
        ("Nature", "http://feeds.nature.com/nature/rss/current"),
    ],
    "ai": [
        ("AI News", "https://www.artificialintelligence-news.com/feed/"),
        ("MIT Tech Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
        ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ],
    "gaming": [
        ("IGN", "https://feeds.feedburner.com/ign/all"),
        ("Kotaku", "https://kotaku.com/rss"),
        ("PC Gamer", "https://www.pcgamer.com/rss/"),
    ],
}

# ============================================================
# NewsAPI Functions
# ============================================================

@st.cache_data(ttl=1800)
def get_top_headlines(
    country: str = "gb",
    category: str = None,
    query: str = None,
    page_size: int = 20
) -> Dict[str, Any]:
    """Get top headlines from NewsAPI."""
    api_key = _newsapi_key()
    if not api_key:
        return {"error": "No NEWSAPI_KEY configured", "articles": []}
    
    params = {
        "country": country,
        "pageSize": page_size,
        "apiKey": api_key,
    }
    
    if category:
        params["category"] = category
    if query:
        params["q"] = query
    
    try:
        r = requests.get(f"{NEWSAPI_BASE}/top-headlines", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        return {
            "status": data.get("status"),
            "total": data.get("totalResults", 0),
            "articles": data.get("articles", [])
        }
    except Exception as e:
        return {"error": str(e), "articles": []}

@st.cache_data(ttl=1800)
def search_news(
    query: str,
    from_date: str = None,
    sort_by: str = "relevancy",
    page_size: int = 20
) -> Dict[str, Any]:
    """Search news articles from NewsAPI."""
    api_key = _newsapi_key()
    if not api_key:
        return {"error": "No NEWSAPI_KEY configured", "articles": []}
    
    params = {
        "q": query,
        "sortBy": sort_by,
        "pageSize": page_size,
        "apiKey": api_key,
        "language": "en",
    }
    
    if from_date:
        params["from"] = from_date
    else:
        # Default to last 7 days
        params["from"] = (datetime.now(TZ) - timedelta(days=7)).strftime("%Y-%m-%d")
    
    try:
        r = requests.get(f"{NEWSAPI_BASE}/everything", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        return {
            "status": data.get("status"),
            "total": data.get("totalResults", 0),
            "articles": data.get("articles", [])
        }
    except Exception as e:
        return {"error": str(e), "articles": []}

# ============================================================
# RSS Feed Functions
# ============================================================

def _extract_image_from_entry(entry) -> Optional[str]:
    """Extract image URL from RSS entry using various methods."""
    # Method 1: media:content
    if hasattr(entry, 'media_content') and entry.media_content:
        for media in entry.media_content:
            if media.get('medium') == 'image' or media.get('type', '').startswith('image'):
                return media.get('url')
            # Some feeds just have url without type
            url = media.get('url', '')
            if url and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                return url
    
    # Method 2: media:thumbnail
    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
        return entry.media_thumbnail[0].get('url')
    
    # Method 3: enclosure
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get('type', '').startswith('image'):
                return enc.get('href') or enc.get('url')
    
    # Method 4: Parse from summary/content HTML
    import re
    content = entry.get('summary', '') or entry.get('content', [{}])[0].get('value', '') if hasattr(entry, 'content') else ''
    if content:
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content)
        if img_match:
            return img_match.group(1)
    
    # Method 5: image tag in feed
    if hasattr(entry, 'image') and entry.image:
        if isinstance(entry.image, dict):
            return entry.image.get('href') or entry.image.get('url')
        return str(entry.image)
    
    return None

def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse various date formats from RSS feeds."""
    from email.utils import parsedate_to_datetime
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except:
        pass
    
    # Try common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None

@st.cache_data(ttl=900)
def get_rss_feed(url: str, max_items: int = 10, max_age_hours: int = 48) -> List[Dict]:
    """Fetch and parse an RSS feed with images and freshness filter."""
    try:
        feed = feedparser.parse(url)
        articles = []
        now = datetime.now(TZ)
        cutoff = now - timedelta(hours=max_age_hours)
        
        for entry in feed.entries[:max_items * 2]:  # Fetch more to account for filtered items
            # Parse and filter by date
            pub_date = _parse_date(entry.get("published", "") or entry.get("updated", ""))
            
            if pub_date:
                # Make timezone-aware if needed
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=TZ)
                
                # Skip old articles
                if pub_date < cutoff:
                    continue
                
                # Format relative time
                age = now - pub_date.astimezone(TZ)
                if age.total_seconds() < 3600:
                    age_str = f"{int(age.total_seconds() / 60)}m ago"
                elif age.total_seconds() < 86400:
                    age_str = f"{int(age.total_seconds() / 3600)}h ago"
                else:
                    age_str = f"{int(age.days)}d ago"
            else:
                age_str = ""
            
            # Extract image
            image_url = _extract_image_from_entry(entry)
            
            article = {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "age": age_str,
                "summary": (entry.get("summary", "") or "")[:200],
                "source": feed.feed.get("title", "Unknown"),
                "image": image_url,
            }
            articles.append(article)
            
            if len(articles) >= max_items:
                break
        
        return articles
    except Exception as e:
        return []

def get_category_news(category: str, max_per_source: int = 5) -> List[Dict]:
    """Get news from RSS feeds for a category."""
    feeds = RSS_FEEDS.get(category, [])
    all_articles = []
    
    for source_name, url in feeds:
        articles = get_rss_feed(url, max_items=max_per_source)
        for article in articles:
            article["source"] = source_name
            all_articles.append(article)
    
    # Sort by published date (newest first)
    all_articles.sort(key=lambda x: x.get("published", ""), reverse=True)
    
    return all_articles

def get_multi_category_news(categories: List[str], max_per_category: int = 10) -> Dict[str, List[Dict]]:
    """Get news from multiple categories."""
    result = {}
    for category in categories:
        result[category] = get_category_news(category, max_per_source=max_per_category // len(RSS_FEEDS.get(category, [("", "")])))
    return result

# ============================================================
# Personalized News
# ============================================================

def get_personalized_feed(
    interests: List[str] = None,
    keywords: List[str] = None
) -> List[Dict]:
    """
    Get a personalized news feed based on interests and keywords.
    """
    if not interests:
        interests = ["tech", "uk_news", "sports"]
    
    all_articles = []
    
    # Get from RSS feeds for interests
    for interest in interests:
        if interest in RSS_FEEDS:
            articles = get_category_news(interest, max_per_source=3)
            all_articles.extend(articles)
    
    # If NewsAPI key available, also search for keywords
    if keywords and _newsapi_key():
        for keyword in keywords[:3]:  # Limit to avoid rate limits
            result = search_news(keyword, page_size=5)
            if result.get("articles"):
                for article in result["articles"]:
                    all_articles.append({
                        "title": article.get("title", ""),
                        "link": article.get("url", ""),
                        "published": article.get("publishedAt", ""),
                        "summary": (article.get("description", "") or "")[:200],
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "image": article.get("urlToImage"),
                    })
    
    # Deduplicate by title
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title = article.get("title", "").lower()[:50]
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)
    
    # Sort by date
    unique_articles.sort(key=lambda x: x.get("published", ""), reverse=True)
    
    return unique_articles[:30]

# ============================================================
# UI Components
# ============================================================

def render_article_card(article: Dict, show_image: bool = False):
    """Render a news article card."""
    title = article.get("title", "Untitled")
    link = article.get("link", "")
    source = article.get("source", "Unknown")
    summary = article.get("summary", "")
    published = article.get("published", "")
    image = article.get("image")
    
    with st.container():
        if show_image and image:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(image, width=120)
            with col2:
                st.markdown(f"**[{title}]({link})**")
                st.caption(f"📰 {source} • {published[:16] if published else 'N/A'}")
                if summary:
                    st.caption(summary)
        else:
            st.markdown(f"**[{title}]({link})**")
            st.caption(f"📰 {source} • {published[:16] if published else 'N/A'}")
            if summary:
                st.caption(summary)
        st.divider()

def render_news_by_category():
    """Render news organized by category."""
    st.markdown("### 📰 News by Category")
    
    available_categories = list(RSS_FEEDS.keys())
    
    selected_category = st.selectbox(
        "Category",
        available_categories,
        format_func=lambda x: x.replace("_", " ").title(),
        key="news_category"
    )
    
    articles = get_category_news(selected_category, max_per_source=5)
    
    if articles:
        for article in articles[:15]:
            render_article_card(article)
    else:
        st.info("No articles found for this category")

def render_headlines():
    """Render top headlines from NewsAPI."""
    st.markdown("### 📰 Top Headlines")
    
    if not _newsapi_key():
        st.warning("Add NEWSAPI_KEY to secrets for more headlines")
        st.info("Get a free API key at: https://newsapi.org/")
        # Fall back to RSS
        articles = get_category_news("uk_news", max_per_source=5)
        for article in articles[:10]:
            render_article_card(article)
        return
    
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox(
            "Country",
            ["gb", "us", "au", "ca"],
            format_func=lambda x: {"gb": "UK", "us": "USA", "au": "Australia", "ca": "Canada"}.get(x, x.upper()),
            key="headlines_country"
        )
    with col2:
        category = st.selectbox(
            "Category",
            ["general"] + NEWS_CATEGORIES[1:],
            format_func=lambda x: x.title(),
            key="headlines_category"
        )
    
    result = get_top_headlines(country=country, category=category if category != "general" else None)
    
    if result.get("error"):
        st.error(f"Error: {result['error']}")
    elif result.get("articles"):
        for article in result["articles"][:15]:
            render_article_card({
                "title": article.get("title", ""),
                "link": article.get("url", ""),
                "published": article.get("publishedAt", ""),
                "summary": (article.get("description", "") or "")[:200],
                "source": article.get("source", {}).get("name", "Unknown"),
                "image": article.get("urlToImage"),
            }, show_image=True)
    else:
        st.info("No headlines found")

def render_news_search():
    """Render news search interface."""
    st.markdown("### 🔍 Search News")
    
    query = st.text_input("Search for...", placeholder="e.g., AI, Manchester United, Tech", key="news_search_query")
    
    if query:
        # Search RSS feeds
        all_results = []
        
        for category, feeds in RSS_FEEDS.items():
            for source_name, url in feeds:
                articles = get_rss_feed(url, max_items=20)
                for article in articles:
                    if query.lower() in article.get("title", "").lower() or query.lower() in article.get("summary", "").lower():
                        article["source"] = source_name
                        all_results.append(article)
        
        # Also search NewsAPI if available
        if _newsapi_key():
            api_result = search_news(query, page_size=10)
            if api_result.get("articles"):
                for article in api_result["articles"]:
                    all_results.append({
                        "title": article.get("title", ""),
                        "link": article.get("url", ""),
                        "published": article.get("publishedAt", ""),
                        "summary": (article.get("description", "") or "")[:200],
                        "source": article.get("source", {}).get("name", "Unknown"),
                    })
        
        # Deduplicate
        seen = set()
        unique = []
        for article in all_results:
            title = article.get("title", "")[:50].lower()
            if title and title not in seen:
                seen.add(title)
                unique.append(article)
        
        if unique:
            st.success(f"Found {len(unique)} results")
            for article in unique[:20]:
                render_article_card(article)
        else:
            st.info("No results found")

def render_personalized_feed():
    """Render personalized news feed."""
    st.markdown("### ⭐ Your Feed")
    
    # Let user select interests
    available_interests = list(RSS_FEEDS.keys())
    
    selected = st.multiselect(
        "Your Interests",
        available_interests,
        default=["tech", "uk_news", "sports"],
        format_func=lambda x: x.replace("_", " ").title(),
        key="user_interests"
    )
    
    keywords = st.text_input(
        "Custom Keywords (comma separated)",
        placeholder="e.g., Manchester United, AI, startups",
        key="user_keywords"
    )
    
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    
    articles = get_personalized_feed(interests=selected, keywords=keyword_list)
    
    if articles:
        for article in articles:
            render_article_card(article)
    else:
        st.info("No articles found. Try selecting more interests!")

def render_quick_feeds():
    """Render quick access to popular feeds."""
    st.markdown("### ⚡ Quick Feeds")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🇬🇧 UK News", key="quick_uk"):
            st.session_state["quick_feed"] = "uk_news"
    with col2:
        if st.button("💻 Tech", key="quick_tech"):
            st.session_state["quick_feed"] = "tech"
    with col3:
        if st.button("⚽ Football", key="quick_football"):
            st.session_state["quick_feed"] = "football"
    with col4:
        if st.button("🤖 AI", key="quick_ai"):
            st.session_state["quick_feed"] = "ai"
    
    if "quick_feed" in st.session_state:
        st.divider()
        articles = get_category_news(st.session_state["quick_feed"], max_per_source=5)
        for article in articles[:10]:
            render_article_card(article)

def render_news_dashboard():
    """Render the main news dashboard."""
    tabs = st.tabs(["⭐ Your Feed", "📰 Headlines", "📂 By Category", "🔍 Search"])
    
    with tabs[0]:
        render_personalized_feed()
    
    with tabs[1]:
        render_headlines()
    
    with tabs[2]:
        render_news_by_category()
    
    with tabs[3]:
        render_news_search()
