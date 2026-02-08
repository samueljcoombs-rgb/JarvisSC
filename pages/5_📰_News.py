# pages/5_📰_News.py
"""
Personalized News Dashboard - Curated news from RSS feeds and NewsAPI.
"""
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import feedparser
import requests

# Import modules
try:
    from modules import sheets_memory as sm
    from modules import news_tools as nt
    from modules import global_styles as gs
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules import sheets_memory as sm
    from modules import news_tools as nt
    from modules import global_styles as gs

TZ = ZoneInfo("Europe/London")

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="News | Jarvis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

gs.inject_global_styles()

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>
.news-header {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 50%, #b45309 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.3);
}
.news-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
}
.news-header p {
    color: rgba(255,255,255,0.85);
    margin: 0.25rem 0 0 0;
}
.article-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}
.article-card:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(245, 158, 11, 0.3);
}
.article-title {
    font-weight: 600;
    color: white;
    font-size: 1.05rem;
    margin-bottom: 0.5rem;
}
.article-meta {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    margin-bottom: 0.5rem;
}
.article-summary {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.7);
    line-height: 1.5;
}
.category-chip {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
}
.category-chip:hover {
    background: rgba(245, 158, 11, 0.2);
    border-color: rgba(245, 158, 11, 0.4);
}
.category-chip.active {
    background: rgba(245, 158, 11, 0.3);
    border-color: #f59e0b;
    color: #fcd34d;
}
.source-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 8px;
    font-size: 0.7rem;
    font-weight: 600;
    background: rgba(139, 92, 246, 0.3);
    color: #c4b5fd;
}
.quick-feed-btn {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
}
.quick-feed-btn:hover {
    background: rgba(255,255,255,0.1);
    border-color: rgba(245, 158, 11, 0.3);
}
.empty-state {
    text-align: center;
    padding: 2rem;
    color: rgba(255,255,255,0.5);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="news-header">
    <h1>📰 News Feed</h1>
    <p>Stay informed with curated news from your favorite sources</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Quick Access Categories
# ============================================================

st.markdown("### Quick Access")
cols = st.columns(6)

categories = [
    ("🇬🇧", "UK News", "uk_news"),
    ("💻", "Tech", "tech"),
    ("⚽", "Football", "football"),
    ("🤖", "AI", "ai"),
    ("🎮", "Gaming", "gaming"),
    ("🔬", "Science", "science")
]

# Initialize session state
if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

for i, (emoji, name, key) in enumerate(categories):
    with cols[i]:
        if st.button(f"{emoji} {name}", key=f"cat_{key}", use_container_width=True):
            st.session_state.selected_category = key

# ============================================================
# Main Tabs
# ============================================================

tab_feed, tab_headlines, tab_category, tab_search = st.tabs([
    "🏠 My Feed", "📡 Headlines", "📂 By Category", "🔍 Search"
])

# ============================================================
# Helper: Render Article
# ============================================================

def render_article(article, show_source=True):
    title = article.get("title", "Untitled")
    link = article.get("link", "#")
    source = article.get("source", "")
    published = article.get("published", "")
    summary = article.get("summary", "")[:200] + "..." if len(article.get("summary", "")) > 200 else article.get("summary", "")
    image = article.get("image", "")
    
    # Format date
    date_str = ""
    if published:
        try:
            # Try various date formats
            for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d"]:
                try:
                    dt = datetime.strptime(published[:19], fmt[:len(published[:19])])
                    date_str = dt.strftime("%d %b %H:%M")
                    break
                except:
                    continue
            if not date_str:
                date_str = published[:16]
        except:
            date_str = published[:16]
    
    col1, col2 = st.columns([4, 1]) if image else (st.container(), None)
    
    with col1:
        st.markdown(f'<div class="article-title">{title}</div>', unsafe_allow_html=True)
        
        meta_parts = []
        if source and show_source:
            meta_parts.append(f'<span class="source-badge">{source}</span>')
        if date_str:
            meta_parts.append(date_str)
        
        if meta_parts:
            st.markdown(f'<div class="article-meta">{" · ".join(meta_parts)}</div>', unsafe_allow_html=True)
        
        if summary:
            st.markdown(f'<div class="article-summary">{summary}</div>', unsafe_allow_html=True)
        
        st.markdown(f"[Read more →]({link})")

# ============================================================
# My Feed Tab
# ============================================================

with tab_feed:
    # Interest selection
    st.markdown("#### Your Interests")
    
    all_interests = ["uk_news", "tech", "football", "ai", "gaming", "science", "business", "sports"]
    interest_labels = {
        "uk_news": "🇬🇧 UK News",
        "tech": "💻 Tech",
        "football": "⚽ Football",
        "ai": "🤖 AI",
        "gaming": "🎮 Gaming",
        "science": "🔬 Science",
        "business": "💼 Business",
        "sports": "🏆 Sports"
    }
    
    selected_interests = st.multiselect(
        "Select topics",
        all_interests,
        default=["uk_news", "tech", "football"],
        format_func=lambda x: interest_labels.get(x, x),
        label_visibility="collapsed"
    )
    
    custom_keywords = st.text_input(
        "Custom keywords (comma-separated)",
        placeholder="e.g., Manchester United, Anthropic, AI",
        key="feed_keywords"
    )
    
    if st.button("🔄 Refresh Feed", key="refresh_feed"):
        st.cache_data.clear()
    
    st.markdown("---")
    
    # Fetch and display news
    if selected_interests or custom_keywords:
        with st.spinner("Loading your personalized feed..."):
            articles = []
            
            # Fetch from RSS feeds for each interest
            for interest in selected_interests:
                try:
                    cat_articles = nt.get_category_news(interest, max_per_source=5)
                    articles.extend(cat_articles)
                except Exception as e:
                    st.warning(f"Could not load {interest}: {e}")
            
            # Add NewsAPI results for keywords if available
            if custom_keywords:
                newsapi_key = st.secrets.get("NEWSAPI_KEY") or os.getenv("NEWSAPI_KEY")
                if newsapi_key:
                    try:
                        for kw in custom_keywords.split(","):
                            kw = kw.strip()
                            if kw:
                                kw_articles = nt.search_news(kw, days_back=7)
                                articles.extend(kw_articles[:3])
                    except Exception as e:
                        st.warning(f"NewsAPI search failed: {e}")
            
            # Sort by date (newest first) and deduplicate
            seen_titles = set()
            unique_articles = []
            for a in articles:
                title = a.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(a)
            
            # Display articles
            if unique_articles:
                for article in unique_articles[:20]:
                    with st.container():
                        st.markdown('<div class="article-card">', unsafe_allow_html=True)
                        render_article(article)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty-state">No articles found. Try different interests or keywords.</div>', unsafe_allow_html=True)
    else:
        st.info("Select some interests above to see your personalized feed!")

# ============================================================
# Headlines Tab
# ============================================================

with tab_headlines:
    st.markdown("### Top Headlines")
    
    col1, col2 = st.columns(2)
    with col1:
        country = st.selectbox(
            "Country",
            ["gb", "us", "au", "ca"],
            format_func=lambda x: {"gb": "🇬🇧 UK", "us": "🇺🇸 US", "au": "🇦🇺 Australia", "ca": "🇨🇦 Canada"}.get(x, x),
            key="headlines_country"
        )
    with col2:
        category = st.selectbox(
            "Category",
            ["general", "business", "technology", "sports", "entertainment", "science", "health"],
            key="headlines_category"
        )
    
    newsapi_key = st.secrets.get("NEWSAPI_KEY") or os.getenv("NEWSAPI_KEY")
    
    if newsapi_key:
        with st.spinner("Fetching headlines..."):
            try:
                headlines = nt.get_top_headlines(country=country, category=category)
                
                if headlines:
                    for article in headlines[:15]:
                        with st.container():
                            st.markdown('<div class="article-card">', unsafe_allow_html=True)
                            render_article(article)
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No headlines found for this selection.")
            except Exception as e:
                st.error(f"Failed to fetch headlines: {e}")
    else:
        st.warning("⚠️ NewsAPI key not configured. Add NEWSAPI_KEY to secrets for headlines.")
        st.info("Using RSS feeds instead:")
        
        # Fallback to RSS
        rss_articles = nt.get_category_news("uk_news", max_per_source=10)
        for article in rss_articles:
            with st.container():
                st.markdown('<div class="article-card">', unsafe_allow_html=True)
                render_article(article)
                st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# By Category Tab
# ============================================================

with tab_category:
    # Use selected category from quick access or dropdown
    cat_options = list(nt.RSS_FEEDS.keys())
    cat_labels = {
        "tech": "💻 Tech",
        "uk_news": "🇬🇧 UK News",
        "business": "💼 Business",
        "sports": "🏆 Sports",
        "football": "⚽ Football",
        "science": "🔬 Science",
        "ai": "🤖 AI",
        "gaming": "🎮 Gaming"
    }
    
    # Check if quick access was used
    if st.session_state.selected_category:
        default_idx = cat_options.index(st.session_state.selected_category) if st.session_state.selected_category in cat_options else 0
    else:
        default_idx = 0
    
    selected_cat = st.selectbox(
        "Choose a category",
        cat_options,
        index=default_idx,
        format_func=lambda x: cat_labels.get(x, x.title()),
        key="browse_category"
    )
    
    # Clear quick access selection after using it
    if st.session_state.selected_category:
        st.session_state.selected_category = None
    
    with st.spinner(f"Loading {selected_cat}..."):
        try:
            cat_articles = nt.get_category_news(selected_cat, max_per_source=15)
            
            if cat_articles:
                # Group by source
                sources = {}
                for article in cat_articles:
                    src = article.get("source", "Other")
                    if src not in sources:
                        sources[src] = []
                    sources[src].append(article)
                
                for source, articles in sources.items():
                    st.markdown(f"#### {source}")
                    for article in articles[:5]:
                        with st.container():
                            st.markdown('<div class="article-card">', unsafe_allow_html=True)
                            render_article(article, show_source=False)
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info(f"No articles found for {selected_cat}")
        except Exception as e:
            st.error(f"Failed to load category: {e}")

# ============================================================
# Search Tab
# ============================================================

with tab_search:
    st.markdown("### 🔍 Search News")
    
    search_query = st.text_input("Search for...", placeholder="e.g., climate change, AI regulations", key="news_search")
    
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.slider("From last N days", 1, 30, 7, key="search_days")
    with col2:
        search_source = st.radio("Search in", ["RSS + NewsAPI", "RSS only", "NewsAPI only"], horizontal=True)
    
    if st.button("🔍 Search", use_container_width=True) and search_query:
        results = []
        
        with st.spinner("Searching..."):
            # RSS search
            if search_source in ["RSS + NewsAPI", "RSS only"]:
                for cat in nt.RSS_FEEDS.keys():
                    try:
                        cat_articles = nt.get_category_news(cat, max_per_source=10)
                        for article in cat_articles:
                            title = article.get("title", "").lower()
                            summary = article.get("summary", "").lower()
                            if search_query.lower() in title or search_query.lower() in summary:
                                results.append(article)
                    except:
                        continue
            
            # NewsAPI search
            if search_source in ["RSS + NewsAPI", "NewsAPI only"]:
                newsapi_key = st.secrets.get("NEWSAPI_KEY") or os.getenv("NEWSAPI_KEY")
                if newsapi_key:
                    try:
                        api_results = nt.search_news(search_query, days_back=days_back)
                        results.extend(api_results)
                    except Exception as e:
                        st.warning(f"NewsAPI search failed: {e}")
                elif search_source == "NewsAPI only":
                    st.warning("NewsAPI key not configured")
            
            # Deduplicate
            seen = set()
            unique = []
            for r in results:
                t = r.get("title", "")
                if t and t not in seen:
                    seen.add(t)
                    unique.append(r)
            
            if unique:
                st.success(f"Found {len(unique)} articles")
                for article in unique[:20]:
                    with st.container():
                        st.markdown('<div class="article-card">', unsafe_allow_html=True)
                        render_article(article)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No articles found matching your search.")

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("💡 News is refreshed from RSS feeds and NewsAPI. Some sources may have a delay.")
