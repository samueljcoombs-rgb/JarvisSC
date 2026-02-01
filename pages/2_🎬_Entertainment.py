# pages/2_🎬_Entertainment.py
"""
Entertainment Dashboard - Movies, TV Shows, Letterboxd integration with AI recommendations.
Premium dashboard layout - no tabs, everything visible.
"""
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import json

# Import modules
try:
    from modules import sheets_memory as sm
    from modules import entertainment_tools as et
    from modules import global_styles as gs
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules import sheets_memory as sm
    from modules import entertainment_tools as et
    from modules import global_styles as gs

TZ = ZoneInfo("Europe/London")
LETTERBOXD_USER = os.getenv("LETTERBOXD_USERNAME") or st.secrets.get("LETTERBOXD_USERNAME", "SamECee")

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Entertainment | Jarvis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

gs.inject_global_styles()

# ============================================================
# OpenAI Client for AI Coach
# ============================================================

def get_openai_client():
    """Get OpenAI client."""
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
    except Exception:
        pass
    return None

# ============================================================
# Custom Styling - Premium Entertainment Theme
# ============================================================

st.markdown("""
<style>
/* Animated gradient header */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

.ent-header {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    padding: 2rem 2.5rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(139, 92, 246, 0.3);
    position: relative;
    overflow: hidden;
}

.ent-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 200%;
    background: radial-gradient(circle, rgba(236, 72, 153, 0.15) 0%, transparent 60%);
    animation: float 6s ease-in-out infinite;
}

.ent-header h1 {
    color: white;
    margin: 0;
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 1;
}

.ent-header p {
    color: rgba(255,255,255,0.8);
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    position: relative;
    z-index: 1;
}

/* News ticker/carousel */
.news-ticker {
    background: linear-gradient(90deg, rgba(236, 72, 153, 0.1), rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    overflow: hidden;
}

.news-item {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-right: 1rem;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
    max-width: 300px;
}

.news-item:hover {
    background: rgba(139, 92, 246, 0.2);
    border-color: rgba(139, 92, 246, 0.4);
    transform: translateY(-2px);
}

.news-item .source {
    font-size: 0.7rem;
    color: #ec4899;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.news-item .title {
    font-size: 0.85rem;
    color: white;
    font-weight: 600;
    margin-top: 0.25rem;
    line-height: 1.3;
}

/* Poster Grid - Letterboxd Style */
.poster-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 1rem;
}

.poster-card {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    aspect-ratio: 2/3;
    background: rgba(255,255,255,0.05);
    transition: all 0.3s ease;
    cursor: pointer;
}

.poster-card:hover {
    transform: scale(1.05);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    z-index: 10;
}

.poster-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.poster-card .overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.9));
    padding: 2rem 0.5rem 0.5rem;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.poster-card:hover .overlay {
    opacity: 1;
}

.poster-card .title {
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    line-height: 1.2;
}

.poster-card .rating {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    font-size: 0.7rem;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 6px;
}

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.section-header .icon {
    font-size: 1.5rem;
}

.section-header .text {
    font-weight: 800;
    font-size: 1.2rem;
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* AI Coach Card */
.ai-coach-card {
    background: linear-gradient(145deg, rgba(139, 92, 246, 0.15) 0%, rgba(236, 72, 153, 0.1) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Letterboxd Activity Card */
.lb-activity {
    background: linear-gradient(135deg, rgba(255, 136, 0, 0.15) 0%, rgba(0, 210, 0, 0.1) 100%);
    border: 1px solid rgba(255, 136, 0, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 1rem;
    align-items: center;
}

.lb-activity:hover {
    transform: translateX(4px);
    border-color: rgba(255, 136, 0, 0.5);
}

.lb-activity .rating-stars {
    color: #00e054;
    font-weight: 700;
}

/* Watchlist Item */
.watchlist-item {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.3s ease;
}

.watchlist-item:hover {
    background: rgba(139, 92, 246, 0.1);
    border-color: rgba(139, 92, 246, 0.3);
}

/* Scrollable Chat Container */
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    padding-right: 0.5rem;
}

.chat-container::-webkit-scrollbar {
    width: 6px;
}

.chat-container::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
}

.chat-container::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.5);
    border-radius: 3px;
}

/* Footer nav */
.footer-nav {
    display: flex;
    justify-content: center;
    gap: 1rem;
    padding: 1rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="ent-header">
    <h1>🎬 Entertainment Hub</h1>
    <p>✨ Your movies, shows, and personalized AI recommendations</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Entertainment News Bar (Top)
# ============================================================

st.markdown("""
<div class="section-header">
    <span class="icon">📰</span>
    <span class="text">Entertainment News</span>
</div>
""", unsafe_allow_html=True)

try:
    news = et.get_entertainment_news()
    if news:
        # Display as horizontal scrollable row
        news_html = '<div class="news-ticker">'
        for article in news[:8]:
            title = article.get("title", "")[:60] + "..." if len(article.get("title", "")) > 60 else article.get("title", "")
            source = article.get("source", "")
            link = article.get("link", "#")
            news_html += f'''
            <a href="{link}" target="_blank" style="text-decoration: none;">
                <div class="news-item">
                    <div class="source">{source}</div>
                    <div class="title">{title}</div>
                </div>
            </a>
            '''
        news_html += '</div>'
        st.markdown(news_html, unsafe_allow_html=True)
    else:
        st.caption("No news available")
except Exception as e:
    st.caption(f"News loading... ({e})")

st.markdown("---")

# ============================================================
# Main 3-Column Layout
# ============================================================

left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

# ============================================================
# LEFT COLUMN: Watchlist & Letterboxd
# ============================================================

with left_col:
    # My Watchlist Section
    st.markdown("""
    <div class="section-header">
        <span class="icon">📋</span>
        <span class="text">My Watchlist</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Get watchlist from sheets
    try:
        watchlist = sm.get_watchlist(status="to_watch")
        if watchlist:
            for item in watchlist[:8]:
                title = item.get("title", "Unknown")
                content_type = item.get("type", "movie")
                icon = "🎬" if content_type == "movie" else "📺"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"{icon} **{title}**")
                with col2:
                    if st.button("✓", key=f"watched_{item.get('id', title)}", help="Mark as watched"):
                        sm.update_watchlist_item(item.get("id"), {"status": "watched"})
                        st.rerun()
        else:
            st.caption("Your watchlist is empty. Search for movies below!")
    except Exception as e:
        st.caption(f"Watchlist: {e}")
    
    # Add to watchlist
    with st.expander("➕ Add to Watchlist"):
        new_title = st.text_input("Movie/Show title", key="add_wl_title")
        new_type = st.selectbox("Type", ["movie", "tv"], key="add_wl_type")
        if st.button("Add", key="add_wl_btn"):
            if new_title:
                sm.add_to_watchlist(title=new_title, content_type=new_type, status="to_watch")
                st.success(f"Added {new_title}!")
                st.rerun()
    
    st.divider()
    
    # Letterboxd Section
    st.markdown("""
    <div class="section-header">
        <span class="icon">🎭</span>
        <span class="text">Letterboxd</span>
    </div>
    """, unsafe_allow_html=True)
    
    if LETTERBOXD_USER:
        st.caption(f"@{LETTERBOXD_USER}")
        
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", []) if isinstance(lb_data, dict) else []
            lb_watchlist = lb_data.get("watchlist", []) if isinstance(lb_data, dict) else []
            
            # Recent Activity
            if activity:
                st.markdown("**Recent Activity:**")
                for item in activity[:5]:
                    title = item.get("title", "")
                    has_rating = item.get("has_rating", False)
                    
                    st.markdown(f"""
                    <div class="lb-activity">
                        <div>
                            <strong>{title}</strong>
                            {' <span class="rating-stars">★</span>' if has_rating else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Letterboxd Watchlist Count
            if lb_watchlist:
                st.caption(f"📋 {len(lb_watchlist)} films in Letterboxd watchlist")
                
        except Exception as e:
            st.caption(f"Could not load: {e}")
    else:
        st.caption("Add LETTERBOXD_USERNAME to secrets")

# ============================================================
# MIDDLE COLUMN: AI Movie Coach
# ============================================================

with mid_col:
    st.markdown("""
    <div class="section-header">
        <span class="icon">🤖</span>
        <span class="text">AI Movie Coach</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "ent_chat" not in st.session_state:
        st.session_state.ent_chat = []
    
    # Quick action buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎯 Recommend", key="ent_recommend", use_container_width=True):
            st.session_state.ent_pending = "Based on my Letterboxd activity and watchlist, recommend 5 movies I should watch next. Consider my taste and what I've enjoyed recently."
    with c2:
        if st.button("🎲 Mood Pick", key="ent_mood", use_container_width=True):
            st.session_state.ent_pending = "I can't decide what to watch tonight. Ask me about my mood and suggest something perfect."
    with c3:
        if st.button("🗑️ Clear", key="ent_clear", use_container_width=True):
            st.session_state.ent_chat = []
            st.rerun()
    
    # Chat input
    user_input = st.chat_input("Ask about movies, get recommendations...", key="ent_chat_input")
    
    # Handle pending button actions
    if "ent_pending" in st.session_state and st.session_state.ent_pending:
        user_input = st.session_state.ent_pending
        st.session_state.ent_pending = None
    
    if user_input:
        st.session_state.ent_chat.append({"role": "user", "content": user_input})
        
        # Build context with Letterboxd data
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", [])[:10] if isinstance(lb_data, dict) else []
            lb_watchlist = lb_data.get("watchlist", [])[:20] if isinstance(lb_data, dict) else []
            
            activity_text = "\n".join([f"- {a.get('title', '')}" for a in activity])
            watchlist_text = "\n".join([f"- {w.get('title', '')} ({w.get('year', '')})" for w in lb_watchlist])
        except:
            activity_text = "No activity available"
            watchlist_text = "No watchlist available"
        
        context = f"""You are an expert film critic and recommendation AI. You have deep knowledge of cinema from all eras and countries.

USER'S LETTERBOXD DATA:
Recent Activity (watched/rated):
{activity_text}

Watchlist (wants to watch):
{watchlist_text}

Based on this viewing history and watchlist, you can understand the user's taste. They seem to enjoy films based on what they've watched and what they want to watch.

When recommending:
- Consider their apparent preferences
- Suggest films similar to ones they've enjoyed
- Mix well-known and hidden gems
- Be specific about WHY they'd like each recommendation
- Group suggestions by mood/genre when appropriate

Be conversational, enthusiastic about film, and helpful."""
        
        # Call AI
        client = get_openai_client()
        if client:
            try:
                messages = [{"role": "system", "content": context}]
                messages.extend(st.session_state.ent_chat)
                
                # Try GPT-5.1 first
                if hasattr(client, "responses"):
                    resp = client.responses.create(
                        model="gpt-5.1",
                        input=messages,
                        max_output_tokens=1500,
                    )
                    if hasattr(resp, 'output_text'):
                        reply = resp.output_text
                    else:
                        reply = resp.output[0].content[0].text if resp.output else "No response"
                else:
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=1500
                    )
                    reply = resp.choices[0].message.content
                
                st.session_state.ent_chat.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.ent_chat.append({"role": "assistant", "content": f"Error: {e}"})
        else:
            st.session_state.ent_chat.append({"role": "assistant", "content": "OpenAI API not configured."})
        
        st.rerun()
    
    # Display chat history
    st.markdown("---")
    if st.session_state.ent_chat:
        chat_container = st.container(height=400)
        with chat_container:
            for msg in reversed(st.session_state.ent_chat):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.5);">
            <p style="font-size: 2rem;">🎬</p>
            <p><strong>Your AI Movie Coach</strong></p>
            <p style="font-size: 0.9rem;">Ask for recommendations, discuss films, or get help deciding what to watch!</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# RIGHT COLUMN: Trending & Search
# ============================================================

with right_col:
    # Trending Movies
    st.markdown("""
    <div class="section-header">
        <span class="icon">🔥</span>
        <span class="text">Trending Now</span>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        trending = et.get_trending_movies()
        if trending:
            # Display as poster grid
            cols = st.columns(3)
            for i, movie in enumerate(trending[:6]):
                with cols[i % 3]:
                    poster_path = movie.get("poster_path")
                    title = movie.get("title", "Unknown")
                    rating = movie.get("vote_average", 0)
                    year = (movie.get("release_date") or "")[:4]
                    
                    if poster_path:
                        poster_url = et.get_poster_url(poster_path, "w342")
                        st.image(poster_url, use_column_width=True)
                    
                    st.caption(f"**{title}** ({year})")
                    st.caption(f"⭐ {rating:.1f}")
    except Exception as e:
        st.caption(f"Could not load trending: {e}")
    
    st.divider()
    
    # Quick Search
    st.markdown("""
    <div class="section-header">
        <span class="icon">🔍</span>
        <span class="text">Quick Search</span>
    </div>
    """, unsafe_allow_html=True)
    
    search_query = st.text_input("Search movies or TV shows...", key="ent_search", label_visibility="collapsed")
    
    if search_query:
        search_type = st.radio("Type", ["Movies", "TV Shows"], horizontal=True, key="search_type")
        
        try:
            if search_type == "Movies":
                results = et.search_movie(search_query)
            else:
                results = et.search_tv(search_query)
            
            if results:
                for item in results[:5]:
                    title = item.get("title") or item.get("name", "Unknown")
                    year = (item.get("release_date") or item.get("first_air_date") or "")[:4]
                    rating = item.get("vote_average", 0)
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{title}** ({year}) ⭐ {rating:.1f}")
                    with col2:
                        if st.button("➕", key=f"add_{item.get('id', title)}", help="Add to watchlist"):
                            sm.add_to_watchlist(
                                title=title,
                                content_type="movie" if search_type == "Movies" else "tv",
                                status="to_watch"
                            )
                            st.success("Added!")
            else:
                st.caption("No results found")
        except Exception as e:
            st.caption(f"Search error: {e}")

# ============================================================
# Footer Navigation
# ============================================================

st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
with col2:
    if st.button("🏋️ Health", use_container_width=True):
        st.switch_page("pages/1_🏋️_Health_Fitness.py")
with col3:
    if st.button("🎯 Goals", use_container_width=True):
        st.switch_page("pages/3_🎯_Goals.py")
with col4:
    if st.button("✈️ Travel", use_container_width=True):
        st.switch_page("pages/4_✈️_Travel.py")
with col5:
    if st.button("📰 News", use_container_width=True):
        st.switch_page("pages/5_📰_News.py")
