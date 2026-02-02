# pages/2_🎬_Entertainment.py
"""
Entertainment Dashboard - Letterboxd, Cinema Releases, Vue Listings, AI Recommendations.
"""
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import os

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

/* Movie release card */
.release-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    display: flex;
    gap: 0.75rem;
    align-items: center;
    transition: all 0.3s ease;
}

.release-card:hover {
    border-color: rgba(139, 92, 246, 0.4);
    transform: translateX(4px);
}

.release-date {
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    color: white;
    padding: 0.5rem;
    border-radius: 8px;
    text-align: center;
    min-width: 50px;
    font-weight: 700;
    font-size: 0.8rem;
}

.release-date .month {
    font-size: 0.65rem;
    text-transform: uppercase;
}

.release-date .day {
    font-size: 1.1rem;
    line-height: 1;
}

/* Vue cinema card */
.vue-card {
    background: linear-gradient(145deg, rgba(0, 82, 147, 0.2) 0%, rgba(0, 50, 100, 0.1) 100%);
    border: 1px solid rgba(0, 82, 147, 0.3);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}

.vue-card:hover {
    border-color: rgba(0, 82, 147, 0.6);
    box-shadow: 0 4px 15px rgba(0, 82, 147, 0.3);
}

/* Letterboxd styles */
.lb-watchlist-item {
    background: linear-gradient(135deg, rgba(255, 136, 0, 0.1) 0%, rgba(0, 210, 0, 0.05) 100%);
    border: 1px solid rgba(255, 136, 0, 0.2);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    transition: all 0.3s ease;
}

.lb-watchlist-item:hover {
    border-color: rgba(255, 136, 0, 0.5);
    transform: translateX(4px);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="ent-header">
    <h1>🎬 Entertainment Hub</h1>
    <p>✨ Cinema releases, Letterboxd, and AI recommendations</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Entertainment News Bar (Top)
# ============================================================

st.markdown("""
<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
    <span style="font-size: 1.5rem;">📰</span>
    <span style="font-weight: 800; font-size: 1.2rem; background: linear-gradient(135deg, #ec4899, #8b5cf6);
                 -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Entertainment News</span>
</div>
""", unsafe_allow_html=True)

try:
    news = et.get_entertainment_news()
    if news:
        news_cols = st.columns(4)
        for i, article in enumerate(news[:8]):
            with news_cols[i % 4]:
                title = article.get("title", "")
                if len(title) > 50:
                    title = title[:50] + "..."
                source = article.get("source", "")
                link = article.get("link", "#")
                
                st.markdown(f"""
                <a href="{link}" target="_blank" style="text-decoration: none;">
                    <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); 
                                border-radius: 12px; padding: 0.75rem; margin-bottom: 0.5rem; min-height: 80px;">
                        <div style="font-size: 0.7rem; color: #ec4899; font-weight: 700; text-transform: uppercase;">{source}</div>
                        <div style="font-size: 0.85rem; font-weight: 600; margin-top: 0.25rem; line-height: 1.3; color: white;">{title}</div>
                    </div>
                </a>
                """, unsafe_allow_html=True)
except Exception as e:
    st.caption(f"News loading...")

st.markdown("---")

# ============================================================
# Main 3-Column Layout
# ============================================================

left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

# ============================================================
# LEFT COLUMN: Letterboxd (Activity + Watchlist)
# ============================================================

with left_col:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">🎭</span>
        <span style="font-weight: 800; font-size: 1.2rem; background: linear-gradient(135deg, #ff8800, #00e054);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Letterboxd</span>
        <span style="font-size: 0.8rem; color: rgba(255,255,255,0.5);">@{LETTERBOXD_USER}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if LETTERBOXD_USER:
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", []) if isinstance(lb_data, dict) else []
            lb_watchlist = lb_data.get("watchlist", []) if isinstance(lb_data, dict) else []
            
            # Watchlist Tab and Activity Tab
            lb_tab1, lb_tab2 = st.tabs(["📋 Watchlist", "🎬 Recent Activity"])
            
            with lb_tab1:
                if lb_watchlist:
                    st.caption(f"{len(lb_watchlist)} films to watch")
                    for item in lb_watchlist[:15]:
                        title = item.get("title", "Unknown")
                        year = item.get("year", "")
                        
                        st.markdown(f"""
                        <div class="lb-watchlist-item">
                            <strong>{title}</strong> <span style="color: rgba(255,255,255,0.5);">({year})</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No watchlist found")
            
            with lb_tab2:
                if activity:
                    for item in activity[:8]:
                        title = item.get("title", "")
                        has_rating = item.get("has_rating", False)
                        rating_html = ' <span style="color: #00e054; font-weight: 700;">★</span>' if has_rating else ''
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(255, 136, 0, 0.15) 0%, rgba(0, 210, 0, 0.1) 100%);
                                    border: 1px solid rgba(255, 136, 0, 0.3); border-radius: 10px; padding: 0.6rem 0.8rem;
                                    margin-bottom: 0.4rem;">
                            <strong>{title}</strong>{rating_html}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No recent activity")
                    
        except Exception as e:
            st.caption(f"Could not load Letterboxd: {e}")
    else:
        st.caption("Add LETTERBOXD_USERNAME to secrets")
    
    st.divider()
    
    # In Cinemas Now (Vue / UK)
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">🎥</span>
        <span style="font-weight: 800; font-size: 1.2rem; background: linear-gradient(135deg, #005293, #00a8e8);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">At Vue Basingstoke</span>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Try Vue API first, fall back to now_playing
        vue_films = et.get_vue_cinema_listings("10032")  # Basingstoke
        
        if not vue_films:
            # Fallback to TMDb now_playing
            now_playing = et.get_now_playing("GB")
            vue_films = [{"title": m.get("title", ""), "rating": f"⭐ {m.get('vote_average', 0):.1f}"} for m in now_playing[:8]]
        
        if vue_films:
            for film in vue_films[:8]:
                title = film.get("title", "Unknown")
                rating = film.get("rating", "")
                
                st.markdown(f"""
                <div class="vue-card">
                    <strong>{title}</strong>
                    <span style="float: right; font-size: 0.8rem; color: rgba(255,255,255,0.6);">{rating}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Could not load cinema listings")
    except Exception as e:
        st.caption(f"Cinema listings: {e}")

# ============================================================
# MIDDLE COLUMN: Coming Soon + AI Coach
# ============================================================

with mid_col:
    # Coming Soon Section
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">🗓️</span>
        <span style="font-weight: 800; font-size: 1.2rem; background: linear-gradient(135deg, #f59e0b, #ef4444);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Coming to Cinemas</span>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        upcoming = et.get_upcoming_movies("GB", pages=3)
        
        if upcoming:
            # Filter to major releases (higher popularity)
            major = [m for m in upcoming if m.get("popularity", 0) > 15][:12]
            
            for movie in major:
                title = movie.get("title", "Unknown")
                release_date = movie.get("release_date", "")
                rating = movie.get("vote_average", 0)
                poster_path = movie.get("poster_path", "")
                
                # Parse date
                if release_date:
                    try:
                        dt = datetime.strptime(release_date, "%Y-%m-%d")
                        month = dt.strftime("%b").upper()
                        day = dt.strftime("%d")
                    except:
                        month = "TBA"
                        day = "?"
                else:
                    month = "TBA"
                    day = "?"
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(f"""
                    <div class="release-date">
                        <div class="month">{month}</div>
                        <div class="day">{day}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**{title}**")
                    if rating > 0:
                        st.caption(f"⭐ {rating:.1f}")
        else:
            st.caption("No upcoming releases found")
    except Exception as e:
        st.caption(f"Could not load upcoming: {e}")
    
    st.divider()
    
    # AI Movie Coach
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">🤖</span>
        <span style="font-weight: 800; font-size: 1.2rem; background: linear-gradient(135deg, #8b5cf6, #ec4899);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">AI Movie Coach</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "ent_chat" not in st.session_state:
        st.session_state.ent_chat = []
    
    # Quick action buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎯 Recommend", key="ent_recommend", use_container_width=True):
            st.session_state.ent_pending = "Based on my Letterboxd activity and watchlist, recommend 5 movies I should watch next. Consider my taste and explain why each would suit me."
    with c2:
        if st.button("🎲 Tonight?", key="ent_mood", use_container_width=True):
            st.session_state.ent_pending = "I need help deciding what to watch tonight. Ask me about my mood and suggest something perfect from my watchlist or something new."
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

Based on this viewing history and watchlist, you can understand the user's taste.

When recommending:
- Consider their apparent preferences
- Suggest films similar to ones they've enjoyed
- Mix well-known and hidden gems
- Be specific about WHY they'd like each recommendation
- Group suggestions by mood/genre when appropriate
- If they ask about their watchlist, help prioritize what to watch

Be conversational, enthusiastic about film, and helpful. Keep responses concise but insightful."""
        
        # Call AI
        client = get_openai_client()
        if client:
            try:
                messages = [{"role": "system", "content": context}]
                messages.extend(st.session_state.ent_chat)
                
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1000
                )
                reply = resp.choices[0].message.content
                st.session_state.ent_chat.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.ent_chat.append({"role": "assistant", "content": f"Error: {e}"})
        else:
            st.session_state.ent_chat.append({"role": "assistant", "content": "OpenAI API not configured."})
        
        st.rerun()
    
    # Display chat history
    if st.session_state.ent_chat:
        chat_container = st.container(height=300)
        with chat_container:
            for msg in reversed(st.session_state.ent_chat):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
    else:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; color: rgba(255,255,255,0.5); 
                    background: rgba(139, 92, 246, 0.1); border-radius: 12px;">
            <p style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎬</p>
            <p><strong>Your AI Movie Coach</strong></p>
            <p style="font-size: 0.85rem;">Get personalized recommendations based on your Letterboxd!</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# RIGHT COLUMN: Trending & Search
# ============================================================

with right_col:
    # Trending Movies
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">🔥</span>
        <span style="font-weight: 800; font-size: 1.2rem; background: linear-gradient(135deg, #f59e0b, #ef4444);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Trending Now</span>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        trending = et.get_trending_movies()
        if trending:
            cols = st.columns(3)
            for i, movie in enumerate(trending[:6]):
                with cols[i % 3]:
                    poster_path = movie.get("poster_path")
                    title = movie.get("title", "Unknown")
                    rating = movie.get("vote_average", 0)
                    
                    if poster_path:
                        poster_url = et.get_poster_url(poster_path, "w342")
                        st.image(poster_url, use_column_width=True)
                    
                    st.caption(f"**{title[:15]}...**" if len(title) > 15 else f"**{title}**")
                    st.caption(f"⭐ {rating:.1f}")
    except Exception as e:
        st.caption(f"Could not load trending: {e}")
    
    st.divider()
    
    # Quick Search
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">🔍</span>
        <span style="font-weight: 800; font-size: 1.2rem; background: linear-gradient(135deg, #60a5fa, #8b5cf6);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Quick Search</span>
    </div>
    """, unsafe_allow_html=True)
    
    search_query = st.text_input("Search movies or TV shows...", key="ent_search", label_visibility="collapsed")
    
    if search_query:
        search_type = st.radio("Type", ["Movies", "TV Shows"], horizontal=True, key="search_type", label_visibility="collapsed")
        
        try:
            if search_type == "Movies":
                results = et.search_movie(search_query)
            else:
                results = et.search_tv(search_query)
            
            if results:
                for item in results[:6]:
                    title = item.get("title") or item.get("name", "Unknown")
                    year = (item.get("release_date") or item.get("first_air_date") or "")[:4]
                    rating = item.get("vote_average", 0)
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); 
                                border-radius: 10px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;">
                        <strong>{title}</strong> <span style="color: rgba(255,255,255,0.5);">({year})</span>
                        <span style="float: right;">⭐ {rating:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
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
