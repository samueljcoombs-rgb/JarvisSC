# pages/2_🎬_Entertainment.py
"""
Entertainment Dashboard - Cinema, Letterboxd, Games, AI Recommendations.
"""
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import re

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

# Vue Basingstoke URLs
VUE_CINEMA_SLUG = "basingstoke-festival-place"
VUE_WHATS_ON_URL = f"https://www.myvue.com/cinema/{VUE_CINEMA_SLUG}/whats-on"

def make_vue_film_url(title: str) -> str:
    """Create Vue film URL from title."""
    # Convert to lowercase, replace spaces with hyphens, remove special chars
    slug = title.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars except hyphens
    slug = re.sub(r'\s+', '-', slug)  # Replace spaces with hyphens
    slug = re.sub(r'-+', '-', slug)  # Remove multiple hyphens
    return f"https://www.myvue.com/cinema/{VUE_CINEMA_SLUG}/film/{slug}"

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Entertainment | Jarvis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

gs.inject_global_styles()

# ============================================================
# OpenAI Client
# ============================================================

def get_openai_client():
    try:
        from openai import OpenAI
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
    except:
        pass
    return None

# ============================================================
# Sidebar - News & Settings
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📰 News Preferences")
    available_sources = ["IGN", "Variety", "The Verge", "Polygon", "Kotaku", "Eurogamer", "Screen Rant", "Collider"]
    news_sources = st.multiselect(
        "Show news from:",
        available_sources,
        default=["IGN", "Variety", "The Verge", "Polygon"],
        key="news_sources"
    )
    news_count = st.slider("Number of articles", 5, 30, 20, key="news_count")
    
    st.divider()
    
    st.subheader("🎥 Cinema")
    st.markdown(f"[🎟️ Vue Basingstoke - What's On]({VUE_WHATS_ON_URL})")
    
    st.divider()
    
    st.subheader("🎭 Letterboxd")
    st.markdown(f"[📋 {LETTERBOXD_USER}'s Watchlist](https://letterboxd.com/{LETTERBOXD_USER}/watchlist/)")
    st.markdown(f"[🎬 {LETTERBOXD_USER}'s Profile](https://letterboxd.com/{LETTERBOXD_USER}/)")
    
    # Debug toggle
    show_debug = st.checkbox("Show debug info", key="show_debug")

# ============================================================
# Header
# ============================================================

st.markdown("""
<div style="background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e); padding: 2rem 2.5rem; 
            border-radius: 24px; margin-bottom: 1.5rem; box-shadow: 0 20px 60px rgba(139, 92, 246, 0.3);">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 900; 
               background: linear-gradient(135deg, #ec4899, #8b5cf6, #60a5fa);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎬 Entertainment Hub</h1>
    <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">✨ Cinema, Letterboxd, Games & AI recommendations</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Entertainment News - Full Width, Scrollable
# ============================================================

with st.expander("📰 **Entertainment News** - Customize in sidebar ⚙️", expanded=True):
    try:
        news = et.get_entertainment_news()
        if news:
            # Filter by selected sources
            if news_sources:
                filtered_news = [n for n in news if n.get("source") in news_sources]
                if len(filtered_news) < 5:
                    filtered_news = news  # Fall back if filter too restrictive
            else:
                filtered_news = news
            
            # Scrollable container with more articles
            news_container = st.container(height=280)
            with news_container:
                # Display in 2 columns for better readability
                cols = st.columns(2)
                for i, article in enumerate(filtered_news[:news_count]):
                    with cols[i % 2]:
                        title = article.get("title", "No title")
                        source = article.get("source", "")
                        link = article.get("link", "")
                        
                        # Source badge + full title
                        if link:
                            st.markdown(f"**{source}** · [{title}]({link})")
                        else:
                            st.markdown(f"**{source}** · {title}")
        else:
            st.info("No news available - check your internet connection")
    except Exception as e:
        st.warning(f"News error: {e}")

st.divider()

# ============================================================
# Main 3-Column Layout
# ============================================================

left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

# ============================================================
# LEFT COLUMN: In Cinemas + Coming Soon
# ============================================================

with left_col:
    # In Cinemas Now with Posters (smaller)
    st.subheader("🎥 In Cinemas Now")
    st.markdown(f"[🎟️ Book at Vue Basingstoke]({VUE_WHATS_ON_URL})")
    
    try:
        now_playing = et.get_now_playing("GB")
        
        if now_playing:
            # Sort by rating (highest first)
            sorted_films = sorted(now_playing, key=lambda x: x.get("vote_average", 0), reverse=True)[:8]
            
            # Display in 3 columns with smaller posters
            cols = st.columns(3)
            for i, film in enumerate(sorted_films):
                with cols[i % 3]:
                    title = film.get("title", "Unknown")
                    rating = film.get("vote_average", 0)
                    poster_path = film.get("poster_path")
                    
                    # Smaller poster (w185 instead of w342)
                    if poster_path:
                        poster_url = et.get_poster_url(poster_path, "w185")
                        st.image(poster_url, use_column_width=True)
                    
                    # Title with link to Vue
                    vue_url = make_vue_film_url(title)
                    short_title = title[:18] + "..." if len(title) > 18 else title
                    st.markdown(f"[**{short_title}**]({vue_url})")
                    st.caption(f"⭐ {rating:.1f}")
        else:
            st.caption("No films found")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # Coming Soon
    st.subheader("🗓️ Coming Soon")
    
    try:
        upcoming = et.get_upcoming_movies("GB", pages=3)
        
        if upcoming:
            major = sorted([m for m in upcoming if m.get("popularity", 0) > 10], 
                          key=lambda x: x.get("release_date", "9999"))[:10]
            
            for movie in major:
                title = movie.get("title", "Unknown")
                release_date = movie.get("release_date", "")
                tmdb_id = movie.get("id")
                
                if release_date:
                    try:
                        dt = datetime.strptime(release_date, "%Y-%m-%d")
                        date_str = dt.strftime("%d %b")
                    except:
                        date_str = "TBA"
                else:
                    date_str = "TBA"
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"**{date_str}**")
                with col2:
                    if tmdb_id:
                        st.markdown(f"[{title}](https://www.themoviedb.org/movie/{tmdb_id})")
                    else:
                        st.write(title)
        else:
            st.caption("No upcoming releases")
    except Exception as e:
        st.caption(f"Error: {e}")

# ============================================================
# MIDDLE COLUMN: AI Coach + Games
# ============================================================

with mid_col:
    st.subheader("🤖 AI Movie Coach")
    
    if "ent_chat" not in st.session_state:
        st.session_state.ent_chat = []
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎯 Recommend", key="ent_rec", use_container_width=True):
            st.session_state.ent_pending = "Based on my Letterboxd, recommend 5 films I'd love. Explain why each suits my taste."
    with c2:
        if st.button("🎲 Tonight?", key="ent_tonight", use_container_width=True):
            st.session_state.ent_pending = "Help me decide what to watch tonight. Ask about my mood first."
    with c3:
        if st.button("🗑️ Clear", key="ent_clear", use_container_width=True):
            st.session_state.ent_chat = []
            st.rerun()
    
    user_input = st.chat_input("Ask about movies...", key="ent_input")
    
    if "ent_pending" in st.session_state and st.session_state.ent_pending:
        user_input = st.session_state.ent_pending
        st.session_state.ent_pending = None
    
    if user_input:
        st.session_state.ent_chat.append({"role": "user", "content": user_input})
        
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", [])[:10] if isinstance(lb_data, dict) else []
            lb_watchlist = lb_data.get("watchlist", [])[:20] if isinstance(lb_data, dict) else []
            activity_text = "\n".join([f"- {a.get('title', '')}" for a in activity])
            watchlist_text = "\n".join([f"- {w.get('title', '')} ({w.get('year', '')})" for w in lb_watchlist])
        except:
            activity_text = "No data"
            watchlist_text = "No data"
        
        context = f"""You're an expert film critic and recommendation AI.

USER'S LETTERBOXD:
Recent (watched/rated):
{activity_text}

Watchlist:
{watchlist_text}

Give personalized recommendations. Be specific about WHY they'd like each film."""
        
        client = get_openai_client()
        if client:
            try:
                messages = [{"role": "system", "content": context}]
                messages.extend(st.session_state.ent_chat)
                resp = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=800)
                reply = resp.choices[0].message.content
                st.session_state.ent_chat.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.ent_chat.append({"role": "assistant", "content": f"Error: {e}"})
        else:
            st.session_state.ent_chat.append({"role": "assistant", "content": "OpenAI not configured."})
        st.rerun()
    
    if st.session_state.ent_chat:
        chat_box = st.container(height=220)
        with chat_box:
            for msg in reversed(st.session_state.ent_chat):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
    else:
        st.info("💡 Ask for recommendations based on your Letterboxd!")
    
    st.divider()
    
    # Games To Play
    st.subheader("🎮 Games To Play")
    
    try:
        games_data = sm.get_watchlist(status="to_watch")
        games = [g for g in games_data if g.get("type") == "game"] if games_data else []
    except:
        games = []
    
    if games:
        for game in games[:6]:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"🎮 {game.get('title', '?')}")
            with col2:
                if st.button("✓", key=f"game_{game.get('id')}"):
                    try:
                        sm.update_watchlist_item(game.get("id"), {"status": "completed"})
                        st.rerun()
                    except:
                        pass
    else:
        st.caption("No games yet")
    
    with st.expander("➕ Add Game"):
        new_game = st.text_input("Game title", key="new_game")
        if st.button("Add", key="add_game"):
            if new_game:
                try:
                    sm.add_to_watchlist(title=new_game, media_type="game")
                    st.success(f"Added!")
                    st.rerun()
                except Exception as e:
                    st.error(f"{e}")

# ============================================================
# RIGHT COLUMN: Letterboxd + Trending + Search
# ============================================================

with right_col:
    st.subheader("🎭 Letterboxd")
    
    # Quick links
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"[📋 Watchlist](https://letterboxd.com/{LETTERBOXD_USER}/watchlist/)")
    with col2:
        st.markdown(f"[🎬 Profile](https://letterboxd.com/{LETTERBOXD_USER}/)")
    
    if LETTERBOXD_USER:
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", []) if isinstance(lb_data, dict) else []
            lb_watchlist = lb_data.get("watchlist", []) if isinstance(lb_data, dict) else []
            
            # Debug info if enabled
            if st.session_state.get("show_debug"):
                with st.expander("🔧 Debug Info"):
                    st.write(f"Activity count: {lb_data.get('activity_entries_count', 'N/A')}")
                    st.write(f"Watchlist count: {lb_data.get('watchlist_entries_count', 'N/A')}")
                    st.write(f"Watchlist URL: {lb_data.get('watchlist_url', 'N/A')}")
                    if lb_data.get('watchlist_sample_keys'):
                        st.write(f"Sample keys: {lb_data.get('watchlist_sample_keys')}")
                    if lb_data.get('watchlist_error'):
                        st.error(f"Error: {lb_data.get('watchlist_error')}")
            
            lb_tab1, lb_tab2 = st.tabs(["📋 Watchlist", "🎬 Activity"])
            
            with lb_tab1:
                if lb_watchlist:
                    st.success(f"{len(lb_watchlist)} films")
                    box = st.container(height=180)
                    with box:
                        for item in lb_watchlist[:20]:
                            title = item.get("title", "?")
                            year = item.get("year", "")
                            link = item.get("link", "")
                            display = f"🎬 {title}" + (f" ({year})" if year else "")
                            if link:
                                st.markdown(f"[{display}]({link})")
                            else:
                                st.write(display)
                else:
                    st.warning("Watchlist empty or not loading")
                    st.caption(f"URL: letterboxd.com/{LETTERBOXD_USER}/watchlist/")
                    st.caption("Ensure it's set to public!")
            
            with lb_tab2:
                if activity:
                    box = st.container(height=180)
                    with box:
                        for item in activity[:10]:
                            title = item.get("title", "")
                            link = item.get("link", "")
                            star = " ⭐" if item.get("has_rating") else ""
                            if link:
                                st.markdown(f"[🎬 {title}]({link}){star}")
                            else:
                                st.write(f"🎬 {title}{star}")
                else:
                    st.caption("No recent activity")
                    
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.divider()
    
    # Trending
    st.subheader("🔥 Trending")
    
    try:
        trending = et.get_trending_movies()
        if trending:
            cols = st.columns(3)
            for i, movie in enumerate(trending[:6]):
                with cols[i % 3]:
                    poster_path = movie.get("poster_path")
                    title = movie.get("title", "?")
                    rating = movie.get("vote_average", 0)
                    tmdb_id = movie.get("id")
                    
                    if poster_path:
                        st.image(et.get_poster_url(poster_path, "w154"), use_column_width=True)
                    
                    short = title[:10] + "..." if len(title) > 10 else title
                    if tmdb_id:
                        st.markdown(f"[{short}](https://www.themoviedb.org/movie/{tmdb_id})")
                    else:
                        st.caption(short)
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # Search (TMDB)
    st.subheader("🔍 Search TMDB")
    st.caption("Search The Movie Database")
    
    search_q = st.text_input("Movie or TV...", key="ent_search", label_visibility="collapsed", placeholder="Search TMDB...")
    
    if search_q:
        search_type = st.radio("", ["Movies", "TV"], horizontal=True, key="s_type", label_visibility="collapsed")
        
        try:
            results = et.search_movie(search_q) if search_type == "Movies" else et.search_tv(search_q)
            
            if results:
                for item in results[:5]:
                    title = item.get("title") or item.get("name", "?")
                    year = (item.get("release_date") or item.get("first_air_date") or "")[:4]
                    rating = item.get("vote_average", 0)
                    tmdb_id = item.get("id")
                    
                    url = f"https://www.themoviedb.org/{'movie' if search_type == 'Movies' else 'tv'}/{tmdb_id}"
                    st.markdown(f"[**{title}**]({url}) ({year}) ⭐{rating:.1f}")
            else:
                st.caption("No results")
        except Exception as e:
            st.caption(f"Error: {e}")

# ============================================================
# Footer
# ============================================================

st.divider()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
with c2:
    if st.button("🏋️ Health", use_container_width=True):
        st.switch_page("pages/1_🏋️_Health_Fitness.py")
with c3:
    if st.button("🎯 Goals", use_container_width=True):
        st.switch_page("pages/3_🎯_Goals.py")
with c4:
    if st.button("✈️ Travel", use_container_width=True):
        st.switch_page("pages/4_✈️_Travel.py")
with c5:
    if st.button("📰 News", use_container_width=True):
        st.switch_page("pages/5_📰_News.py")
