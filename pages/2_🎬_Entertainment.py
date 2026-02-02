# pages/2_🎬_Entertainment.py
"""
Entertainment Dashboard - Letterboxd, Cinema Releases, Games, AI Recommendations.
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
# Entertainment News (Top Row) - Better Display
# ============================================================

with st.expander("📰 **Entertainment News**", expanded=True):
    try:
        news = et.get_entertainment_news()
        if news:
            for i, article in enumerate(news[:6]):
                title = article.get("title", "No title")
                source = article.get("source", "")
                link = article.get("link", "")
                
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.caption(f"**{source}**")
                with col2:
                    if link:
                        st.markdown(f"[{title}]({link})")
                    else:
                        st.write(title)
    except Exception as e:
        st.caption(f"News unavailable")

st.divider()

# ============================================================
# Main 3-Column Layout
# ============================================================

left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

# ============================================================
# LEFT COLUMN: Letterboxd + Games
# ============================================================

with left_col:
    # Letterboxd Section
    st.subheader("🎭 Letterboxd")
    st.caption(f"@{LETTERBOXD_USER}")
    
    if LETTERBOXD_USER:
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", []) if isinstance(lb_data, dict) else []
            lb_watchlist = lb_data.get("watchlist", []) if isinstance(lb_data, dict) else []
            
            # Debug info
            if lb_data.get("watchlist_error"):
                st.caption(f"⚠️ Watchlist error: {lb_data.get('watchlist_error')}")
            
            # Tabs
            lb_tab1, lb_tab2 = st.tabs(["📋 Watchlist", "🎬 Activity"])
            
            with lb_tab1:
                if lb_watchlist:
                    st.success(f"{len(lb_watchlist)} films to watch")
                    for item in lb_watchlist[:15]:
                        title = item.get("title", "Unknown")
                        year = item.get("year", "")
                        year_str = f" ({year})" if year else ""
                        st.write(f"🎬 {title}{year_str}")
                else:
                    st.warning("No watchlist found. Make sure your Letterboxd watchlist is public!")
                    st.caption(f"Checking: letterboxd.com/{LETTERBOXD_USER}/watchlist/rss/")
            
            with lb_tab2:
                if activity:
                    for item in activity[:8]:
                        title = item.get("title", "")
                        has_rating = item.get("has_rating", False)
                        star = " ⭐" if has_rating else ""
                        st.write(f"🎬 {title}{star}")
                else:
                    st.caption("No recent activity")
                    
        except Exception as e:
            st.error(f"Letterboxd error: {e}")
    
    st.divider()
    
    # Games To Play Section
    st.subheader("🎮 Games To Play")
    
    # Get games from sheets (we'll add this functionality)
    if "games_list" not in st.session_state:
        st.session_state.games_list = []
    
    # Try to load from sheets
    try:
        games_data = sm.get_watchlist(status="to_watch")
        games = [g for g in games_data if g.get("type") == "game"] if games_data else []
    except:
        games = []
    
    if games:
        for game in games[:8]:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"🎮 {game.get('title', '?')}")
            with col2:
                if st.button("✓", key=f"game_done_{game.get('id', game.get('title'))}"):
                    try:
                        sm.update_watchlist_item(game.get("id"), {"status": "completed"})
                        st.rerun()
                    except:
                        pass
    else:
        st.caption("No games in your list yet")
    
    # Add game form
    with st.expander("➕ Add Game"):
        new_game = st.text_input("Game title", key="new_game_title")
        if st.button("Add Game", key="add_game_btn"):
            if new_game:
                try:
                    sm.add_to_watchlist(title=new_game, media_type="game")
                    st.success(f"Added {new_game}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================================
# MIDDLE COLUMN: In Cinemas + Coming Soon + AI Coach
# ============================================================

with mid_col:
    # In Cinemas Now - Sorted by Rating
    st.subheader("🎥 In Cinemas Now")
    
    try:
        now_playing = et.get_now_playing("GB")
        
        if now_playing:
            # Sort by rating (highest first)
            sorted_films = sorted(now_playing, key=lambda x: x.get("vote_average", 0), reverse=True)
            
            for film in sorted_films[:8]:
                title = film.get("title", "Unknown")
                rating = film.get("vote_average", 0)
                release = film.get("release_date", "")
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{title}**")
                with col2:
                    st.write(f"⭐ {rating:.1f}")
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
            # Filter to major releases and sort by date
            major = sorted([m for m in upcoming if m.get("popularity", 0) > 10], 
                          key=lambda x: x.get("release_date", "9999"))[:8]
            
            for movie in major:
                title = movie.get("title", "Unknown")
                release_date = movie.get("release_date", "")
                rating = movie.get("vote_average", 0)
                
                # Format date
                if release_date:
                    try:
                        dt = datetime.strptime(release_date, "%Y-%m-%d")
                        date_str = dt.strftime("%d %b")
                    except:
                        date_str = "TBA"
                else:
                    date_str = "TBA"
                
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.write(f"**{date_str}**")
                with col2:
                    st.write(title)
                with col3:
                    if rating > 0:
                        st.write(f"⭐{rating:.1f}")
        else:
            st.caption("No upcoming releases")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # AI Movie Coach
    st.subheader("🤖 AI Movie Coach")
    
    # Initialize chat
    if "ent_chat" not in st.session_state:
        st.session_state.ent_chat = []
    
    # Quick buttons
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
    
    # Chat input
    user_input = st.chat_input("Ask about movies...", key="ent_input")
    
    # Handle pending
    if "ent_pending" in st.session_state and st.session_state.ent_pending:
        user_input = st.session_state.ent_pending
        st.session_state.ent_pending = None
    
    if user_input:
        st.session_state.ent_chat.append({"role": "user", "content": user_input})
        
        # Build context
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

Give personalized recommendations. Be specific about WHY they'd like each film. Keep responses concise but insightful."""
        
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
    
    # Display chat
    if st.session_state.ent_chat:
        chat_box = st.container(height=200)
        with chat_box:
            for msg in reversed(st.session_state.ent_chat):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
    else:
        st.info("💡 Ask for recommendations based on your Letterboxd!")

# ============================================================
# RIGHT COLUMN: Trending + Search
# ============================================================

with right_col:
    # Trending
    st.subheader("🔥 Trending Now")
    
    try:
        trending = et.get_trending_movies()
        if trending:
            cols = st.columns(3)
            for i, movie in enumerate(trending[:6]):
                with cols[i % 3]:
                    poster_path = movie.get("poster_path")
                    title = movie.get("title", "?")
                    rating = movie.get("vote_average", 0)
                    
                    if poster_path:
                        poster_url = et.get_poster_url(poster_path, "w342")
                        st.image(poster_url, use_column_width=True)
                    
                    short_title = title[:12] + "..." if len(title) > 12 else title
                    st.caption(f"**{short_title}**")
                    st.caption(f"⭐ {rating:.1f}")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # Search
    st.subheader("🔍 Search")
    
    search_q = st.text_input("Movie or TV show...", key="ent_search", label_visibility="collapsed", placeholder="Search...")
    
    if search_q:
        search_type = st.radio("", ["Movies", "TV"], horizontal=True, key="search_type", label_visibility="collapsed")
        
        try:
            results = et.search_movie(search_q) if search_type == "Movies" else et.search_tv(search_q)
            
            if results:
                for item in results[:5]:
                    title = item.get("title") or item.get("name", "?")
                    year = (item.get("release_date") or item.get("first_air_date") or "")[:4]
                    rating = item.get("vote_average", 0)
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{title}** ({year})")
                    with col2:
                        st.write(f"⭐{rating:.1f}")
            else:
                st.caption("No results")
        except Exception as e:
            st.caption(f"Search error: {e}")

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
