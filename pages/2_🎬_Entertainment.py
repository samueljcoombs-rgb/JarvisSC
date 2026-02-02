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

# Vue Basingstoke
VUE_CINEMA_SLUG = "basingstoke-festival-place"
VUE_WHATS_ON_URL = f"https://www.myvue.com/cinema/{VUE_CINEMA_SLUG}/whats-on"

def make_vue_film_url(title: str) -> str:
    """Create Vue film URL from title."""
    slug = title.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
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
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📰🎮 News")
    available_sources = ["IGN", "Polygon", "Kotaku", "Eurogamer", "Variety", "The Verge", "Screen Rant", "Collider", "Deadline"]
    news_sources = st.multiselect(
        "Sources:",
        available_sources,
        default=["IGN", "Polygon", "Kotaku", "Variety", "Eurogamer"],
        key="news_sources"
    )
    news_count = st.slider("Articles", 10, 40, 25, key="news_count")
    
    st.divider()
    
    st.subheader("🔗 Quick Links")
    st.markdown(f"🎟️ [Vue Basingstoke]({VUE_WHATS_ON_URL})")
    st.markdown(f"🎭 [Letterboxd @{LETTERBOXD_USER}](https://letterboxd.com/{LETTERBOXD_USER}/)")
    st.markdown(f"📋 [My Watchlist](https://letterboxd.com/{LETTERBOXD_USER}/watchlist/)")
    
    st.divider()
    show_debug = st.checkbox("Debug mode", key="debug")

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
# Entertainment & Gaming News - Full Width with Cards
# ============================================================

with st.expander("📰🎮 **Entertainment & Gaming News** - Configure in sidebar", expanded=True):
    try:
        news = et.get_entertainment_news()
        if news:
            # Filter by selected sources
            if news_sources:
                filtered = [n for n in news if n.get("source") in news_sources]
                if len(filtered) < 5:
                    filtered = news
            else:
                filtered = news
            
            # Scrollable container with card layout
            news_box = st.container(height=350)
            with news_box:
                # 3-column grid
                cols = st.columns(3)
                for i, article in enumerate(filtered[:news_count]):
                    with cols[i % 3]:
                        title = article.get("title", "")
                        source = article.get("source", "")
                        link = article.get("link", "")
                        
                        # Gaming vs Entertainment badge
                        if source in ["IGN", "Polygon", "Kotaku", "Eurogamer"]:
                            badge_color = "#10b981"  # Green for gaming
                            badge_icon = "🎮"
                        else:
                            badge_color = "#ec4899"  # Pink for entertainment
                            badge_icon = "🎬"
                        
                        # Truncate title
                        short_title = title[:75] + "..." if len(title) > 75 else title
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); 
                                    border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; 
                                    padding: 0.8rem; margin-bottom: 0.6rem; min-height: 80px;
                                    transition: all 0.2s ease;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.4rem;">
                                <span style="background: {badge_color}; color: white; padding: 2px 8px; 
                                             border-radius: 4px; font-size: 0.65rem; font-weight: bold;">
                                    {badge_icon} {source}
                                </span>
                            </div>
                            <a href="{link}" target="_blank" style="color: #e0e0e0; text-decoration: none; 
                                       font-size: 0.85rem; line-height: 1.4; display: block;">
                                {short_title}
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No news available")
    except Exception as e:
        st.warning(f"News error: {e}")

st.divider()

# ============================================================
# Main 3-Column Layout
# ============================================================

left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

# ============================================================
# LEFT COLUMN: Coming Soon (TOP) + In Cinemas Now
# ============================================================

with left_col:
    # COMING SOON - Full Year, Scrollable (MOVED TO TOP)
    st.subheader("🗓️ Coming Soon")
    st.caption("Major UK releases this year")
    
    try:
        # Get lots of pages for full year
        upcoming = et.get_upcoming_movies("GB", pages=8)
        
        if upcoming:
            # Filter and sort
            major = sorted(
                [m for m in upcoming if m.get("popularity", 0) > 5],
                key=lambda x: x.get("release_date", "9999")
            )
            
            # Scrollable for whole year
            coming_box = st.container(height=350)
            with coming_box:
                current_month = None
                for movie in major[:50]:
                    title = movie.get("title", "Unknown")
                    release_date = movie.get("release_date", "")
                    tmdb_id = movie.get("id")
                    
                    if release_date:
                        try:
                            dt = datetime.strptime(release_date, "%Y-%m-%d")
                            month_year = dt.strftime("%B %Y")
                            date_str = dt.strftime("%d")
                            
                            # Month header
                            if month_year != current_month:
                                current_month = month_year
                                st.markdown(f"### 📅 {month_year}")
                        except:
                            date_str = "?"
                    else:
                        date_str = "?"
                    
                    c1, c2 = st.columns([1, 5])
                    with c1:
                        st.markdown(f"**{date_str}**")
                    with c2:
                        if tmdb_id:
                            st.markdown(f"[{title}](https://www.themoviedb.org/movie/{tmdb_id})")
                        else:
                            st.write(title)
        else:
            st.caption("No upcoming releases")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # IN CINEMAS NOW - Try Vue scraping first, fallback to TMDB
    st.subheader("🎥 In Cinemas Now")
    st.markdown(f"[🎟️ Vue Basingstoke]({VUE_WHATS_ON_URL})")
    
    # Try Vue scraping
    vue_films = []
    try:
        vue_films = et.get_vue_cinema_listings(VUE_CINEMA_SLUG)
        if vue_films and not vue_films[0].get("error"):
            st.success(f"Showing {len(vue_films)} films at Vue")
    except:
        vue_films = []
    
    if vue_films and not (len(vue_films) == 1 and vue_films[0].get("error")):
        # Show Vue films
        cols = st.columns(3)
        for i, film in enumerate(vue_films[:9]):
            with cols[i % 3]:
                title = film.get("title", "?")
                link = film.get("link", make_vue_film_url(title))
                img = film.get("image", "")
                
                if img:
                    st.image(img, use_column_width=True)
                
                short = title[:15] + "..." if len(title) > 15 else title
                st.markdown(f"[{short}]({link})")
    else:
        # Fallback to TMDB
        try:
            now_playing = et.get_now_playing("GB")
            if now_playing:
                # Filter high-rated English films
                filtered = sorted(
                    [f for f in now_playing if f.get("vote_average", 0) > 5.5 and f.get("original_language") == "en"],
                    key=lambda x: x.get("vote_average", 0),
                    reverse=True
                )[:9]
                
                cols = st.columns(3)
                for i, film in enumerate(filtered):
                    with cols[i % 3]:
                        title = film.get("title", "?")
                        rating = film.get("vote_average", 0)
                        poster = film.get("poster_path")
                        
                        if poster:
                            st.image(et.get_poster_url(poster, "w154"), use_column_width=True)
                        
                        vue_url = make_vue_film_url(title)
                        short = title[:15] + "..." if len(title) > 15 else title
                        st.markdown(f"[{short}]({vue_url})")
                        st.caption(f"⭐ {rating:.1f}")
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
        if st.button("🎯 Recommend", key="rec", use_container_width=True):
            st.session_state.ent_pending = "Based on my Letterboxd, recommend 5 films I'd love. Explain why."
    with c2:
        if st.button("🎲 Tonight?", key="tonight", use_container_width=True):
            st.session_state.ent_pending = "Help me decide what to watch tonight."
    with c3:
        if st.button("🗑️ Clear", key="clear", use_container_width=True):
            st.session_state.ent_chat = []
            st.rerun()
    
    user_input = st.chat_input("Ask about movies...", key="chat_input")
    
    if "ent_pending" in st.session_state and st.session_state.ent_pending:
        user_input = st.session_state.ent_pending
        st.session_state.ent_pending = None
    
    if user_input:
        st.session_state.ent_chat.append({"role": "user", "content": user_input})
        
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", [])[:15]
            lb_watchlist = lb_data.get("watchlist", [])[:20]
            activity_text = "\n".join([f"- {a.get('title', '')} ({a.get('year', '')})" for a in activity])
            watchlist_text = "\n".join([f"- {w.get('title', '')} ({w.get('year', '')})" for w in lb_watchlist])
        except:
            activity_text = "No data"
            watchlist_text = "No data"
        
        context = f"""You're an expert film critic. USER'S LETTERBOXD:
Recently watched: {activity_text}
Watchlist: {watchlist_text}
Give personalized recommendations with reasons."""
        
        client = get_openai_client()
        if client:
            try:
                messages = [{"role": "system", "content": context}]
                messages.extend(st.session_state.ent_chat)
                resp = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=800)
                st.session_state.ent_chat.append({"role": "assistant", "content": resp.choices[0].message.content})
            except Exception as e:
                st.session_state.ent_chat.append({"role": "assistant", "content": f"Error: {e}"})
        else:
            st.session_state.ent_chat.append({"role": "assistant", "content": "OpenAI not configured."})
        st.rerun()
    
    if st.session_state.ent_chat:
        chat_box = st.container(height=200)
        with chat_box:
            for msg in reversed(st.session_state.ent_chat):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
    else:
        st.info("💡 Get AI recommendations based on your taste!")
    
    st.divider()
    
    # Games
    st.subheader("🎮 Games To Play")
    
    try:
        games_data = sm.get_watchlist(status="to_watch")
        games = [g for g in games_data if g.get("type") == "game"] if games_data else []
    except:
        games = []
    
    if games:
        for game in games[:8]:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.write(f"🎮 {game.get('title', '?')}")
            with c2:
                if st.button("✓", key=f"g_{game.get('id')}"):
                    try:
                        sm.update_watchlist_item(game.get("id"), {"status": "completed"})
                        st.rerun()
                    except:
                        pass
    else:
        st.caption("No games yet")
    
    with st.expander("➕ Add Game"):
        new_game = st.text_input("Title", key="new_game")
        if st.button("Add", key="add_g"):
            if new_game:
                try:
                    sm.add_to_watchlist(title=new_game, media_type="game")
                    st.success("Added!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

# ============================================================
# RIGHT COLUMN: Letterboxd + Trending
# ============================================================

with right_col:
    st.subheader("🎭 Letterboxd")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"[📋 Watchlist](https://letterboxd.com/{LETTERBOXD_USER}/watchlist/)")
    with c2:
        st.markdown(f"[🎬 Profile](https://letterboxd.com/{LETTERBOXD_USER}/)")
    
    try:
        lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
        activity = lb_data.get("activity", [])
        lb_watchlist = lb_data.get("watchlist", [])
        
        # Debug
        if st.session_state.get("debug"):
            with st.expander("🔧 Debug"):
                st.write(f"Activity count: {len(activity)}")
                st.write(f"Watchlist count: {len(lb_watchlist)}")
                if activity:
                    st.write("First activity:", activity[0])
                if lb_watchlist:
                    st.write("First watchlist:", lb_watchlist[0])
                if lb_data.get('watchlist_error'):
                    st.error(lb_data.get('watchlist_error'))
        
        tab1, tab2 = st.tabs(["📋 Watchlist", "🎬 Activity"])
        
        with tab1:
            if lb_watchlist:
                st.success(f"{len(lb_watchlist)} films")
                box = st.container(height=250)
                with box:
                    for item in lb_watchlist[:30]:
                        title = item.get("title", "?")
                        year = item.get("year", "")
                        link = item.get("link", "")
                        text = f"🎬 {title}" + (f" ({year})" if year else "")
                        if link:
                            st.markdown(f"[{text}]({link})")
                        else:
                            st.write(text)
            else:
                st.warning("Watchlist not loading")
                st.caption(f"RSS: letterboxd.com/{LETTERBOXD_USER}/watchlist/rss/")
                st.caption("Ensure it's public!")
        
        with tab2:
            if activity:
                st.success(f"{len(activity)} recent")
                box = st.container(height=250)
                with box:
                    for item in activity[:20]:
                        title = item.get("title", "?")
                        year = item.get("year", "")
                        link = item.get("link", "")
                        star = " ⭐" if item.get("has_rating") else ""
                        text = f"🎬 {title}" + (f" ({year})" if year else "") + star
                        if link:
                            st.markdown(f"[{text}]({link})")
                        else:
                            st.write(text)
            else:
                st.caption("No activity")
                
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
                    poster = movie.get("poster_path")
                    title = movie.get("title", "?")
                    tmdb_id = movie.get("id")
                    
                    if poster:
                        st.image(et.get_poster_url(poster, "w154"), use_column_width=True)
                    
                    short = title[:10] + "..." if len(title) > 10 else title
                    if tmdb_id:
                        st.markdown(f"[{short}](https://www.themoviedb.org/movie/{tmdb_id})")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # Search
    st.subheader("🔍 Search TMDB")
    
    q = st.text_input("Search...", key="search", label_visibility="collapsed", placeholder="Search movies/TV...")
    
    if q:
        stype = st.radio("", ["Movies", "TV"], horizontal=True, key="stype", label_visibility="collapsed")
        results = et.search_movie(q) if stype == "Movies" else et.search_tv(q)
        
        if results:
            for r in results[:5]:
                title = r.get("title") or r.get("name", "?")
                year = (r.get("release_date") or r.get("first_air_date") or "")[:4]
                tid = r.get("id")
                url = f"https://www.themoviedb.org/{'movie' if stype == 'Movies' else 'tv'}/{tid}"
                st.markdown(f"[**{title}**]({url}) ({year})")

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
