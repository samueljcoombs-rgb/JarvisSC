# pages/2_🎬_Entertainment.py
"""
Entertainment Dashboard - PURE STREAMLIT, NO HTML
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
    
    st.subheader("📰 News Sources")
    gaming_sources = st.multiselect(
        "🎮 Gaming",
        ["IGN", "Polygon", "Kotaku", "Eurogamer"],
        default=["IGN", "Polygon", "Kotaku"],
    )
    ent_sources = st.multiselect(
        "🎬 Entertainment", 
        ["Variety", "The Verge", "Screen Rant", "Collider", "Deadline"],
        default=["Variety", "Collider"],
    )
    news_count = st.slider("Max articles", 10, 40, 20)
    
    st.divider()
    
    st.subheader("🔗 Quick Links")
    st.write(f"🎟️ [Vue Basingstoke]({VUE_WHATS_ON_URL})")
    st.write(f"🎭 [Letterboxd @{LETTERBOXD_USER}](https://letterboxd.com/{LETTERBOXD_USER}/)")
    st.write(f"📋 [My Watchlist](https://letterboxd.com/{LETTERBOXD_USER}/watchlist/)")
    
    st.divider()
    show_debug = st.checkbox("🔧 Debug mode")

# ============================================================
# Header
# ============================================================

st.title("🎬 Entertainment Hub")
st.caption("Cinema • Letterboxd • Games • AI Recommendations")

st.divider()

# ============================================================
# Entertainment & Gaming News
# ============================================================

with st.expander("📰🎮 Entertainment & Gaming News", expanded=True):
    try:
        news = et.get_entertainment_news()
        if news:
            all_sources = (gaming_sources or []) + (ent_sources or [])
            if all_sources:
                filtered = [n for n in news if n.get("source") in all_sources]
                if len(filtered) < 5:
                    filtered = news
            else:
                filtered = news
            
            news_box = st.container(height=350)
            with news_box:
                # 3 columns for smaller items
                for i in range(0, min(len(filtered), news_count), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(filtered) and idx < news_count:
                            article = filtered[idx]
                            with col:
                                title = article.get("title", "")
                                source = article.get("source", "")
                                link = article.get("link", "")
                                image = article.get("image", "")
                                
                                icon = "🎮" if source in ["IGN", "Polygon", "Kotaku", "Eurogamer"] else "🎬"
                                
                                # Small image thumbnail
                                if image:
                                    try:
                                        img_col, txt_col = st.columns([1, 2])
                                        with img_col:
                                            st.image(image, width=80)
                                        with txt_col:
                                            st.caption(f"{icon} {source}")
                                            short_title = title[:50] + "..." if len(title) > 50 else title
                                            st.write(f"[{short_title}]({link})")
                                    except:
                                        st.caption(f"{icon} {source}")
                                        short_title = title[:60] + "..." if len(title) > 60 else title
                                        st.write(f"[{short_title}]({link})")
                                else:
                                    st.caption(f"{icon} {source}")
                                    short_title = title[:60] + "..." if len(title) > 60 else title
                                    st.write(f"[{short_title}]({link})")
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
# LEFT COLUMN: Coming Soon + In Cinemas
# ============================================================

with left_col:
    st.subheader("🗓️ Coming Soon")
    st.caption("Major UK cinema releases")
    
    try:
        upcoming = et.get_upcoming_movies("GB", pages=6)
        
        if upcoming:
            # ENGLISH ONLY, sorted by date
            major = sorted(
                [m for m in upcoming if m.get("popularity", 0) > 5 and m.get("original_language") == "en"],
                key=lambda x: x.get("release_date", "9999")
            )
            
            coming_box = st.container(height=350)
            with coming_box:
                current_month = None
                for movie in major[:40]:
                    title = movie.get("title", "Unknown")
                    release_date = movie.get("release_date", "")
                    tmdb_id = movie.get("id")
                    
                    if release_date:
                        try:
                            dt = datetime.strptime(release_date, "%Y-%m-%d")
                            month_year = dt.strftime("%B %Y")
                            date_str = dt.strftime("%d")
                            
                            if month_year != current_month:
                                current_month = month_year
                                st.markdown(f"**📅 {month_year}**")
                        except:
                            date_str = "?"
                    else:
                        date_str = "?"
                    
                    c1, c2 = st.columns([1, 5])
                    with c1:
                        st.write(f"**{date_str}**")
                    with c2:
                        if tmdb_id:
                            st.write(f"[{title}](https://www.themoviedb.org/movie/{tmdb_id})")
                        else:
                            st.write(title)
        else:
            st.info("No upcoming releases")
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.divider()
    
    # IN CINEMAS NOW - ENGLISH ONLY
    st.subheader("🎥 In Cinemas Now")
    st.write(f"[🎟️ Vue Basingstoke]({VUE_WHATS_ON_URL})")
    
    try:
        now_playing = et.get_now_playing("GB")
        
        if now_playing:
            # ENGLISH ONLY, good rating
            filtered = sorted(
                [f for f in now_playing 
                 if f.get("vote_average", 0) > 5.0 
                 and f.get("original_language") == "en"],
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
                    short = title[:14] + "…" if len(title) > 14 else title
                    st.write(f"[{short}]({vue_url})")
                    st.caption(f"⭐ {rating:.1f}")
        else:
            st.info("No films found")
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================================
# MIDDLE COLUMN: AI Coach + Games
# ============================================================

with mid_col:
    st.subheader("🤖 AI Movie Coach")
    
    if "ent_chat" not in st.session_state:
        st.session_state.ent_chat = []
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎯 Recommend", use_container_width=True):
            st.session_state.ent_pending = "Based on my Letterboxd, recommend 5 films I'd love. Explain why."
    with c2:
        if st.button("🎲 Tonight?", use_container_width=True):
            st.session_state.ent_pending = "Help me pick something to watch tonight."
    with c3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.ent_chat = []
            st.rerun()
    
    user_input = st.chat_input("Ask about movies...")
    
    if "ent_pending" in st.session_state and st.session_state.ent_pending:
        user_input = st.session_state.ent_pending
        st.session_state.ent_pending = None
    
    if user_input:
        st.session_state.ent_chat.append({"role": "user", "content": user_input})
        
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", [])[:15]
            watchlist = lb_data.get("watchlist", [])[:20]
            activity_text = "\n".join([f"- {a.get('title', '')} ({a.get('year', '')})" for a in activity])
            watchlist_text = "\n".join([f"- {w.get('title', '')} ({w.get('year', '')})" for w in watchlist])
        except:
            activity_text = "No data"
            watchlist_text = "No data"
        
        context = f"""Expert film critic. USER'S LETTERBOXD:
Recently watched: {activity_text}
Watchlist: {watchlist_text}
Give personalized recommendations explaining why they'd like each."""
        
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
        st.info("💡 Get AI recommendations based on your Letterboxd!")
    
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
        st.caption("No games in list yet")
    
    with st.expander("➕ Add Game"):
        new_game = st.text_input("Game title", key="new_game")
        if st.button("Add Game"):
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
        st.write(f"[📋 Watchlist](https://letterboxd.com/{LETTERBOXD_USER}/watchlist/)")
    with c2:
        st.write(f"[🎬 Profile](https://letterboxd.com/{LETTERBOXD_USER}/)")
    
    # Force fresh fetch for testing
    if st.button("🔄 Refresh", key="refresh_lb"):
        st.cache_data.clear()
        st.rerun()
    
    try:
        lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
        activity = lb_data.get("activity", [])
        watchlist = lb_data.get("watchlist", [])
        
        # Debug panel
        if show_debug:
            with st.expander("🔧 Debug Info", expanded=True):
                st.write(f"Activity status: {lb_data.get('activity_status', 'N/A')}")
                st.write(f"Activity items: {len(activity)}")
                st.write(f"Watchlist URL: {lb_data.get('watchlist_url', 'N/A')}")
                st.write(f"Watchlist HTTP status: {lb_data.get('watchlist_status', 'N/A')}")
                st.write(f"Watchlist response length: {lb_data.get('watchlist_length', 'N/A')} bytes")
                st.write(f"Watchlist entries raw: {lb_data.get('watchlist_entries_raw', 'N/A')}")
                st.write(f"Watchlist items parsed: {len(watchlist)}")
                
                if lb_data.get('trying_manual_parse'):
                    st.info("Tried manual XML parsing")
                if lb_data.get('xml_parse_error'):
                    st.error(f"XML error: {lb_data.get('xml_parse_error')}")
                if lb_data.get('watchlist_error'):
                    st.error(f"Error: {lb_data.get('watchlist_error')}")
                
                if activity:
                    st.write("First activity:", activity[0])
                if watchlist:
                    st.write("First watchlist:", watchlist[0])
        
        tab1, tab2 = st.tabs(["📋 Watchlist", "🎬 Activity"])
        
        with tab1:
            if watchlist:
                st.success(f"✅ {len(watchlist)} films to watch")
                box = st.container(height=280)
                with box:
                    for item in watchlist[:30]:
                        title = item.get("title", "?")
                        year = item.get("year", "")
                        link = item.get("link", "")
                        display = f"🎬 {title}" + (f" ({year})" if year else "")
                        if link:
                            st.write(f"[{display}]({link})")
                        else:
                            st.write(display)
            else:
                st.warning("⚠️ Watchlist not loading")
                st.write(f"RSS: letterboxd.com/{LETTERBOXD_USER}/watchlist/rss/")
                st.write("Make sure your watchlist is PUBLIC!")
                st.write("Try clicking 🔄 Refresh above")
        
        with tab2:
            if activity:
                st.success(f"✅ {len(activity)} recent films")
                box = st.container(height=280)
                with box:
                    for item in activity[:20]:
                        title = item.get("title", "?")
                        year = item.get("year", "")
                        link = item.get("link", "")
                        star = " ⭐" if item.get("has_rating") else ""
                        display = f"🎬 {title}" + (f" ({year})" if year else "") + star
                        if link:
                            st.write(f"[{display}]({link})")
                        else:
                            st.write(display)
            else:
                st.info("No recent activity")
                
    except Exception as e:
        st.error(f"Letterboxd error: {e}")
    
    st.divider()
    
    # Trending
    st.subheader("🔥 Trending Now")
    
    try:
        trending = et.get_trending_movies()
        if trending:
            cols = st.columns(3)
            for i, movie in enumerate(trending[:6]):
                with cols[i % 3]:
                    poster = movie.get("poster_path")
                    title = movie.get("title", "?")
                    tmdb_id = movie.get("id")
                    rating = movie.get("vote_average", 0)
                    
                    if poster:
                        st.image(et.get_poster_url(poster, "w154"), use_column_width=True)
                    
                    short = title[:10] + "…" if len(title) > 10 else title
                    if tmdb_id:
                        st.write(f"[{short}](https://www.themoviedb.org/movie/{tmdb_id})")
                    st.caption(f"⭐ {rating:.1f}")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # Search
    st.subheader("🔍 Search TMDB")
    
    q = st.text_input("Search movies or TV...", label_visibility="collapsed", placeholder="Search...")
    
    if q:
        stype = st.radio("Type", ["Movies", "TV"], horizontal=True, label_visibility="collapsed")
        results = et.search_movie(q) if stype == "Movies" else et.search_tv(q)
        
        if results:
            for r in results[:5]:
                title = r.get("title") or r.get("name", "?")
                year = (r.get("release_date") or r.get("first_air_date") or "")[:4]
                tid = r.get("id")
                url = f"https://www.themoviedb.org/{'movie' if stype == 'Movies' else 'tv'}/{tid}"
                st.write(f"**[{title}]({url})** ({year})")
        else:
            st.caption("No results")

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
