# pages/2_🎬_Entertainment.py
"""
Entertainment Dashboard - Films & Gaming Hub
"""
import streamlit as st
import streamlit.components.v1 as components
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
# Page Config - SIDEBAR COLLAPSED BY DEFAULT
# ============================================================

st.set_page_config(
    page_title="Entertainment | Jarvis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

gs.inject_global_styles()

# ============================================================
# Premium CSS (matching homepage)
# ============================================================

st.markdown("""
<style>
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.ent-header {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    padding: 1.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 15px 40px rgba(139, 92, 246, 0.3);
}

.ent-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ec4899, #8b5cf6, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.ent-header p {
    color: rgba(255,255,255,0.7);
    margin: 0.3rem 0 0 0;
    font-size: 0.95rem;
}

.section-card {
    background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.film-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
    gap: 0.5rem;
}

.quick-link {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(236,72,153,0.2));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 8px;
    color: #a78bfa;
    text-decoration: none;
    font-size: 0.85rem;
    margin: 0.2rem;
    transition: all 0.2s;
}

.quick-link:hover {
    background: linear-gradient(135deg, rgba(139,92,246,0.4), rgba(236,72,153,0.4));
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

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
# Sidebar (collapsed by default)
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📰 News Sources")
    gaming_sources = st.multiselect(
        "🎮 Gaming",
        ["IGN", "Polygon", "Kotaku", "Eurogamer"],
        default=["IGN", "Polygon"],
    )
    ent_sources = st.multiselect(
        "🎬 Entertainment", 
        ["Variety", "The Verge", "Screen Rant", "Collider", "Deadline"],
        default=["Variety", "Collider"],
    )
    news_count = st.slider("Max articles", 10, 40, 20)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="ent-header">
    <h1>🎬 Entertainment Hub</h1>
    <p>🎥 Films • 📺 TV Shows • 🎮 Games • 🤖 AI Recommendations</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Entertainment News - Full Width (2-row tiles)
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
            
            # Build 2-row scrolling grid
            html_content = '''
            <style>
            .news-grid {
                display: grid;
                grid-template-rows: repeat(2, 1fr);
                grid-auto-flow: column;
                grid-auto-columns: 240px;
                gap: 0.8rem;
                overflow-x: scroll;
                padding: 0.5rem 0.5rem 1rem 0.5rem;
                scroll-behavior: smooth;
            }
            .news-grid::-webkit-scrollbar { height: 10px; }
            .news-grid::-webkit-scrollbar-track { background: rgba(128,128,128,0.2); border-radius: 5px; }
            .news-grid::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.6); border-radius: 5px; }
            .news-card {
                border-radius: 10px;
                overflow: hidden;
                background: #1e1e2e;
                border: 1px solid rgba(128,128,128,0.3);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .news-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                border-color: #8b5cf6;
            }
            .news-card img { width: 100%; height: 100px; object-fit: cover; }
            .news-card-body { padding: 0.5rem; }
            .news-badge {
                display: inline-block;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.6rem;
                font-weight: 600;
                margin-bottom: 0.3rem;
            }
            .news-badge.gaming { background: linear-gradient(135deg, #10b981, #059669); color: white; }
            .news-badge.entertainment { background: linear-gradient(135deg, #ec4899, #db2777); color: white; }
            .news-title {
                font-size: 0.75rem;
                line-height: 1.25;
                margin: 0;
                color: #e0e0e0;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            .news-title a { color: #e0e0e0; text-decoration: none; }
            .news-title a:hover { color: #a78bfa; }
            </style>
            <div class="news-grid">
            '''
            
            for article in filtered[:news_count]:
                title = article.get("title", "").replace('"', '&quot;').replace('<', '&lt;')
                source = article.get("source", "")
                link = article.get("link", "")
                image = article.get("image", "") or "https://placehold.co/240x100/1a1a2e/8b5cf6?text=News"
                
                is_gaming = source in ["IGN", "Polygon", "Kotaku", "Eurogamer"]
                badge_class = "gaming" if is_gaming else "entertainment"
                badge_icon = "🎮" if is_gaming else "🎬"
                
                html_content += f'''
                <div class="news-card">
                    <img src="{image}" alt="" onerror="this.src='https://placehold.co/240x100/1a1a2e/8b5cf6?text=News'">
                    <div class="news-card-body">
                        <span class="news-badge {badge_class}">{badge_icon} {source}</span>
                        <p class="news-title"><a href="{link}" target="_blank">{title}</a></p>
                    </div>
                </div>
                '''
            
            html_content += '</div>'
            components.html(html_content, height=400, scrolling=True)
        else:
            st.info("No news available")
    except Exception as e:
        st.warning(f"News error: {e}")

st.divider()

# ============================================================
# Main 2-Column Layout: LEFT = Films, RIGHT = Games
# ============================================================

left_col, mid_col, right_col = st.columns([4, 4, 3], gap="large")

# ============================================================
# LEFT COLUMN: All Film/TV Content
# ============================================================

with left_col:
    # --- LETTERBOXD ---
    st.subheader("🎭 Letterboxd")
    
    try:
        lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
        activity = lb_data.get("activity", [])
        watchlist = lb_data.get("watchlist", [])
        
        tab1, tab2 = st.tabs([f"📋 Watchlist ({len(watchlist)})", f"🎬 Activity ({len(activity)})"])
        
        with tab1:
            if watchlist:
                box = st.container(height=200)
                with box:
                    for item in watchlist[:25]:
                        title = item.get("title", "?")
                        link = item.get("link", "")
                        if link:
                            st.write(f"🎬 [{title}]({link})")
                        else:
                            st.write(f"🎬 {title}")
            else:
                st.caption("No watchlist items")
            st.caption(f"[View full watchlist →](https://letterboxd.com/{LETTERBOXD_USER}/watchlist/)")
        
        with tab2:
            if activity:
                box = st.container(height=200)
                with box:
                    for item in activity[:15]:
                        title = item.get("title", "")
                        link = item.get("link", "")
                        star = " ⭐" if item.get("has_rating") else ""
                        if link:
                            st.write(f"🎬 [{title}]({link}){star}")
                        else:
                            st.write(f"🎬 {title}{star}")
            else:
                st.caption("No recent activity")
            st.caption(f"[View profile →](https://letterboxd.com/{LETTERBOXD_USER}/)")
    except Exception as e:
        st.error(f"Letterboxd error: {e}")
    
    st.divider()
    
    # --- IN CINEMAS NOW ---
    st.subheader("🎥 In Cinemas Now")
    st.caption(f"[🎟️ Book at Vue Basingstoke]({VUE_WHATS_ON_URL})")
    
    try:
        now_playing = et.get_now_playing("GB")
        if now_playing:
            # Filter to English, sort by popularity
            english_films = [f for f in now_playing if f.get("original_language") == "en"]
            sorted_films = sorted(english_films, key=lambda x: x.get("popularity", 0), reverse=True)[:8]
            
            cols = st.columns(4)
            for i, film in enumerate(sorted_films):
                with cols[i % 4]:
                    poster = film.get("poster_path")
                    title = film.get("title", "?")
                    rating = film.get("vote_average", 0)
                    if poster:
                        st.image(et.get_poster_url(poster, "w154"), use_column_width=True)
                    short = title[:12] + "..." if len(title) > 12 else title
                    st.caption(f"[{short}]({make_vue_film_url(title)}) ⭐{rating:.1f}")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # --- COMING SOON (2026 Major Releases) ---
    st.subheader("🗓️ Coming Soon 2026")
    st.caption("[Full list at Movie Insider →](https://www.movieinsider.com/movies/2026)")
    
    # Curated 2026 releases (inline to avoid module caching issues)
    releases_2026 = [
        {"title": "Scream 7", "date": "2026-02-27", "studio": "Paramount"},
        {"title": "The Bride!", "date": "2026-03-20", "studio": "Warner Bros"},
        {"title": "Project Hail Mary", "date": "2026-03-20", "studio": "MGM"},
        {"title": "The Super Mario Galaxy Movie", "date": "2026-04-03", "studio": "Universal"},
        {"title": "Michael (MJ Biopic)", "date": "2026-04-18", "studio": "Lionsgate"},
        {"title": "The Mandalorian and Grogu", "date": "2026-05-22", "studio": "Disney"},
        {"title": "The Devil Wears Prada 2", "date": "2026-05-29", "studio": "Disney"},
        {"title": "Mortal Kombat II", "date": "2026-05-29", "studio": "Warner Bros"},
        {"title": "Toy Story 5", "date": "2026-06-19", "studio": "Pixar"},
        {"title": "Supergirl: Woman of Tomorrow", "date": "2026-06-26", "studio": "DC"},
        {"title": "The Odyssey (Christopher Nolan)", "date": "2026-07-17", "studio": "Universal"},
        {"title": "Spider-Man: Brand New Day", "date": "2026-07-24", "studio": "Sony/Marvel"},
        {"title": "Minions 3", "date": "2026-07-31", "studio": "Illumination"},
        {"title": "Digger (Tom Cruise)", "date": "2026-10-02", "studio": "Warner Bros"},
        {"title": "Street Fighter", "date": "2026-10-09", "studio": "Legendary"},
        {"title": "The Hunger Games: Sunrise on the Reaping", "date": "2026-11-20", "studio": "Lionsgate"},
        {"title": "Narnia (Greta Gerwig)", "date": "2026-11-26", "studio": "Netflix"},
        {"title": "Avengers: Doomsday", "date": "2026-12-01", "studio": "Marvel"},
        {"title": "Jumanji 4", "date": "2026-12-11", "studio": "Sony"},
        {"title": "Dune: Part Three", "date": "2026-12-18", "studio": "Warner Bros"},
    ]
    
    today = datetime.now().strftime("%Y-%m-%d")
    future = [r for r in releases_2026 if r.get("date", "0000") >= today]
    
    if future:
        box = st.container(height=280)
        with box:
            for movie in future:
                title = movie.get("title", "?")
                date = movie.get("date", "")
                studio = movie.get("studio", "")
                if date:
                    try:
                        dt = datetime.strptime(date, "%Y-%m-%d")
                        date_str = dt.strftime("%d %b")
                    except:
                        date_str = date
                else:
                    date_str = "TBA"
                
                st.write(f"**{date_str}** · {title} *({studio})*")
    else:
        st.caption("No upcoming releases")
    
    st.divider()
    
    # --- TRENDING TV ---
    st.subheader("📺 Trending TV Shows")
    st.caption("[View all on TMDB →](https://www.themoviedb.org/tv)")
    
    try:
        trending_tv = et.get_trending_tv()
        if trending_tv:
            cols = st.columns(4)
            for i, show in enumerate(trending_tv[:8]):
                with cols[i % 4]:
                    poster = show.get("poster_path")
                    name = show.get("name", "?")
                    rating = show.get("vote_average", 0)
                    tmdb_id = show.get("id")
                    if poster:
                        st.image(et.get_poster_url(poster, "w154"), use_column_width=True)
                    short = name[:12] + "..." if len(name) > 12 else name
                    url = f"https://www.themoviedb.org/tv/{tmdb_id}"
                    st.caption(f"[{short}]({url}) ⭐{rating:.1f}")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # --- TV WATCHLIST ---
    st.subheader("📺 TV Watchlist")
    
    try:
        tv_data = sm.get_watchlist(status="to_watch")
        tv_shows = [t for t in tv_data if t.get("type") == "tv"] if tv_data else []
    except:
        tv_shows = []
    
    if tv_shows:
        for show in tv_shows[:8]:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"📺 {show.get('title', '?')}")
            with col2:
                if st.button("✓", key=f"tv_{show.get('id')}"):
                    try:
                        sm.update_watchlist_item(show.get("id"), {"status": "completed"})
                        st.rerun()
                    except:
                        pass
    else:
        st.caption("No TV shows in watchlist")
    
    with st.expander("➕ Add TV Show"):
        new_tv = st.text_input("Show title", key="new_tv")
        if st.button("Add", key="add_tv"):
            if new_tv:
                try:
                    sm.add_to_watchlist(title=new_tv, media_type="tv")
                    st.success("Added!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    
    st.divider()
    
    # --- TRENDING FILMS ---
    st.subheader("🔥 Trending Films")
    st.caption("[View all on TMDB →](https://www.themoviedb.org/movie)")
    
    try:
        trending = et.get_trending_movies()
        if trending:
            cols = st.columns(4)
            for i, movie in enumerate(trending[:8]):
                with cols[i % 4]:
                    poster = movie.get("poster_path")
                    title = movie.get("title", "?")
                    tmdb_id = movie.get("id")
                    if poster:
                        st.image(et.get_poster_url(poster, "w154"), use_column_width=True)
                    short = title[:12] + "..." if len(title) > 12 else title
                    url = f"https://www.themoviedb.org/movie/{tmdb_id}"
                    st.caption(f"[{short}]({url})")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # --- SEARCH TMDB ---
    st.subheader("🔍 Search TMDB")
    
    search_q = st.text_input("Search movies or TV...", key="tmdb_search", placeholder="Enter title...")
    
    if search_q:
        search_type = st.radio("Type", ["Movies", "TV"], horizontal=True, key="search_type")
        
        try:
            results = et.search_movie(search_q) if search_type == "Movies" else et.search_tv(search_q)
            if results:
                for item in results[:5]:
                    title = item.get("title") or item.get("name", "?")
                    year = (item.get("release_date") or item.get("first_air_date") or "")[:4]
                    rating = item.get("vote_average", 0)
                    tmdb_id = item.get("id")
                    media = "movie" if search_type == "Movies" else "tv"
                    url = f"https://www.themoviedb.org/{media}/{tmdb_id}"
                    st.write(f"[**{title}**]({url}) ({year}) ⭐{rating:.1f}")
            else:
                st.caption("No results")
        except Exception as e:
            st.caption(f"Error: {e}")
    
    st.divider()
    
    # --- IMDB TOP 250 LINKS ---
    st.subheader("🏆 IMDB Top 250")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("[🎬 Top 250 Films](https://www.imdb.com/chart/top/)")
    with c2:
        st.markdown("[📺 Top 250 TV Shows](https://www.imdb.com/chart/toptv/)")

# ============================================================
# MIDDLE COLUMN: AI Coach
# ============================================================

with mid_col:
    st.subheader("🤖 AI Entertainment Coach")
    
    if "ent_chat" not in st.session_state:
        st.session_state.ent_chat = []
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎯 Recommend", key="ent_rec", use_container_width=True):
            st.session_state.ent_pending = "Based on my Letterboxd, recommend 5 films I'd love."
    with c2:
        if st.button("🎲 Tonight?", key="ent_tonight", use_container_width=True):
            st.session_state.ent_pending = "Help me decide what to watch tonight."
    with c3:
        if st.button("🗑️ Clear", key="ent_clear", use_container_width=True):
            st.session_state.ent_chat = []
            st.rerun()
    
    user_input = st.chat_input("Ask about movies, TV, games...", key="ent_input")
    
    if "ent_pending" in st.session_state and st.session_state.ent_pending:
        user_input = st.session_state.ent_pending
        st.session_state.ent_pending = None
    
    if user_input:
        st.session_state.ent_chat.append({"role": "user", "content": user_input})
        
        try:
            lb_data = et.get_letterboxd_activity(LETTERBOXD_USER)
            activity = lb_data.get("activity", [])[:10]
            lb_watchlist = lb_data.get("watchlist", [])[:20]
            activity_text = "\n".join([f"- {a.get('title', '')}" for a in activity])
            watchlist_text = "\n".join([f"- {w.get('title', '')}" for w in lb_watchlist])
        except:
            activity_text = "No data"
            watchlist_text = "No data"
        
        context = f"""You're an expert entertainment critic and recommendation AI.

USER'S LETTERBOXD:
Recent (watched/rated):
{activity_text}

Watchlist:
{watchlist_text}

Give personalized recommendations. Be specific about WHY they'd like each one."""
        
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
        chat_box = st.container(height=400)
        with chat_box:
            for msg in st.session_state.ent_chat:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
    else:
        st.info("💡 Ask for recommendations based on your taste!")

# ============================================================
# RIGHT COLUMN: Games
# ============================================================

with right_col:
    # --- GAMES TO PLAY ---
    st.subheader("🎮 Games To Play")
    
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
                if st.button("✓", key=f"game_{game.get('id')}"):
                    try:
                        sm.update_watchlist_item(game.get("id"), {"status": "completed"})
                        st.rerun()
                    except:
                        pass
    else:
        st.caption("No games in list")
    
    with st.expander("➕ Add Game"):
        new_game = st.text_input("Game title", key="new_game")
        if st.button("Add", key="add_game"):
            if new_game:
                try:
                    sm.add_to_watchlist(title=new_game, media_type="game")
                    st.success("Added!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    
    st.divider()
    
    # --- STEAM TOP 20 ---
    st.subheader("🔥 Steam Popular Games")
    st.caption("[View live chart on Steam →](https://store.steampowered.com/charts/topselling/global)")
    
    # Curated list of popular Steam games (API blocked on Streamlit Cloud)
    steam_popular = [
        {"name": "Counter-Strike 2", "link": "https://store.steampowered.com/app/730"},
        {"name": "Dota 2", "link": "https://store.steampowered.com/app/570"},
        {"name": "PUBG: Battlegrounds", "link": "https://store.steampowered.com/app/578080"},
        {"name": "Elden Ring", "link": "https://store.steampowered.com/app/1245620"},
        {"name": "Baldur's Gate 3", "link": "https://store.steampowered.com/app/1086940"},
        {"name": "Grand Theft Auto V", "link": "https://store.steampowered.com/app/271590"},
        {"name": "Apex Legends", "link": "https://store.steampowered.com/app/1172470"},
        {"name": "Rust", "link": "https://store.steampowered.com/app/252490"},
        {"name": "Team Fortress 2", "link": "https://store.steampowered.com/app/440"},
        {"name": "Destiny 2", "link": "https://store.steampowered.com/app/1085660"},
        {"name": "Path of Exile", "link": "https://store.steampowered.com/app/238960"},
        {"name": "Warframe", "link": "https://store.steampowered.com/app/230410"},
        {"name": "Monster Hunter: World", "link": "https://store.steampowered.com/app/582010"},
        {"name": "Cyberpunk 2077", "link": "https://store.steampowered.com/app/1091500"},
        {"name": "The Witcher 3", "link": "https://store.steampowered.com/app/292030"},
        {"name": "Hogwarts Legacy", "link": "https://store.steampowered.com/app/990080"},
        {"name": "Palworld", "link": "https://store.steampowered.com/app/1623730"},
        {"name": "Lethal Company", "link": "https://store.steampowered.com/app/1966720"},
        {"name": "Red Dead Redemption 2", "link": "https://store.steampowered.com/app/1174180"},
        {"name": "Stardew Valley", "link": "https://store.steampowered.com/app/413150"},
    ]
    
    box = st.container(height=350)
    with box:
        for i, game in enumerate(steam_popular, 1):
            name = game.get("name", "?")
            link = game.get("link", "")
            st.write(f"**{i}.** [{name}]({link})")
    
    st.divider()
    
    # --- GAME RELEASES ---
    st.subheader("📅 Game News & Releases")
    
    try:
        releases = et.get_major_game_releases()
        if releases:
            for release in releases[:8]:
                title = release.get("title", "?")
                link = release.get("link", "")
                source = release.get("source", "")
                st.write(f"[{title}]({link}) · *{source}*")
        else:
            st.caption("No release news")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # --- MMORPG NEWS ---
    st.subheader("⚔️ MMORPG News")
    
    try:
        mmo_news = et.get_mmorpg_news()
        if mmo_news:
            for news in mmo_news[:8]:
                title = news.get("title", "?")
                link = news.get("link", "")
                source = news.get("source", "")
                st.write(f"[{title}]({link}) · *{source}*")
        else:
            st.caption("No MMO news")
    except Exception as e:
        st.caption(f"Error: {e}")
    
    st.divider()
    
    # --- QUICK LINKS ---
    st.subheader("🔗 Gaming Links")
    st.markdown("""
- [Steam Store](https://store.steampowered.com/)
- [IGN](https://www.ign.com/)
- [Metacritic](https://www.metacritic.com/game)
- [HowLongToBeat](https://howlongtobeat.com/)
    """)

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
