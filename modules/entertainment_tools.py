# modules/entertainment_tools.py
"""
Entertainment Tools for Jarvis
Movies, TV Shows, Gaming, Letterboxd integration, and watchlist management.
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

try:
    from modules import sheets_memory as sm
except ImportError:
    import sheets_memory as sm

TZ = ZoneInfo("Europe/London")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p"

# ============================================================
# API Helpers
# ============================================================

def _tmdb_api_key() -> Optional[str]:
    return st.secrets.get("TMDB_API_KEY") or os.getenv("TMDB_API_KEY")

def _letterboxd_username() -> Optional[str]:
    return st.secrets.get("LETTERBOXD_USERNAME") or os.getenv("LETTERBOXD_USERNAME")

def _tmdb_get(endpoint: str, params: Dict = None) -> Optional[Dict]:
    """Make a TMDB API request."""
    api_key = _tmdb_api_key()
    if not api_key:
        return None
    
    params = params or {}
    params["api_key"] = api_key
    
    try:
        r = requests.get(f"{TMDB_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"TMDB API error: {e}")
        return None

# ============================================================
# Movie Functions
# ============================================================

@st.cache_data(ttl=3600)
def get_upcoming_movies(region: str = "GB", pages: int = 2) -> List[Dict]:
    """Get upcoming movies from TMDB."""
    movies = []
    for page in range(1, pages + 1):
        data = _tmdb_get("/movie/upcoming", {
            "region": region,
            "page": page,
            "language": "en-GB"
        })
        if data and "results" in data:
            movies.extend(data["results"])
    
    # Sort by release date
    movies.sort(key=lambda x: x.get("release_date", "9999"))
    return movies

@st.cache_data(ttl=3600)
def get_now_playing(region: str = "GB") -> List[Dict]:
    """Get movies currently in theaters."""
    data = _tmdb_get("/movie/now_playing", {
        "region": region,
        "language": "en-GB"
    })
    return data.get("results", []) if data else []

@st.cache_data(ttl=3600)
def get_trending_movies(time_window: str = "week") -> List[Dict]:
    """Get trending movies."""
    data = _tmdb_get(f"/trending/movie/{time_window}")
    return data.get("results", []) if data else []

@st.cache_data(ttl=3600)
def search_movie(query: str) -> List[Dict]:
    """Search for a movie."""
    data = _tmdb_get("/search/movie", {"query": query})
    return data.get("results", []) if data else []

def get_movie_details(movie_id: int) -> Optional[Dict]:
    """Get detailed movie info."""
    return _tmdb_get(f"/movie/{movie_id}", {
        "append_to_response": "credits,videos,watch/providers"
    })

# ============================================================
# TV Show Functions
# ============================================================

@st.cache_data(ttl=3600)
def get_trending_tv(time_window: str = "week") -> List[Dict]:
    """Get trending TV shows."""
    data = _tmdb_get(f"/trending/tv/{time_window}")
    return data.get("results", []) if data else []

@st.cache_data(ttl=3600)
def get_popular_tv() -> List[Dict]:
    """Get popular TV shows."""
    data = _tmdb_get("/tv/popular")
    return data.get("results", []) if data else []

@st.cache_data(ttl=3600)
def search_tv(query: str) -> List[Dict]:
    """Search for a TV show."""
    data = _tmdb_get("/search/tv", {"query": query})
    return data.get("results", []) if data else []

def get_tv_details(tv_id: int) -> Optional[Dict]:
    """Get detailed TV show info."""
    return _tmdb_get(f"/tv/{tv_id}", {
        "append_to_response": "credits,videos"
    })

@st.cache_data(ttl=3600)
def get_tv_season(tv_id: int, season_num: int) -> Optional[Dict]:
    """Get season details including episodes."""
    return _tmdb_get(f"/tv/{tv_id}/season/{season_num}")

# ============================================================
# Letterboxd Integration (via RSS)
# ============================================================

@st.cache_data(ttl=1800)
def get_letterboxd_activity(username: str = None) -> Dict[str, List[Dict]]:
    """Get Letterboxd activity via RSS feed."""
    username = username or _letterboxd_username()
    if not username:
        return {"error": "No Letterboxd username configured", "activity": [], "watchlist": []}
    
    result = {"activity": [], "watchlist": [], "username": username}
    
    # Activity feed (diary/reviews)
    try:
        feed = feedparser.parse(f"https://letterboxd.com/{username}/rss/")
        for entry in feed.entries[:20]:
            item = {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "description": entry.get("description", "")[:200] if entry.get("description") else ""
            }
            # Try to extract rating if present
            if "★" in entry.get("title", ""):
                item["has_rating"] = True
            result["activity"].append(item)
    except Exception as e:
        result["activity_error"] = str(e)
    
    # Watchlist feed
    try:
        watchlist_feed = feedparser.parse(f"https://letterboxd.com/{username}/watchlist/rss/")
        for entry in watchlist_feed.entries[:20]:
            item = {
                "title": entry.get("letterboxd_filmtitle", entry.get("title", "")),
                "link": entry.get("link", ""),
                "year": entry.get("letterboxd_filmyear", "")
            }
            result["watchlist"].append(item)
    except Exception as e:
        result["watchlist_error"] = str(e)
    
    return result

# ============================================================
# Entertainment News
# ============================================================

@st.cache_data(ttl=1800)
def get_entertainment_news() -> List[Dict]:
    """Get entertainment news from various RSS sources."""
    news = []
    
    # Entertainment RSS feeds
    feeds = [
        ("Variety", "https://variety.com/feed/"),
        ("IGN", "https://feeds.feedburner.com/ign/all"),
        ("The Verge (Entertainment)", "https://www.theverge.com/rss/entertainment/index.xml"),
    ]
    
    for source, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                news.append({
                    "source": source,
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": (entry.get("summary", "") or "")[:150]
                })
        except Exception:
            continue
    
    # Sort by date (newest first)
    news.sort(key=lambda x: x.get("published", ""), reverse=True)
    return news[:15]

# ============================================================
# Watchlist Management
# ============================================================

def add_to_watchlist(title: str, media_type: str = "movie", 
                     tmdb_id: int = None, platform: str = "", notes: str = "") -> Dict:
    """Add item to watchlist in Google Sheets."""
    return sm.add_to_watchlist(
        title=title,
        media_type=media_type,
        platform=platform,
        notes=notes
    )

def get_watchlist(status: str = None, media_type: str = None) -> List[Dict]:
    """Get watchlist from Google Sheets."""
    return sm.get_watchlist(status=status, media_type=media_type)

def update_watchlist_item(item_id: str, updates: Dict) -> Dict:
    """Update a watchlist item."""
    return sm.update_watchlist_item(item_id, updates)

def mark_as_watched(item_id: str) -> Dict:
    """Mark a watchlist item as watched."""
    return sm.update_watchlist_item(item_id, {"status": "watched"})

# ============================================================
# TV Episode Tracking
# ============================================================

def track_show(title: str, current_season: int = 1, current_episode: int = 1,
               platform: str = "", notes: str = "") -> Dict:
    """Add a show to track."""
    return sm.add_to_watchlist(
        title=title,
        media_type="tv_tracking",
        season=current_season,
        episode=current_episode,
        platform=platform,
        notes=notes
    )

def update_show_progress(item_id: str, season: int, episode: int) -> Dict:
    """Update show progress."""
    return sm.update_watchlist_item(item_id, {
        "season": str(season),
        "episode": str(episode)
    })

def get_tracked_shows() -> List[Dict]:
    """Get shows being tracked."""
    return sm.get_watchlist(media_type="tv_tracking")

# ============================================================
# UI Components
# ============================================================

def _poster_url(path: str, size: str = "w342") -> str:
    """Get full poster URL."""
    if not path:
        return ""
    return f"{TMDB_IMG_BASE}/{size}{path}"

def render_movie_card(movie: Dict, show_add_btn: bool = True):
    """Render a movie card."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        poster = _poster_url(movie.get("poster_path"), "w185")
        if poster:
            st.image(poster, width=120)
        else:
            st.markdown("🎬")
    
    with col2:
        title = movie.get("title", "Unknown")
        year = (movie.get("release_date") or "")[:4]
        rating = movie.get("vote_average", 0)
        
        st.markdown(f"**{title}** ({year})")
        
        if rating:
            stars = "⭐" * int(rating / 2)
            st.caption(f"{stars} {rating:.1f}/10")
        
        overview = movie.get("overview", "")
        if overview:
            st.caption(overview[:150] + "..." if len(overview) > 150 else overview)
        
        if show_add_btn:
            if st.button("➕ Add to Watchlist", key=f"add_movie_{movie.get('id')}"):
                result = add_to_watchlist(
                    title=f"{title} ({year})",
                    media_type="movie",
                    tmdb_id=movie.get("id")
                )
                if result.get("ok"):
                    st.success("Added to watchlist!")
                else:
                    st.error("Failed to add")

def render_tv_card(show: Dict, show_add_btn: bool = True):
    """Render a TV show card."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        poster = _poster_url(show.get("poster_path"), "w185")
        if poster:
            st.image(poster, width=120)
        else:
            st.markdown("📺")
    
    with col2:
        title = show.get("name", "Unknown")
        year = (show.get("first_air_date") or "")[:4]
        rating = show.get("vote_average", 0)
        
        st.markdown(f"**{title}** ({year})")
        
        if rating:
            st.caption(f"⭐ {rating:.1f}/10")
        
        overview = show.get("overview", "")
        if overview:
            st.caption(overview[:150] + "..." if len(overview) > 150 else overview)
        
        if show_add_btn:
            if st.button("📺 Track Show", key=f"track_tv_{show.get('id')}"):
                result = track_show(
                    title=f"{title} ({year})",
                    platform=""
                )
                if result.get("ok"):
                    st.success("Now tracking!")
                else:
                    st.error("Failed to add")

def render_upcoming_movies():
    """Render upcoming movies section."""
    st.markdown("### 🎬 Upcoming Releases")
    
    movies = get_upcoming_movies()
    if not movies:
        st.info("No upcoming movies found. Check your TMDB_API_KEY.")
        return
    
    # Filter to actually upcoming
    today = datetime.now().date().isoformat()
    upcoming = [m for m in movies if (m.get("release_date") or "") >= today][:10]
    
    for movie in upcoming:
        with st.container():
            render_movie_card(movie)
            release = movie.get("release_date", "TBA")
            st.caption(f"📅 Release: {release}")
            st.divider()

def render_trending():
    """Render trending movies and TV."""
    tab1, tab2 = st.tabs(["🎬 Movies", "📺 TV Shows"])
    
    with tab1:
        movies = get_trending_movies()
        if movies:
            for movie in movies[:8]:
                render_movie_card(movie)
                st.divider()
        else:
            st.info("Could not load trending movies")
    
    with tab2:
        shows = get_trending_tv()
        if shows:
            for show in shows[:8]:
                render_tv_card(show)
                st.divider()
        else:
            st.info("Could not load trending shows")

def render_watchlist_manager():
    """Render watchlist management UI."""
    st.markdown("### 📝 My Watchlist")
    
    tab1, tab2, tab3 = st.tabs(["To Watch", "Watched", "TV Tracking"])
    
    with tab1:
        items = get_watchlist(status="to_watch")
        movies = [i for i in items if i.get("type") == "movie"]
        
        if movies:
            for item in movies:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"🎬 **{item.get('title')}**")
                    if item.get("notes"):
                        st.caption(item["notes"])
                with col2:
                    if item.get("platform"):
                        st.caption(f"📍 {item['platform']}")
                with col3:
                    if st.button("✅", key=f"watched_{item.get('id')}"):
                        mark_as_watched(item.get("id"))
                        st.rerun()
                st.divider()
        else:
            st.info("No movies in watchlist")
    
    with tab2:
        items = get_watchlist(status="watched")
        if items:
            for item in items[-10:]:
                st.markdown(f"✅ ~~{item.get('title')}~~")
        else:
            st.info("No watched items")
    
    with tab3:
        shows = get_tracked_shows()
        if shows:
            for show in shows:
                col1, col2 = st.columns([3, 1])
                with col1:
                    season = show.get("season", "1")
                    episode = show.get("episode", "1")
                    st.markdown(f"📺 **{show.get('title')}**")
                    st.caption(f"Season {season}, Episode {episode}")
                with col2:
                    new_ep = st.number_input(
                        "Ep",
                        min_value=1,
                        value=int(episode) if episode else 1,
                        key=f"ep_{show.get('id')}"
                    )
                    if new_ep != int(episode or 1):
                        if st.button("💾", key=f"save_ep_{show.get('id')}"):
                            update_show_progress(show.get("id"), int(season), new_ep)
                            st.rerun()
                st.divider()
        else:
            st.info("No shows being tracked")

def render_letterboxd():
    """Render Letterboxd integration."""
    st.markdown("### 🎬 Letterboxd")
    
    data = get_letterboxd_activity()
    
    if "error" in data:
        st.warning(data["error"])
        st.info("Add your username to LETTERBOXD_USERNAME in secrets")
        return
    
    tab1, tab2 = st.tabs(["📝 Recent Activity", "👀 Watchlist"])
    
    with tab1:
        if data.get("activity"):
            for item in data["activity"][:10]:
                title = item.get("title", "")
                link = item.get("link", "")
                st.markdown(f"[{title}]({link})")
                if item.get("description"):
                    st.caption(item["description"][:100])
                st.divider()
        else:
            st.info("No recent activity")
    
    with tab2:
        if data.get("watchlist"):
            for item in data["watchlist"]:
                title = item.get("title", "")
                year = item.get("year", "")
                link = item.get("link", "")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"[{title} ({year})]({link})")
                with col2:
                    if st.button("➕", key=f"lb_{hash(title)}"):
                        add_to_watchlist(f"{title} ({year})", "movie")
                        st.success("Added!")
        else:
            st.info("Watchlist empty or not public")

def render_search():
    """Render search functionality."""
    st.markdown("### 🔍 Search")
    
    query = st.text_input("Search movies or TV shows...", key="entertainment_search")
    search_type = st.radio("Type", ["Movies", "TV Shows"], horizontal=True, key="search_type")
    
    if query:
        if search_type == "Movies":
            results = search_movie(query)
            if results:
                for movie in results[:5]:
                    render_movie_card(movie)
                    st.divider()
            else:
                st.info("No movies found")
        else:
            results = search_tv(query)
            if results:
                for show in results[:5]:
                    render_tv_card(show)
                    st.divider()
            else:
                st.info("No TV shows found")

def render_entertainment_news():
    """Render entertainment news."""
    st.markdown("### 📰 Entertainment News")
    
    news = get_entertainment_news()
    
    if news:
        for item in news[:10]:
            source = item.get("source", "")
            title = item.get("title", "")
            link = item.get("link", "")
            
            st.markdown(f"**[{title}]({link})**")
            st.caption(f"📰 {source}")
            st.divider()
    else:
        st.info("Could not load news")
