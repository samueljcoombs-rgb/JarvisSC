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
# Image/Poster Helpers
# ============================================================

def get_poster_url(poster_path: str, size: str = "w500") -> Optional[str]:
    """Get full poster URL from TMDB poster path.
    
    Args:
        poster_path: The poster path from TMDB (e.g., "/abc123.jpg")
        size: Image size - w92, w154, w185, w342, w500, w780, original
    
    Returns:
        Full URL to the poster image, or None if no path provided
    """
    if not poster_path:
        return None
    return f"{TMDB_IMG_BASE}/{size}{poster_path}"

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
def get_vue_cinema_listings(venue_id: str = "10032") -> List[Dict]:
    """
    Get what's showing at a Vue cinema.
    Default venue_id 10032 = Vue Basingstoke
    Other venues: 10029=Reading, 10084=Camberley, 10031=Farnborough
    """
    try:
        # Vue's API endpoint for cinema listings
        url = f"https://www.myvue.com/api/microservice/showings/cinemas/{venue_id}/films"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            films = []
            for film in data.get("films", data) if isinstance(data, dict) else data:
                if isinstance(film, dict):
                    films.append({
                        "title": film.get("title", film.get("name", "Unknown")),
                        "poster": film.get("posterUrl", film.get("poster", "")),
                        "rating": film.get("rating", ""),
                        "runtime": film.get("runningTime", film.get("runtime", "")),
                        "synopsis": film.get("synopsis", "")[:150] if film.get("synopsis") else ""
                    })
            return films
    except Exception as e:
        pass
    
    # Fallback: return now_playing as proxy for Vue listings
    return []

@st.cache_data(ttl=3600)
def get_upcoming_releases_2025(region: str = "GB") -> List[Dict]:
    """Get upcoming major releases for the rest of the year with release dates."""
    movies = []
    # Get multiple pages of upcoming
    for page in range(1, 5):
        data = _tmdb_get("/movie/upcoming", {
            "region": region,
            "page": page,
            "language": "en-GB"
        })
        if data and "results" in data:
            movies.extend(data["results"])
    
    # Filter to only include movies with decent popularity (major releases)
    major_releases = [m for m in movies if m.get("popularity", 0) > 20]
    
    # Sort by release date
    major_releases.sort(key=lambda x: x.get("release_date", "9999"))
    return major_releases

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
# Letterboxd Integration (via RSS + HTML scraping)
# ============================================================

@st.cache_data(ttl=600)
def get_letterboxd_activity(username: str = None) -> Dict[str, List[Dict]]:
    """Get Letterboxd activity via RSS and watchlist via HTML scraping (with pagination)."""
    import requests
    from bs4 import BeautifulSoup
    
    username = username or _letterboxd_username()
    if not username:
        return {"error": "No Letterboxd username configured", "activity": [], "watchlist": []}
    
    result = {"activity": [], "watchlist": [], "username": username}
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    # Activity feed via RSS (this works!)
    try:
        activity_url = f"https://letterboxd.com/{username}/rss/"
        resp = requests.get(activity_url, headers=headers, timeout=10)
        result["activity_status"] = resp.status_code
        
        if resp.status_code == 200:
            feed = feedparser.parse(resp.text)
            for entry in feed.entries[:25]:
                film_title = getattr(entry, 'letterboxd_filmtitle', '') or ''
                film_year = getattr(entry, 'letterboxd_filmyear', '') or ''
                
                if not film_title:
                    raw_title = entry.get("title", "")
                    if ", " in raw_title:
                        film_title = raw_title.split(", ")[0]
                    else:
                        film_title = raw_title
                
                if film_title:
                    result["activity"].append({
                        "title": str(film_title).strip(),
                        "year": str(film_year).strip() if film_year else "",
                        "link": entry.get("link", ""),
                        "has_rating": "★" in entry.get("title", "")
                    })
    except Exception as e:
        result["activity_error"] = str(e)
    
    # WATCHLIST via HTML scraping with PAGINATION
    try:
        result["pages_scraped"] = 0
        
        for page_num in range(1, 6):  # Scrape up to 5 pages (280 films max)
            if page_num == 1:
                watchlist_url = f"https://letterboxd.com/{username}/watchlist/"
            else:
                watchlist_url = f"https://letterboxd.com/{username}/watchlist/page/{page_num}/"
            
            result["watchlist_url"] = watchlist_url
            
            resp = requests.get(watchlist_url, headers=headers, timeout=10)
            result["watchlist_status"] = resp.status_code
            
            if resp.status_code != 200:
                break
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Find all film posters on this page
            posters = soup.select('li.poster-container')
            
            if not posters:
                # No more films, stop paginating
                break
            
            result["pages_scraped"] = page_num
            
            for poster in posters:
                film_div = poster.select_one('div.film-poster')
                if film_div:
                    slug = film_div.get('data-film-slug', '')
                    img = film_div.select_one('img')
                    title = img.get('alt', '') if img else ''
                    
                    if not title and slug:
                        title = slug.replace('-', ' ').title()
                    
                    link = f"https://letterboxd.com/film/{slug}/" if slug else ""
                    
                    if title:
                        result["watchlist"].append({
                            "title": title.strip(),
                            "year": "",
                            "link": link
                        })
            
            # Check if there's a next page
            next_link = soup.select_one('.paginate-nextprev a.next')
            if not next_link:
                break
                        
    except Exception as e:
        result["watchlist_error"] = str(e)
    
    result["posters_found"] = len(result["watchlist"])
    return result

# ============================================================
# Entertainment News
# ============================================================

@st.cache_data(ttl=1800)
def get_entertainment_news() -> List[Dict]:
    """Get entertainment news from various RSS sources with images."""
    import re
    news = []
    
    # Entertainment RSS feeds - more sources
    feeds = [
        ("Variety", "https://variety.com/feed/"),
        ("IGN", "https://feeds.feedburner.com/ign/all"),
        ("The Verge", "https://www.theverge.com/rss/entertainment/index.xml"),
        ("Polygon", "https://www.polygon.com/rss/index.xml"),
        ("Kotaku", "https://kotaku.com/rss"),
        ("Eurogamer", "https://www.eurogamer.net/feed"),
        ("Screen Rant", "https://screenrant.com/feed/"),
        ("Collider", "https://collider.com/feed/"),
        ("Deadline", "https://deadline.com/feed/"),
    ]
    
    for source, url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                # Try to extract image from various sources
                image = None
                
                # 1. Check media_content
                if hasattr(entry, 'media_content') and entry.media_content:
                    for media in entry.media_content:
                        if media.get('medium') == 'image' or media.get('type', '').startswith('image'):
                            image = media.get('url')
                            break
                
                # 2. Check media_thumbnail
                if not image and hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
                    image = entry.media_thumbnail[0].get('url')
                
                # 3. Check enclosures
                if not image and hasattr(entry, 'enclosures') and entry.enclosures:
                    for enc in entry.enclosures:
                        if enc.get('type', '').startswith('image'):
                            image = enc.get('href') or enc.get('url')
                            break
                
                # 4. Try to extract from description/content HTML
                if not image:
                    content = entry.get('summary', '') or entry.get('description', '')
                    img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content)
                    if img_match:
                        image = img_match.group(1)
                
                news.append({
                    "source": source,
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": (entry.get("summary", "") or "")[:200],
                    "image": image
                })
        except Exception:
            continue
    
    # Sort by date (newest first)
    news.sort(key=lambda x: x.get("published", ""), reverse=True)
    return news[:60]

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

# ============================================================
# Vue Cinema Scraping
# ============================================================

@st.cache_data(ttl=3600)
def get_vue_cinema_listings(cinema_slug: str = "basingstoke-festival-place") -> List[Dict]:
    """
    Scrape Vue cinema listings from their website.
    Returns list of films with title, link, and image.
    """
    import requests
    from bs4 import BeautifulSoup
    
    url = f"https://www.myvue.com/cinema/{cinema_slug}/whats-on"
    films = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Vue uses film cards - look for common patterns
        # Try multiple selectors
        film_cards = soup.select('.film-card, .movie-card, [data-film], .film-poster-wrapper, article')
        
        for card in film_cards[:20]:
            film = {}
            
            # Try to get title
            title_el = card.select_one('h2, h3, .film-title, .movie-title, [data-title]')
            if title_el:
                film['title'] = title_el.get_text(strip=True)
            
            # Try to get link
            link_el = card.select_one('a[href*="/film/"]')
            if link_el:
                href = link_el.get('href', '')
                if href.startswith('/'):
                    href = f"https://www.myvue.com{href}"
                film['link'] = href
            
            # Try to get image
            img_el = card.select_one('img')
            if img_el:
                film['image'] = img_el.get('src') or img_el.get('data-src', '')
            
            if film.get('title'):
                films.append(film)
        
        # Deduplicate by title
        seen = set()
        unique_films = []
        for f in films:
            if f['title'] not in seen:
                seen.add(f['title'])
                unique_films.append(f)
        
        return unique_films
        
    except Exception as e:
        return [{"error": str(e)}]
