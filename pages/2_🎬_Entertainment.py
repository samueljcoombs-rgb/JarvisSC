# pages/2_🎬_Entertainment.py
"""
Entertainment Dashboard - Movies, TV Shows, Gaming, and Letterboxd integration.
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
# Custom Styling - Premium Design
# ============================================================

st.markdown("""
<style>
/* Animated gradient background */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

.ent-header {
    background: linear-gradient(-45deg, #ec4899, #8b5cf6, #6366f1, #a855f7);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    padding: 2rem 2.5rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 60px rgba(139, 92, 246, 0.4);
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
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
    animation: float 6s ease-in-out infinite;
}

.ent-header h1 {
    color: white;
    margin: 0;
    font-size: 2.5rem;
    font-weight: 900;
    text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.ent-header p {
    color: rgba(255,255,255,0.9);
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    font-weight: 500;
    position: relative;
    z-index: 1;
}

/* Movie/TV Cards - Premium Glassmorphism */
.movie-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.02) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}

.movie-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: left 0.5s;
}

.movie-card:hover::before {
    left: 100%;
}

.movie-card:hover {
    border-color: rgba(139, 92, 246, 0.6);
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 40px rgba(139, 92, 246, 0.3);
}

.movie-poster {
    width: 100%;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}

.movie-title {
    font-size: 1rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.25rem;
}

.movie-meta {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.6);
}

.rating-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
}

/* Watchlist Card */
.watchlist-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}

.watchlist-card:hover {
    border-color: rgba(59, 130, 246, 0.4);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
}

/* Letterboxd Card - Orange/Green gradient (Letterboxd colors) */
.letterboxd-card {
    background: linear-gradient(135deg, rgba(255, 128, 0, 0.2) 0%, rgba(0, 224, 0, 0.15) 100%);
    border: 1px solid rgba(255, 128, 0, 0.4);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.letterboxd-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255, 128, 0, 0.3);
}

.letterboxd-card a {
    color: #ff8000;
    text-decoration: none;
    font-weight: 600;
}

.letterboxd-card a:hover {
    text-decoration: underline;
}

/* News Card */
.news-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
}

.news-card:hover {
    border-color: rgba(236, 72, 153, 0.4);
    box-shadow: 0 8px 25px rgba(236, 72, 153, 0.2);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    border: 1px solid rgba(255,255,255,0.1);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(236, 72, 153, 0.3));
    border-color: rgba(139, 92, 246, 0.5);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="ent-header">
    <h1>🎬 Entertainment</h1>
    <p>✨ Movies, TV shows, gaming news, and your watchlist</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Main Layout
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Movies",
    "📺 TV Shows", 
    "📋 My Watchlist",
    "🎭 Letterboxd",
    "📰 Entertainment News"
])

# ============================================================
# Tab 1: Movies
# ============================================================

with tab1:
    st.subheader("🎬 Movies")
    
    # Search
    search_col, spacer = st.columns([2, 3])
    with search_col:
        movie_search = st.text_input("🔍 Search movies", placeholder="e.g., Inception", key="movie_search")
    
    if movie_search:
        with st.spinner("Searching..."):
            try:
                results = et.search_movie(movie_search)
                if results:
                    cols = st.columns(5)
                    for i, movie in enumerate(results[:10]):
                        with cols[i % 5]:
                            poster = et.get_poster_url(movie.get("poster_path"))
                            title = movie.get("title", "Unknown")
                            year = (movie.get("release_date") or "")[:4]
                            rating = movie.get("vote_average", 0)
                            movie_id = movie.get("id")
                            
                            if poster:
                                st.image(poster, use_container_width=True)
                            st.markdown(f"**{title}** ({year})")
                            st.markdown(f"<span class='rating-badge'>⭐ {rating:.1f}</span>", unsafe_allow_html=True)
                            
                            if st.button("➕ Add to Watchlist", key=f"add_movie_{movie_id}"):
                                result = sm.add_to_watchlist(
                                    title=title,
                                    content_type="movie",
                                    status="to_watch"
                                )
                                if result.get("ok"):
                                    st.success("Added!")
                else:
                    st.info("No movies found")
            except Exception as e:
                st.error(f"Error searching: {e}")
    else:
        # Show trending/upcoming
        sub_tab1, sub_tab2 = st.tabs(["🔥 Trending", "📅 Upcoming"])
        
        with sub_tab1:
            try:
                trending = et.get_trending_movies()
                if trending:
                    cols = st.columns(5)
                    for i, movie in enumerate(trending[:10]):
                        with cols[i % 5]:
                            poster = et.get_poster_url(movie.get("poster_path"))
                            title = movie.get("title", "Unknown")
                            year = (movie.get("release_date") or "")[:4]
                            rating = movie.get("vote_average", 0)
                            movie_id = movie.get("id")
                            
                            if poster:
                                st.image(poster, use_container_width=True)
                            st.markdown(f"**{title}**")
                            st.caption(f"{year} • ⭐ {rating:.1f}")
                            
                            if st.button("➕ Watchlist", key=f"trend_movie_{movie_id}"):
                                sm.add_to_watchlist(title=title, content_type="movie", status="to_watch")
                                st.success("Added!")
            except Exception as e:
                st.warning(f"Could not load trending movies: {e}")
        
        with sub_tab2:
            try:
                upcoming = et.get_upcoming_movies()
                if upcoming:
                    cols = st.columns(5)
                    for i, movie in enumerate(upcoming[:10]):
                        with cols[i % 5]:
                            poster = et.get_poster_url(movie.get("poster_path"))
                            title = movie.get("title", "Unknown")
                            release = movie.get("release_date", "TBA")
                            movie_id = movie.get("id")
                            
                            if poster:
                                st.image(poster, use_container_width=True)
                            st.markdown(f"**{title}**")
                            st.caption(f"📅 {release}")
                            
                            if st.button("➕ Watchlist", key=f"up_movie_{movie_id}"):
                                sm.add_to_watchlist(title=title, content_type="movie", status="to_watch")
                                st.success("Added!")
            except Exception as e:
                st.warning(f"Could not load upcoming movies: {e}")

# ============================================================
# Tab 2: TV Shows
# ============================================================

with tab2:
    st.subheader("📺 TV Shows")
    
    # Search
    search_col, spacer = st.columns([2, 3])
    with search_col:
        tv_search = st.text_input("🔍 Search TV shows", placeholder="e.g., Breaking Bad", key="tv_search")
    
    if tv_search:
        with st.spinner("Searching..."):
            try:
                results = et.search_tv(tv_search)
                if results:
                    cols = st.columns(5)
                    for i, show in enumerate(results[:10]):
                        with cols[i % 5]:
                            poster = et.get_poster_url(show.get("poster_path"))
                            title = show.get("name", "Unknown")
                            year = (show.get("first_air_date") or "")[:4]
                            rating = show.get("vote_average", 0)
                            show_id = show.get("id")
                            
                            if poster:
                                st.image(poster, use_container_width=True)
                            st.markdown(f"**{title}** ({year})")
                            st.markdown(f"<span class='rating-badge'>⭐ {rating:.1f}</span>", unsafe_allow_html=True)
                            
                            if st.button("➕ Track Show", key=f"add_tv_{show_id}"):
                                result = sm.add_to_watchlist(
                                    title=title,
                                    content_type="tv",
                                    status="watching",
                                    season=1,
                                    episode=1
                                )
                                if result.get("ok"):
                                    st.success("Added!")
                else:
                    st.info("No shows found")
            except Exception as e:
                st.error(f"Error searching: {e}")
    else:
        # Show trending/popular
        sub_tab1, sub_tab2 = st.tabs(["🔥 Trending", "⭐ Popular"])
        
        with sub_tab1:
            try:
                trending = et.get_trending_tv()
                if trending:
                    cols = st.columns(5)
                    for i, show in enumerate(trending[:10]):
                        with cols[i % 5]:
                            poster = et.get_poster_url(show.get("poster_path"))
                            title = show.get("name", "Unknown")
                            year = (show.get("first_air_date") or "")[:4]
                            rating = show.get("vote_average", 0)
                            show_id = show.get("id")
                            
                            if poster:
                                st.image(poster, use_container_width=True)
                            st.markdown(f"**{title}**")
                            st.caption(f"{year} • ⭐ {rating:.1f}")
                            
                            if st.button("➕ Track", key=f"trend_tv_{show_id}"):
                                sm.add_to_watchlist(title=title, content_type="tv", status="watching", season=1, episode=1)
                                st.success("Added!")
            except Exception as e:
                st.warning(f"Could not load trending shows: {e}")
        
        with sub_tab2:
            try:
                popular = et.get_popular_tv()
                if popular:
                    cols = st.columns(5)
                    for i, show in enumerate(popular[:10]):
                        with cols[i % 5]:
                            poster = et.get_poster_url(show.get("poster_path"))
                            title = show.get("name", "Unknown")
                            year = (show.get("first_air_date") or "")[:4]
                            rating = show.get("vote_average", 0)
                            show_id = show.get("id")
                            
                            if poster:
                                st.image(poster, use_container_width=True)
                            st.markdown(f"**{title}**")
                            st.caption(f"{year} • ⭐ {rating:.1f}")
                            
                            if st.button("➕ Track", key=f"pop_tv_{show_id}"):
                                sm.add_to_watchlist(title=title, content_type="tv", status="watching", season=1, episode=1)
                                st.success("Added!")
            except Exception as e:
                st.warning(f"Could not load popular shows: {e}")

# ============================================================
# Tab 3: My Watchlist
# ============================================================

with tab3:
    st.subheader("📋 My Watchlist")
    
    # Add manually
    with st.expander("➕ Add Manually", expanded=False):
        with st.form("add_watchlist", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Title", key="wl_title")
                content_type = st.selectbox("Type", ["movie", "tv", "documentary"], key="wl_type")
            with col2:
                status = st.selectbox("Status", ["to_watch", "watching", "watched"], key="wl_status")
                platform = st.text_input("Platform (optional)", placeholder="Netflix, Disney+", key="wl_platform")
            
            if content_type == "tv":
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    season = st.number_input("Season", min_value=1, value=1, key="wl_season")
                with tcol2:
                    episode = st.number_input("Episode", min_value=1, value=1, key="wl_episode")
            else:
                season, episode = None, None
            
            notes = st.text_area("Notes (optional)", key="wl_notes", height=60)
            
            submitted = st.form_submit_button("➕ Add to Watchlist", use_container_width=True)
            if submitted and title:
                result = sm.add_to_watchlist(
                    title=title,
                    content_type=content_type,
                    status=status,
                    platform=platform if platform else None,
                    season=season,
                    episode=episode,
                    notes=notes if notes else None
                )
                if result.get("ok"):
                    st.success("Added!")
                    st.rerun()
    
    st.divider()
    
    # Display watchlist
    try:
        watchlist = sm.get_watchlist()
        
        if watchlist:
            # Group by status
            to_watch = [w for w in watchlist if w.get("status") == "to_watch"]
            watching = [w for w in watchlist if w.get("status") == "watching"]
            watched = [w for w in watchlist if w.get("status") == "watched"]
            
            wl_tab1, wl_tab2, wl_tab3 = st.tabs([
                f"📋 To Watch ({len(to_watch)})",
                f"▶️ Watching ({len(watching)})",
                f"✅ Watched ({len(watched)})"
            ])
            
            with wl_tab1:
                if to_watch:
                    for item in to_watch:
                        item_id = item.get("id", "")
                        title = item.get("title", "Unknown")
                        content_type = item.get("type", "movie")
                        platform = item.get("platform", "")
                        
                        st.markdown(f"""
                        <div class="watchlist-card">
                            <strong>{title}</strong>
                            <span style="opacity:0.6">({content_type.upper()})</span>
                            {f'<br><small>📺 {platform}</small>' if platform else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if st.button("▶️ Start Watching", key=f"start_{item_id}"):
                                sm.update_watchlist_item(item_id, status="watching")
                                st.rerun()
                        with col2:
                            if st.button("✅ Mark Watched", key=f"watched_{item_id}"):
                                sm.update_watchlist_item(item_id, status="watched")
                                st.rerun()
                        with col3:
                            if st.button("🗑️ Remove", key=f"remove_{item_id}"):
                                sm.delete_watchlist_item(item_id)
                                st.rerun()
                else:
                    st.info("Nothing in your watchlist yet!")
            
            with wl_tab2:
                if watching:
                    for item in watching:
                        item_id = item.get("id", "")
                        title = item.get("title", "Unknown")
                        content_type = item.get("type", "movie")
                        season = item.get("season", "")
                        episode = item.get("episode", "")
                        
                        progress_text = f"S{season}E{episode}" if season else ""
                        
                        st.markdown(f"""
                        <div class="watchlist-card">
                            <strong>{title}</strong>
                            <span style="opacity:0.6">({content_type.upper()})</span>
                            {f'<br><small>Progress: {progress_text}</small>' if progress_text else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if content_type == "tv":
                            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                            with col1:
                                new_season = st.number_input("S", min_value=1, value=int(season) if season else 1, key=f"s_{item_id}", label_visibility="collapsed")
                            with col2:
                                new_ep = st.number_input("E", min_value=1, value=int(episode) if episode else 1, key=f"e_{item_id}", label_visibility="collapsed")
                            with col3:
                                if st.button("📝 Update", key=f"upd_{item_id}"):
                                    sm.update_watchlist_item(item_id, season=new_season, episode=new_ep)
                                    st.rerun()
                            with col4:
                                if st.button("✅ Done", key=f"done_{item_id}"):
                                    sm.update_watchlist_item(item_id, status="watched")
                                    st.rerun()
                        else:
                            if st.button("✅ Mark as Watched", key=f"finish_{item_id}"):
                                sm.update_watchlist_item(item_id, status="watched")
                                st.rerun()
                else:
                    st.info("Nothing currently watching")
            
            with wl_tab3:
                if watched:
                    for item in watched:
                        title = item.get("title", "Unknown")
                        content_type = item.get("type", "movie")
                        st.write(f"✅ **{title}** ({content_type})")
                else:
                    st.info("No watched items yet")
        else:
            st.info("Your watchlist is empty. Search for movies or TV shows to add!")
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")

# ============================================================
# Tab 4: Letterboxd
# ============================================================

with tab4:
    st.subheader("🎭 Letterboxd Integration")
    
    # Get username from secrets/env
    letterboxd_user = os.getenv("LETTERBOXD_USERNAME") or st.secrets.get("LETTERBOXD_USERNAME", "")
    
    if not letterboxd_user:
        st.warning("Add LETTERBOXD_USERNAME to your secrets to see your activity!")
        letterboxd_user = st.text_input("Or enter your Letterboxd username:", key="lb_user")
    
    if letterboxd_user:
        st.write(f"**User:** @{letterboxd_user}")
        
        try:
            data = et.get_letterboxd_activity(letterboxd_user)
            
            # Get the activity list from the dict
            activity = data.get("activity", []) if isinstance(data, dict) else []
            
            if activity:
                st.write("**Recent Activity:**")
                for item in activity[:10]:
                    title = item.get("title", "Unknown")
                    link = item.get("link", "")
                    published = item.get("published", "")
                    has_rating = item.get("has_rating", False)
                    
                    st.markdown(f"""
                    <div class="letterboxd-card">
                        <strong>{title}</strong>
                        <br><small>{published}</small>
                        {f'<a href="{link}" target="_blank">View on Letterboxd</a>' if link else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Extract film title from full title (remove rating stars etc)
                    film_title = title.split(" - ")[0] if " - " in title else title
                    
                    # Add to watchlist button
                    if st.button(f"➕ Add to Watchlist", key=f"lb_{hash(title[:50])}"):
                        sm.add_to_watchlist(title=film_title, content_type="movie", status="to_watch")
                        st.success("Added!")
            else:
                if data.get("activity_error"):
                    st.warning(f"Could not fetch activity: {data.get('activity_error')}")
                else:
                    st.info("No recent activity found")
                    
            # Also show watchlist if available
            watchlist = data.get("watchlist", []) if isinstance(data, dict) else []
            if watchlist:
                with st.expander(f"📋 Letterboxd Watchlist ({len(watchlist)} films)"):
                    for item in watchlist[:20]:
                        title = item.get("title", "Unknown")
                        year = item.get("year", "")
                        st.write(f"• {title} ({year})" if year else f"• {title}")
                        
        except Exception as e:
            st.error(f"Error loading Letterboxd: {e}")
    
    st.divider()
    st.markdown("""
    **Note:** Letterboxd integration uses the public RSS feed. Make sure your diary is set to public.
    
    To set up:
    1. Go to your Letterboxd settings
    2. Make sure your profile/diary is public
    3. Add `LETTERBOXD_USERNAME` to your Streamlit secrets
    """)

# ============================================================
# Tab 5: Entertainment News
# ============================================================

with tab5:
    st.subheader("📰 Entertainment News")
    
    try:
        news = et.get_entertainment_news()
        
        if news:
            for article in news[:15]:
                title = article.get("title", "Unknown")
                source = article.get("source", "")
                date = article.get("date", "")
                link = article.get("link", "")
                summary = article.get("summary", "")[:200] + "..." if article.get("summary") else ""
                
                st.markdown(f"""
                <div class="news-card">
                    <strong>{title}</strong>
                    <br><small>{source} • {date}</small>
                    <br><span style="opacity:0.8">{summary}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if link:
                    st.link_button("Read More →", link, key=f"news_{hash(title)}")
                st.write("")
        else:
            st.info("No entertainment news available")
    except Exception as e:
        st.warning(f"Could not load entertainment news: {e}")

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("🎬 Quick Stats")
    
    try:
        watchlist = sm.get_watchlist()
        to_watch = len([w for w in watchlist if w.get("status") == "to_watch"])
        watching = len([w for w in watchlist if w.get("status") == "watching"])
        watched = len([w for w in watchlist if w.get("status") == "watched"])
        
        st.metric("To Watch", to_watch)
        st.metric("Currently Watching", watching)
        st.metric("Watched", watched)
    except Exception:
        st.info("Start adding to your watchlist!")
    
    st.divider()
    st.page_link("app.py", label="🏠 Home", icon="🏠")
