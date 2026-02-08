# pages/4_✈️_Travel.py
"""
Travel Planning Dashboard - Plan trips, track flights, and discover deals.
"""
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os

# Import modules
try:
    from modules import sheets_memory as sm
    from modules import travel_tools as tt
    from modules import global_styles as gs
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules import sheets_memory as sm
    from modules import travel_tools as tt
    from modules import global_styles as gs

TZ = ZoneInfo("Europe/London")

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Travel | Jarvis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

gs.inject_global_styles()

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>
.travel-header {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0369a1 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(14, 165, 233, 0.3);
}
.travel-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
}
.travel-header p {
    color: rgba(255,255,255,0.85);
    margin: 0.25rem 0 0 0;
}
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
}
.stat-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #38bdf8;
}
.stat-card .label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.25rem;
}
.trip-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}
.trip-card:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(14, 165, 233, 0.3);
}
.trip-card.planning { border-left: 4px solid #f59e0b; }
.trip-card.booked { border-left: 4px solid #10b981; }
.trip-card.ongoing { border-left: 4px solid #3b82f6; }
.trip-card.completed { border-left: 4px solid #8b5cf6; opacity: 0.8; }
.flight-result {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(14, 165, 233, 0.15) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.flight-price {
    font-size: 1.5rem;
    font-weight: 800;
    color: #10b981;
}
.alert-card {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.deal-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.empty-state {
    text-align: center;
    padding: 2rem;
    color: rgba(255,255,255,0.5);
}
.status-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-planning { background: rgba(245, 158, 11, 0.3); color: #fcd34d; }
.status-booked { background: rgba(16, 185, 129, 0.3); color: #6ee7b7; }
.status-ongoing { background: rgba(59, 130, 246, 0.3); color: #93c5fd; }
.status-completed { background: rgba(139, 92, 246, 0.3); color: #c4b5fd; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="travel-header">
    <h1>✈️ Travel Hub</h1>
    <p>Plan adventures, track flights, find deals</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Quick Stats
# ============================================================

def render_stats():
    trips = sm.read_all_rows("travel_plans")
    alerts = sm.read_all_rows("flight_alerts")
    
    total_trips = len(trips)
    completed = sum(1 for t in trips if t.get("status") == "completed")
    upcoming = sum(1 for t in trips if t.get("status") in ["planning", "booked"])
    booked = sum(1 for t in trips if t.get("status") == "booked")
    active_alerts = sum(1 for a in alerts if a.get("status") == "active")
    
    # Unique destinations
    destinations = set(t.get("destination", "") for t in trips if t.get("destination"))
    
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="value">{total_trips}</div>
            <div class="label">Total Trips</div>
        </div>
        <div class="stat-card">
            <div class="value">{completed}</div>
            <div class="label">Completed</div>
        </div>
        <div class="stat-card">
            <div class="value">{booked}</div>
            <div class="label">Booked</div>
        </div>
        <div class="stat-card">
            <div class="value">{upcoming}</div>
            <div class="label">Upcoming</div>
        </div>
        <div class="stat-card">
            <div class="value">{len(destinations)}</div>
            <div class="label">Destinations</div>
        </div>
        <div class="stat-card">
            <div class="value">{active_alerts}</div>
            <div class="label">Price Alerts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_stats()

# ============================================================
# Tabs
# ============================================================

tab_trips, tab_search, tab_alerts, tab_plan = st.tabs([
    "🗺️ My Trips", "🔍 Search Flights", "🔔 Price Alerts", "➕ Plan Trip"
])

# ============================================================
# My Trips Tab
# ============================================================

with tab_trips:
    trips = sm.read_all_rows("travel_plans")
    
    # Filter
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "Planning", "Booked", "Ongoing", "Completed", "Cancelled"],
        key="trip_filter"
    )
    
    if status_filter != "All":
        trips = [t for t in trips if t.get("status", "").title() == status_filter]
    
    if not trips:
        st.markdown('<div class="empty-state">No trips found. Start planning your next adventure!</div>', unsafe_allow_html=True)
    else:
        # Sort by start date (upcoming first)
        def sort_key(t):
            try:
                return datetime.fromisoformat(t.get("start_date", "2099-12-31"))
            except:
                return datetime(2099, 12, 31)
        
        trips = sorted(trips, key=sort_key)
        
        for trip in trips:
            trip_id = trip.get("trip_id", "")
            status = trip.get("status", "planning")
            destination = trip.get("destination", "Unknown")
            start = trip.get("start_date", "")
            end = trip.get("end_date", "")
            budget = trip.get("budget", "")
            notes = trip.get("notes", "")
            
            # Format dates
            date_str = ""
            if start:
                try:
                    start_dt = datetime.fromisoformat(start)
                    date_str = start_dt.strftime("%d %b %Y")
                    if end:
                        end_dt = datetime.fromisoformat(end)
                        date_str += f" - {end_dt.strftime('%d %b %Y')}"
                        days = (end_dt - start_dt).days + 1
                        date_str += f" ({days} days)"
                except:
                    date_str = f"{start} to {end}"
            
            with st.expander(f"{'✈️' if status == 'booked' else '📍'} {destination}", expanded=status in ["booked", "ongoing"]):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f'<span class="status-badge status-{status}">{status.title()}</span>', unsafe_allow_html=True)
                    
                    if date_str:
                        st.markdown(f"📅 {date_str}")
                    if budget:
                        st.markdown(f"💰 Budget: £{budget}")
                    if notes:
                        st.markdown(f"📝 {notes}")
                
                with col2:
                    new_status = st.selectbox(
                        "Update Status",
                        ["planning", "booked", "ongoing", "completed", "cancelled"],
                        index=["planning", "booked", "ongoing", "completed", "cancelled"].index(status),
                        key=f"trip_status_{trip_id}"
                    )
                    if new_status != status:
                        sm.update_row_by_id("travel_plans", "trip_id", trip_id, {"status": new_status})
                        st.rerun()

# ============================================================
# Search Flights Tab
# ============================================================

with tab_search:
    st.markdown("### 🔍 Flight Search")
    st.caption("Search for flights using Google Flights data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        origin = st.selectbox(
            "From",
            list(tt.UK_AIRPORTS.keys()),
            format_func=lambda x: f"{x} - {tt.UK_AIRPORTS[x]}",
            key="search_origin"
        )
    
    with col2:
        dest_options = list(tt.POPULAR_DESTINATIONS.keys())
        destination = st.selectbox(
            "To",
            dest_options,
            format_func=lambda x: f"{x} - {tt.POPULAR_DESTINATIONS[x]}",
            key="search_dest"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        outbound_date = st.date_input(
            "Departure",
            value=datetime.now(TZ).date() + timedelta(days=30),
            min_value=datetime.now(TZ).date(),
            key="search_out"
        )
    with col2:
        return_date = st.date_input(
            "Return",
            value=datetime.now(TZ).date() + timedelta(days=37),
            min_value=datetime.now(TZ).date(),
            key="search_return"
        )
    
    if st.button("🔍 Search Flights", use_container_width=True):
        serpapi_key = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
        
        if not serpapi_key:
            st.warning("⚠️ SerpAPI key not configured. Add SERPAPI_KEY to secrets for live flight search.")
            st.info("💡 For now, check these sites manually:")
            st.markdown("""
            - [Google Flights](https://www.google.com/flights)
            - [Skyscanner](https://www.skyscanner.net)
            - [Kayak](https://www.kayak.co.uk)
            """)
        else:
            with st.spinner("Searching flights..."):
                try:
                    results = tt.search_flights_serpapi(
                        origin, destination,
                        outbound_date.isoformat(),
                        return_date.isoformat() if return_date else None
                    )
                    
                    # Extract flights list from results dict
                    flights = results.get("flights", []) if isinstance(results, dict) else []
                    
                    if flights:
                        st.success(f"Found {len(flights)} flights!")
                        for flight in flights[:5]:
                            st.markdown(f"""
                            <div class="flight-result">
                                <div class="flight-price">£{flight.get('price', 'N/A')}</div>
                                <div><strong>{flight.get('airline', 'Unknown')}</strong></div>
                                <div>Duration: {flight.get('duration', 'N/A')} mins</div>
                                <div>Stops: {flight.get('stops', 'N/A')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    elif results.get("error"):
                        st.error(f"Search error: {results.get('error')}")
                    else:
                        st.info("No flights found. Try different dates.")
                except Exception as e:
                    st.error(f"Search error: {e}")

# ============================================================
# Price Alerts Tab
# ============================================================

with tab_alerts:
    st.markdown("### 🔔 Flight Price Alerts")
    st.caption("Get notified when prices drop below your target")
    
    alerts = sm.read_all_rows("flight_alerts")
    active_alerts = [a for a in alerts if a.get("status") == "active"]
    
    # Add new alert
    with st.expander("➕ Create New Alert", expanded=len(active_alerts) == 0):
        col1, col2 = st.columns(2)
        
        with col1:
            alert_origin = st.selectbox(
                "From",
                list(tt.UK_AIRPORTS.keys()),
                format_func=lambda x: f"{x} - {tt.UK_AIRPORTS[x]}",
                key="alert_origin"
            )
        
        with col2:
            alert_dest = st.selectbox(
                "To",
                list(tt.POPULAR_DESTINATIONS.keys()),
                format_func=lambda x: f"{x} - {tt.POPULAR_DESTINATIONS[x]}",
                key="alert_dest"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            max_price = st.number_input("Max Price (£)", min_value=1, value=100, step=10)
        
        if st.button("🔔 Create Alert", use_container_width=True):
            sm.append_row("flight_alerts", {
                "alert_id": str(int(datetime.now().timestamp() * 1000)),
                "origin": alert_origin,
                "destination": alert_dest,
                "max_price": str(max_price),
                "currency": "GBP",
                "status": "active",
                "last_checked": "",
                "created": datetime.now(TZ).isoformat()
            })
            st.success("Price alert created! ✅")
            st.rerun()
    
    # Active alerts
    if active_alerts:
        st.markdown("### Active Alerts")
        
        for alert in active_alerts:
            alert_id = alert.get("alert_id", "")
            origin = alert.get("origin", "")
            dest = alert.get("destination", "")
            max_p = alert.get("max_price", "")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="alert-card">
                    <strong>{origin} → {dest}</strong><br>
                    <span style="color: #10b981;">Target: £{max_p}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("❌", key=f"del_alert_{alert_id}"):
                    sm.update_row_by_id("flight_alerts", "alert_id", alert_id, {"status": "cancelled"})
                    st.rerun()
        
        # Check prices button
        if st.button("🔄 Check All Prices Now", use_container_width=True):
            serpapi_key = st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")
            if not serpapi_key:
                st.warning("SerpAPI key needed for price checks")
            else:
                with st.spinner("Checking prices..."):
                    triggered = tt.check_flight_alerts(send_notifications=True)
                    if triggered:
                        # Check if Telegram is configured
                        tg_token = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
                        tg_chat = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
                        
                        if tg_token and tg_chat:
                            st.success(f"🔔 {len(triggered)} deal(s) found! Telegram notification sent.")
                        else:
                            st.success(f"🎉 {len(triggered)} deal(s) found!")
                        
                        for deal in triggered:
                            st.markdown(f"""
                            <div class="deal-card">
                                <strong>🎉 Deal Found!</strong><br>
                                {deal.get('route')} - £{deal.get('current_price')}<br>
                                <small>Your target: £{deal.get('max_price')} • Save £{deal.get('savings')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No deals found yet. Prices are above your targets.")
        
        # Telegram status
        st.markdown("---")
        tg_token = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        tg_chat = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
        
        if tg_token and tg_chat:
            st.success("✅ Telegram notifications enabled")
            if st.button("🧪 Test Telegram", key="test_telegram"):
                if tt.send_telegram_message("🧪 Test from Jarvis! Flight alerts are working."):
                    st.success("Test message sent!")
                else:
                    st.error("Failed to send test message")
        else:
            with st.expander("📱 Enable Telegram Notifications"):
                st.info("""
                **To get notified when deals are found:**
                
                1. Message [@BotFather](https://t.me/botfather) on Telegram
                2. Send `/newbot` and follow the instructions
                3. Copy the **Bot Token** you receive
                4. Message [@userinfobot](https://t.me/userinfobot) to get your **Chat ID**
                5. Add to Streamlit secrets:
                ```
                TELEGRAM_BOT_TOKEN = "your-bot-token"
                TELEGRAM_CHAT_ID = "your-chat-id"
                ```
                """)
    else:
        st.markdown('<div class="empty-state">No active alerts. Create one above!</div>', unsafe_allow_html=True)

# ============================================================
# Plan Trip Tab
# ============================================================

with tab_plan:
    st.markdown("### ➕ Plan a New Trip")
    
    with st.form("add_trip"):
        destination = st.text_input("Destination", placeholder="e.g., Barcelona, Spain")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=None,
                min_value=datetime.now(TZ).date(),
                key="plan_start"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=None,
                min_value=datetime.now(TZ).date(),
                key="plan_end"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Budget (£)", min_value=0, value=0, step=50)
        with col2:
            status = st.selectbox("Status", ["planning", "booked"])
        
        notes = st.text_area("Trip Notes", placeholder="Flights, accommodation, things to do...")
        
        watch_flights = st.checkbox("Watch flight prices for this destination")
        
        if st.form_submit_button("✈️ Add Trip", use_container_width=True):
            if destination:
                sm.append_row("travel_plans", {
                    "trip_id": str(int(datetime.now().timestamp() * 1000)),
                    "destination": destination,
                    "start_date": start_date.isoformat() if start_date else "",
                    "end_date": end_date.isoformat() if end_date else "",
                    "status": status,
                    "budget": str(budget) if budget else "",
                    "notes": notes,
                    "flight_watched": "true" if watch_flights else "false",
                    "created": datetime.now(TZ).isoformat()
                })
                st.success(f"Trip to {destination} added! 🎉")
                st.rerun()
            else:
                st.error("Please enter a destination")
    
    # Inspiration section
    st.markdown("---")
    st.markdown("### 🌍 Destination Inspiration")
    
    cols = st.columns(4)
    inspiration = [
        ("🇮🇹", "Rome", "History & Food"),
        ("🇯🇵", "Tokyo", "Culture & Tech"),
        ("🇬🇷", "Santorini", "Beaches & Views"),
        ("🇮🇸", "Iceland", "Nature & Aurora"),
        ("🇵🇹", "Lisbon", "Coastal Vibes"),
        ("🇹🇭", "Thailand", "Beaches & Temples"),
        ("🇪🇸", "Barcelona", "Art & Nightlife"),
        ("🇳🇴", "Norway", "Fjords & Hiking")
    ]
    
    for i, (flag, place, desc) in enumerate(inspiration):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 0.5rem;">
                <div style="font-size: 2rem;">{flag}</div>
                <div style="font-weight: 600;">{place}</div>
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.6);">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("💡 Tip: Set up price alerts for destinations you're flexible on dates for!")
