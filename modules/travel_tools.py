# modules/travel_tools.py
"""
Travel Tools for Jarvis
Trip planning, flight price tracking, and travel management.
"""
from __future__ import annotations
import os
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
import streamlit as st

try:
    from modules import sheets_memory as sm
except ImportError:
    import sheets_memory as sm

TZ = ZoneInfo("Europe/London")

# ============================================================
# API Configuration
# ============================================================

def _serpapi_key() -> Optional[str]:
    """Get SerpAPI key for flight searches."""
    return st.secrets.get("SERPAPI_KEY") or os.getenv("SERPAPI_KEY")

# Common airports
UK_AIRPORTS = {
    "LHR": "London Heathrow",
    "LGW": "London Gatwick",
    "STN": "London Stansted",
    "LTN": "London Luton",
    "MAN": "Manchester",
    "BHX": "Birmingham",
    "EDI": "Edinburgh",
    "BRS": "Bristol",
    "GLA": "Glasgow",
    "NCL": "Newcastle",
    "LBA": "Leeds Bradford",
    "EMA": "East Midlands",
    "SOU": "Southampton",
}

POPULAR_DESTINATIONS = {
    "AMS": "Amsterdam",
    "BCN": "Barcelona",
    "CDG": "Paris CDG",
    "FCO": "Rome",
    "DUB": "Dublin",
    "LIS": "Lisbon",
    "ATH": "Athens",
    "PMI": "Palma Mallorca",
    "AGP": "Malaga",
    "PRG": "Prague",
    "BUD": "Budapest",
    "VIE": "Vienna",
    "CPH": "Copenhagen",
    "DXB": "Dubai",
    "JFK": "New York JFK",
    "LAX": "Los Angeles",
    "TYO": "Tokyo",
    "BKK": "Bangkok",
}

TRIP_STATUSES = ["planning", "booked", "ongoing", "completed", "cancelled"]

# ============================================================
# Flight Search (via SerpAPI)
# ============================================================

@st.cache_data(ttl=3600)
def search_flights_serpapi(
    origin: str,
    destination: str,
    outbound_date: str,
    return_date: str = None,
    adults: int = 1
) -> Dict[str, Any]:
    """
    Search flights using SerpAPI Google Flights.
    Returns flight options with prices.
    """
    api_key = _serpapi_key()
    if not api_key:
        return {"error": "No SERPAPI_KEY configured", "flights": []}
    
    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": outbound_date,
        "currency": "GBP",
        "hl": "en",
        "api_key": api_key,
        "adults": adults,
    }
    
    if return_date:
        params["return_date"] = return_date
        params["type"] = "1"  # Round trip
    else:
        params["type"] = "2"  # One way
    
    try:
        r = requests.get("https://serpapi.com/search", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        flights = []
        
        # Parse best flights
        for flight in data.get("best_flights", [])[:5]:
            flights.append({
                "price": flight.get("price"),
                "airline": flight.get("flights", [{}])[0].get("airline", "Unknown"),
                "duration": flight.get("total_duration"),
                "stops": len(flight.get("flights", [])) - 1,
                "departure": flight.get("flights", [{}])[0].get("departure_airport", {}).get("time"),
                "arrival": flight.get("flights", [{}])[-1].get("arrival_airport", {}).get("time"),
            })
        
        # Parse other flights
        for flight in data.get("other_flights", [])[:5]:
            flights.append({
                "price": flight.get("price"),
                "airline": flight.get("flights", [{}])[0].get("airline", "Unknown"),
                "duration": flight.get("total_duration"),
                "stops": len(flight.get("flights", [])) - 1,
                "departure": flight.get("flights", [{}])[0].get("departure_airport", {}).get("time"),
                "arrival": flight.get("flights", [{}])[-1].get("arrival_airport", {}).get("time"),
            })
        
        # Sort by price
        flights.sort(key=lambda x: x.get("price", 99999))
        
        return {
            "origin": origin,
            "destination": destination,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "flights": flights,
            "price_insights": data.get("price_insights", {})
        }
        
    except Exception as e:
        return {"error": str(e), "flights": []}

def get_flight_price_estimate(origin: str, destination: str) -> Dict[str, Any]:
    """Get a rough price estimate for a route (cached)."""
    # Search for flights 2 weeks from now
    future_date = (datetime.now(TZ) + timedelta(days=14)).strftime("%Y-%m-%d")
    return_date = (datetime.now(TZ) + timedelta(days=21)).strftime("%Y-%m-%d")
    
    result = search_flights_serpapi(origin, destination, future_date, return_date)
    
    if result.get("flights"):
        prices = [f.get("price", 0) for f in result["flights"] if f.get("price")]
        if prices:
            return {
                "origin": origin,
                "destination": destination,
                "min_price": min(prices),
                "avg_price": round(sum(prices) / len(prices)),
                "max_price": max(prices),
                "currency": "GBP",
                "sample_date": future_date
            }
    
    return {"error": "Could not get price estimate", "origin": origin, "destination": destination}

# ============================================================
# Travel Plans Management
# ============================================================

def add_travel_plan(
    destination: str,
    start_date: str = "",
    end_date: str = "",
    budget: str = "",
    notes: str = ""
) -> Dict:
    """Add a new travel plan."""
    return sm.add_travel_plan(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        budget=budget,
        notes=notes
    )

def get_travel_plans(status: str = None) -> List[Dict]:
    """Get travel plans."""
    return sm.get_travel_plans(status=status)

def update_travel_plan(trip_id: str, updates: Dict) -> Dict:
    """Update a travel plan."""
    return sm.update_travel_plan(trip_id, updates)

def get_upcoming_trips() -> List[Dict]:
    """Get upcoming trips (booked or planning, future dates)."""
    plans = get_travel_plans()
    today = datetime.now(TZ).date().isoformat()
    
    upcoming = []
    for plan in plans:
        status = plan.get("status", "")
        start = plan.get("start_date", "")
        
        if status in ("booked", "planning"):
            if not start or start >= today:
                upcoming.append(plan)
    
    # Sort by start date
    upcoming.sort(key=lambda x: x.get("start_date", "9999"))
    return upcoming

# ============================================================
# Flight Alerts
# ============================================================

def add_flight_alert(
    origin: str,
    destination: str,
    max_price: float,
    currency: str = "GBP"
) -> Dict:
    """Add a flight price alert."""
    return sm.add_flight_alert(
        origin=origin,
        destination=destination,
        max_price=max_price,
        currency=currency
    )

def get_flight_alerts(status: str = "active") -> List[Dict]:
    """Get flight alerts."""
    return sm.get_flight_alerts(status=status)

def update_flight_alert(alert_id: str, updates: Dict) -> Dict:
    """Update a flight alert."""
    return sm.update_flight_alert(alert_id, updates)

def check_flight_alerts() -> List[Dict]:
    """
    Check all active flight alerts against current prices.
    Returns alerts where current price is below max_price.
    """
    alerts = get_flight_alerts(status="active")
    triggered = []
    
    for alert in alerts:
        origin = alert.get("origin", "")
        destination = alert.get("destination", "")
        max_price = float(alert.get("max_price", 0) or 0)
        
        if not origin or not destination or max_price <= 0:
            continue
        
        # Get current price estimate
        estimate = get_flight_price_estimate(origin, destination)
        
        if estimate.get("min_price"):
            current_price = estimate["min_price"]
            
            if current_price <= max_price:
                triggered.append({
                    "alert": alert,
                    "current_price": current_price,
                    "max_price": max_price,
                    "savings": round(max_price - current_price, 2),
                    "origin": origin,
                    "destination": destination
                })
            
            # Update last checked
            sm.update_flight_alert(alert.get("alert_id"), {
                "last_checked": datetime.now(TZ).isoformat()
            })
    
    return triggered

# ============================================================
# Jack's Flight Club Integration (RSS)
# ============================================================

@st.cache_data(ttl=1800)
def get_jacks_flight_club_deals() -> List[Dict]:
    """
    Get deals from Jack's Flight Club.
    Note: This requires a public RSS feed or scraping - limited without account.
    For now, we'll provide instructions for manual integration.
    """
    # Jack's Flight Club doesn't have a public API or RSS
    # Users receive deals via email
    # This is a placeholder for manual entry
    return []

# ============================================================
# UI Components
# ============================================================

def render_trip_planner():
    """Render trip planning form."""
    st.markdown("### ✈️ Plan a Trip")
    
    destination = st.text_input("Destination", placeholder="e.g., Barcelona, Spain", key="trip_destination")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None, key="trip_start")
    with col2:
        end_date = st.date_input("End Date", value=None, key="trip_end")
    
    col3, col4 = st.columns(2)
    with col3:
        budget = st.text_input("Budget (optional)", placeholder="e.g., £1000", key="trip_budget")
    with col4:
        status = st.selectbox("Status", ["planning", "booked"], key="trip_status")
    
    notes = st.text_area("Notes", placeholder="Flight details, accommodation, activities...", key="trip_notes")
    
    if st.button("💾 Save Trip", key="save_trip_btn"):
        if destination:
            result = add_travel_plan(
                destination=destination,
                start_date=start_date.isoformat() if start_date else "",
                end_date=end_date.isoformat() if end_date else "",
                budget=budget,
                notes=notes
            )
            if result.get("ok"):
                st.success("Trip saved! ✈️")
                # Update status if booked
                if status == "booked" and result.get("tab"):
                    # The add function returns trip_id, update status
                    pass
                st.rerun()
            else:
                st.error(f"Failed: {result.get('error')}")
        else:
            st.warning("Please enter a destination")

def render_my_trips():
    """Render list of trips."""
    st.markdown("### 🗺️ My Trips")
    
    tab1, tab2, tab3 = st.tabs(["📅 Upcoming", "✅ Past", "🗓️ All"])
    
    with tab1:
        upcoming = get_upcoming_trips()
        if upcoming:
            for trip in upcoming:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        status_emoji = {"planning": "📋", "booked": "✅"}.get(trip.get("status"), "📋")
                        st.markdown(f"{status_emoji} **{trip.get('destination')}**")
                        
                        dates = []
                        if trip.get("start_date"):
                            dates.append(f"From: {trip['start_date']}")
                        if trip.get("end_date"):
                            dates.append(f"To: {trip['end_date']}")
                        if dates:
                            st.caption(" | ".join(dates))
                        
                        if trip.get("budget"):
                            st.caption(f"💰 Budget: {trip['budget']}")
                        
                        if trip.get("notes"):
                            with st.expander("Notes"):
                                st.write(trip["notes"])
                    
                    with col2:
                        new_status = st.selectbox(
                            "Status",
                            TRIP_STATUSES,
                            index=TRIP_STATUSES.index(trip.get("status", "planning")),
                            key=f"trip_status_{trip.get('trip_id')}",
                            label_visibility="collapsed"
                        )
                        if new_status != trip.get("status"):
                            update_travel_plan(trip.get("trip_id"), {"status": new_status})
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No upcoming trips. Plan one! ✈️")
    
    with tab2:
        completed = get_travel_plans(status="completed")
        if completed:
            for trip in completed[-10:]:
                st.markdown(f"✅ **{trip.get('destination')}** - {trip.get('start_date', 'N/A')}")
        else:
            st.info("No past trips recorded")
    
    with tab3:
        all_trips = get_travel_plans()
        if all_trips:
            for trip in all_trips:
                status = trip.get("status", "planning")
                st.markdown(f"{'✅' if status == 'completed' else '📋'} {trip.get('destination')} ({status})")
        else:
            st.info("No trips yet")

def render_flight_search():
    """Render flight search interface."""
    st.markdown("### 🔍 Search Flights")
    
    if not _serpapi_key():
        st.warning("Add SERPAPI_KEY to secrets to enable flight search")
        st.info("Get a free API key at: https://serpapi.com/")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox(
            "From",
            list(UK_AIRPORTS.keys()),
            format_func=lambda x: f"{x} - {UK_AIRPORTS[x]}",
            key="flight_origin"
        )
    with col2:
        dest_options = list(POPULAR_DESTINATIONS.keys())
        destination = st.selectbox(
            "To",
            dest_options,
            format_func=lambda x: f"{x} - {POPULAR_DESTINATIONS[x]}",
            key="flight_dest"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        outbound = st.date_input(
            "Outbound",
            value=datetime.now(TZ).date() + timedelta(days=14),
            key="flight_outbound"
        )
    with col4:
        return_date = st.date_input(
            "Return (optional)",
            value=None,
            key="flight_return"
        )
    
    if st.button("🔍 Search Flights", key="search_flights_btn"):
        with st.spinner("Searching flights..."):
            result = search_flights_serpapi(
                origin=origin,
                destination=destination,
                outbound_date=outbound.isoformat(),
                return_date=return_date.isoformat() if return_date else None
            )
            
            if result.get("error"):
                st.error(f"Search failed: {result['error']}")
            elif result.get("flights"):
                st.success(f"Found {len(result['flights'])} flights!")
                
                for flight in result["flights"][:5]:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.markdown(f"✈️ **{flight.get('airline', 'Unknown')}**")
                            st.caption(f"⏱️ {flight.get('duration', 'N/A')} mins | {flight.get('stops', 0)} stops")
                        with col2:
                            st.caption(f"🛫 {flight.get('departure', 'N/A')}")
                            st.caption(f"🛬 {flight.get('arrival', 'N/A')}")
                        with col3:
                            price = flight.get("price", "N/A")
                            st.markdown(f"**£{price}**")
                        st.divider()
            else:
                st.info("No flights found for this route/date")

def render_flight_alerts():
    """Render flight alerts management."""
    st.markdown("### 🔔 Flight Price Alerts")
    
    # Add new alert
    with st.expander("➕ Add New Alert"):
        col1, col2 = st.columns(2)
        with col1:
            alert_origin = st.selectbox(
                "From",
                list(UK_AIRPORTS.keys()),
                format_func=lambda x: f"{x} - {UK_AIRPORTS[x]}",
                key="alert_origin"
            )
        with col2:
            alert_dest = st.selectbox(
                "To",
                list(POPULAR_DESTINATIONS.keys()),
                format_func=lambda x: f"{x} - {POPULAR_DESTINATIONS[x]}",
                key="alert_dest"
            )
        
        max_price = st.number_input("Max Price (£)", min_value=1, value=100, key="alert_max_price")
        
        if st.button("🔔 Create Alert", key="create_alert_btn"):
            result = add_flight_alert(
                origin=alert_origin,
                destination=alert_dest,
                max_price=max_price
            )
            if result.get("ok"):
                st.success("Alert created!")
                st.rerun()
            else:
                st.error(f"Failed: {result.get('error')}")
    
    # Active alerts
    alerts = get_flight_alerts(status="active")
    
    if alerts:
        st.markdown("#### Active Alerts")
        
        for alert in alerts:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                origin = alert.get("origin", "")
                dest = alert.get("destination", "")
                origin_name = UK_AIRPORTS.get(origin, origin)
                dest_name = POPULAR_DESTINATIONS.get(dest, dest)
                st.markdown(f"✈️ **{origin_name}** → **{dest_name}**")
                st.caption(f"Max: £{alert.get('max_price', 'N/A')}")
            with col2:
                if alert.get("last_checked"):
                    st.caption(f"Last checked: {alert['last_checked'][:10]}")
            with col3:
                if st.button("🗑️", key=f"delete_alert_{alert.get('alert_id')}"):
                    update_flight_alert(alert.get("alert_id"), {"status": "inactive"})
                    st.rerun()
            st.divider()
        
        # Check alerts button
        if st.button("🔍 Check All Alerts Now", key="check_alerts_btn"):
            with st.spinner("Checking prices..."):
                triggered = check_flight_alerts()
                if triggered:
                    st.success(f"🎉 {len(triggered)} deals found!")
                    for deal in triggered:
                        st.markdown(f"**{deal['origin']} → {deal['destination']}**: £{deal['current_price']} (Save £{deal['savings']})")
                else:
                    st.info("No deals found below your price targets yet")
    else:
        st.info("No active alerts. Create one to track prices!")

def render_jacks_flight_club():
    """Render Jack's Flight Club section."""
    st.markdown("### 🎟️ Jack's Flight Club Deals")
    
    st.info("""
    **Jack's Flight Club** sends deals via email - there's no public API.
    
    **To integrate:**
    1. Sign up at [jacksflightclub.com](https://jacksflightclub.com)
    2. When you see a good deal in your email, add it as a trip here!
    
    **Tips:**
    - Set up email filters to highlight their deals
    - Act fast - best deals go quickly!
    - Use the flight search above to verify prices
    """)
    
    # Manual deal entry
    with st.expander("📝 Add a Deal from Email"):
        deal_dest = st.text_input("Destination", key="jfc_dest")
        deal_price = st.text_input("Price", placeholder="e.g., £199 return", key="jfc_price")
        deal_dates = st.text_input("Valid Dates", placeholder="e.g., Jan-Mar 2025", key="jfc_dates")
        deal_notes = st.text_area("Deal Details", key="jfc_notes")
        
        if st.button("💾 Save Deal", key="save_jfc_deal"):
            if deal_dest:
                result = add_travel_plan(
                    destination=deal_dest,
                    budget=deal_price,
                    notes=f"JFC Deal - {deal_dates}\n{deal_notes}"
                )
                if result.get("ok"):
                    st.success("Deal saved!")
                    st.rerun()

def render_travel_stats():
    """Render travel statistics."""
    st.markdown("### 📊 Travel Stats")
    
    all_trips = get_travel_plans()
    completed = [t for t in all_trips if t.get("status") == "completed"]
    upcoming = get_upcoming_trips()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trips", len(all_trips))
    with col2:
        st.metric("Completed", len(completed))
    with col3:
        st.metric("Upcoming", len(upcoming))
    
    # Destination count
    if completed:
        destinations = [t.get("destination", "Unknown").split(",")[0] for t in completed]
        unique_destinations = len(set(destinations))
        st.metric("Unique Destinations", unique_destinations)
