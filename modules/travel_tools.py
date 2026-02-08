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

def _telegram_config() -> tuple[Optional[str], Optional[str]]:
    """Get Telegram bot token and chat ID."""
    token = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    return token, chat_id

# ============================================================
# Telegram Notifications
# ============================================================

def send_telegram_message(message: str, parse_mode: str = "HTML") -> bool:
    """Send a message via Telegram bot."""
    token, chat_id = _telegram_config()
    if not token or not chat_id:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        return False

def send_flight_deal_notification(deal: Dict) -> bool:
    """Send a flight deal notification via Telegram."""
    route = deal.get("route", "Unknown route")
    current_price = deal.get("current_price", "?")
    max_price = deal.get("max_price", "?")
    airline = deal.get("airline", "")
    
    message = f"""✈️ <b>Flight Deal Alert!</b>

🛫 <b>{route}</b>
💰 Current Price: <b>£{current_price}</b>
🎯 Your Target: £{max_price}
{"✈️ Airline: " + airline if airline else ""}

🔗 <a href="https://www.google.com/flights">Book on Google Flights</a>"""
    
    return send_telegram_message(message)

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

def check_flight_alerts(send_notifications: bool = True) -> List[Dict]:
    """
    Check all active flight alerts against current prices.
    Returns alerts where current price is below max_price.
    Optionally sends Telegram notifications for deals found.
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
                # Get airport names for nicer display
                origin_name = UK_AIRPORTS.get(origin, origin)
                dest_name = POPULAR_DESTINATIONS.get(destination, destination)
                route = f"{origin_name} → {dest_name}"
                
                deal = {
                    "alert": alert,
                    "current_price": current_price,
                    "max_price": max_price,
                    "savings": round(max_price - current_price, 2),
                    "origin": origin,
                    "destination": destination,
                    "route": route,
                    "airline": estimate.get("airline", "")
                }
                triggered.append(deal)
                
                # Send Telegram notification
                if send_notifications:
                    send_flight_deal_notification(deal)
            
            # Update last checked
            sm.update_flight_alert(alert.get("alert_id"), {
                "last_checked": datetime.now(TZ).isoformat()
            })
    
    return triggered


def check_flight_alerts_silent() -> List[Dict]:
    """
    Check flight alerts without sending notifications.
    Used for homepage banner display.
    """
    return check_flight_alerts(send_notifications=False)

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
