"""
FOOTBALL RESEARCH DASHBOARD
============================

This is a DISPLAY-ONLY dashboard. No long-running operations.
The actual research runs in background_researcher.py

Architecture:
- background_researcher.py runs continuously, executing research
- This dashboard just polls Supabase and displays results
- Clicking buttons just updates state in Supabase
- No computation happens in Streamlit = NO REFRESH ISSUES

Usage:
    streamlit run research_dashboard.py
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st

st.set_page_config(
    page_title="Football Research Dashboard", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION
# ============================================================

OUTCOME_COLUMNS = ["BO 2.5 PL", "BTTS PL", "SHG PL", "SHG 2+ PL", "LU1.5 PL", "LFGHU0.5 PL", "BO1.5 FHG PL", "PL"]
REFRESH_INTERVAL = 3  # seconds

# ============================================================
# SUPABASE CLIENT
# ============================================================

@st.cache_resource
def get_supabase():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


# ============================================================
# STATE FUNCTIONS (all read/write to Supabase)
# ============================================================

def get_all_sessions() -> List[Dict]:
    """Get all research sessions."""
    try:
        sb = get_supabase()
        if not sb:
            return []
        result = sb.table("research_sessions").select("*").order("updated_at", desc=True).limit(20).execute()
        
        sessions = []
        for row in result.data or []:
            try:
                state = json.loads(row.get("state_json", "{}"))
                sessions.append({
                    "session_id": row.get("session_id"),
                    "updated_at": row.get("updated_at"),
                    "state": state,
                })
            except:
                pass
        return sessions
    except Exception as e:
        st.error(f"Error loading sessions: {e}")
        return []


def get_session_state(session_id: str) -> Dict:
    """Get state for a specific session."""
    try:
        sb = get_supabase()
        if not sb:
            return {}
        result = sb.table("research_sessions").select("*").eq("session_id", session_id).execute()
        if result.data:
            return json.loads(result.data[0].get("state_json", "{}"))
    except Exception as e:
        st.error(f"Error loading session: {e}")
    return {}


def create_new_session(pl_column: str) -> str:
    """Create a new research session."""
    import uuid
    session_id = str(uuid.uuid4())[:8]
    
    state = {
        "session_id": session_id,
        "agent_phase": "running",  # Start immediately
        "agent_iteration": 0,
        "target_pl_column": pl_column,
        "bible": None,
        "avenues_to_explore": [],
        "avenues_explored": [],
        "accumulated_learnings": [],
        "strategies_found": [],
        "research_log": [{"ts": datetime.utcnow().strftime("%H:%M:%S"), "level": "header", "content": "Research session created"}],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    
    try:
        sb = get_supabase()
        sb.table("research_sessions").insert({
            "session_id": session_id,
            "state_json": json.dumps(state, default=str),
            "updated_at": datetime.utcnow().isoformat(),
        }).execute()
        return session_id
    except Exception as e:
        st.error(f"Error creating session: {e}")
        return None


def update_session_phase(session_id: str, phase: str) -> bool:
    """Update just the phase of a session (pause/resume)."""
    try:
        state = get_session_state(session_id)
        if state:
            state["agent_phase"] = phase
            state["updated_at"] = datetime.utcnow().isoformat()
            
            sb = get_supabase()
            sb.table("research_sessions").update({
                "state_json": json.dumps(state, default=str),
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("session_id", session_id).execute()
            return True
    except Exception as e:
        st.error(f"Error updating session: {e}")
    return False


def delete_session(session_id: str) -> bool:
    """Delete a research session."""
    try:
        sb = get_supabase()
        sb.table("research_sessions").delete().eq("session_id", session_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting session: {e}")
    return False


# ============================================================
# UI COMPONENTS
# ============================================================

def display_research_log(log: List[Dict]):
    """Display the research log with nice formatting."""
    if not log:
        st.info("No activity yet...")
        return
    
    # Create scrollable container
    log_html = []
    for entry in log[-50:]:  # Last 50 entries
        ts = entry.get("ts", "")
        level = entry.get("level", "info")
        content = entry.get("content", "").replace("<", "&lt;").replace(">", "&gt;")
        
        if level == "header":
            log_html.append(f'<div style="margin: 8px 0; padding: 6px 10px; background: #1e3a5f; color: white; border-radius: 4px; font-weight: bold;">[{ts}] 📌 {content}</div>')
        elif level == "strategy":
            log_html.append(f'<div style="margin: 8px 0; padding: 8px 10px; background: #d4edda; border: 2px solid #28a745; border-radius: 6px; font-weight: bold;">[{ts}] 🎉 {content}</div>')
        elif level == "success":
            log_html.append(f'<div style="margin: 4px 0; padding: 4px 10px; background: #e8f5e9; border-left: 3px solid #4caf50;">[{ts}] ✅ {content}</div>')
        elif level == "warning":
            log_html.append(f'<div style="margin: 4px 0; padding: 4px 10px; background: #fff3e0; border-left: 3px solid #ff9800;">[{ts}] ⚠️ {content}</div>')
        elif level == "error":
            log_html.append(f'<div style="margin: 4px 0; padding: 4px 10px; background: #ffebee; border-left: 3px solid #f44336;">[{ts}] ❌ {content}</div>')
        elif level == "iteration":
            log_html.append(f'<div style="margin: 6px 0; padding: 4px 10px; background: #fff8e1; border-left: 3px solid #ffc107;">[{ts}] 🔄 {content}</div>')
        else:
            log_html.append(f'<div style="margin: 2px 0; padding: 2px 10px; color: #555;">[{ts}] {content}</div>')
    
    st.markdown(
        f'<div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fafafa;">{"".join(log_html)}</div>',
        unsafe_allow_html=True
    )


def display_strategies(strategies: List[Dict]):
    """Display found strategies."""
    if not strategies:
        st.info("No strategies found yet...")
        return
    
    for i, strategy in enumerate(strategies, 1):
        with st.expander(f"Strategy {i}: {strategy.get('test_roi', 0)*100:.2f}% ROI", expanded=(i==1)):
            col1, col2, col3 = st.columns(3)
            col1.metric("Train ROI", f"{strategy.get('train_roi', 0)*100:.2f}%")
            col2.metric("Val ROI", f"{strategy.get('val_roi', 0)*100:.2f}%")
            col3.metric("Test ROI", f"{strategy.get('test_roi', 0)*100:.2f}%")
            
            st.markdown(f"**Test Rows:** {strategy.get('test_rows', 0)}")
            st.markdown(f"**Found at iteration:** {strategy.get('found_at_iteration', '?')}")
            st.markdown(f"**Avenue:** {strategy.get('avenue', 'Unknown')}")
            
            st.code(json.dumps(strategy.get("filters", []), indent=2), language="json")


def display_session_dashboard(session_id: str, state: Dict):
    """Display dashboard for a specific session."""
    
    phase = state.get("agent_phase", "unknown")
    iteration = state.get("agent_iteration", 0)
    strategies = state.get("strategies_found", [])
    learnings = state.get("accumulated_learnings", [])
    log = state.get("research_log", [])
    
    # Status bar
    if phase == "running":
        st.success(f"🟢 **RUNNING** - Iteration {iteration}")
    elif phase == "paused":
        st.warning(f"⏸️ **PAUSED** at iteration {iteration}")
    elif phase == "complete":
        st.info(f"✅ **COMPLETE** - {iteration} iterations")
    else:
        st.error(f"❓ Unknown phase: {phase}")
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if phase == "running":
            if st.button("⏸️ Pause", use_container_width=True):
                update_session_phase(session_id, "paused")
                st.rerun()
        elif phase == "paused":
            if st.button("▶️ Resume", type="primary", use_container_width=True):
                update_session_phase(session_id, "running")
                st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col3:
        st.metric("Strategies", len(strategies))
    
    with col4:
        st.metric("Learnings", len(learnings))
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📜 Live Log", "🎯 Strategies", "📚 Learnings", "📊 Details"])
    
    with tab1:
        st.markdown("### Research Activity")
        display_research_log(log)
    
    with tab2:
        st.markdown("### Found Strategies")
        display_strategies(strategies)
    
    with tab3:
        st.markdown("### Accumulated Learnings")
        if learnings:
            for learning in learnings[-20:]:
                if isinstance(learning, dict):
                    st.markdown(f"- **Iter {learning.get('iteration', '?')}:** {learning.get('learning', '')}")
                else:
                    st.markdown(f"- {learning}")
        else:
            st.info("No learnings yet...")
    
    with tab4:
        st.markdown("### Session Details")
        st.json({
            "session_id": session_id,
            "phase": phase,
            "iteration": iteration,
            "target_market": state.get("target_pl_column", "?"),
            "avenues_explored": len(state.get("avenues_explored", [])),
            "avenues_remaining": len(state.get("avenues_to_explore", [])),
            "strategies_found": len(strategies),
            "updated_at": state.get("updated_at", "?"),
        })


# ============================================================
# MAIN APP
# ============================================================

st.title("⚽ Football Research Dashboard")

# Check Supabase connection
sb = get_supabase()
if not sb:
    st.error("❌ Cannot connect to Supabase!")
    st.markdown("""
    Add to your Streamlit secrets:
    ```
    SUPABASE_URL = "https://xxx.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY = "xxx"
    ```
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")
    
    st.success("🟢 Supabase Connected")
    
    st.divider()
    
    # New session
    st.markdown("### Start New Research")
    pl_col = st.selectbox("Target Market", OUTCOME_COLUMNS, index=0)
    
    if st.button("🚀 Start New Session", type="primary", use_container_width=True):
        new_id = create_new_session(pl_col)
        if new_id:
            st.session_state.active_session = new_id
            st.success(f"Created session: {new_id}")
            st.rerun()
    
    st.divider()
    
    # Session list
    st.markdown("### Sessions")
    sessions = get_all_sessions()
    
    for sess in sessions[:10]:
        session_id = sess.get("session_id")
        state = sess.get("state", {})
        phase = state.get("agent_phase", "?")
        iteration = state.get("agent_iteration", 0)
        strategies = len(state.get("strategies_found", []))
        
        # Phase emoji
        phase_emoji = "🟢" if phase == "running" else "⏸️" if phase == "paused" else "✅" if phase == "complete" else "❓"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"{phase_emoji} {session_id}", key=f"select_{session_id}", use_container_width=True):
                st.session_state.active_session = session_id
                st.rerun()
        with col2:
            st.caption(f"i{iteration}")
    
    st.divider()
    
    # Instructions
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        1. **Start** a new research session
        2. **Run** `background_researcher.py` in terminal
        3. **Watch** progress here (auto-refreshes)
        4. **Pause/Resume** anytime
        5. Sessions survive browser refresh!
        
        ```bash
        python3 background_researcher.py
        ```
        """)

# Main content
active_session = st.session_state.get("active_session")

if active_session:
    state = get_session_state(active_session)
    if state:
        st.markdown(f"### Session: `{active_session}`")
        display_session_dashboard(active_session, state)
        
        # Auto-refresh for running sessions
        if state.get("agent_phase") == "running":
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
    else:
        st.error(f"Session {active_session} not found")
        st.session_state.active_session = None
else:
    st.info("👈 Select a session from the sidebar or start a new one")
    
    # Show recent sessions
    sessions = get_all_sessions()
    if sessions:
        st.markdown("### Recent Sessions")
        
        for sess in sessions[:5]:
            session_id = sess.get("session_id")
            state = sess.get("state", {})
            phase = state.get("agent_phase", "?")
            iteration = state.get("agent_iteration", 0)
            strategies = len(state.get("strategies_found", []))
            updated = sess.get("updated_at", "")[:19].replace("T", " ")
            
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
            col1.markdown(f"**{session_id}**")
            col2.markdown(f"Phase: {phase}")
            col3.markdown(f"Iter: {iteration}")
            col4.markdown(f"Strategies: {strategies}")
            col5.markdown(f"Updated: {updated}")
