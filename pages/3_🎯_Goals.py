# pages/3_🎯_Goals.py
"""
Goals & Bucket List Dashboard - Track life goals, yearly objectives, and get AI coaching.
"""
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import os

# Import modules
try:
    from modules import sheets_memory as sm
    from modules import goals_tools as gt
    from modules import global_styles as gs
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules import sheets_memory as sm
    from modules import goals_tools as gt
    from modules import global_styles as gs

TZ = ZoneInfo("Europe/London")

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Goals & Dreams | Jarvis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

gs.inject_global_styles()

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>
.goals-header {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
}
.goals-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
}
.goals-header p {
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
    color: #a78bfa;
}
.stat-card .label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.25rem;
}
.bucket-item {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}
.bucket-item:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(139, 92, 246, 0.3);
}
.bucket-item.high-priority {
    border-left: 4px solid #ef4444;
}
.bucket-item.medium-priority {
    border-left: 4px solid #f59e0b;
}
.bucket-item.low-priority {
    border-left: 4px solid #10b981;
}
.bucket-item.completed {
    opacity: 0.7;
    border-left-color: #10b981 !important;
}
.yearly-goal {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.progress-bar-container {
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #8b5cf6 0%, #a78bfa 100%);
    transition: width 0.5s ease;
}
.category-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
.category-travel { background: rgba(59, 130, 246, 0.3); color: #93c5fd; }
.category-adventure { background: rgba(245, 158, 11, 0.3); color: #fcd34d; }
.category-career { background: rgba(16, 185, 129, 0.3); color: #6ee7b7; }
.category-personal { background: rgba(139, 92, 246, 0.3); color: #c4b5fd; }
.category-financial { background: rgba(34, 197, 94, 0.3); color: #86efac; }
.category-health { background: rgba(236, 72, 153, 0.3); color: #f9a8d4; }
.category-learning { background: rgba(14, 165, 233, 0.3); color: #7dd3fc; }
.category-creative { background: rgba(244, 63, 94, 0.3); color: #fda4af; }
.category-relationships { background: rgba(168, 85, 247, 0.3); color: #d8b4fe; }
.category-experiences { background: rgba(251, 146, 60, 0.3); color: #fdba74; }
.advice-card {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.empty-state {
    text-align: center;
    padding: 2rem;
    color: rgba(255,255,255,0.5);
}
.quarterly-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
    margin-top: 0.75rem;
}
.quarter-box {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 0.5rem;
    text-align: center;
    font-size: 0.8rem;
}
.quarter-box .label {
    color: rgba(255,255,255,0.5);
    font-size: 0.7rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="goals-header">
    <h1>🎯 Goals & Dreams</h1>
    <p>Track your bucket list and yearly objectives</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Quick Stats
# ============================================================

def render_stats():
    bucket_list = sm.read_all_rows("bucket_list")
    yearly_goals = sm.read_all_rows("yearly_goals")
    current_year = datetime.now(TZ).year
    
    # Bucket list stats
    total_bucket = len(bucket_list)
    completed_bucket = sum(1 for b in bucket_list if b.get("status") == "completed")
    in_progress = sum(1 for b in bucket_list if b.get("status") == "in_progress")
    
    # Yearly goals stats
    this_year = [g for g in yearly_goals if g.get("year") == str(current_year)]
    yearly_completed = sum(1 for g in this_year if g.get("status") == "completed")
    yearly_total = len(this_year)
    avg_progress = 0
    if this_year:
        progs = [int(g.get("progress", 0) or 0) for g in this_year]
        avg_progress = sum(progs) // len(progs) if progs else 0
    
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="value">{total_bucket}</div>
            <div class="label">Bucket List Items</div>
        </div>
        <div class="stat-card">
            <div class="value">{completed_bucket}</div>
            <div class="label">Dreams Achieved</div>
        </div>
        <div class="stat-card">
            <div class="value">{in_progress}</div>
            <div class="label">In Progress</div>
        </div>
        <div class="stat-card">
            <div class="value">{yearly_total}</div>
            <div class="label">{current_year} Goals</div>
        </div>
        <div class="stat-card">
            <div class="value">{yearly_completed}</div>
            <div class="label">Year Goals Done</div>
        </div>
        <div class="stat-card">
            <div class="value">{avg_progress}%</div>
            <div class="label">Avg Progress</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

render_stats()

# ============================================================
# Tabs
# ============================================================

tab_bucket, tab_yearly, tab_add, tab_ai = st.tabs([
    "🌟 Bucket List", "📅 This Year", "➕ Add New", "🤖 AI Coach"
])

# ============================================================
# Bucket List Tab
# ============================================================

with tab_bucket:
    bucket_list = sm.read_all_rows("bucket_list")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "Pending", "In Progress", "Completed"],
            key="bucket_status"
        )
    with col2:
        categories = ["All"] + gt.BUCKET_LIST_CATEGORIES
        cat_filter = st.selectbox("Category", categories, key="bucket_cat")
    with col3:
        priorities = ["All", "High", "Medium", "Low"]
        pri_filter = st.selectbox("Priority", priorities, key="bucket_pri")
    
    # Filter the list
    filtered = bucket_list
    if status_filter != "All":
        filtered = [b for b in filtered if b.get("status", "pending").replace("_", " ").title() == status_filter]
    if cat_filter != "All":
        filtered = [b for b in filtered if b.get("category") == cat_filter]
    if pri_filter != "All":
        filtered = [b for b in filtered if b.get("priority", "medium").title() == pri_filter]
    
    if not filtered:
        st.markdown('<div class="empty-state">No items match your filters. Add some dreams!</div>', unsafe_allow_html=True)
    else:
        # Group by priority
        for priority in ["high", "medium", "low"]:
            items = [b for b in filtered if b.get("priority", "medium") == priority]
            if items:
                st.markdown(f"### {'🔴' if priority == 'high' else '🟡' if priority == 'medium' else '🟢'} {priority.title()} Priority")
                
                for item in items:
                    item_id = item.get("id", "")
                    status = item.get("status", "pending")
                    is_completed = status == "completed"
                    category = item.get("category", "personal")
                    
                    status_class = "completed" if is_completed else ""
                    
                    with st.expander(f"{'✅' if is_completed else '⬜'} {item.get('item', 'Untitled')}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <span class="category-badge category-{category}">{category.title()}</span>
                            """, unsafe_allow_html=True)
                            
                            if item.get("target_year"):
                                st.caption(f"🎯 Target: {item.get('target_year')}")
                            if item.get("notes"):
                                st.markdown(f"📝 {item.get('notes')}")
                            if item.get("completed_date"):
                                st.success(f"✨ Completed on {item.get('completed_date')}")
                        
                        with col2:
                            if not is_completed:
                                new_status = st.selectbox(
                                    "Status",
                                    ["pending", "in_progress", "completed"],
                                    index=["pending", "in_progress", "completed"].index(status),
                                    key=f"status_{item_id}"
                                )
                                if new_status != status:
                                    if new_status == "completed":
                                        sm.update_row_by_id(
                                            "bucket_list", "id", item_id,
                                            {"status": "completed", "completed_date": datetime.now(TZ).date().isoformat()}
                                        )
                                    else:
                                        sm.update_row_by_id("bucket_list", "id", item_id, {"status": new_status})
                                    st.rerun()

# ============================================================
# Yearly Goals Tab
# ============================================================

with tab_yearly:
    current_year = datetime.now(TZ).year
    
    # Year selector
    col1, col2 = st.columns([1, 3])
    with col1:
        years = list(range(current_year - 2, current_year + 2))
        selected_year = st.selectbox("Year", years, index=years.index(current_year), key="year_sel")
    
    yearly_goals = sm.read_all_rows("yearly_goals")
    year_goals = [g for g in yearly_goals if g.get("year") == str(selected_year)]
    
    if not year_goals:
        st.markdown('<div class="empty-state">No goals set for this year yet. Add some!</div>', unsafe_allow_html=True)
    else:
        # Group by category
        categories_with_goals = {}
        for g in year_goals:
            cat = g.get("category", "personal")
            if cat not in categories_with_goals:
                categories_with_goals[cat] = []
            categories_with_goals[cat].append(g)
        
        for cat, goals in categories_with_goals.items():
            st.markdown(f"### {cat.title()}")
            
            for goal in goals:
                goal_id = goal.get("goal_id", "")
                status = goal.get("status", "pending")
                progress = int(goal.get("progress", 0) or 0)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="yearly-goal">
                        <strong>{'✅' if status == 'completed' else '🎯'} {goal.get('goal', 'Untitled')}</strong>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: {progress}%"></div>
                        </div>
                        <small style="color: rgba(255,255,255,0.6);">{progress}% complete</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quarterly targets
                    q1 = goal.get("q1_target", "")
                    q2 = goal.get("q2_target", "")
                    q3 = goal.get("q3_target", "")
                    q4 = goal.get("q4_target", "")
                    
                    if any([q1, q2, q3, q4]):
                        st.markdown(f"""
                        <div class="quarterly-grid">
                            <div class="quarter-box"><div class="label">Q1</div>{q1 or '-'}</div>
                            <div class="quarter-box"><div class="label">Q2</div>{q2 or '-'}</div>
                            <div class="quarter-box"><div class="label">Q3</div>{q3 or '-'}</div>
                            <div class="quarter-box"><div class="label">Q4</div>{q4 or '-'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if goal.get("notes"):
                        st.caption(f"📝 {goal.get('notes')}")
                
                with col2:
                    if status != "completed":
                        new_progress = st.slider(
                            "Progress",
                            0, 100, progress,
                            key=f"prog_{goal_id}",
                            label_visibility="collapsed"
                        )
                        if new_progress != progress:
                            updates = {"progress": str(new_progress)}
                            if new_progress == 100:
                                updates["status"] = "completed"
                            sm.update_row_by_id("yearly_goals", "goal_id", goal_id, updates)
                            st.rerun()

# ============================================================
# Add New Tab
# ============================================================

with tab_add:
    add_type = st.radio("What would you like to add?", ["Bucket List Item", "Yearly Goal"], horizontal=True)
    
    if add_type == "Bucket List Item":
        st.markdown("### 🌟 Add to Bucket List")
        
        with st.form("add_bucket"):
            item = st.text_input("What's your dream?", placeholder="e.g., Visit the Northern Lights")
            
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category", gt.BUCKET_LIST_CATEGORIES)
                priority = st.selectbox("Priority", ["high", "medium", "low"], index=1)
            with col2:
                target_year = st.number_input(
                    "Target Year (optional)",
                    min_value=datetime.now(TZ).year,
                    max_value=datetime.now(TZ).year + 50,
                    value=None,
                    step=1
                )
            
            notes = st.text_area("Notes (optional)", placeholder="Any details, why this matters to you...")
            
            if st.form_submit_button("✨ Add Dream", use_container_width=True):
                if item:
                    sm.append_row("bucket_list", {
                        "id": str(int(datetime.now().timestamp() * 1000)),
                        "item": item,
                        "category": category,
                        "priority": priority,
                        "status": "pending",
                        "target_year": str(target_year) if target_year else "",
                        "notes": notes,
                        "created": datetime.now(TZ).isoformat(),
                        "completed_date": ""
                    })
                    st.success("Dream added to your bucket list! 🌟")
                    st.rerun()
                else:
                    st.error("Please enter your dream")
    
    else:
        st.markdown("### 📅 Add Yearly Goal")
        
        with st.form("add_yearly"):
            goal = st.text_input("What's your goal?", placeholder="e.g., Read 24 books")
            
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input(
                    "Year",
                    min_value=datetime.now(TZ).year - 1,
                    max_value=datetime.now(TZ).year + 5,
                    value=datetime.now(TZ).year
                )
                category = st.selectbox("Category", gt.YEARLY_GOAL_CATEGORIES, key="yearly_cat")
            
            st.markdown("**Quarterly Milestones (optional)**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                q1 = st.text_input("Q1", placeholder="e.g., 6 books")
            with col2:
                q2 = st.text_input("Q2", placeholder="e.g., 12 books")
            with col3:
                q3 = st.text_input("Q3", placeholder="e.g., 18 books")
            with col4:
                q4 = st.text_input("Q4", placeholder="e.g., 24 books")
            
            notes = st.text_area("Notes (optional)", placeholder="How will you achieve this?", key="yearly_notes")
            
            if st.form_submit_button("🎯 Add Goal", use_container_width=True):
                if goal:
                    sm.append_row("yearly_goals", {
                        "year": str(year),
                        "goal_id": str(int(datetime.now().timestamp() * 1000)),
                        "goal": goal,
                        "category": category,
                        "status": "pending",
                        "progress": "0",
                        "q1_target": q1,
                        "q2_target": q2,
                        "q3_target": q3,
                        "q4_target": q4,
                        "notes": notes
                    })
                    st.success(f"Goal added for {year}! 🎯")
                    st.rerun()
                else:
                    st.error("Please enter your goal")

# ============================================================
# AI Coach Tab
# ============================================================

with tab_ai:
    st.markdown("### 🤖 AI Goal Coach")
    st.markdown("Get personalized advice on achieving your dreams and staying on track.")
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌟 Dream Inspiration", use_container_width=True):
            st.session_state.goal_question = "Based on my current bucket list and goals, what are some inspiring new dreams I should consider adding?"
    
    with col2:
        if st.button("📊 Progress Review", use_container_width=True):
            st.session_state.goal_question = "Review my yearly goals progress and suggest specific actions to get back on track or accelerate."
    
    with col3:
        if st.button("🎯 Prioritization", use_container_width=True):
            st.session_state.goal_question = "Help me prioritize my bucket list and yearly goals. What should I focus on right now?"
    
    st.markdown("---")
    
    # Custom question
    question = st.text_area(
        "Ask your goal coach",
        value=st.session_state.get("goal_question", ""),
        placeholder="e.g., How can I make progress on my travel bucket list items while working full-time?",
        key="goal_q_input"
    )
    
    if st.button("💬 Get Advice", use_container_width=True):
        if question:
            with st.spinner("Your coach is thinking..."):
                try:
                    advice = gt.get_goal_advice(question)
                    st.markdown(f"""
                    <div class="advice-card">
                        <h4>🎯 Coach's Advice</h4>
                        <p>{advice}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Couldn't get advice: {e}")
            st.session_state.goal_question = ""
        else:
            st.warning("Please enter a question")
    
    # Show context
    with st.expander("📋 Your Current Goals Summary"):
        bucket = sm.read_all_rows("bucket_list")
        yearly = sm.read_all_rows("yearly_goals")
        
        st.markdown(f"**Bucket List:** {len(bucket)} items ({sum(1 for b in bucket if b.get('status') == 'completed')} completed)")
        
        current_year = datetime.now(TZ).year
        this_year = [g for g in yearly if g.get("year") == str(current_year)]
        st.markdown(f"**{current_year} Goals:** {len(this_year)} goals")
        
        if bucket:
            in_progress = [b for b in bucket if b.get("status") == "in_progress"]
            if in_progress:
                st.markdown("**Currently Working On:**")
                for b in in_progress[:5]:
                    st.markdown(f"- {b.get('item')}")

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("💡 Tip: Break big dreams into yearly goals, and yearly goals into quarterly milestones!")
