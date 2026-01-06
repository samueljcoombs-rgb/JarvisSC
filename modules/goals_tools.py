# modules/goals_tools.py
"""
Goals & Bucket List Tools for Jarvis
Lifelong bucket list, yearly goals, and AI-powered goal coaching.
"""
from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import streamlit as st
from openai import OpenAI

try:
    from modules import sheets_memory as sm
except ImportError:
    import sheets_memory as sm

TZ = ZoneInfo("Europe/London")

# ============================================================
# Categories and Priorities
# ============================================================

BUCKET_LIST_CATEGORIES = [
    "travel",
    "adventure",
    "career",
    "personal",
    "financial",
    "relationships",
    "health",
    "learning",
    "creative",
    "experiences",
    "other"
]

YEARLY_GOAL_CATEGORIES = [
    "health",
    "career",
    "financial",
    "personal",
    "relationships",
    "learning",
    "travel",
    "creative",
    "other"
]

PRIORITIES = ["high", "medium", "low"]
STATUSES = ["pending", "in_progress", "completed", "abandoned"]

# ============================================================
# OpenAI Client
# ============================================================

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

# ============================================================
# Bucket List Functions
# ============================================================

def add_bucket_list_item(item: str, category: str = "other", 
                         priority: str = "medium", target_year: str = "",
                         notes: str = "") -> Dict:
    """Add an item to the bucket list."""
    return sm.add_bucket_list_item(
        item=item,
        category=category,
        priority=priority,
        target_year=target_year,
        notes=notes
    )

def get_bucket_list(status: str = None, category: str = None) -> List[Dict]:
    """Get bucket list items."""
    return sm.get_bucket_list(status=status, category=category)

def update_bucket_list_item(item_id: str, updates: Dict) -> Dict:
    """Update a bucket list item."""
    return sm.update_bucket_list_item(item_id, updates)

def complete_bucket_list_item(item_id: str) -> Dict:
    """Mark a bucket list item as completed."""
    return sm.update_bucket_list_item(item_id, {"status": "completed"})

def get_bucket_list_stats() -> Dict:
    """Get statistics about the bucket list."""
    items = get_bucket_list()
    
    total = len(items)
    completed = len([i for i in items if i.get("status") == "completed"])
    in_progress = len([i for i in items if i.get("status") == "in_progress"])
    pending = len([i for i in items if i.get("status") == "pending"])
    
    by_category = {}
    for item in items:
        cat = item.get("category", "other")
        by_category[cat] = by_category.get(cat, 0) + 1
    
    by_priority = {}
    for item in items:
        pri = item.get("priority", "medium")
        by_priority[pri] = by_priority.get(pri, 0) + 1
    
    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "pending": pending,
        "completion_rate": round(completed / total * 100, 1) if total > 0 else 0,
        "by_category": by_category,
        "by_priority": by_priority
    }

# ============================================================
# Yearly Goals Functions
# ============================================================

def add_yearly_goal(year: int, goal: str, category: str = "other",
                    q1: str = "", q2: str = "", q3: str = "", q4: str = "") -> Dict:
    """Add a yearly goal."""
    return sm.add_yearly_goal(
        year=year,
        goal=goal,
        category=category,
        q1=q1, q2=q2, q3=q3, q4=q4
    )

def get_yearly_goals(year: int = None) -> List[Dict]:
    """Get yearly goals."""
    if year is None:
        year = datetime.now(TZ).year
    return sm.get_yearly_goals(year=year)

def update_yearly_goal(goal_id: str, updates: Dict) -> Dict:
    """Update a yearly goal."""
    return sm.update_yearly_goal(goal_id, updates)

def get_yearly_stats(year: int = None) -> Dict:
    """Get statistics for yearly goals."""
    if year is None:
        year = datetime.now(TZ).year
    
    goals = get_yearly_goals(year)
    
    total = len(goals)
    completed = len([g for g in goals if g.get("status") == "completed"])
    active = len([g for g in goals if g.get("status") == "active"])
    
    # Calculate average progress
    progress_values = []
    for g in goals:
        try:
            progress_values.append(int(g.get("progress", 0)))
        except (ValueError, TypeError):
            pass
    
    avg_progress = round(sum(progress_values) / len(progress_values), 1) if progress_values else 0
    
    by_category = {}
    for g in goals:
        cat = g.get("category", "other")
        by_category[cat] = by_category.get(cat, 0) + 1
    
    return {
        "year": year,
        "total": total,
        "completed": completed,
        "active": active,
        "average_progress": avg_progress,
        "by_category": by_category
    }

# ============================================================
# AI Goal Coach
# ============================================================

def get_goal_advice(query: str = "", context: str = "general") -> str:
    """Get AI-powered goal advice."""
    bucket_stats = get_bucket_list_stats()
    yearly_stats = get_yearly_stats()
    bucket_items = get_bucket_list()
    yearly_goals = get_yearly_goals()
    
    # Build context
    context_parts = []
    
    # Bucket list summary
    context_parts.append(f"Bucket List: {bucket_stats['total']} items total, {bucket_stats['completed']} completed ({bucket_stats['completion_rate']}%)")
    
    # Sample bucket items
    pending_bucket = [i for i in bucket_items if i.get("status") in ("pending", "in_progress")][:5]
    if pending_bucket:
        bucket_text = "\n".join([f"- {i.get('item')} ({i.get('category')}, {i.get('priority')} priority)" for i in pending_bucket])
        context_parts.append(f"Current bucket list items:\n{bucket_text}")
    
    # Yearly goals summary
    context_parts.append(f"\n{yearly_stats['year']} Goals: {yearly_stats['total']} goals, {yearly_stats['average_progress']}% avg progress")
    
    # Active yearly goals
    active_yearly = [g for g in yearly_goals if g.get("status") == "active"][:5]
    if active_yearly:
        yearly_text = "\n".join([f"- {g.get('goal')} ({g.get('progress', 0)}% complete)" for g in active_yearly])
        context_parts.append(f"Active yearly goals:\n{yearly_text}")
    
    full_context = "\n".join(context_parts)
    
    system_prompt = f"""You are an encouraging and insightful goal coach AI.
You help users achieve their bucket list dreams and yearly goals.

USER'S GOALS DATA:
{full_context}

Guidelines:
- Be encouraging but realistic
- Suggest actionable next steps
- Help prioritize when asked
- Connect different goals when relevant
- Consider time constraints and priorities
- Keep responses concise and practical
"""

    client = _get_openai_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query if query else f"Give me advice about my {context} goals"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't generate advice. Error: {e}"

# ============================================================
# UI Components
# ============================================================

def render_bucket_list_add():
    """Render the bucket list add form."""
    st.markdown("### ➕ Add to Bucket List")
    
    item = st.text_input("What's on your bucket list?", placeholder="e.g., Visit Japan", key="bucket_item")
    
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", BUCKET_LIST_CATEGORIES, key="bucket_category")
    with col2:
        priority = st.selectbox("Priority", PRIORITIES, index=1, key="bucket_priority")
    
    col3, col4 = st.columns(2)
    with col3:
        target_year = st.text_input("Target Year (optional)", placeholder="e.g., 2025", key="bucket_year")
    with col4:
        notes = st.text_input("Notes (optional)", key="bucket_notes")
    
    if st.button("✨ Add to Bucket List", key="add_bucket_btn"):
        if item:
            result = add_bucket_list_item(
                item=item,
                category=category,
                priority=priority,
                target_year=target_year,
                notes=notes
            )
            if result.get("ok"):
                st.success("Added to bucket list! 🎉")
                st.rerun()
            else:
                st.error(f"Failed: {result.get('error')}")
        else:
            st.warning("Please enter an item")

def render_bucket_list_view():
    """Render the bucket list view."""
    st.markdown("### 🪣 My Bucket List")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        filter_status = st.selectbox(
            "Status",
            ["all", "pending", "in_progress", "completed"],
            key="bucket_filter_status"
        )
    with col2:
        filter_category = st.selectbox(
            "Category",
            ["all"] + BUCKET_LIST_CATEGORIES,
            key="bucket_filter_category"
        )
    
    # Get items
    status_filter = None if filter_status == "all" else filter_status
    category_filter = None if filter_category == "all" else filter_category
    items = get_bucket_list(status=status_filter, category=category_filter)
    
    if not items:
        st.info("No bucket list items found. Add some dreams! ✨")
        return
    
    # Group by priority for display
    high = [i for i in items if i.get("priority") == "high"]
    medium = [i for i in items if i.get("priority") == "medium"]
    low = [i for i in items if i.get("priority") == "low"]
    
    def render_item(item):
        status = item.get("status", "pending")
        status_emoji = {"pending": "⬜", "in_progress": "🔄", "completed": "✅", "abandoned": "❌"}.get(status, "⬜")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            text = item.get("item", "")
            if status == "completed":
                st.markdown(f"{status_emoji} ~~{text}~~")
            else:
                st.markdown(f"{status_emoji} **{text}**")
            
            details = []
            if item.get("category"):
                details.append(f"📂 {item['category']}")
            if item.get("target_year"):
                details.append(f"📅 {item['target_year']}")
            if item.get("notes"):
                details.append(f"📝 {item['notes']}")
            if details:
                st.caption(" | ".join(details))
        
        with col2:
            if status != "completed":
                new_status = st.selectbox(
                    "Status",
                    ["pending", "in_progress", "completed"],
                    index=["pending", "in_progress", "completed"].index(status),
                    key=f"status_{item.get('id')}",
                    label_visibility="collapsed"
                )
                if new_status != status:
                    update_bucket_list_item(item.get("id"), {"status": new_status})
                    st.rerun()
        
        with col3:
            if status != "completed":
                if st.button("✅", key=f"complete_bucket_{item.get('id')}"):
                    complete_bucket_list_item(item.get("id"))
                    st.rerun()
    
    if high:
        st.markdown("#### 🔴 High Priority")
        for item in high:
            render_item(item)
            st.divider()
    
    if medium:
        st.markdown("#### 🟡 Medium Priority")
        for item in medium:
            render_item(item)
            st.divider()
    
    if low:
        st.markdown("#### 🟢 Low Priority")
        for item in low:
            render_item(item)
            st.divider()

def render_yearly_goals_add():
    """Render yearly goals add form."""
    st.markdown("### ➕ Add Yearly Goal")
    
    current_year = datetime.now(TZ).year
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=current_year, max_value=current_year + 5, value=current_year, key="goal_year")
    with col2:
        category = st.selectbox("Category", YEARLY_GOAL_CATEGORIES, key="goal_category")
    
    goal = st.text_input("Goal", placeholder="e.g., Read 24 books", key="goal_text")
    
    st.markdown("**Quarterly Targets (optional)**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        q1 = st.text_input("Q1", key="goal_q1")
    with col2:
        q2 = st.text_input("Q2", key="goal_q2")
    with col3:
        q3 = st.text_input("Q3", key="goal_q3")
    with col4:
        q4 = st.text_input("Q4", key="goal_q4")
    
    if st.button("🎯 Add Goal", key="add_goal_btn"):
        if goal:
            result = add_yearly_goal(
                year=year,
                goal=goal,
                category=category,
                q1=q1, q2=q2, q3=q3, q4=q4
            )
            if result.get("ok"):
                st.success("Goal added! 🎯")
                st.rerun()
            else:
                st.error(f"Failed: {result.get('error')}")
        else:
            st.warning("Please enter a goal")

def render_yearly_goals_view():
    """Render yearly goals view."""
    current_year = datetime.now(TZ).year
    
    year = st.selectbox(
        "Year",
        list(range(current_year - 2, current_year + 3)),
        index=2,
        key="view_year"
    )
    
    st.markdown(f"### 🎯 {year} Goals")
    
    goals = get_yearly_goals(year)
    
    if not goals:
        st.info(f"No goals set for {year}. Add some! 🎯")
        return
    
    # Stats
    stats = get_yearly_stats(year)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Goals", stats["total"])
    with col2:
        st.metric("Completed", stats["completed"])
    with col3:
        st.metric("Avg Progress", f"{stats['average_progress']}%")
    
    st.divider()
    
    # Goals list
    for goal in goals:
        status = goal.get("status", "active")
        progress = int(goal.get("progress", 0) or 0)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            status_emoji = {"active": "🎯", "completed": "✅", "abandoned": "❌"}.get(status, "🎯")
            
            if status == "completed":
                st.markdown(f"{status_emoji} ~~{goal.get('goal')}~~")
            else:
                st.markdown(f"{status_emoji} **{goal.get('goal')}**")
            
            st.progress(min(progress / 100, 1.0))
            
            # Quarterly targets
            quarters = []
            for q in ["q1_target", "q2_target", "q3_target", "q4_target"]:
                if goal.get(q):
                    quarters.append(f"{q[:2].upper()}: {goal[q]}")
            if quarters:
                st.caption(" | ".join(quarters))
        
        with col2:
            if status == "active":
                new_progress = st.slider(
                    "Progress",
                    0, 100, progress,
                    key=f"progress_{goal.get('goal_id')}"
                )
                if new_progress != progress:
                    if st.button("💾", key=f"save_{goal.get('goal_id')}"):
                        updates = {"progress": str(new_progress)}
                        if new_progress >= 100:
                            updates["status"] = "completed"
                        update_yearly_goal(goal.get("goal_id"), updates)
                        st.rerun()
        
        st.divider()

def render_stats_dashboard():
    """Render a combined stats dashboard."""
    st.markdown("### 📊 Goals Overview")
    
    bucket_stats = get_bucket_list_stats()
    yearly_stats = get_yearly_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🪣 Bucket List")
        st.metric("Total Items", bucket_stats["total"])
        st.metric("Completed", f"{bucket_stats['completed']} ({bucket_stats['completion_rate']}%)")
        st.metric("In Progress", bucket_stats["in_progress"])
    
    with col2:
        st.markdown(f"#### 🎯 {yearly_stats['year']} Goals")
        st.metric("Total Goals", yearly_stats["total"])
        st.metric("Completed", yearly_stats["completed"])
        st.metric("Avg Progress", f"{yearly_stats['average_progress']}%")

def render_ai_coach():
    """Render AI goal coach."""
    st.markdown("### 🤖 Goal Coach")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💡 Get Inspiration", key="coach_inspire"):
            with st.spinner("Thinking..."):
                advice = get_goal_advice("What should I focus on next? Give me some inspiration.", "bucket_list")
                st.session_state["goal_coach_response"] = advice
    with col2:
        if st.button("📊 Review Progress", key="coach_progress"):
            with st.spinner("Thinking..."):
                advice = get_goal_advice("How am I doing on my goals? Any suggestions?", "yearly")
                st.session_state["goal_coach_response"] = advice
    with col3:
        if st.button("🎯 Prioritize", key="coach_prioritize"):
            with st.spinner("Thinking..."):
                advice = get_goal_advice("Help me prioritize my goals and bucket list items.", "general")
                st.session_state["goal_coach_response"] = advice
    
    user_q = st.text_input("Ask your goal coach...", placeholder="e.g., What's a good goal for Q2?", key="goal_coach_q")
    if user_q:
        if st.button("Ask", key="ask_goal_coach"):
            with st.spinner("Thinking..."):
                advice = get_goal_advice(user_q)
                st.session_state["goal_coach_response"] = advice
    
    if "goal_coach_response" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["goal_coach_response"])
