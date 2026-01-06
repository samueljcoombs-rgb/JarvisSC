# pages/1_🏋️_Health_Fitness.py
"""
Health & Fitness Dashboard - Track weight, workouts, nutrition, and get AI advice.
"""
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import os

# Import modules
try:
    from modules import sheets_memory as sm
    from modules import health_tools as ht
    from modules import global_styles as gs
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules import sheets_memory as sm
    from modules import health_tools as ht
    from modules import global_styles as gs

TZ = ZoneInfo("Europe/London")

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="Health & Fitness | Jarvis",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

gs.inject_global_styles()

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>
.health-header {
    background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
}
.health-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
}
.health-header p {
    color: rgba(255,255,255,0.85);
    margin: 0.25rem 0 0 0;
}
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
    color: #10b981;
}
.stat-card .label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.25rem;
}
.workout-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.goal-card {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}
.advice-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="health-header">
    <h1>🏋️ Health & Fitness</h1>
    <p>Track your progress, log workouts, and get personalized AI advice</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Main Layout
# ============================================================

# Tab navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "⚖️ Log Health", 
    "💪 Log Workout",
    "🎯 Goals",
    "🤖 AI Coach"
])

# ============================================================
# Tab 1: Dashboard
# ============================================================

with tab1:
    st.subheader("📊 Your Stats (Last 30 Days)")
    
    # Get summary data
    try:
        summary = ht.get_health_summary(days=30)
    except Exception as e:
        summary = {}
        st.warning(f"Could not load health data: {e}")
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_weight = summary.get("avg_weight", 0)
        weight_change = summary.get("weight_change", 0)
        change_text = f"({'+' if weight_change > 0 else ''}{weight_change:.1f})" if weight_change != 0 else ""
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{avg_weight:.1f} st</div>
            <div class="label">Avg Weight {change_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        workout_count = summary.get("workout_count", 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{workout_count}</div>
            <div class="label">Workout Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_calories = summary.get("avg_calories", 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{avg_calories:,.0f}</div>
            <div class="label">Avg Daily Calories</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_protein = summary.get("avg_protein", 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{avg_protein:.0f}g</div>
            <div class="label">Avg Daily Protein</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Recent activity
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📅 Recent Health Logs")
        try:
            health_logs = sm.get_health_logs(days=7)
            if health_logs:
                for log in reversed(health_logs[-7:]):
                    date = log.get("date", "Unknown")
                    w_st = log.get("weight_stone", "-")
                    w_lb = log.get("weight_lbs", "0")
                    cal = log.get("calories", "-")
                    prot = log.get("protein_g", "-")
                    
                    weight_str = f"{w_st}st {w_lb}lb" if w_st and w_st != "-" else "Not logged"
                    cal_str = f"{cal} kcal" if cal and cal != "-" else ""
                    prot_str = f"{prot}g protein" if prot and prot != "-" else ""
                    
                    st.markdown(f"""
                    <div class="workout-card">
                        <strong>{date}</strong><br>
                        ⚖️ {weight_str} &nbsp; {f'🔥 {cal_str}' if cal_str else ''} &nbsp; {f'🥩 {prot_str}' if prot_str else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No health logs yet. Start tracking!")
        except Exception as e:
            st.error(f"Error loading health logs: {e}")
    
    with col_right:
        st.subheader("💪 Recent Workouts")
        try:
            workout_logs = sm.get_workout_logs(days=7)
            if workout_logs:
                for log in reversed(workout_logs[-10:]):
                    date = log.get("date", "Unknown")
                    exercise = log.get("exercise", "Unknown")
                    workout_type = log.get("type", "")
                    
                    if workout_type == "running":
                        dist = log.get("distance_km", "-")
                        pace = log.get("pace_per_km", "-")
                        details = f"🏃 {dist}km @ {pace}/km"
                    elif workout_type == "strength":
                        sets = log.get("sets", "-")
                        reps = log.get("reps", "-")
                        weight = log.get("weight_kg", "-")
                        details = f"💪 {sets}x{reps} @ {weight}kg"
                    else:
                        duration = log.get("duration_mins", "-")
                        details = f"⏱️ {duration} mins"
                    
                    st.markdown(f"""
                    <div class="workout-card">
                        <strong>{date}</strong> - {exercise}<br>
                        {details}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No workouts logged yet. Start tracking!")
        except Exception as e:
            st.error(f"Error loading workouts: {e}")

# ============================================================
# Tab 2: Log Health
# ============================================================

with tab2:
    st.subheader("⚖️ Log Daily Health")
    
    with st.form("health_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            log_date = st.date_input(
                "Date",
                value=datetime.now(TZ).date(),
                key="health_date"
            )
            
            st.write("**Weight**")
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                weight_stone = st.number_input(
                    "Stone",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1,
                    key="weight_stone"
                )
            with wcol2:
                weight_lbs = st.number_input(
                    "Pounds",
                    min_value=0,
                    max_value=13,
                    value=0,
                    step=1,
                    key="weight_lbs"
                )
        
        with col2:
            calories = st.number_input(
                "Calories",
                min_value=0,
                max_value=10000,
                value=0,
                step=50,
                key="calories"
            )
            
            protein = st.number_input(
                "Protein (g)",
                min_value=0,
                max_value=500,
                value=0,
                step=5,
                key="protein"
            )
        
        notes = st.text_area("Notes (optional)", key="health_notes", height=80)
        
        submitted = st.form_submit_button("💾 Save Health Log", use_container_width=True)
        
        if submitted:
            if weight_stone == 0 and calories == 0 and protein == 0:
                st.warning("Please enter at least one measurement")
            else:
                try:
                    result = sm.log_health(
                        date=str(log_date),
                        weight_stone=weight_stone if weight_stone > 0 else None,
                        weight_lbs=weight_lbs if weight_stone > 0 else None,
                        calories=calories if calories > 0 else None,
                        protein_g=protein if protein > 0 else None,
                        notes=notes if notes else None
                    )
                    if result.get("ok"):
                        st.success("✅ Health log saved!")
                        st.rerun()
                    else:
                        st.error(f"Error saving: {result.get('error')}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================================
# Tab 3: Log Workout
# ============================================================

with tab3:
    st.subheader("💪 Log Workout")
    
    workout_type = st.radio(
        "Workout Type",
        ["Strength Training", "Running", "Cardio"],
        horizontal=True,
        key="workout_type_radio"
    )
    
    with st.form("workout_form", clear_on_submit=True):
        workout_date = st.date_input(
            "Date",
            value=datetime.now(TZ).date(),
            key="workout_date"
        )
        
        if workout_type == "Strength Training":
            exercise = st.text_input(
                "Exercise Name",
                placeholder="e.g., Bench Press, Squats, Deadlift",
                key="exercise_name"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                sets = st.number_input("Sets", min_value=1, max_value=20, value=3, key="sets")
            with col2:
                reps = st.number_input("Reps", min_value=1, max_value=100, value=10, key="reps")
            with col3:
                weight_kg = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=0.0, step=2.5, key="weight_kg")
            
            notes = st.text_area("Notes (optional)", key="strength_notes", height=80)
            
            submitted = st.form_submit_button("💾 Save Workout", use_container_width=True)
            
            if submitted:
                if not exercise:
                    st.warning("Please enter an exercise name")
                else:
                    try:
                        result = sm.log_workout(
                            date=str(workout_date),
                            exercise=exercise,
                            sets=sets,
                            reps=reps,
                            weight_kg=weight_kg,
                            workout_type="strength",
                            notes=notes if notes else None
                        )
                        if result.get("ok"):
                            st.success(f"✅ {exercise} logged!")
                            st.rerun()
                        else:
                            st.error(f"Error: {result.get('error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        elif workout_type == "Running":
            col1, col2 = st.columns(2)
            with col1:
                distance_km = st.number_input(
                    "Distance (km)",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=0.5,
                    key="distance_km"
                )
            with col2:
                pace_mins = st.number_input(
                    "Pace (mins per km)",
                    min_value=0,
                    max_value=20,
                    value=5,
                    key="pace_mins"
                )
                pace_secs = st.number_input(
                    "Pace (secs)",
                    min_value=0,
                    max_value=59,
                    value=30,
                    key="pace_secs"
                )
            
            duration_mins = st.number_input(
                "Duration (minutes)",
                min_value=0,
                max_value=300,
                value=int(distance_km * (pace_mins + pace_secs/60)),
                key="run_duration"
            )
            
            notes = st.text_area("Notes (optional)", key="run_notes", height=80)
            
            submitted = st.form_submit_button("💾 Save Run", use_container_width=True)
            
            if submitted:
                try:
                    pace_str = f"{pace_mins}:{pace_secs:02d}"
                    result = sm.log_workout(
                        date=str(workout_date),
                        exercise="Running",
                        distance_km=distance_km,
                        duration_mins=duration_mins,
                        pace_per_km=pace_str,
                        workout_type="running",
                        notes=notes if notes else None
                    )
                    if result.get("ok"):
                        st.success("✅ Run logged!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result.get('error')}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        else:  # Cardio
            exercise = st.text_input(
                "Exercise",
                placeholder="e.g., Cycling, Swimming, Rowing",
                key="cardio_exercise"
            )
            
            duration_mins = st.number_input(
                "Duration (minutes)",
                min_value=0,
                max_value=300,
                value=30,
                key="cardio_duration"
            )
            
            notes = st.text_area("Notes (optional)", key="cardio_notes", height=80)
            
            submitted = st.form_submit_button("💾 Save Cardio", use_container_width=True)
            
            if submitted:
                if not exercise:
                    st.warning("Please enter an exercise name")
                else:
                    try:
                        result = sm.log_workout(
                            date=str(workout_date),
                            exercise=exercise,
                            duration_mins=duration_mins,
                            workout_type="cardio",
                            notes=notes if notes else None
                        )
                        if result.get("ok"):
                            st.success(f"✅ {exercise} logged!")
                            st.rerun()
                        else:
                            st.error(f"Error: {result.get('error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")

# ============================================================
# Tab 4: Goals
# ============================================================

with tab4:
    st.subheader("🎯 Fitness Goals")
    
    # Add new goal
    with st.expander("➕ Add New Goal", expanded=False):
        with st.form("goal_form", clear_on_submit=True):
            goal_text = st.text_input("Goal", placeholder="e.g., Bench press 100kg")
            
            col1, col2 = st.columns(2)
            with col1:
                target_date = st.date_input(
                    "Target Date",
                    value=datetime.now(TZ).date() + timedelta(days=90),
                    key="goal_target"
                )
            with col2:
                progress = st.slider("Current Progress (%)", 0, 100, 0, key="goal_progress")
            
            submitted = st.form_submit_button("💾 Add Goal", use_container_width=True)
            
            if submitted:
                if not goal_text:
                    st.warning("Please enter a goal")
                else:
                    try:
                        result = sm.add_fitness_goal(
                            goal=goal_text,
                            target_date=str(target_date),
                            progress=progress
                        )
                        if result.get("ok"):
                            st.success("✅ Goal added!")
                            st.rerun()
                        else:
                            st.error(f"Error: {result.get('error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    st.divider()
    
    # Display existing goals
    try:
        goals = sm.get_fitness_goals()
        
        if goals:
            active_goals = [g for g in goals if g.get("status") in ("active", "")]
            completed_goals = [g for g in goals if g.get("status") == "completed"]
            
            st.write(f"**Active Goals ({len(active_goals)})**")
            for goal in active_goals:
                goal_id = goal.get("goal_id", "")
                goal_text = goal.get("goal", "Unknown goal")
                target = goal.get("target_date", "No date")
                progress = int(float(goal.get("progress", 0) or 0))
                
                st.markdown(f"""
                <div class="goal-card">
                    <strong>{goal_text}</strong><br>
                    <small>Target: {target}</small>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    new_progress = st.slider(
                        "Progress",
                        0, 100, progress,
                        key=f"progress_{goal_id}",
                        label_visibility="collapsed"
                    )
                with col2:
                    if st.button("Update", key=f"update_{goal_id}"):
                        sm.update_fitness_goal(goal_id, progress=new_progress)
                        st.rerun()
                with col3:
                    if st.button("✅ Complete", key=f"complete_{goal_id}"):
                        sm.update_fitness_goal(goal_id, status="completed", progress=100)
                        st.success("Goal completed! 🎉")
                        st.rerun()
                
                st.progress(progress / 100)
                st.write("")
            
            if completed_goals:
                with st.expander(f"✅ Completed Goals ({len(completed_goals)})"):
                    for goal in completed_goals:
                        st.write(f"✓ {goal.get('goal', 'Unknown')}")
        else:
            st.info("No goals yet. Add your first fitness goal above!")
    except Exception as e:
        st.error(f"Error loading goals: {e}")

# ============================================================
# Tab 5: AI Coach
# ============================================================

with tab5:
    st.subheader("🤖 AI Fitness Coach")
    
    st.markdown("""
    <div class="advice-card">
        <strong>Your Personal AI Coach</strong><br>
        Get personalized advice based on your logged data. I can help with:
        <ul>
            <li>💪 Strength training progression</li>
            <li>🏃 Running improvement</li>
            <li>🍎 Nutrition advice</li>
            <li>🎯 Goal setting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # Quick advice buttons
    st.write("**Quick Advice:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💪 Strength Progress", use_container_width=True):
            st.session_state["coach_question"] = "Based on my workout logs, how should I progress with my strength training? Should I increase weights?"
    
    with col2:
        if st.button("🏃 Running Tips", use_container_width=True):
            st.session_state["coach_question"] = "Based on my running logs, how can I improve my pace and endurance?"
    
    with col3:
        if st.button("🍎 Nutrition", use_container_width=True):
            st.session_state["coach_question"] = "Based on my calorie and protein logs, am I eating enough to support my training goals?"
    
    with col4:
        if st.button("⚖️ Weight Goals", use_container_width=True):
            st.session_state["coach_question"] = "Based on my weight logs, am I making good progress? What should I adjust?"
    
    st.divider()
    
    # Custom question
    question = st.text_area(
        "Ask your AI coach anything:",
        value=st.session_state.get("coach_question", ""),
        placeholder="e.g., Should I increase my bench press weight?",
        key="coach_input"
    )
    
    if st.button("🤖 Get Advice", use_container_width=True, type="primary"):
        if question:
            with st.spinner("Analyzing your data and generating advice..."):
                try:
                    advice = ht.get_fitness_advice(question)
                    st.markdown(f"""
                    <div class="advice-card">
                        <strong>🤖 Coach's Advice:</strong><br><br>
                        {advice}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error getting advice: {e}")
        else:
            st.warning("Please enter a question")
    
    # Clear the session state question after displaying
    if "coach_question" in st.session_state:
        del st.session_state["coach_question"]

# ============================================================
# Sidebar - Quick Stats
# ============================================================

with st.sidebar:
    st.header("🏋️ Quick Stats")
    
    try:
        summary = ht.get_health_summary(days=7)
        
        st.metric("Workouts This Week", summary.get("workout_count", 0))
        
        if summary.get("avg_weight", 0) > 0:
            st.metric(
                "Current Weight",
                f"{summary.get('avg_weight', 0):.1f} st",
                delta=f"{summary.get('weight_change', 0):.1f} st" if summary.get('weight_change') else None
            )
        
        if summary.get("avg_calories", 0) > 0:
            st.metric("Avg Calories", f"{summary.get('avg_calories', 0):,.0f}")
        
        if summary.get("avg_protein", 0) > 0:
            st.metric("Avg Protein", f"{summary.get('avg_protein', 0):.0f}g")
    except Exception:
        st.info("Log your first health data to see stats!")
    
    st.divider()
    
    st.page_link("app.py", label="🏠 Home", icon="🏠")
