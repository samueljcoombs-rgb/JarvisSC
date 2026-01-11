# pages/1_🏋️_Health_Fitness.py
"""
Health & Fitness Dashboard - Consolidated view with today's workout,
AI coach, and quick logging.
"""
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
from openai import OpenAI

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
# User Profile - Edit this to personalize AI coaching
# ============================================================

USER_PROFILE = """
- Age: 30 years old
- Gender: Male
- Location: UK
- Height: 5ft 9in (175cm)
- Occupation: Office job (sedentary during work hours)
- Training Experience: ~6 months of gym training (beginner/intermediate)
"""

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
# OpenAI Client
# ============================================================

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            api_key = None
    if api_key:
        return OpenAI(api_key=api_key)
    return None

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>
.health-header {
    background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
}
.health-header h1 {
    color: white;
    margin: 0;
    font-size: 1.8rem;
    font-weight: 800;
}
.health-header p {
    color: rgba(255,255,255,0.85);
    margin: 0.25rem 0 0 0;
    font-size: 0.95rem;
}
.workout-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.workout-card.completed {
    opacity: 0.6;
    text-decoration: line-through;
}
.stat-mini {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.stat-mini .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #10b981;
}
.stat-mini .label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.6);
}
.log-section {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.coach-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 1.25rem;
    margin-top: 1rem;
}
.goal-item {
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 10px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}
.progress-bar {
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
    height: 6px;
    overflow: hidden;
    margin-top: 0.4rem;
}
.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    border-radius: 6px;
}
.analyze-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 12px !important;
    border: none !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

today_str = datetime.now(TZ).strftime("%A, %d %B")
st.markdown(f"""
<div class="health-header">
    <h1>🏋️ Health & Fitness</h1>
    <p>{today_str} • Track, Train, Transform</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================

if "health_chat" not in st.session_state:
    st.session_state.health_chat = []

# ============================================================
# Helper: Get Today's Workout from Todos Sheet
# ============================================================

def get_todays_workout():
    """Get today's gym exercises from the todo sheet."""
    try:
        # Try to read from the existing todo sheet gym tab
        import gspread
        from google.oauth2.service_account import Credentials
        import json
        
        raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not raw:
            return []
        
        data = json.loads(raw) if isinstance(raw, str) else raw
        creds = Credentials.from_service_account_info(data, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ])
        gc = gspread.authorize(creds)
        
        # Try the TODO sheet first
        todo_url = st.secrets.get("TODO_SHEET_URL") or os.getenv("TODO_SHEET_URL")
        if todo_url:
            sh = gc.open_by_url(todo_url)
            try:
                ws = sh.worksheet("gym")
                rows = ws.get_all_values()
                if len(rows) > 1:
                    headers = rows[0]
                    exercises = []
                    for row in rows[1:]:
                        if row and any(cell.strip() for cell in row):
                            ex = {}
                            for i, h in enumerate(headers):
                                ex[h.lower().strip()] = row[i] if i < len(row) else ""
                            exercises.append(ex)
                    return exercises
            except:
                pass
        return []
    except Exception as e:
        return []

# ============================================================
# Helper: AI Coach Analysis
# ============================================================

def get_coach_analysis():
    """Get comprehensive AI coach analysis of health data."""
    client = get_openai_client()
    if not client:
        return "❌ OpenAI API key not configured."
    
    # Gather all the data
    try:
        summary = ht.get_health_summary(days=30)
    except:
        summary = {}
    
    try:
        health_logs = sm.get_health_logs(days=14)
    except:
        health_logs = []
    
    try:
        workout_logs = sm.get_workout_logs(days=14)
    except:
        workout_logs = []
    
    try:
        goals = sm.get_fitness_goals(status="active")
    except:
        goals = []
    
    # Build context
    weight_data = summary.get("weight", {})
    nutrition_data = summary.get("nutrition", {})
    workout_data = summary.get("workouts", {})
    exercise_progress = summary.get("exercise_progress", {})
    
    context = f"""
USER PROFILE:
{USER_PROFILE}

USER'S FITNESS DATA (Last 30 days):

WEIGHT SUMMARY:
- Current: {weight_data.get('current', 'Not logged')} stone
- Average: {weight_data.get('average', 'N/A')} stone  
- Change: {weight_data.get('change', 'N/A')} stone
- Entries: {weight_data.get('entries', 0)}

NUTRITION SUMMARY:
- Average Calories: {nutrition_data.get('avg_calories', 'Not logged')} kcal
- Average Protein: {nutrition_data.get('avg_protein', 'Not logged')}g
- Days tracked: {nutrition_data.get('entries', 0)}

DAILY HEALTH LOGS (most recent):
"""
    # Add actual daily health log entries
    for log in health_logs[-10:]:
        date = log.get("date", "Unknown")
        w_st = log.get("weight_stone", "")
        w_lb = log.get("weight_lbs", "")
        cal = log.get("calories", "")
        prot = log.get("protein_g", "")
        notes = log.get("notes", "")
        
        weight_str = f"{w_st}st {w_lb}lb" if w_st else "Not logged"
        cal_str = f"{cal} kcal" if cal else "N/A"
        prot_str = f"{prot}g protein" if prot else "N/A"
        
        context += f"- {date}: Weight: {weight_str}, Calories: {cal_str}, Protein: {prot_str}\n"
    
    context += f"""
WORKOUT SUMMARY:
- Strength sessions: {workout_data.get('strength_sessions', 0)}
- Cardio sessions: {workout_data.get('cardio_sessions', 0)}
- Total running distance: {workout_data.get('total_distance_km', 0)} km

EXERCISE PROGRESS (weight lifted over time):
"""
    
    for exercise, data in exercise_progress.items():
        if data:
            weights = [d.get('weight', 0) for d in data if d.get('weight')]
            if weights:
                context += f"- {exercise}: Started at {weights[0]}kg, now at {weights[-1]}kg\n"
    
    context += "\nACTIVE GOALS:\n"
    if goals:
        for g in goals:
            context += f"- {g.get('goal', 'Unknown')} (Progress: {g.get('progress', 0)}%, Target: {g.get('target_date', 'No date')})\n"
    else:
        context += "- No active goals set\n"
    
    context += f"\nRECENT WORKOUTS (detailed):\n"
    for w in workout_logs[-10:]:
        wtype = w.get('type', 'strength')
        if wtype == 'strength':
            context += f"- {w.get('date')}: {w.get('exercise')} - {w.get('sets')}x{w.get('reps')} @ {w.get('weight_kg')}kg\n"
        else:
            context += f"- {w.get('date')}: {w.get('exercise')} - {w.get('distance_km')}km in {w.get('duration_mins')}min\n"
    
    prompt = f"""You are a supportive but direct fitness coach. Analyze this user's ACTUAL logged data and provide:

1. **Performance Summary** - How are they doing overall? Reference their SPECIFIC numbers from the logs.
2. **Progress on Goals** - Are they on track? What needs attention?
3. **Motivation** - Acknowledge wins (be specific about what they've achieved), be encouraging but honest
4. **Actionable Advice**:
   - Should they increase any weights? Which exercises and by how much?
   - Based on their logged calories ({nutrition_data.get('avg_calories', 'N/A')}) and protein ({nutrition_data.get('avg_protein', 'N/A')}g), do they need adjustments?
   - What should they focus on this week?

IMPORTANT: Use their ACTUAL logged data. Don't say data is missing if it's there. Be specific with numbers.

{context}"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "You are a knowledgeable, supportive fitness coach working with a 30-year-old male beginner (6 months training) who has an office job. Give specific, data-driven advice tailored to his experience level. Don't suggest advanced techniques - focus on progressive overload fundamentals. Don't say tracking is inconsistent if there are multiple log entries."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error getting analysis: {e}"

def chat_with_coach(question: str):
    """Chat with the AI fitness coach."""
    client = get_openai_client()
    if not client:
        return "❌ OpenAI API key not configured."
    
    # Get current context
    try:
        summary = ht.get_health_summary(days=14)
    except:
        summary = {}
    
    try:
        health_logs = sm.get_health_logs(days=7)
    except:
        health_logs = []
    
    try:
        workout_logs = sm.get_workout_logs(days=7)
    except:
        workout_logs = []
    
    try:
        goals = sm.get_fitness_goals(status="active")
    except:
        goals = []
    
    context = f"""USER PROFILE:
{USER_PROFILE}

User's recent fitness data (last 14 days):
- Current Weight: {summary.get('weight', {}).get('current', 'N/A')} stone
- Weight Change: {summary.get('weight', {}).get('change', 'N/A')} stone
- Avg Calories: {summary.get('nutrition', {}).get('avg_calories', 'N/A')} kcal
- Avg Protein: {summary.get('nutrition', {}).get('avg_protein', 'N/A')}g
- Strength sessions: {summary.get('workouts', {}).get('strength_sessions', 0)}
- Cardio sessions: {summary.get('workouts', {}).get('cardio_sessions', 0)}

Recent Daily Logs:
"""
    for log in health_logs[-5:]:
        context += f"- {log.get('date')}: {log.get('weight_stone', '')}st {log.get('weight_lbs', '')}lb, {log.get('calories', '')}cal, {log.get('protein_g', '')}g protein\n"
    
    context += "\nRecent Workouts:\n"
    for w in workout_logs[-5:]:
        context += f"- {w.get('date')}: {w.get('exercise')} - {w.get('sets')}x{w.get('reps')} @ {w.get('weight_kg')}kg\n"
    
    context += "\nActive Goals:\n"
    for g in goals[:3]:
        context += f"- {g.get('goal')} ({g.get('progress', 0)}% done)\n"
    
    messages = [
        {"role": "system", "content": f"You are a helpful fitness coach for a 30yo male beginner who works an office job. Be concise and actionable. Reference specific numbers from the user's data. Tailor advice to their experience level.\n\nUser's Data:\n{context}"},
    ]
    
    # Add chat history
    for msg in st.session_state.health_chat[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=messages,
            max_completion_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {e}"

# ============================================================
# Main Layout: 3 Columns
# ============================================================

left_col, mid_col, right_col = st.columns([3, 4, 3], gap="medium")

# ============================================================
# LEFT COLUMN: Today's Workout + Quick Log
# ============================================================

with left_col:
    st.markdown("### 📋 Today's Workout")
    
    workout_exercises = get_todays_workout()
    
    if workout_exercises:
        for ex in workout_exercises:
            name = ex.get("exercise", ex.get("name", "Exercise"))
            sets = ex.get("sets", "")
            reps = ex.get("reps", "")
            weight = ex.get("weight", ex.get("weight_kg", ""))
            
            details = []
            if sets:
                details.append(f"{sets} sets")
            if reps:
                details.append(f"{reps} reps")
            if weight:
                details.append(f"{weight}kg")
            
            detail_str = " • ".join(details) if details else ""
            
            st.markdown(f"""
            <div class="workout-card">
                <strong>💪 {name}</strong><br>
                <small style="color: rgba(255,255,255,0.6);">{detail_str}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No workout scheduled today. Check your Gym sheet or add exercises!")
    
    st.markdown("---")
    
    # Quick Log Section
    st.markdown("### ⚡ Quick Log")
    
    with st.expander("📝 Log Today's Health", expanded=False):
        with st.form("quick_health_log"):
            col1, col2 = st.columns(2)
            with col1:
                weight_st = st.number_input("Weight (stone)", min_value=0, max_value=30, value=0, key="qh_st")
            with col2:
                weight_lb = st.number_input("lbs", min_value=0, max_value=13, value=0, key="qh_lb")
            
            col1, col2 = st.columns(2)
            with col1:
                calories = st.number_input("Calories", min_value=0, max_value=10000, value=0, step=100, key="qh_cal")
            with col2:
                protein = st.number_input("Protein (g)", min_value=0, max_value=500, value=0, step=10, key="qh_prot")
            
            notes = st.text_input("Notes", placeholder="How do you feel?", key="qh_notes")
            
            if st.form_submit_button("💾 Save Health Log", use_container_width=True):
                try:
                    sm.log_health(
                        date=datetime.now(TZ).date().isoformat(),
                        weight_stone=weight_st if weight_st > 0 else None,
                        weight_lbs=weight_lb if weight_lb > 0 else None,
                        calories=calories if calories > 0 else None,
                        protein_g=protein if protein > 0 else None,
                        notes=notes
                    )
                    st.success("✅ Logged!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with st.expander("🏋️ Log Workout", expanded=False):
        with st.form("quick_workout_log"):
            exercise = st.text_input("Exercise", placeholder="e.g., Bench Press", key="qw_ex")
            
            workout_type = st.radio("Type", ["Strength", "Running", "Cardio"], horizontal=True, key="qw_type")
            
            if workout_type == "Strength":
                col1, col2, col3 = st.columns(3)
                with col1:
                    sets = st.number_input("Sets", min_value=0, max_value=20, value=3, key="qw_sets")
                with col2:
                    reps = st.number_input("Reps", min_value=0, max_value=100, value=10, key="qw_reps")
                with col3:
                    weight_kg = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=0.0, step=2.5, key="qw_weight")
                distance = 0
                duration = 0
                pace = ""
            elif workout_type == "Running":
                col1, col2 = st.columns(2)
                with col1:
                    distance = st.number_input("Distance (km)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="qw_dist")
                with col2:
                    duration = st.number_input("Time (mins)", min_value=0, max_value=300, value=0, key="qw_dur")
                pace = st.text_input("Pace (min/km)", placeholder="e.g., 5:30", key="qw_pace")
                sets = reps = 0
                weight_kg = 0
            else:
                duration = st.number_input("Duration (mins)", min_value=0, max_value=300, value=0, key="qw_cardio_dur")
                sets = reps = 0
                weight_kg = distance = 0
                pace = ""
            
            if st.form_submit_button("💾 Save Workout", use_container_width=True):
                if exercise:
                    try:
                        sm.log_workout(
                            date=datetime.now(TZ).date().isoformat(),
                            exercise=exercise,
                            sets=sets,
                            reps=reps,
                            weight_kg=weight_kg,
                            duration_mins=duration,
                            distance_km=distance,
                            pace_per_km=pace,
                            workout_type=workout_type.lower()
                        )
                        st.success("✅ Workout logged!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Enter exercise name")
    
    with st.expander("🎯 Add Goal", expanded=False):
        with st.form("quick_goal"):
            goal_text = st.text_input("Goal", placeholder="e.g., Bench 100kg", key="qg_text")
            target_date = st.date_input("Target Date", value=datetime.now(TZ).date() + timedelta(days=90), key="qg_date")
            
            if st.form_submit_button("🎯 Add Goal", use_container_width=True):
                if goal_text:
                    try:
                        sm.add_fitness_goal(goal=goal_text, target_date=str(target_date))
                        st.success("✅ Goal added!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Enter a goal")

# ============================================================
# MIDDLE COLUMN: AI Coach Chat
# ============================================================

with mid_col:
    st.markdown("### 🤖 AI Fitness Coach")
    
    # Big Analyze Button
    if st.button("📊 Analyze My Performance & Give Advice", use_container_width=True, type="primary"):
        with st.spinner("Analyzing your fitness data..."):
            analysis = get_coach_analysis()
            st.session_state.health_chat.append({"role": "assistant", "content": analysis})
    
    st.markdown("---")
    
    # Chat display
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.health_chat:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.5);">
                <p>👋 I'm your AI fitness coach!</p>
                <p style="font-size: 0.9rem;">Hit the button above for a full analysis, or ask me anything about your training.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.health_chat:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask your coach anything...", key="coach_input")
    
    if user_input:
        st.session_state.health_chat.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = chat_with_coach(user_input)
            st.session_state.health_chat.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Quick question buttons
    st.markdown("#### Quick Questions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💪 Should I increase weights?", use_container_width=True):
            st.session_state.health_chat.append({"role": "user", "content": "Based on my recent workouts, should I increase any weights? Which exercises and by how much?"})
            with st.spinner("Analyzing..."):
                response = chat_with_coach("Based on my recent workouts, should I increase any weights? Which exercises and by how much?")
                st.session_state.health_chat.append({"role": "assistant", "content": response})
            st.rerun()
    with col2:
        if st.button("🍽️ Nutrition advice", use_container_width=True):
            st.session_state.health_chat.append({"role": "user", "content": "Based on my goals and current data, should I adjust my calories or protein intake?"})
            with st.spinner("Analyzing..."):
                response = chat_with_coach("Based on my goals and current data, should I adjust my calories or protein intake?")
                st.session_state.health_chat.append({"role": "assistant", "content": response})
            st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Am I on track?", use_container_width=True):
            st.session_state.health_chat.append({"role": "user", "content": "Am I on track with my fitness goals? What should I focus on?"})
            with st.spinner("Analyzing..."):
                response = chat_with_coach("Am I on track with my fitness goals? What should I focus on?")
                st.session_state.health_chat.append({"role": "assistant", "content": response})
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.health_chat = []
            st.rerun()

# ============================================================
# RIGHT COLUMN: Stats & Goals
# ============================================================

with right_col:
    st.markdown("### 📊 Quick Stats")
    
    try:
        summary = ht.get_health_summary(days=30)
        weight_data = summary.get("weight", {})
        nutrition_data = summary.get("nutrition", {})
        workout_data = summary.get("workouts", {})
        
        col1, col2 = st.columns(2)
        with col1:
            current_weight = weight_data.get("current")
            if current_weight:
                st.markdown(f"""
                <div class="stat-mini">
                    <div class="value">{current_weight:.1f}</div>
                    <div class="label">Current (stone)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-mini">
                    <div class="value">--</div>
                    <div class="label">Current Weight</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            change = weight_data.get("change")
            if change is not None:
                color = "#10b981" if change <= 0 else "#f59e0b"
                st.markdown(f"""
                <div class="stat-mini">
                    <div class="value" style="color: {color};">{'+' if change > 0 else ''}{change:.1f}</div>
                    <div class="label">30d Change</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-mini">
                    <div class="value">--</div>
                    <div class="label">30d Change</div>
                </div>
                """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            total_workouts = workout_data.get("strength_sessions", 0) + workout_data.get("cardio_sessions", 0)
            st.markdown(f"""
            <div class="stat-mini">
                <div class="value">{total_workouts}</div>
                <div class="label">Workouts (30d)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_cal = nutrition_data.get("avg_calories")
            st.markdown(f"""
            <div class="stat-mini">
                <div class="value">{avg_cal or '--'}</div>
                <div class="label">Avg Calories</div>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            avg_prot = nutrition_data.get("avg_protein")
            st.markdown(f"""
            <div class="stat-mini">
                <div class="value">{avg_prot or '--'}g</div>
                <div class="label">Avg Protein</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            distance = workout_data.get("total_distance_km", 0)
            st.markdown(f"""
            <div class="stat-mini">
                <div class="value">{distance:.1f}</div>
                <div class="label">km Run (30d)</div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.info("Start logging to see your stats!")
    
    st.markdown("---")
    
    # Active Goals
    st.markdown("### 🎯 Active Goals")
    
    try:
        goals = sm.get_fitness_goals(status="active")
        
        if goals:
            for goal in goals[:5]:
                goal_text = goal.get("goal", "Unknown")
                progress = int(float(goal.get("progress", 0) or 0))
                target = goal.get("target_date", "")
                goal_id = goal.get("goal_id", "")
                
                st.markdown(f"""
                <div class="goal-item">
                    <strong>{goal_text}</strong>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress}%;"></div>
                    </div>
                    <small style="color: rgba(255,255,255,0.5);">{progress}% • Target: {target or 'No date'}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Update progress
                new_prog = st.slider(
                    f"Update {goal_text[:20]}...",
                    0, 100, progress,
                    key=f"goal_prog_{goal_id}",
                    label_visibility="collapsed"
                )
                if new_prog != progress:
                    try:
                        updates = {"progress": str(new_prog)}
                        if new_prog == 100:
                            updates["status"] = "completed"
                        sm.update_fitness_goal(goal_id, updates)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating: {e}")
        else:
            st.info("No active goals. Add one!")
    except Exception as e:
        st.error(f"Error loading goals: {e}")
    
    st.markdown("---")
    
    # Recent Activity
    st.markdown("### 📅 Recent Activity")
    
    try:
        workouts = sm.get_workout_logs(days=7)
        if workouts:
            for w in workouts[-5:]:
                date = w.get("date", "")
                exercise = w.get("exercise", "Workout")
                wtype = w.get("type", "")
                
                if wtype == "strength":
                    detail = f"{w.get('sets', '')}x{w.get('reps', '')} @ {w.get('weight_kg', '')}kg"
                elif wtype in ("running", "cardio"):
                    detail = f"{w.get('distance_km', '')}km in {w.get('duration_mins', '')}min"
                else:
                    detail = ""
                
                st.markdown(f"**{date}**: {exercise} - {detail}")
        else:
            st.caption("No recent workouts logged")
    except:
        st.caption("Log workouts to see history")

# ============================================================
# Footer
# ============================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.page_link("app.py", label="🏠 Back to Home", icon="🏠")
