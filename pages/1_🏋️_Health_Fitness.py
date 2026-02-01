# pages/1_🏋️_Health_Fitness.py
"""
Health & Fitness Dashboard v2 - Complete rebuild with:
- Auto-populated workouts from Google Sheets
- Day-specific routines (Mon/Wed/Fri/Sat = Gym, Tue/Thu/Sun = Run)
- Smart logging with pre-filled exercises
- AI coach with proper data analysis
- Sexy UI design
"""
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import random
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
TODAY = datetime.now(TZ)
DAY_NAME = TODAY.strftime("%A")

# ============================================================
# User Profile
# ============================================================

USER_PROFILE = """
- Age: 30 years old
- Gender: Male
- Location: UK
- Height: 5ft 9in (175cm)
- Occupation: Office job (sedentary during work hours)
- Training Experience: ~6 months of gym training (beginner/intermediate)
- Current Goal: Reach 12 stone by March 23rd 2026
- Training Split: Mon (Push), Wed (Legs), Fri (Pull), Sat (Full Body), Tue/Thu/Sun (Running)
"""

# ============================================================
# Day Configuration
# ============================================================

GYM_DAYS = ["Monday", "Wednesday", "Friday", "Saturday"]
RUN_DAYS = ["Tuesday", "Thursday", "Sunday"]
IS_GYM_DAY = DAY_NAME in GYM_DAYS
IS_RUN_DAY = DAY_NAME in RUN_DAYS

# Tomorrow's day for preview
TOMORROW = TODAY + timedelta(days=1)
TOMORROW_NAME = TOMORROW.strftime("%A")
IS_TOMORROW_GYM = TOMORROW_NAME in GYM_DAYS

# ============================================================
# Motivational Quotes
# ============================================================

MOTIVATIONAL_QUOTES = [
    "Every rep counts. Every step matters.",
    "You're building the body you deserve.",
    "Discipline beats motivation every time.",
    "Small progress is still progress.",
    "Your only competition is yesterday's you.",
    "Sweat now, shine later.",
    "The pain you feel today is the strength you feel tomorrow.",
    "Champions are made when no one is watching.",
    "Your body can stand almost anything. It's your mind you have to convince.",
    "The only bad workout is the one that didn't happen.",
    "Be stronger than your excuses.",
    "Results happen over time, not overnight.",
    "Push yourself because no one else is going to do it for you.",
    "Wake up. Work out. Look hot. Kick ass.",
    "Sore today, strong tomorrow.",
]

TODAYS_QUOTE = random.choice(MOTIVATIONAL_QUOTES)

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
# Custom Styling - Sexy Dark Theme
# ============================================================

st.markdown("""
<style>
/* Main Header - Gradient based on day type */
.health-header {
    background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
    padding: 1.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(16, 185, 129, 0.4);
    position: relative;
    overflow: hidden;
}
.health-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
}
.health-header.run-day {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 50%, #60a5fa 100%);
    box-shadow: 0 10px 40px rgba(59, 130, 246, 0.4);
}
.health-header h1 {
    color: white;
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.health-header .subtitle {
    color: rgba(255,255,255,0.9);
    margin: 0.3rem 0 0 0;
    font-size: 1rem;
    font-weight: 500;
}
.health-header .quote {
    color: rgba(255,255,255,0.8);
    margin: 0.5rem 0 0 0;
    font-size: 0.9rem;
    font-style: italic;
}
.health-header .day-badge {
    position: absolute;
    top: 1rem;
    right: 1.5rem;
    background: rgba(255,255,255,0.2);
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    color: white;
    backdrop-filter: blur(10px);
}

/* Workout Card - Premium Look */
.workout-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.workout-card:hover {
    transform: translateY(-2px);
    border-color: rgba(16, 185, 129, 0.4);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
}
.workout-card .exercise-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.5rem;
}
.workout-card .exercise-details {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}
.workout-card .detail-chip {
    background: rgba(16, 185, 129, 0.2);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    color: #34d399;
    font-weight: 600;
}
.workout-card .detail-chip.rpe {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
}
.workout-card .detail-chip.time {
    background: rgba(139, 92, 246, 0.2);
    color: #a78bfa;
}
.workout-card .detail-chip.weight {
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    font-weight: 700;
}
.workout-card.finisher {
    border-left: 4px solid #f59e0b;
    background: linear-gradient(145deg, rgba(251, 191, 36, 0.1) 0%, rgba(255,255,255,0.02) 100%);
}

/* Run Day Card */
.run-card {
    background: linear-gradient(145deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.run-card h3 {
    color: #60a5fa;
    margin: 0 0 0.5rem 0;
    font-size: 1.5rem;
}
.run-card p {
    color: rgba(255,255,255,0.7);
    margin: 0;
}

/* Stat Cards - Glassmorphism */
.stat-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.stat-card .value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-card .label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}

/* Goal Card */
.goal-card {
    background: linear-gradient(145deg, rgba(139, 92, 246, 0.15) 0%, rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.goal-card .goal-text {
    color: #c4b5fd;
    font-weight: 600;
    font-size: 0.95rem;
}
.goal-card .goal-target {
    color: rgba(255,255,255,0.5);
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

/* Log Section */
.log-section {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.02) 100%);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* AI Coach Section */
.coach-section {
    background: linear-gradient(145deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 20px;
    padding: 1.5rem;
}

/* Scrollable Chat Container */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding-right: 0.5rem;
}
.chat-container::-webkit-scrollbar {
    width: 6px;
}
.chat-container::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
}
.chat-container::-webkit-scrollbar-thumb {
    background: rgba(16, 185, 129, 0.4);
    border-radius: 3px;
}
.chat-container::-webkit-scrollbar-thumb:hover {
    background: rgba(16, 185, 129, 0.6);
}

/* Activity Item */
.activity-item {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid #10b981;
}
.activity-item.run {
    border-left-color: #3b82f6;
}

/* Section Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.section-header h3 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

header_class = "run-day" if IS_RUN_DAY else ""
day_type = "🏃 Run Day" if IS_RUN_DAY else "💪 Gym Day"
date_str = TODAY.strftime("%A, %d %B")

st.markdown(f"""
<div class="health-header {header_class}">
    <div class="day-badge">{day_type}</div>
    <h1>🏋️ Health & Fitness</h1>
    <p class="subtitle">{date_str} • Track, Train, Transform</p>
    <p class="quote">"{TODAYS_QUOTE}"</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================

if "health_chat" not in st.session_state:
    st.session_state.health_chat = []

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
# Helper: Get Today's Workout from Google Sheets
# ============================================================

def get_gym_workout(day_name: str = None):
    """Get workout for a specific day from the gym_workout sheet in Jarvis_Data."""
    if day_name is None:
        day_name = DAY_NAME
    
    try:
        # Use sheets_memory's connection to Jarvis_Data
        # Read directly from gym_workout tab
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
        
        # Use JARVIS_DATA_SHEET_URL (same as sheets_memory uses)
        jarvis_url = st.secrets.get("JARVIS_DATA_SHEET_URL") or os.getenv("JARVIS_DATA_SHEET_URL")
        if not jarvis_url:
            # Fallback
            jarvis_url = st.secrets.get("TODO_SHEET_URL") or os.getenv("TODO_SHEET_URL")
        
        if jarvis_url:
            sh = gc.open_by_url(jarvis_url)
            
            # Look for gym_workout sheet
            try:
                ws = sh.worksheet("gym_workout")
            except:
                return []
            
            rows = ws.get_all_values()
            if len(rows) > 1:
                headers = [h.lower().strip() for h in rows[0]]
                exercises = []
                for row in rows[1:]:
                    if row and len(row) > 0:
                        row_day = row[0].strip() if row[0] else ""
                        # Case-insensitive match
                        if row_day.lower() == day_name.lower():
                            ex = {}
                            for i, h in enumerate(headers):
                                ex[h] = row[i] if i < len(row) else ""
                            exercises.append(ex)
                return exercises
        return []
    except Exception as e:
        return []
    except Exception as e:
        return []

def get_last_weight_for_exercise(exercise_name: str) -> dict:
    """Get the last logged weight for a specific exercise."""
    try:
        # Get all workout logs and find the most recent one for this exercise
        workout_logs = sm.get_workout_logs(days=90)  # Look back 90 days
        
        # Filter for this exercise (case-insensitive partial match)
        exercise_lower = exercise_name.lower().strip()
        matching_logs = []
        
        for log in workout_logs:
            log_exercise = (log.get("exercise") or "").lower().strip()
            # Match if the exercise names are similar
            if exercise_lower in log_exercise or log_exercise in exercise_lower:
                weight = log.get("weight_kg")
                if weight and float(weight) > 0:
                    matching_logs.append({
                        "date": log.get("date", ""),
                        "weight": float(weight),
                        "sets": log.get("sets", ""),
                        "reps": log.get("reps", ""),
                    })
        
        # Sort by date (most recent first) and return the latest
        if matching_logs:
            matching_logs.sort(key=lambda x: x["date"], reverse=True)
            return matching_logs[0]
        
        return {}
    except Exception as e:
        return {}

def is_time_based(reps_str: str) -> bool:
    """Check if the reps value is time-based (seconds, steps, etc.)."""
    if not reps_str:
        return False
    reps_lower = str(reps_str).lower()
    return any(keyword in reps_lower for keyword in ["second", "sec", "step", "max", "minute", "min", "/side"])

# ============================================================
# AI Coach Function
# ============================================================

def chat_with_coach(question: str):
    """Chat with the AI fitness coach using GPT-5.1 Responses API."""
    try:
        client = get_openai_client()
        if not client:
            return "❌ OpenAI API key not configured."
        
        # Get current context
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
        
        # Build comprehensive context
        weight_data = summary.get("weight", {})
        nutrition_data = summary.get("nutrition", {})
        
        # Count workouts properly
        strength_exercises = [w for w in workout_logs if w.get('type') == 'strength']
        cardio_sessions = [w for w in workout_logs if w.get('type') in ('running', 'cardio')]
        total_distance = sum(float(w.get('distance_km', 0) or 0) for w in cardio_sessions)
        
        # Group strength exercises by date to count sessions
        workout_dates = set(w.get('date') for w in strength_exercises)
        
        context = f"""USER PROFILE:
{USER_PROFILE}

=== CURRENT STATUS ===
Today: {DAY_NAME} ({'Gym Day' if IS_GYM_DAY else 'Run Day'})
Current Weight: {weight_data.get('current', 'N/A')} stone
Starting Weight (30d ago): {weight_data.get('start', 'N/A')} stone
Weight Change: {weight_data.get('change', 'N/A')} stone

=== NUTRITION (Last 14 days) ===
Average Calories: {nutrition_data.get('avg_calories', 'N/A')} kcal/day
Average Protein: {nutrition_data.get('avg_protein', 'N/A')}g/day
Days Tracked: {nutrition_data.get('entries', 0)}

=== TRAINING (Last 14 days) ===
Gym Sessions: {len(workout_dates)} (total {len(strength_exercises)} exercises logged)
Running Sessions: {len(cardio_sessions)}
Total Distance Run: {total_distance:.1f} km

=== RECENT HEALTH LOGS ===
"""
        for log in health_logs[-7:]:
            context += f"- {log.get('date')}: {log.get('weight_stone', '')}st {log.get('weight_lbs', '')}lb, {log.get('calories', '')} cal, {log.get('protein_g', '')}g protein\n"
        
        context += "\n=== RECENT STRENGTH EXERCISES ===\n"
        for w in strength_exercises[-15:]:
            context += f"- {w.get('date')}: {w.get('exercise')} - {w.get('sets')}x{w.get('reps')} @ {w.get('weight_kg')}kg\n"
        
        context += "\n=== RECENT RUNS ===\n"
        for w in cardio_sessions[-5:]:
            context += f"- {w.get('date')}: {w.get('distance_km')}km in {w.get('duration_mins')} mins\n"
        
        context += "\n=== ACTIVE GOALS ===\n"
        for g in goals[:5]:
            context += f"- {g.get('goal')} (Target: {g.get('target_date', 'N/A')})\n"
        
        system_prompt = """You are an expert fitness coach analyzing a client's data. You have their:
- Weight logs (in stone/lbs - 14 lbs = 1 stone)
- Nutrition logs (calories, protein)
- Workout logs (exercises with sets, reps, weight)
- Running logs (distance, time)
- Goals with target dates

Your job is to:
1. Analyze their data with SPECIFIC numbers
2. Calculate if they're on track for their goals (e.g., losing X stone by date Y)
3. Recommend weight increases for exercises showing consistent performance
4. Advise on nutrition adjustments based on their goals
5. Be encouraging but data-driven and specific

IMPORTANT:
- Each workout log entry is a separate EXERCISE, not a separate session
- Multiple exercises on same date = one gym session
- Use their actual numbers, don't say data is missing if it's there
- For weight loss: ~0.5kg/week is healthy, calculate timeline
- For strength: suggest 2.5-5kg increases when ready"""
        
        user_text = f"{context}\n\n---\nQuestion: {question}"
        
        # GPT-5.1 uses the Responses API
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model="gpt-5.1",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_output_tokens=2000,
            )
            
            content = getattr(resp, "output_text", None)
            if content is None:
                content_parts = []
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", None) == "message":
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", None) in ("output_text", "text"):
                                content_parts.append(getattr(c, "text", "") or "")
                content = "".join(content_parts).strip()
            
            return content if content else "❌ Empty response from AI"
        else:
            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_completion_tokens=2000
            )
            return response.choices[0].message.content or "❌ Empty response"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================
# Main Layout: 3 Columns
# ============================================================

left_col, mid_col, right_col = st.columns([3, 4, 3], gap="large")

# ============================================================
# LEFT COLUMN: Today's Workout / Run + Quick Log
# ============================================================

with left_col:
    if IS_GYM_DAY:
        st.markdown(f"### 💪 Today's Workout")
        
        exercises = get_gym_workout(DAY_NAME)
        
        if exercises:
            for ex in exercises:
                name = ex.get("exercise", "Exercise")
                sets = ex.get("sets", "")
                reps = ex.get("reps", "")
                rpe = ex.get("rpe", "")
                
                is_finisher = "finisher" in name.lower()
                is_time = is_time_based(reps)
                card_class = "finisher" if is_finisher else ""
                
                # Get last weight for this exercise
                last_data = get_last_weight_for_exercise(name)
                last_weight = last_data.get("weight", 0) if last_data else 0
                last_date = last_data.get("date", "") if last_data else ""
                
                # Format the reps/time display
                if is_time:
                    reps_display = f'<span class="detail-chip time">{reps}</span>'
                else:
                    reps_display = f'<span class="detail-chip">{reps} reps</span>'
                
                # Last weight badge (only for non-time exercises)
                weight_badge = ""
                if last_weight > 0 and not is_time:
                    weight_badge = f'<span class="detail-chip weight">Last: {last_weight}kg</span>'
                
                st.markdown(f"""
                <div class="workout-card {card_class}">
                    <div class="exercise-name">{'🔥 ' if is_finisher else ''}{'⏱️ ' if is_time else ''}{name}</div>
                    <div class="exercise-details">
                        <span class="detail-chip">{sets} sets</span>
                        {reps_display}
                        {f'<span class="detail-chip rpe">RPE {rpe}</span>' if rpe else ''}
                        {weight_badge}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No workout found for {DAY_NAME}. Check 'gym_workout' sheet in Jarvis_Data.")
    
    else:  # Run Day
        st.markdown(f"### 🏃 Today's Run")
        st.markdown("""
        <div class="run-card">
            <h3>🏃‍♂️ Running Day</h3>
            <p>Time to hit the road! Log your run below.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tomorrow's workout preview (collapsed)
        if IS_TOMORROW_GYM:
            with st.expander(f"📅 Tomorrow's Workout ({TOMORROW_NAME})", expanded=False):
                tomorrow_exercises = get_gym_workout(TOMORROW_NAME)
                
                if tomorrow_exercises:
                    for ex in tomorrow_exercises:
                        name = ex.get("exercise", "Exercise")
                        sets = ex.get("sets", "")
                        reps = ex.get("reps", "")
                        is_finisher = "finisher" in name.lower()
                        is_time = is_time_based(reps)
                        
                        if is_time:
                            st.markdown(f"{'🔥' if is_finisher else '⏱️'} **{name}** - {sets} sets × {reps}")
                        else:
                            st.markdown(f"{'🔥' if is_finisher else '💪'} **{name}** - {sets} sets × {reps} reps")
                else:
                    st.caption(f"No workout found for {TOMORROW_NAME}")
    
    st.markdown("---")
    
    # ============================================================
    # Quick Log Section
    # ============================================================
    
    st.markdown("### ⚡ Quick Log")
    
    # Health Log (always available)
    with st.expander("📝 Log Health", expanded=False):
        with st.form("health_log_form"):
            c1, c2 = st.columns(2)
            with c1:
                weight_st = st.number_input("Stone", min_value=0, max_value=30, value=13, key="hl_st")
            with c2:
                weight_lb = st.number_input("lbs", min_value=0, max_value=13, value=0, key="hl_lb")
            
            c1, c2 = st.columns(2)
            with c1:
                calories = st.number_input("Calories", min_value=0, max_value=10000, value=2000, step=50, key="hl_cal")
            with c2:
                protein = st.number_input("Protein (g)", min_value=0, max_value=500, value=180, step=5, key="hl_prot")
            
            notes = st.text_input("Notes", placeholder="How do you feel?", key="hl_notes")
            
            if st.form_submit_button("💾 Save", use_container_width=True, type="primary"):
                try:
                    sm.log_health(
                        date=TODAY.date().isoformat(),
                        weight_stone=weight_st if weight_st > 0 else None,
                        weight_lbs=weight_lb,
                        calories=calories if calories > 0 else None,
                        protein_g=protein if protein > 0 else None,
                        notes=notes
                    )
                    st.success("✅ Health logged!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Workout/Run Log based on day
    if IS_GYM_DAY:
        with st.expander("🏋️ Log Workout", expanded=True):
            exercises = get_gym_workout(DAY_NAME)
            
            # Pre-populate with today's exercises
            if exercises:
                st.caption("Select exercise to log:")
                exercise_names = [ex.get("exercise", "") for ex in exercises]
                exercise_names.append("➕ Custom Exercise")
                
                selected_ex = st.selectbox("Exercise", exercise_names, key="wl_select", label_visibility="collapsed")
                
                if selected_ex == "➕ Custom Exercise":
                    exercise_name = st.text_input("Exercise Name", key="wl_custom")
                    default_sets = 3
                    default_reps = "10"
                    default_weight = 0.0
                    is_time_exercise = False
                else:
                    exercise_name = selected_ex
                    # Get defaults from workout plan
                    ex_data = next((e for e in exercises if e.get("exercise") == selected_ex), {})
                    default_sets = int(ex_data.get("sets", "3").split("-")[0]) if ex_data.get("sets") else 3
                    default_reps = ex_data.get("reps", "10")
                    is_time_exercise = is_time_based(default_reps)
                    
                    # Get last weight for this exercise
                    last_data = get_last_weight_for_exercise(selected_ex)
                    default_weight = last_data.get("weight", 0.0) if last_data else 0.0
                    
                    # Show last weight info
                    if default_weight > 0:
                        last_date = last_data.get("date", "")
                        st.info(f"💡 Last time: **{default_weight}kg** (on {last_date})")
                
                with st.form("workout_log_form"):
                    if is_time_exercise:
                        # Time-based exercise - different UI
                        c1, c2 = st.columns(2)
                        with c1:
                            sets = st.number_input("Sets", min_value=1, max_value=10, value=default_sets, key="wl_sets")
                        with c2:
                            time_input = st.text_input("Duration/Reps", value=str(default_reps), key="wl_time", help="e.g., 60 seconds, 20 steps")
                        
                        # Some time exercises still have weight (like weighted planks)
                        weight = st.number_input("Weight (kg, if any)", min_value=0.0, max_value=500.0, value=default_weight, step=2.5, key="wl_weight")
                        reps = 0  # Will store time_input in notes
                    else:
                        # Rep-based exercise
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            sets = st.number_input("Sets", min_value=1, max_value=10, value=default_sets, key="wl_sets")
                        with c2:
                            # Parse default reps from string like "8-10"
                            try:
                                if "-" in str(default_reps):
                                    reps_val = int(default_reps.split("-")[0])
                                elif str(default_reps).isdigit():
                                    reps_val = int(default_reps)
                                else:
                                    reps_val = 10
                            except:
                                reps_val = 10
                            reps = st.number_input("Reps", min_value=1, max_value=100, value=reps_val, key="wl_reps")
                        with c3:
                            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=default_weight, step=2.5, key="wl_weight")
                        time_input = None
                    
                    if st.form_submit_button("💾 Log Exercise", use_container_width=True, type="primary"):
                        ex_to_log = exercise_name if selected_ex != "➕ Custom Exercise" else exercise_name
                        if ex_to_log:
                            try:
                                if is_time_exercise:
                                    # Log time-based exercise with duration in notes
                                    sm.log_workout(
                                        date=TODAY.date().isoformat(),
                                        exercise=ex_to_log,
                                        sets=sets,
                                        reps=0,
                                        weight_kg=weight if weight > 0 else None,
                                        workout_type="strength",
                                        notes=f"Duration: {time_input}"
                                    )
                                else:
                                    sm.log_workout(
                                        date=TODAY.date().isoformat(),
                                        exercise=ex_to_log,
                                        sets=sets,
                                        reps=reps,
                                        weight_kg=weight,
                                        workout_type="strength"
                                    )
                                st.success(f"✅ {ex_to_log} logged!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.warning("Enter exercise name")
            else:
                st.info("No workout plan found. Add custom exercise:")
                with st.form("custom_workout_form"):
                    exercise_name = st.text_input("Exercise", key="cwl_ex")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        sets = st.number_input("Sets", min_value=1, max_value=10, value=3, key="cwl_sets")
                    with c2:
                        reps = st.number_input("Reps", min_value=1, max_value=100, value=10, key="cwl_reps")
                    with c3:
                        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=0.0, step=2.5, key="cwl_weight")
                    
                    if st.form_submit_button("💾 Log", use_container_width=True, type="primary"):
                        if exercise_name:
                            sm.log_workout(
                                date=TODAY.date().isoformat(),
                                exercise=exercise_name,
                                sets=sets,
                                reps=reps,
                                weight_kg=weight,
                                workout_type="strength"
                            )
                            st.success("✅ Logged!")
                            st.rerun()
    
    else:  # Run Day
        with st.expander("🏃 Log Run", expanded=True):
            with st.form("run_log_form"):
                c1, c2 = st.columns(2)
                with c1:
                    distance = st.number_input("Distance (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="rl_dist")
                with c2:
                    duration = st.number_input("Time (mins)", min_value=0, max_value=300, value=30, key="rl_dur")
                
                pace = st.text_input("Pace (min/km)", placeholder="e.g., 6:00", key="rl_pace")
                notes = st.text_input("Notes", placeholder="How did it feel?", key="rl_notes")
                
                if st.form_submit_button("💾 Log Run", use_container_width=True, type="primary"):
                    try:
                        sm.log_workout(
                            date=TODAY.date().isoformat(),
                            exercise="Running",
                            distance_km=distance,
                            duration_mins=duration,
                            pace_per_km=pace,
                            workout_type="running",
                            notes=notes
                        )
                        st.success("✅ Run logged!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

# ============================================================
# MIDDLE COLUMN: AI Coach
# ============================================================

with mid_col:
    st.markdown("### 🤖 AI Fitness Coach")
    
    # Action buttons at TOP
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 Full Analysis", use_container_width=True, type="primary"):
            prompt = """Give me a comprehensive analysis:

1. WEIGHT PROGRESS: Calculate exactly - am I on track for 12 stone by March 23rd? Show the math (current weight, weeks remaining, required loss rate).

2. STRENGTH PROGRESS: Review each of my recent exercises. For each one, should I increase weight next session? By how much?

3. NUTRITION: Based on my logged calories and protein, are they right for my weight loss goal? Specific recommendations.

4. THIS WEEK: What should I focus on?

Be specific with my actual numbers!"""
            
            st.session_state.health_chat.append({"role": "user", "content": "Full analysis"})
            with st.spinner("Analyzing your data..."):
                response = chat_with_coach(prompt)
            st.session_state.health_chat.append({"role": "assistant", "content": response})
            st.rerun()
    
    with c2:
        if st.button("💪 Weight Check", use_container_width=True):
            prompt = "Review each of my strength exercises from the last 2 weeks. For EACH exercise, tell me: current weight I'm using, whether I should increase it, and by how much (2.5kg or 5kg). Be specific!"
            st.session_state.health_chat.append({"role": "user", "content": "Should I increase weights?"})
            with st.spinner("Checking..."):
                response = chat_with_coach(prompt)
            st.session_state.health_chat.append({"role": "assistant", "content": response})
            st.rerun()
    
    with c3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.health_chat = []
            st.rerun()
    
    # Text input at TOP (below buttons)
    user_q = st.text_input("Ask a question:", placeholder="e.g., Am I eating enough protein?", key="coach_q", label_visibility="collapsed")
    if st.button("Send", key="send_q", use_container_width=True) and user_q:
        st.session_state.health_chat.append({"role": "user", "content": user_q})
        with st.spinner("Thinking..."):
            response = chat_with_coach(user_q)
        st.session_state.health_chat.append({"role": "assistant", "content": response})
        st.rerun()
    
    st.markdown("---")
    
    # Chat history in scrollable container
    if st.session_state.health_chat:
        # Create scrollable container
        chat_container = st.container(height=500)
        with chat_container:
            # Show newest first
            for msg in reversed(st.session_state.health_chat):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
    else:
        st.info("👋 Ask me anything! I'll analyze your data and give specific advice.")

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
        
        # Row 1: Weight
        c1, c2 = st.columns(2)
        with c1:
            current = weight_data.get("current")
            st.markdown(f"""
            <div class="stat-card">
                <div class="value">{f'{current:.1f}' if current else '--'}</div>
                <div class="label">Current (st)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            change = weight_data.get("change")
            if change is not None:
                sign = "+" if change > 0 else ""
                st.markdown(f"""
                <div class="stat-card">
                    <div class="value" style="color: {'#10b981' if change <= 0 else '#f59e0b'};">{sign}{change:.1f}</div>
                    <div class="label">30d Change</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-card">
                    <div class="value">--</div>
                    <div class="label">30d Change</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Row 2: Nutrition
        c1, c2 = st.columns(2)
        with c1:
            avg_cal = nutrition_data.get("avg_calories")
            st.markdown(f"""
            <div class="stat-card">
                <div class="value">{avg_cal if avg_cal else '--'}</div>
                <div class="label">Avg Cal/day</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            avg_prot = nutrition_data.get("avg_protein")
            st.markdown(f"""
            <div class="stat-card">
                <div class="value">{avg_prot if avg_prot else '--'}g</div>
                <div class="label">Avg Protein</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 3: Workouts
        c1, c2 = st.columns(2)
        with c1:
            strength = workout_data.get("strength_sessions", 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="value">{strength}</div>
                <div class="label">Gym Sessions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            distance = workout_data.get("total_distance_km", 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="value">{distance:.1f}</div>
                <div class="label">km Run</div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.info("Start logging to see your stats!")
    
    st.markdown("---")
    
    # Goals (no progress bars)
    st.markdown("### 🎯 Goals")
    
    try:
        goals = sm.get_fitness_goals(status="active")
        if goals:
            for g in goals[:5]:
                target = g.get("target_date", "")
                st.markdown(f"""
                <div class="goal-card">
                    <div class="goal-text">🎯 {g.get('goal', 'Unknown')}</div>
                    <div class="goal-target">Target: {target if target else 'No date set'}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active goals")
    except:
        st.info("Add some goals!")
    
    # Add goal button
    with st.expander("➕ Add Goal"):
        with st.form("add_goal_form"):
            goal_text = st.text_input("Goal", placeholder="e.g., Bench 100kg")
            target = st.date_input("Target Date", value=TODAY.date() + timedelta(days=90))
            if st.form_submit_button("Add Goal", use_container_width=True):
                if goal_text:
                    sm.add_fitness_goal(goal=goal_text, target_date=str(target))
                    st.success("✅ Goal added!")
                    st.rerun()
    
    st.markdown("---")
    
    # Recent Activity
    st.markdown("### 📅 Recent Activity")
    
    try:
        workouts = sm.get_workout_logs(days=7)
        if workouts:
            for w in workouts[-6:]:
                wtype = w.get("type", "strength")
                date = w.get("date", "")
                
                if wtype in ("running", "cardio"):
                    st.markdown(f"""
                    <div class="activity-item run">
                        <strong>🏃 {date}</strong>: {w.get('distance_km', 0)}km in {w.get('duration_mins', 0)} mins
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="activity-item">
                        <strong>💪 {date}</strong>: {w.get('exercise', '')} - {w.get('sets')}×{w.get('reps')} @ {w.get('weight_kg', 0)}kg
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("No recent activity")
    except:
        st.caption("Log workouts to see history")

# ============================================================
# Footer
# ============================================================

st.markdown("---")
c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    st.page_link("app.py", label="Back to Home", icon="🏠")
