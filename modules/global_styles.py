# modules/global_styles.py
"""
Global Styling for Jarvis Dashboard
Beautiful, consistent Apple-inspired dark theme with glassmorphism effects.
"""
import streamlit as st

# ============================================================
# Color Palette
# ============================================================

COLORS = {
    "bg_dark": "#0f0f1a",
    "bg_card": "rgba(30, 30, 46, 0.8)",
    "bg_glass": "rgba(255, 255, 255, 0.05)",
    "accent_blue": "#3b82f6",
    "accent_purple": "#8b5cf6",
    "accent_green": "#10b981",
    "accent_orange": "#f97316",
    "accent_red": "#ef4444",
    "accent_pink": "#ec4899",
    "text_primary": "rgba(255, 255, 255, 0.95)",
    "text_secondary": "rgba(255, 255, 255, 0.7)",
    "text_muted": "rgba(255, 255, 255, 0.5)",
    "border": "rgba(255, 255, 255, 0.1)",
    "border_hover": "rgba(255, 255, 255, 0.2)",
    "shadow": "rgba(0, 0, 0, 0.3)",
}

# ============================================================
# CSS Injection
# ============================================================

def inject_global_styles():
    """Inject global CSS styles for the entire app."""
    if st.session_state.get("_global_styles_injected"):
        return
    st.session_state["_global_styles_injected"] = True
    
    st.markdown("""
<style>
/* ============================================
   ROOT & VARIABLES
   ============================================ */
:root {
    --bg-dark: #0f0f1a;
    --bg-card: rgba(30, 30, 46, 0.8);
    --bg-glass: rgba(255, 255, 255, 0.05);
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-green: #10b981;
    --accent-orange: #f97316;
    --accent-red: #ef4444;
    --accent-pink: #ec4899;
    --text-primary: rgba(255, 255, 255, 0.95);
    --text-secondary: rgba(255, 255, 255, 0.7);
    --text-muted: rgba(255, 255, 255, 0.5);
    --border: rgba(255, 255, 255, 0.1);
    --border-hover: rgba(255, 255, 255, 0.2);
    --shadow: rgba(0, 0, 0, 0.3);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
}

/* ============================================
   MAIN CONTAINER
   ============================================ */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* ============================================
   GLASS CARD COMPONENT
   ============================================ */
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-lg);
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(255, 255, 255, 0.2);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    transform: translateY(-2px);
}

/* ============================================
   METRIC CARDS
   ============================================ */
.metric-card {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
    border-radius: var(--radius-lg);
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
}

.metric-card.green {
    background: linear-gradient(135deg, var(--accent-green) 0%, #059669 100%);
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
}

.metric-card.orange {
    background: linear-gradient(135deg, var(--accent-orange) 0%, #ea580c 100%);
    box-shadow: 0 8px 24px rgba(249, 115, 22, 0.3);
}

.metric-card.purple {
    background: linear-gradient(135deg, var(--accent-purple) 0%, #7c3aed 100%);
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.3);
}

.metric-card.pink {
    background: linear-gradient(135deg, var(--accent-pink) 0%, #db2777 100%);
    box-shadow: 0 8px 24px rgba(236, 72, 153, 0.3);
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    margin: 0;
}

.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.85);
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ============================================
   SECTION HEADERS
   ============================================ */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.section-header .emoji {
    font-size: 1.2rem;
}

/* ============================================
   LIST ITEMS
   ============================================ */
.list-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
}

.list-item:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: var(--border-hover);
}

.list-item .icon {
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.list-item .content {
    flex: 1;
    min-width: 0;
}

.list-item .title {
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.list-item .subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* ============================================
   STATUS BADGES
   ============================================ */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.badge.success {
    background: rgba(16, 185, 129, 0.2);
    color: var(--accent-green);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge.warning {
    background: rgba(249, 115, 22, 0.2);
    color: var(--accent-orange);
    border: 1px solid rgba(249, 115, 22, 0.3);
}

.badge.danger {
    background: rgba(239, 68, 68, 0.2);
    color: var(--accent-red);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge.info {
    background: rgba(59, 130, 246, 0.2);
    color: var(--accent-blue);
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.badge.purple {
    background: rgba(139, 92, 246, 0.2);
    color: var(--accent-purple);
    border: 1px solid rgba(139, 92, 246, 0.3);
}

/* ============================================
   PROGRESS BARS
   ============================================ */
.progress-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.progress-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    transition: width 0.5s ease;
}

.progress-bar.green {
    background: linear-gradient(90deg, var(--accent-green), #34d399);
}

.progress-bar.orange {
    background: linear-gradient(90deg, var(--accent-orange), #fb923c);
}

/* ============================================
   BUTTONS
   ============================================ */
.stButton > button {
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

/* Primary action button */
.action-btn {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius-md) !important;
    cursor: pointer;
    transition: all 0.2s ease;
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
}

/* ============================================
   INPUT FIELDS
   ============================================ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-glass) !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}

/* ============================================
   CHAT MESSAGES
   ============================================ */
.chat-message {
    padding: 1rem;
    border-radius: var(--radius-lg);
    margin-bottom: 0.75rem;
    max-width: 85%;
}

.chat-message.user {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}

.chat-message.assistant {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
}

/* ============================================
   TABS
   ============================================ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-md);
    padding: 0.5rem 1rem;
    background: var(--bg-glass);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%) !important;
    color: white !important;
    border-color: transparent !important;
}

/* ============================================
   EXPANDERS
   ============================================ */
.streamlit-expanderHeader {
    background: var(--bg-glass) !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
}

/* ============================================
   SIDEBAR
   ============================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ============================================
   CUSTOM SCROLLBAR
   ============================================ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* ============================================
   ANIMATIONS
   ============================================ */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.animate-fade-in {
    animation: fadeIn 0.3s ease-out;
}

.animate-pulse {
    animation: pulse 2s infinite;
}

/* ============================================
   LOADING SKELETON
   ============================================ */
.skeleton {
    background: linear-gradient(90deg, 
        rgba(255,255,255,0.05) 25%, 
        rgba(255,255,255,0.1) 50%, 
        rgba(255,255,255,0.05) 75%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: var(--radius-md);
}

/* ============================================
   RESPONSIVE ADJUSTMENTS
   ============================================ */
@media (max-width: 768px) {
    .metric-value {
        font-size: 1.5rem;
    }
    
    .glass-card {
        padding: 1rem;
    }
    
    .chat-message {
        max-width: 95%;
    }
}

/* ============================================
   DARK MODE FIXES
   ============================================ */
.stMarkdown, .stText {
    color: var(--text-primary);
}

hr {
    border-color: var(--border) !important;
    opacity: 0.5;
}
</style>
    """, unsafe_allow_html=True)


# ============================================================
# Helper Functions for Styled Components
# ============================================================

def glass_card(content: str, extra_class: str = "") -> str:
    """Wrap content in a glass card."""
    return f'<div class="glass-card {extra_class}">{content}</div>'

def metric_card(value: str, label: str, color: str = "blue") -> str:
    """Create a metric card."""
    return f'''
    <div class="metric-card {color}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    '''

def section_header(emoji: str, title: str) -> str:
    """Create a section header."""
    return f'<div class="section-header"><span class="emoji">{emoji}</span>{title}</div>'

def badge(text: str, style: str = "info") -> str:
    """Create a status badge."""
    return f'<span class="badge {style}">{text}</span>'

def progress_bar(percent: float, color: str = "blue") -> str:
    """Create a progress bar."""
    return f'''
    <div class="progress-container">
        <div class="progress-bar {color}" style="width: {min(100, max(0, percent))}%;"></div>
    </div>
    '''

def list_item(icon: str, title: str, subtitle: str = "", icon_bg: str = "var(--accent-blue)") -> str:
    """Create a styled list item."""
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f'''
    <div class="list-item">
        <div class="icon" style="background: {icon_bg};">{icon}</div>
        <div class="content">
            <div class="title">{title}</div>
            {subtitle_html}
        </div>
    </div>
    '''


# ============================================================
# Page Configuration
# ============================================================

def setup_page(title: str = "Jarvis", icon: str = "🤖", layout: str = "wide"):
    """Set up page configuration with consistent styling."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="collapsed",
    )
    inject_global_styles()


def page_header(title: str, subtitle: str = "", emoji: str = ""):
    """Render a consistent page header."""
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin: 0; font-size: 2rem; font-weight: 800; 
                   background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   background-clip: text;">
            {emoji} {title}
        </h1>
        {f'<p style="margin: 0.25rem 0 0 0; color: rgba(255,255,255,0.6); font-size: 1rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_top_bar():
    """Render a slim top navigation bar."""
    from datetime import datetime
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;
                padding: 0.5rem 0; margin-bottom: 0.5rem;
                border-bottom: 1px solid rgba(255,255,255,0.1);">
        <div style="font-weight: 700; font-size: 1.1rem; 
                    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🤖 JARVIS
        </div>
        <div style="color: rgba(255,255,255,0.5); font-size: 0.9rem;">
            {today_str}
        </div>
    </div>
    """, unsafe_allow_html=True)
