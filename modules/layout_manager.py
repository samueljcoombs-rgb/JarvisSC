# modules/layout_manager.py
# Streamlit layout manager for Jarvis
# - Preserves the classic layout: Weather + Spotify + Chat
# - Adds Athletic Feed (code lives in modules/athletic_feed.py)
# - Safe, lazy imports so missing modules won't crash the app
# - Sidebar toggles are persisted in st.session_state
# - Each panel renders inside cards with consistent spacing

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import streamlit as st


@dataclass
class ModuleSpec:
    key: str
    title: str
    icon: str
    module_name: str
    # Preferred callable names in the module (checked in order)
    callables: tuple[str, ...] = ("render",)
    # Optional help text for tooltips
    help: Optional[str] = None


# ---- Registry: keep the “classic” panels and add Athletic Feed ----
# If a module is missing, we skip it gracefully.
MODULES: Dict[str, ModuleSpec] = {
    "weather": ModuleSpec(
        key="weather",
        title="Weather",
        icon=":sun_small_cloud:",
        module_name="modules.weather_panel",
        callables=("render_weather_panel", "render"),
        help="Local forecast and conditions",
    ),
    "spotify": ModuleSpec(
        key="spotify",
        title="Spotify",
        icon=":notes:",
        module_name="modules.spotify_panel",
        callables=("render_spotify_panel", "render_spotify_widget", "render"),
        help="Now playing & quick controls",
    ),
    "athletic": ModuleSpec(
        key="athletic",
        title="Athletic Feed",
        icon=":trophy:",
        module_name="modules.athletic_feed",
        callables=("render_athletic_feed", "render"),
        help="Live sports headlines & fixtures",
    ),
    "chat": ModuleSpec(
        key="chat",
        title="Chat",
        icon=":robot_face:",
        module_name="modules.chat_ui",
        callables=("render_chat_ui", "render"),
        help="Your Jarvis conversation window",
    ),
}


# ---- Utilities ----
def _lazy_load_callable(module_name: str, candidates: tuple[str, ...]) -> Optional[Callable]:
    """
    Try to import a module and return the first available callable from candidates.
    Returns None if neither the module nor any candidate callable exist.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        st.debug(f"[layout] Skipping '{module_name}': import failed: {e}")
        return None

    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn

    st.debug(f"[layout] Module '{module_name}' has no callable in {candidates}")
    return None


def _card(title: str, icon: str, body_fn: Callable[[], None]):
    """
    Consistent card styling wrapper for panels.
    """
    with st.container(border=True):
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.5rem'>"
            f"<span style='font-size:1.1rem'>{icon}</span>"
            f"<h3 style='margin:0'>{title}</h3>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        body_fn()


def _init_state():
    """
    Initialize persistent toggle state for module visibility & layout.
    """
    if "lm_initialized" in st.session_state:
        return

    # Default visibility: keep the classic feel on first run
    defaults = {
        "weather": True,
        "spotify": True,
        "athletic": True,   # new panel on by default
        "chat": True,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(f"show_{k}", v)

    # Default layout mode
    st.session_state.setdefault("layout_mode", "Dashboard")  # "Dashboard" | "Focus"

    st.session_state["lm_initialized"] = True


# ---- Public API ----
def render_header(app_title: str = "Jarvis"):
    """
    Renders the top header with a compact identity bar.
    Call this once near the top of app.py.
    """
    _init_state()
    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            justify-content:space-between;
            padding:.6rem 0 .2rem 0;">
            <div style="display:flex;align-items:center;gap:.6rem">
                <span style="font-size:1.6rem">🤖</span>
                <h1 style="margin:0;font-size:1.6rem">{app_title}</h1>
            </div>
            <div style="opacity:.7;font-size:.9rem">
                Modular AI dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_sidebar():
    """
    Renders the global controls on the sidebar:
    - Layout mode
    - Module visibility toggles
    - Helpful links/notes
    """
    _init_state()
    with st.sidebar:
        st.subheader("Layout")
        st.radio(
            "Mode",
            options=["Dashboard", "Focus"],
            index=0 if st.session_state.get("layout_mode") == "Dashboard" else 1,
            key="layout_mode",
            help="Dashboard shows multiple panels. Focus shows a single panel of your choice.",
        )

        st.subheader("Panels")
        for spec in MODULES.values():
            st.checkbox(
                f"{spec.title} {spec.icon}",
                value=st.session_state.get(f"show_{spec.key}", True),
                key=f"show_{spec.key}",
                help=spec.help,
            )

        if st.session_state.get("layout_mode") == "Focus":
            visible_keys = [k for k, s in MODULES.items() if st.session_state.get(f"show_{k}")]
            if not visible_keys:
                visible_keys = ["chat"]  # fallback
            default_focus = visible_keys[0]
            st.selectbox(
                "Focus panel",
                options=visible_keys,
                index=visible_keys.index(st.session_state.get("focus_key", default_focus))
                if st.session_state.get("focus_key") in visible_keys
                else 0,
                key="focus_key",
                format_func=lambda k: MODULES[k].title,
                help="Choose which single panel to show in Focus mode.",
            )

        st.markdown("---")
        st.caption(
            "Tip: Toggle panels on/off here. Missing a module? "
            "Make sure its file exists under `modules/`."
        )


def render_main():
    """
    Renders the main page according to the selected layout mode and toggles.
    - Dashboard: Weather + Athletic across top, Spotify + Chat below
    - Focus: Only the selected panel fills the page
    """
    _init_state()

    # Resolve available renderers (soft import)
    renderers: Dict[str, Callable] = {}
    for key, spec in MODULES.items():
        if st.session_state.get(f"show_{key}", True):
            fn = _lazy_load_callable(spec.module_name, spec.callables)
            if fn is not None:
                renderers[key] = fn

    # Nothing to show?
    if not renderers:
        st.info("No panels are enabled or available. Turn some on in the sidebar.")
        return

    # Focus mode: render only one chosen panel
    if st.session_state.get("layout_mode") == "Focus":
        focus_key = st.session_state.get("focus_key")
        if focus_key not in renderers:
            # Pick the first available as fallback
            focus_key = next(iter(renderers.keys()))
            st.session_state["focus_key"] = focus_key
        spec = MODULES[focus_key]
        fn = renderers[focus_key]
        _card(spec.title, spec.icon, lambda: _safe_call(fn, spec))
        return

    # Dashboard mode:
    # Row 1: Weather | Athletic
    # Row 2: Spotify | Chat
    # (Panels render only if enabled & available. Empty columns compact gracefully.)
    row1_keys = [k for k in ("weather", "athletic") if k in renderers]
    row2_keys = [k for k in ("spotify", "chat") if k in renderers]

    # If some expected panels are missing, still try to display what we have
    rows = [row1_keys, row2_keys]
    for row_keys in rows:
        if not row_keys:
            continue
        cols = st.columns(len(row_keys))
        for i, k in enumerate(row_keys):
            spec = MODULES[k]
            fn = renderers[k]
            with cols[i]:
                _card(spec.title, spec.icon, lambda fn=fn, spec=spec: _safe_call(fn, spec))


def _safe_call(fn: Callable, spec: ModuleSpec):
    """
    Execute a panel's render function safely so one panel cannot crash the page.
    """
    try:
        fn()
    except Exception as e:
        with st.expander(f"⚠️ {spec.title} failed to render (click for details)", expanded=False):
            st.exception(e)


# ---- Convenience: single entry to render full page ----
def render(app_title: str = "Jarvis"):
    """
    Call this from app.py:
        from modules import layout_manager as lm
        lm.render("Jarvis")
    """
    render_header(app_title)
    render_sidebar()
    render_main()
