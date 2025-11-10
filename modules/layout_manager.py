# modules/layout_manager.py
# Jarvis Layout Manager — streamlit sidebar router for modular panels
from __future__ import annotations
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any, Tuple

import streamlit as st

# ---- Config -----------------------------------------------------------------

APP_TITLE = "Jarvis — AI Dashboard"
SIDEBAR_TITLE = "Modules"
SESSION_SELECTED_KEY = "layout_selected_key"
SESSION_QUERY_PARAMS_LOCK = "layout_query_lock"

# Known-first modules (loaded first if present; order matters)
PREFERRED_MODULE_ORDER = [
    "chat_ui",
    "athletic_feed",
    "weather_panel",
]

# Module package to scan
MODULES_PACKAGE = "modules"

# ---- Data model --------------------------------------------------------------

@dataclass
class Panel:
    key: str                 # stable internal key (module name)
    title: str               # display title in nav
    render: Callable[[], None]  # function to render the panel
    icon: Optional[str] = None  # optional emoji/icon prefix

# ---- Registry ----------------------------------------------------------------

def _try_import_module(mod_name: str):
    try:
        return importlib.import_module(f"{MODULES_PACKAGE}.{mod_name}")
    except Exception:
        return None

def _extract_panel(module, fallback_key: str) -> Optional[Panel]:
    """Derive a Panel from a module if it exposes a render() function."""
    if module is None:
        return None
    render_fn = getattr(module, "render", None)
    if not callable(render_fn):
        return None

    # Prefer module.TITLE, else Title Case from key
    raw_title = getattr(module, "TITLE", None) or fallback_key.replace("_", " ").title()
    icon = getattr(module, "ICON", None)  # optional; modules can set ICON = "🧠"
    title = f"{icon} {raw_title}" if icon else raw_title

    # Normalize key
    key = getattr(module, "KEY", None) or fallback_key
    return Panel(key=key, title=title, render=render_fn, icon=icon)

def _discover_modules() -> List[Panel]:
    """
    Discover panels by importing Python modules under the `modules` package
    that expose a callable `render()` function. The order is:
    1) PREFERRED_MODULE_ORDER (if found)
    2) All other discovered modules A–Z
    """
    discovered: Dict[str, Panel] = {}

    # 1) Preferred order
    for name in PREFERRED_MODULE_ORDER:
        mod = _try_import_module(name)
        p = _extract_panel(mod, name)
        if p:
            discovered[p.key] = p

    # 2) Scan all submodules in /modules (lightweight)
    package = importlib.import_module(MODULES_PACKAGE)
    for finder, name, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg:
            continue
        if name in discovered:
            continue
        mod = _try_import_module(name)
        p = _extract_panel(mod, name)
        if p:
            discovered[p.key] = p

    # Sort: keep preferred order first, then alphabetical by title
    preferred_keys = [k for k in PREFERRED_MODULE_ORDER if k in discovered]
    rest = sorted([k for k in discovered.keys() if k not in preferred_keys],
                  key=lambda k: discovered[k].title.lower())
    ordered_keys = preferred_keys + rest
    return [discovered[k] for k in ordered_keys]

def _sync_query_params(selected_key: str):
    """
    Keep ?panel=<key> in the URL so deep links work. We gate updates to avoid
    Streamlit re-run storms when both sidebar and URL change at once.
    """
    qp = st.query_params
    current = qp.get("panel", [None])
    current = current[0] if isinstance(current, list) else current
    lock = st.session_state.get(SESSION_QUERY_PARAMS_LOCK, False)

    if not lock and current != selected_key:
        st.session_state[SESSION_QUERY_PARAMS_LOCK] = True
        st.query_params["panel"] = selected_key
        # release lock next run
    elif lock:
        # One pass after we set it, unlock
        st.session_state[SESSION_QUERY_PARAMS_LOCK] = False

def _initial_selection(panels: List[Panel]) -> str:
    """Pick initial selected panel using URL ?panel=<key> if present."""
    if not panels:
        return ""
    qp = st.query_params
    url_key = qp.get("panel", [None])
    url_key = url_key[0] if isinstance(url_key, list) else url_key
    keys = [p.key for p in panels]
    if url_key in keys:
        return url_key
    # Fallback to first panel
    return keys[0]

# ---- Public API --------------------------------------------------------------

def render_layout():
    """
    Render the full Jarvis layout: top header, sidebar navigation,
    and the selected panel content area.
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Discover panels
    panels = _discover_modules()

    # App Header
    st.title(APP_TITLE)
    st.caption("Modular, resilient, and memory-aware. ⚙️")

    # Early exit if nothing to show
    if not panels:
        st.info("No modules with a `render()` function were found under `/modules`.")
        return

    # ---- Sidebar Nav ----
    with st.sidebar:
        st.header(SIDEBAR_TITLE)

        # Build options for radio
        titles_by_key = {p.key: p.title for p in panels}
        keys = [p.key for p in panels]
        titles = [titles_by_key[k] for k in keys]

        # Seed selection from state or URL
        if SESSION_SELECTED_KEY not in st.session_state:
            st.session_state[SESSION_SELECTED_KEY] = _initial_selection(panels)

        # Optional quick filter
        filter_text = st.text_input("Filter", placeholder="Type to filter modules…").strip().lower()
        if filter_text:
            filtered_pairs = [(k, t) for k, t in zip(keys, titles) if filter_text in t.lower()]
            if filtered_pairs:
                keys, titles = zip(*filtered_pairs)  # type: ignore[assignment]
                keys, titles = list(keys), list(titles)
            else:
                st.warning("No matches.")
                keys, titles = [], []

        # Radio for panel selection
        if keys:
            default_index = 0
            if st.session_state[SESSION_SELECTED_KEY] in keys:
                default_index = keys.index(st.session_state[SESSION_SELECTED_KEY])

            choice_title = st.radio(
                "Choose a panel",
                titles,
                index=default_index,
                label_visibility="collapsed",
            )
            # Map back to key
            selected_key = keys[titles.index(choice_title)]
        else:
            selected_key = ""

        # Persist selection
        st.session_state[SESSION_SELECTED_KEY] = selected_key

        # Keep URL param in sync
        if selected_key:
            _sync_query_params(selected_key)

        st.divider()
        st.caption("Tip: Use the filter to quickly jump between modules.")

    # ---- Main Content ----
    selected_panel: Optional[Panel] = next((p for p in panels if p.key == st.session_state[SESSION_SELECTED_KEY]), None)

    if not selected_panel:
        st.info("Select a module from the sidebar to get started.")
        return

    # Render the chosen panel inside an error boundary
    with st.container():
        st.subheader(selected_panel.title)
        try:
            selected_panel.render()
        except Exception as e:
            st.error(f"Module `{selected_panel.key}` failed to render:\n\n{e}")
            with st.expander("Traceback / Debug"):
                st.exception(e)

    # Footer
    st.divider()
    st.caption("Jarvis layout manager • modules are auto-discovered from `/modules` that expose `render()`.")

# For convenience if someone imports as a module and calls main()
def main():
    render_layout()

# If run directly (optional), render layout.
if __name__ == "__main__":
    main()
