"""
Football Research Agent v6 - COMPREHENSIVE EDITION

Includes ALL v5 features:
1. BATCH TESTING: Test 5-10 variations per avenue, not just 1
2. DEEP ANALYSIS: LLM analyzes results after EVERY phase
3. NO PREMATURE CONCLUSIONS: Must explore ALL avenues before giving up
4. COMBINATION SCANNING: When base filter shows promise, auto-scan for additive filters
5. LEARNING ACCUMULATION: Track insights across iterations
6. AVENUE TRACKING: Explicitly track which avenues explored vs remaining

PLUS v6 Enhancements:
7. LOCAL COMPUTE: FastAPI server instead of Modal
8. 38 TOOLS: Expanded tool arsenal including ML
9. SUPABASE PERSISTENCE: Strategies, learnings, checkpoints
10. AUTO-PROMOTION: draft → candidate → promising → validated lifecycle
11. PAUSE/RESUME: Control research flow
12. ASK AGENT: Query when paused
13. DEEP THINKING SYSTEM: OBSERVE → CONNECT → HYPOTHESIZE → PLAN → DECIDE → RECORD
14. CRASH RECOVERY: Checkpoint-based state restore

Usage: streamlit run football_researcher_v6.py
Requires: python3 local_compute.py running on port 8000
"""

from __future__ import annotations
import os, json, uuid, re, time, hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
import streamlit as st

st.set_page_config(page_title="Football Research Agent v6", page_icon="⚽", layout="wide")

# ============================================================
# Configuration
# ============================================================

MAX_ITERATIONS = 999  # Keep going until user stops!
MAX_MESSAGES = 200
JOB_TIMEOUT = 300

# Default gates - SINGLE SOURCE OF TRUTH
# These are passed to local_compute for all test_filter calls
DEFAULT_GATES = {
    "min_train_rows": 300,
    "min_val_rows": 60,
    "min_test_rows": 60,
    "max_train_val_gap_roi": 0.40,
    "max_test_drawdown": -50,
    "max_rolling_dd": -50,
    "max_test_losing_streak_bets": 50,
}

# ============================================================
# COMPLEXITY PENALTY SYSTEM
# ============================================================
# A strategy with fewer filters is MORE likely to be real/robust
# Penalty reduces the "adjusted score" used for ranking strategies
COMPLEXITY_PENALTY_PER_FILTER = 0.005  # 0.5% penalty per filter
TRAIN_VAL_GAP_PENALTY = 0.02  # 2% penalty per 1% train-val gap
MIN_FILTERS_NO_PENALTY = 2  # First 2 filters are "free"

def _calculate_adjusted_score(test_roi: float, num_filters: int, train_val_gap: float = 0) -> float:
    """
    Calculate complexity-adjusted score for ranking strategies.
    
    A simpler strategy (fewer filters) with slightly lower ROI 
    should be preferred over a complex strategy with higher ROI.
    
    Formula: adjusted_score = test_roi - (filter_penalty) - (gap_penalty)
    """
    # Filter penalty (first 2 filters are free)
    extra_filters = max(0, num_filters - MIN_FILTERS_NO_PENALTY)
    filter_penalty = extra_filters * COMPLEXITY_PENALTY_PER_FILTER
    
    # Train-val gap penalty (bigger gap = more overfitting risk)
    gap_penalty = abs(train_val_gap) * TRAIN_VAL_GAP_PENALTY
    
    adjusted = test_roi - filter_penalty - gap_penalty
    return adjusted

def _count_filters(filters: List[Dict]) -> int:
    """Count number of filter conditions (excluding base market/mode filters)."""
    if not filters:
        return 0
    return len(filters)

# ============================================================
# FINAL HOLDOUT CONFIGURATION
# ============================================================
# Reserve the most recent data as a FINAL holdout that is NEVER touched
# during exploration. Only used for final strategy validation.
# Using 8% as a conservative amount since data doesn't go super far back
FINAL_HOLDOUT_FRACTION = 0.08  # Last 8% of data by time
ENABLE_FINAL_HOLDOUT = True  # Set to False to disable

# Outcome columns (NEVER use as features)
OUTCOME_COLUMNS = ["BO 2.5 PL", "BTTS PL", "SHG PL", "SHG 2+ PL", "LU1.5 PL", "LFGHU0.5 PL", "BO1.5 FHG PL", "PL"]

# Banned columns (NEVER use as features - data leakage or irrelevant)
# These columns reveal match outcomes or are identifiers - using them is CHEATING!
BANNED_COLUMNS = [
    # Match outcomes - ABSOLUTE DATA LEAKAGE!
    "HT Score", "FT Score", "Result", "BET RESULT",
    "1H GT", "2H GT",  # Goals scored - outcome data!
    "RETURN",  # This is the betting return - outcome!
    
    # Identifiers - not predictive features
    "ID", "Home", "Away", "HOME TEAM", "AWAY TEAM",
    "Date", "DATE", "Time", "TIME",
    
    # Form columns to ignore per Bible
    "HOME FORM", "AWAY FORM",
]

# Convert to set for fast lookup
BANNED_COLUMNS_SET = set(col.upper() for col in BANNED_COLUMNS)

# ============================================================
# Helpers
# ============================================================

def _is_banned_column(col_name: str) -> bool:
    """Check if a column is banned (data leakage or identifier)."""
    if not col_name:
        return False
    col_upper = col_name.upper().strip()
    # Check exact match
    if col_upper in BANNED_COLUMNS_SET:
        return True
    # Check if it's a PL column
    if "PL" in col_upper and col_upper not in ["IMPLIED", "IMPL"]:
        return True
    return False

def _filter_banned_from_results(results: Dict) -> Dict:
    """Remove any suggestions that use banned columns."""
    if not results:
        return results
    
    # Filter suggested_filters
    if "suggested_filters" in results:
        results["suggested_filters"] = [
            f for f in results.get("suggested_filters", [])
            if not _is_banned_column(f.get("col", ""))
        ]
    
    # Filter feature_importance
    if "feature_importance" in results:
        results["feature_importance"] = [
            f for f in results.get("feature_importance", [])
            if not _is_banned_column(f.get("feature", "") if isinstance(f, dict) else str(f))
        ]
    
    return results

def _validate_avenue_filters(filters: List[Dict]) -> Tuple[bool, str]:
    """Validate that filters don't use banned columns. Returns (is_valid, error_message)."""
    for f in filters:
        col = f.get("col", "")
        if _is_banned_column(col):
            return False, f"BANNED COLUMN: {col} - this is data leakage!"
    return True, ""

def _safe_json(obj: Any, max_len: int = 5000) -> str:
    try:
        result = json.dumps(obj, indent=2, default=str)
        return result[:max_len] + "\n...(truncated)" if len(result) > max_len else result
    except:
        return '{"error": "serialize failed"}'

def _filter_hash(filters: List[Dict]) -> str:
    return hashlib.md5(json.dumps(filters, sort_keys=True, default=str).encode()).hexdigest()[:12]

# ============================================================
# OpenAI Client (GPT-5)
# ============================================================

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

@st.cache_resource
def _get_client() -> Optional[OpenAI]:
    if not HAS_OPENAI:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def _get_model() -> str:
    """Get the OpenAI model to use - defaults to gpt-5.1, can override with env var."""
    return os.getenv("OPENAI_MODEL") or os.getenv("PREFERRED_OPENAI_MODEL") or "gpt-5.1"

# ============================================================
# Session State
# ============================================================

def _init_state():
    defaults = {
        "messages": [], 
        "session_id": None,
        "bible": None,
        "agent_phase": "idle",  # idle, running, paused, complete
        "agent_iteration": 0, 
        "agent_findings": [],
        "target_pl_column": "BO 2.5 PL", 
        "exploration_results": {}, 
        "sweep_results": {},
        "exploration_analysis": {},
        "past_failures": [], 
        "near_misses": [], 
        "tested_filter_hashes": set(),
        "run_requested": False, 
        "log": [],
        # v5 features
        "avenues_to_explore": [],
        "avenues_explored": [],
        "accumulated_learnings": [],
        "promising_bases": [],
        # v6 additions
        "strategies_found": [],
        "current_thinking": "",
        "last_checkpoint": None,
        # v6.1 - Hypothesis tracking & validation improvements
        "hypothesis_count": 0,  # Total hypotheses tested this session
        "tests_per_iteration": [],  # Track tests per iteration for analysis
        "final_holdout_tested": False,  # Have we used final holdout?
        "strategies_pending_final_validation": [],  # Strategies awaiting final holdout test
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def _log(msg: str):
    try:
        if "log" not in st.session_state:
            st.session_state.log = []
        st.session_state.log.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")
        st.session_state.log = st.session_state.log[-200:]  # Keep last 200
    except Exception:
        pass

def _append(role: str, content: str):
    try:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})
        if len(st.session_state.messages) > MAX_MESSAGES:
            st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
    except Exception:
        pass

def _research_log(content: str, level: str = "info"):
    """Add entry to the persistent research log that survives page rerenders.
    
    Levels: info, success, warning, error, header, strategy, phase
    """
    if "research_log" not in st.session_state:
        st.session_state.research_log = []
    
    entry = {
        "ts": datetime.utcnow().strftime("%H:%M:%S"),
        "level": level,
        "content": content,
    }
    st.session_state.research_log.append(entry)
    
    # Keep last 500 entries
    if len(st.session_state.research_log) > 500:
        st.session_state.research_log = st.session_state.research_log[-500:]

def _display_research_log():
    """Display the full research log in a scrollable container."""
    if "research_log" not in st.session_state or not st.session_state.research_log:
        st.info("No research activity yet. Click **Start Research** to begin.")
        return
    
    st.markdown("### 📜 Full Research Log (Scrollable)")
    st.caption(f"{len(st.session_state.research_log)} entries")
    
    # Create scrollable container with CSS
    log_lines = []
    for entry in st.session_state.research_log:
        ts = entry.get("ts", "")
        level = entry.get("level", "info")
        content = entry.get("content", "").replace("<", "&lt;").replace(">", "&gt;")
        
        # Style based on level
        if level == "header":
            log_lines.append(f'<div style="margin: 10px 0; padding: 8px; background: #1e3a5f; color: white; border-radius: 4px; font-weight: bold;">[{ts}] 📌 {content}</div>')
        elif level == "phase":
            log_lines.append(f'<div style="margin: 8px 0; padding: 6px; background: #e3f2fd; border-left: 4px solid #2196f3; font-weight: bold;">[{ts}] {content}</div>')
        elif level == "success":
            log_lines.append(f'<div style="margin: 4px 0; padding: 4px 8px; background: #e8f5e9; border-left: 3px solid #4caf50;">[{ts}] ✅ {content}</div>')
        elif level == "warning":
            log_lines.append(f'<div style="margin: 4px 0; padding: 4px 8px; background: #fff3e0; border-left: 3px solid #ff9800;">[{ts}] ⚠️ {content}</div>')
        elif level == "error":
            log_lines.append(f'<div style="margin: 4px 0; padding: 4px 8px; background: #ffebee; border-left: 3px solid #f44336;">[{ts}] ❌ {content}</div>')
        elif level == "strategy":
            log_lines.append(f'<div style="margin: 10px 0; padding: 10px; background: #f3e5f5; border: 2px solid #9c27b0; border-radius: 8px; font-weight: bold;">[{ts}] 🎉 STRATEGY FOUND: {content}</div>')
        elif level == "iteration":
            log_lines.append(f'<div style="margin: 8px 0; padding: 6px; background: #fff8e1; border-left: 4px solid #ffc107;">[{ts}] 🔄 {content}</div>')
        else:
            log_lines.append(f'<div style="margin: 2px 0; padding: 2px 8px; color: #333;">[{ts}] {content}</div>')
    
    log_html = f'''
    <div id="research-log" style="
        height: 500px; 
        overflow-y: auto; 
        border: 1px solid #ddd; 
        padding: 10px; 
        border-radius: 8px; 
        background: #fafafa;
        font-family: monospace;
        font-size: 13px;
    ">
        {"".join(log_lines)}
    </div>
    <script>
        var logDiv = document.getElementById('research-log');
        if (logDiv) logDiv.scrollTop = logDiv.scrollHeight;
    </script>
    '''
    
    st.markdown(log_html, unsafe_allow_html=True)

def _add_learning(learning: str):
    """Accumulate learnings across iterations."""
    try:
        if "accumulated_learnings" not in st.session_state:
            st.session_state.accumulated_learnings = []
        if "agent_iteration" not in st.session_state:
            st.session_state.agent_iteration = 0
        st.session_state.accumulated_learnings.append({
            "iteration": st.session_state.agent_iteration,
            "learning": learning,
            "ts": datetime.utcnow().isoformat()
        })
    except Exception:
        pass

# ============================================================
# Local Compute Client - Submit jobs to Supabase
# ============================================================

def _check_server() -> Dict:
    """Check if we can connect to Supabase (jobs table)."""
    try:
        sb = _sb_cached()
        sb.table("jobs").select("job_id").limit(1).execute()
        return {"status": "healthy", "data_rows": 0}  # Can't easily get row count
    except Exception as e:
        return {"status": "offline", "error": str(e)}

def _run_tool(name: str, args: Optional[Dict] = None, timeout: int = JOB_TIMEOUT) -> Any:
    """Submit job to Supabase and wait for result."""
    args = args or {}
    
    # ====== HYPOTHESIS TRACKING ======
    # Track how many filter tests we're running (for multiple testing awareness)
    if name == "test_filter":
        if "hypothesis_count" not in st.session_state:
            st.session_state.hypothesis_count = 0
        st.session_state.hypothesis_count += 1
        
        # Log milestone counts
        if st.session_state.hypothesis_count % 50 == 0:
            _log(f"📊 Hypothesis count: {st.session_state.hypothesis_count} tests run this session")
    
    # Get storage config
    storage_bucket = os.getenv("DATA_STORAGE_BUCKET") or st.secrets.get("DATA_STORAGE_BUCKET", "football-data")
    storage_path = os.getenv("DATA_STORAGE_PATH") or st.secrets.get("DATA_STORAGE_PATH", "football_ai_NNIA.csv")
    results_bucket = os.getenv("RESULTS_BUCKET") or st.secrets.get("RESULTS_BUCKET", "football-results")
    
    # Add storage config to params
    params = {
        "storage_bucket": storage_bucket,
        "storage_path": storage_path,
        "_results_bucket": results_bucket,
        **args
    }
    
    try:
        sb = _sb_cached()
        
        # Submit job
        _log(f"Submitting {name}...")
        row = {
            "status": "queued",
            "task_type": name,
            "params": params,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        res = sb.table("jobs").insert(row).execute()
        if not res.data:
            return {"error": "Failed to submit job"}
        
        job_id = res.data[0]["job_id"]
        _log(f"Job {job_id[:8]}... queued")
        
        # Wait for completion
        start = time.time()
        while time.time() - start < timeout:
            job = sb.table("jobs").select("*").eq("job_id", job_id).limit(1).execute().data
            if not job:
                return {"error": "Job not found"}
            
            job = job[0]
            status = (job.get("status") or "").lower()
            
            if status == "done":
                result_path = job.get("result_path")
                if result_path:
                    # Download result from storage
                    try:
                        raw = sb.storage.from_(results_bucket).download(result_path)
                        return json.loads(raw.decode("utf-8"))
                    except Exception as e:
                        return {"error": f"Failed to download result: {e}"}
                return job
            
            elif status == "error":
                return {"error": job.get("error", "Job failed")}
            
            time.sleep(3)
        
        return {"error": f"Timeout after {timeout}s"}
        
    except Exception as e:
        _log(f"Error in {name}: {e}")
        return {"error": str(e)}

@st.cache_resource
def _sb_cached():
    """Get cached Supabase client."""
    url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    from supabase import create_client
    return create_client(url, key)

# ============================================================
# Bible (Research Context)
# ============================================================

def _load_bible() -> Dict:
    """Load research context from Google Sheets (Bible) directly."""
    try:
        if "bible" not in st.session_state:
            st.session_state.bible = None
        if st.session_state.bible:
            return st.session_state.bible
        
        _log("Loading Bible from Google Sheets...")
        bible = None
        bible_source = "unknown"
        
        # METHOD 1: Try importing football_tools (multiple approaches for different folder structures)
        football_tools = None
        
        # Approach 1a: Direct import (if in same folder or PYTHONPATH)
        try:
            import football_tools as ft
            football_tools = ft
            _log("✅ Imported football_tools directly")
        except ImportError:
            pass
        
        # Approach 1b: Import from modules folder (pages/researcher.py → modules/football_tools.py)
        if not football_tools:
            try:
                from modules import football_tools as ft
                football_tools = ft
                _log("✅ Imported football_tools from modules/")
            except ImportError:
                pass
        
        # Approach 1c: Import from parent's modules folder
        if not football_tools:
            try:
                import sys
                from pathlib import Path
                # If we're in pages/, add parent directory to find modules/
                script_dir = Path(__file__).parent.resolve()
                parent_dir = script_dir.parent
                modules_dir = parent_dir / "modules"
                
                if modules_dir.exists():
                    if str(parent_dir) not in sys.path:
                        sys.path.insert(0, str(parent_dir))
                    if str(modules_dir) not in sys.path:
                        sys.path.insert(0, str(modules_dir))
                    _log(f"Added {parent_dir} and {modules_dir} to Python path")
                    
                    import football_tools as ft
                    football_tools = ft
                    _log("✅ Imported football_tools after path fix")
            except Exception as e:
                _log(f"Path approach failed: {e}")
        
        # Approach 1d: Try explicit file import as last resort
        if not football_tools:
            try:
                import importlib.util
                from pathlib import Path
                
                # Try multiple possible locations
                possible_paths = [
                    Path(__file__).parent / "football_tools.py",  # Same folder
                    Path(__file__).parent.parent / "modules" / "football_tools.py",  # pages/../modules/
                    Path(__file__).parent.parent / "football_tools.py",  # pages/../
                ]
                
                for ft_path in possible_paths:
                    if ft_path.exists():
                        spec = importlib.util.spec_from_file_location("football_tools", ft_path)
                        football_tools = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(football_tools)
                        _log(f"✅ Imported football_tools from {ft_path}")
                        break
                
                if not football_tools:
                    _log(f"❌ football_tools.py not found in any expected location")
                    _log(f"Searched: {[str(p) for p in possible_paths]}")
            except Exception as e:
                _log(f"Explicit import failed: {e}")
        
        # Now try to use the imported module
        if football_tools:
            if hasattr(football_tools, 'get_bible_from_sheets'):
                _log("Calling get_bible_from_sheets...")
                try:
                    bible = football_tools.get_bible_from_sheets(limit_notes=20)
                    
                    if bible and bible.get("ok"):
                        bible_source = "google_sheets_direct"
                        n_rules = len(bible.get('research_rules', []))
                        n_cols = len(bible.get('column_definitions', []))
                        _log(f"✅ Bible loaded from Google Sheets: {n_rules} rules, {n_cols} column defs")
                        st.success(f"✅ Bible loaded from Google Sheets! ({n_rules} rules, {n_cols} columns)")
                    else:
                        error_msg = bible.get('error', 'Unknown error') if bible else 'No response'
                        _log(f"❌ get_bible_from_sheets returned: {error_msg}")
                        st.warning(f"⚠️ Google Sheets returned: {error_msg}")
                        bible = None
                except Exception as e:
                    _log(f"❌ Error calling get_bible_from_sheets: {type(e).__name__}: {e}")
                    st.error(f"❌ Google Sheets error: {type(e).__name__}: {e}")
            else:
                _log("❌ get_bible_from_sheets not found - you may have OLD football_tools.py")
                st.error("❌ get_bible_from_sheets not found - UPDATE football_tools.py!")
                funcs = [f for f in dir(football_tools) if not f.startswith('_') and callable(getattr(football_tools, f, None))]
                _log(f"Available functions: {funcs[:15]}")
        else:
            st.error("❌ Could not import football_tools.py - check folder structure!")
            st.info("Expected: modules/football_tools.py OR same folder as this script")
        
        # METHOD 2: Fallback to job queue (local_compute) - THIS IS BAD, MEANS SHEETS NOT CONNECTED
        if not bible:
            st.warning("⚠️ Falling back to job queue - NOT connected to Google Sheets!")
            st.info("The Bible will have LIMITED rules. Fix the import to get full Bible.")
            _log("Falling back to job queue for Bible...")
            bible = _run_tool("get_research_context", {"pl_column": st.session_state.get("target_pl_column", "BO 2.5 PL")})
            bible_source = "job_queue"
            
            if bible and not bible.get("error"):
                _log(f"Bible loaded from job queue: {len(bible.get('research_rules', []))} rules")
            else:
                _log(f"Job queue Bible failed: {bible}")
        
        # METHOD 3: Minimal fallback
        if not bible or bible.get("error") or not bible.get("research_rules"):
            _log("⚠️ Creating minimal Bible fallback - GOOGLE SHEETS NOT CONNECTED!")
            bible_source = "fallback"
            bible = {
                "dataset_overview": {"primary_goal": "Find profitable betting strategies", "note": "FALLBACK - Google Sheets not connected"},
                "gates": DEFAULT_GATES,
                "derived": {"outcome_columns": OUTCOME_COLUMNS},
                "research_rules": [
                    {"rule": "NEVER use PL columns as features - this is data leakage"},
                    {"rule": "Split by TIME: train on older data, test on newer data"},
                    {"rule": "Minimum 300 train rows for statistical significance"},
                    {"rule": "Require positive test ROI for any strategy"},
                    {"rule": "Forward walk validation required: >60% windows positive"},
                    {"rule": "Statistical significance: p-value < 0.10"},
                ],
                "column_definitions": [],
                "evaluation_framework": [],
            }
        
        # Add source info
        bible["_source"] = bible_source
        st.session_state.bible = bible
        return bible
        
    except Exception as e:
        _log(f"❌ Critical error loading bible: {type(e).__name__}: {e}")
        return {"gates": DEFAULT_GATES, "_source": "error", "_error": str(e)}

def _format_bible(bible: Dict) -> str:
    overview = bible.get("dataset_overview") or {}
    gates = bible.get("gates") or DEFAULT_GATES
    derived = bible.get("derived") or {}
    rules = bible.get("research_rules") or []
    
    outcome_cols = derived.get('outcome_columns', OUTCOME_COLUMNS)
    outcome_str = ', '.join(str(c) for c in outcome_cols) if isinstance(outcome_cols, list) else str(outcome_cols)
    banned_str = ', '.join(BANNED_COLUMNS)
    
    # Format rules
    rules_text = ""
    if rules:
        rules_list = []
        for r in rules[:10]:  # Show first 10 rules
            rule_text = r.get("rule") or r.get("Rule") or str(r)
            if rule_text:
                rules_list.append(f"- {rule_text}")
        if rules_list:
            rules_text = "\n**Key Rules:**\n" + "\n".join(rules_list)
        if len(rules) > 10:
            rules_text += f"\n*...and {len(rules) - 10} more rules*"
    
    return f"""## 📖 Bible Loaded
**Goal:** {overview.get('primary_goal', 'Find profitable strategies')}
**Approach:** {overview.get('approach', 'ML for discovery, filter rules for output')}
**Gates:** min_train={gates.get('min_train_rows', 300)}, min_val={gates.get('min_val_rows', 60)}, min_test={gates.get('min_test_rows', 60)}, max_gap={gates.get('max_train_val_gap_roi', 0.4)}
**Outcome Columns (NEVER features):** {outcome_str}
**Banned Columns (NEVER use):** {banned_str}
{rules_text}"""

# ============================================================
# LLM Functions (GPT-5)
# ============================================================

SYSTEM_PROMPT = """You are a SENIOR QUANTITATIVE RESEARCHER at a top hedge fund. 
Your job is to find profitable, stable betting strategies through DEEP, METHODICAL analysis.

## YOUR CORE PHILOSOPHY

### 70% THINKING, 30% TESTING
- Before EVERY action, think deeply about WHY
- Never rush to test - understand the hypothesis first
- Every action must be justified with expected value reasoning
- Quality of thinking > quantity of tests

### CONNECT EVERYTHING
- Every new finding must be connected to past learnings
- Look for patterns across different tests
- Build a mental model of what's working and WHY
- Contradictions are valuable - they reveal deeper truths

### SKEPTICISM IS ESSENTIAL
- Assume every positive result is noise until proven
- Ask: "Would this have worked 2 years ago? Will it work next year?"
- Overfitting is your enemy - simple rules beat complex ones
- Statistical significance is necessary but not sufficient

## DEEP THINKING CYCLE (USE THIS FOR EVERY DECISION)

### 1. OBSERVE (What just happened?)
- Summarize the results objectively
- Note anything surprising or unexpected
- Identify the key numbers: ROI, sample size, p-value
- What worked? What didn't?

### 2. CONNECT (How does this relate to past learnings?)
- Query your accumulated learnings
- Does this confirm or contradict previous findings?
- Have you seen similar patterns before?
- What strategies have worked in related markets?

### 3. HYPOTHESIZE (Why might this work? What market inefficiency?)
- Propose a SPECIFIC market inefficiency that explains the edge
- Example: "DRIFT IN captures late informed money before odds shorten"
- Example: "Low odds O2.5 bets are undervalued because public overestimates draws"
- If you can't explain WHY, the edge probably isn't real

### 4. PLAN (Generate candidate actions, score by expected value)
Generate 5-8 candidate next actions. For EACH action:
- What tool would you use?
- What parameters?
- What's the expected outcome?
- What's the probability of success?
- What will you learn even if it fails?
- Expected Value Score (1-10)

### 5. DECIDE (Pick the best action with full justification)
- Choose the action with highest expected value
- Explain your full reasoning
- State what would change your mind
- Plan your next 3-5 steps

### 6. RECORD (Save insights for future reference)
- What key insight should be remembered?
- What hypothesis was confirmed/refuted?
- What should be explored further?
- What should be avoided?

## WORLD-CLASS QUANT RESEARCH PROCESS

### Phase 1: HYPOTHESIS GENERATION
- Domain expertise: "Why might this market be inefficient?"
- Let data suggest patterns (feature importance, subgroup scans)
- ML discovery to find patterns humans miss

### Phase 2: RIGOROUS TESTING
- Train/Val/Test split (by TIME, not random)
- Walk-forward analysis (simulate real trading)
- Monte Carlo for confidence intervals
- Check for data leakage

### Phase 3: ALPHA DECAY ANALYSIS
- Does edge decay over time?
- Is it being arbitraged away?
- What regime does it work in? Fail in?

### Phase 4: VALIDATION GATES
- p-value < 0.10 for statistical significance
- Forward walk > 60% positive periods
- Train/val gap < 0.4 (no overfitting)
- Minimum sample sizes met

## EXPLOITATION VS EXPLORATION

At each decision point, consider:
- **Exploit**: Refine a promising strategy that's close to validation
- **Explore**: Try new territory to find different edges
- Early in research: 70% explore, 30% exploit
- Later in research: 30% explore, 70% exploit

## CRITICAL: DRILL DOWN ON WINNERS! 🎯

When you find a profitable combination (e.g., MODE=Quick League, MARKET=FHGO0.5 Back with positive test ROI):

**DO NOT immediately move to the next avenue!**

Instead, DRILL DOWN to find the optimal parameters:

1. **Filter the data** to just that MODE+MARKET combination
2. **Run bracket_sweep** on numeric columns:
   - ACTUAL ODDS: Find the sweet spot (e.g., 1.8-2.2 vs 2.5-3.0)
   - % DRIFT: Does higher drift improve results?
   - IMPLIED ODDS, DIFF
3. **Test categorical splits**:
   - DRIFT IN vs OUT vs SAME - which is best?
   - Which LEAGUEs perform best within this filter?
   - Any xG BRACKET patterns?
4. **Combine the best**:
   - If DRIFT IN improves it, add that filter
   - If ACTUAL ODDS 1.8-2.5 is the sweet spot, add that
   - Build up the optimal filter combination

**Example workflow when Quick League + FHGO0.5 Back shows +1% test ROI:**
```
Step 1: bracket_sweep ACTUAL ODDS within this filter → finds 1.5-2.0 best
Step 2: Test DRIFT IN vs OUT → DRIFT IN adds +1.5%
Step 3: bracket_sweep % DRIFT → finds >= 3% adds +0.5%
Step 4: Final filter: Quick League + FHGO0.5 Back + DRIFT IN + ODDS 1.5-2.0 + % DRIFT >= 3
Step 5: Validate with forward_walk
```

This is how you turn a weak +1% edge into a strong +5% edge!

## WHEN THINGS DON'T WORK

- Don't give up after one failure
- Ask: "What did I learn? What hypothesis is now refuted?"
- Try the OPPOSITE (if DRIFT IN failed, try DRIFT OUT)
- Move to different market/mode combination
- Use ML to suggest new directions

## OUTPUT FORMAT FOR ANALYSIS

When analyzing results, ALWAYS structure your thinking:

```
### SITUATION ASSESSMENT
[2-3 paragraphs on current state of research]
[What have we tried? What's worked? What's failed?]
[What patterns are emerging?]

### KEY INSIGHTS FROM THIS TEST
[What did we learn?]
[Does this confirm or contradict previous findings?]
[What market inefficiency might explain this?]

### CONNECTING TO PAST LEARNINGS
[Link to specific past learnings: "Learning #X showed that..."]
[Identify patterns: "This is the 3rd time we've seen..."]
[Note contradictions: "This contradicts learning #Y because..."]

### CANDIDATE ACTIONS (ranked by expected value)

1. **[Action Name]** (EV: 8/10)
   - Tool: {tool_name}
   - Reasoning: [Why this action?]
   - Expected outcome: [What do I expect?]
   - Learning value: [What will I learn?]

2. **[Action Name]** (EV: 7/10)
   ...

### DECISION
**Chosen Action:** [Action name]
**Justification:** [Full reasoning]
**What Would Change My Mind:** [If I see X, I'll pivot to Y]

### NEXT 3-5 STEPS
1. [Step 1]
2. [Step 2]
3. [Step 3]
```

## CRITICAL RULES - VIOLATION = IMMEDIATE FAILURE

### ABSOLUTELY BANNED COLUMNS (DATA LEAKAGE - NEVER USE!)
These columns reveal match outcomes - using them is CHEATING and makes strategies useless:
- **1H GT, 2H GT** - Goals scored in each half (OUTCOME!)
- **HT Score, FT Score** - Half-time/Full-time scores (OUTCOME!)
- **Result, BET RESULT** - Match result (OUTCOME!)
- **RETURN** - Betting return (OUTCOME!)
- **WINS** - Win count (OUTCOME!)
- **ID** - Row identifier (NOT PREDICTIVE!)
- **Home, Away, HOME TEAM, AWAY TEAM** - Team names (IDENTIFIERS!)
- **Date, TIME** - When match played (IDENTIFIERS!)
- **HOME FORM, AWAY FORM, Home Form Rag, Away Form Rag** - Per Bible rules

### WHY THIS MATTERS
If you use 1H GT >= 50, you're saying "bet on matches where >50 goals were scored in first half" - 
but you DON'T KNOW this before the match! This is data leakage and will NEVER work in production.

### VALID FEATURES (Can use)
- MODE, MARKET, DRIFT IN / OUT, % DRIFT
- ACTUAL ODDS, IMPLIED ODDS, DIFF
- LEAGUE, BRACKET, xG MARKET, xG BRACKET
- Home Avg Points, Away Avg Points, Points Diff
- O2.5 Odds, U1.5 Odds, BTTS Y Odds, etc.

### OTHER RULES
1. NEVER use PL columns as features (data leakage!)
2. Split by TIME: train older, test newer (never random!)
3. Explain WHY a filter exploits market inefficiency
4. Simple > complex - prefer fewer filters
5. Sample size matters - need 300+ train rows, 60+ test rows
6. p-value < 0.10 for statistical significance
7. Forward walk must show >60% positive periods
8. When you find a strategy, KEEP SEARCHING for more!
9. NEVER STOP until user tells you to stop!

## AVAILABLE TOOLS

### Exploration
- **query_data**: Aggregations, distributions
- **feature_importance**: Find predictive columns
- **univariate_scan**: Best single filters
- **bracket_sweep**: Numeric range testing
- **subgroup_scan**: Categorical combinations

### Testing
- **test_filter**: Train/val/test validation
- **forward_walk**: Walk-forward analysis
- **monte_carlo_sim**: Confidence intervals
- **statistical_significance**: P-value calculation
- **time_decay_analysis**: Alpha decay check

### ML Discovery
- **train_catboost**: Best for categoricals
- **train_xgboost**: Robust general purpose
- **train_logistic**: Interpretable coefficients
- **shap_explain**: Convert ML to filter rules

### Memory
- **save_learning**: Store insight
- **query_learnings**: Search past insights
- **save_strategy**: Store strategy
- **query_strategies**: Find similar strategies

Remember: You are a SENIOR QUANT. Think deeply. Connect dots. Be skeptical. Find real edges.
"""

def _llm(context: str, question: str, max_tokens: int = 3000) -> str:
    """Call OpenAI for analysis with robust compatibility across GPT-5 / reasoning models.

    - Uses the Responses API for GPT-5* and o-series models when available (recommended).
    - Falls back to Chat Completions for older models / older SDKs.
    """
    client = _get_client()
    if not client:
        _log("ERROR: OpenAI client not available - check OPENAI_API_KEY")
        return '{"error": "OpenAI client not available - check OPENAI_API_KEY"}'

    model = _get_model()
    effort = os.getenv("OPENAI_REASONING_EFFORT") or "medium"
    user_text = f"{context}\n\n---\n{question}"

    # Heuristic: treat GPT-5* + o-series as reasoning models
    is_reasoning_model = model.startswith(("gpt-5", "o"))

    _log(f"Calling LLM: model={model}, context_len={len(context)}, question_len={len(question)}")

    # ---------------------------
    # Preferred path: Responses API
    # ---------------------------
    if is_reasoning_model and hasattr(client, "responses"):
        try:
            # Keep payload simple; avoid unsupported sampling params for reasoning models.
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                max_output_tokens=max_tokens,
                reasoning={"effort": effort},
            )

            # Text extraction (robust across SDK variants)
            content = getattr(resp, "output_text", None)
            if content is None:
                # Try to assemble from output items
                content_parts = []
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", None) == "message":
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", None) in ("output_text", "text"):
                                content_parts.append(getattr(c, "text", "") or "")
                content = "".join(content_parts).strip()

            _log(f"Served by model: {getattr(resp, 'model', model)}")
            _log(f"LLM response received: {len(content) if content else 0} chars")
            return content or ""

        except Exception as e:
            error_msg = str(e)
            _log(f"Responses API ERROR: {error_msg}")
            # Fall through to Chat Completions attempt below

    # ---------------------------
    # Fallback: Chat Completions
    # ---------------------------
    try:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        }

        # Avoid sampling params for reasoning models
        if not is_reasoning_model:
            params["temperature"] = 0.7

        # Token limits:
        # - Newer reasoning models prefer `max_completion_tokens`
        # - Legacy models accept `max_tokens`
        if is_reasoning_model:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content

        _log(f"Served by model: {getattr(response, 'model', model)}")
        _log(f"LLM response received: {len(content) if content else 0} chars")
        return content or ""

    except Exception as e:
        error_msg = str(e)
        _log(f"LLM ERROR: {error_msg}")

        # Helpful hints for common access/billing/model issues
        lower = error_msg.lower()
        if "does not exist" in lower or "not found" in lower or "model" in lower and "access" in lower:
            _log("HINT: This usually means the model is not enabled for this project/org, or the name is wrong.")
        if "insufficient_quota" in lower or "quota" in lower or "billing" in lower:
            _log("HINT: This usually means API billing/credits aren't active for this project/org.")

        return f'{{"error": "{error_msg}"}}'


def _parse_json(resp: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not resp:
        return None
    
    # First, try direct parse
    try:
        return json.loads(resp)
    except:
        pass
    
    # Strip markdown code blocks
    cleaned = resp.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned)
    cleaned = re.sub(r'^```\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    try:
        return json.loads(cleaned)
    except:
        pass
    
    # Find JSON object in response
    try:
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            return json.loads(match.group())
    except:
        pass
    
    return None

# ============================================================
# Final Holdout Validation
# ============================================================

def _validate_on_final_holdout(filters: List[Dict], pl_column: str, bible: Dict) -> Dict:
    """
    Validate a strategy on the FINAL HOLDOUT - data never seen during exploration.
    
    This is the ultimate test. The final holdout is the most recent ~8% of data
    that was NEVER used during any exploration, drill down, or testing.
    
    Rules:
    - This should only be called ONCE per strategy
    - If it fails here, the strategy should be rejected
    - This is the closest we can get to "real" out-of-sample validation
    """
    if not ENABLE_FINAL_HOLDOUT:
        return {"skipped": True, "reason": "Final holdout disabled"}
    
    _log(f"🔒 FINAL HOLDOUT VALIDATION for {len(filters)} filters")
    
    gates = bible.get("gates", DEFAULT_GATES)
    
    # Call test_filter with final_holdout flag
    result = _run_tool("test_filter", {
        "filters": filters,
        "pl_column": pl_column,
        "enforcement": gates,
        "use_final_holdout": True,  # Special flag for local_compute
        "final_holdout_fraction": FINAL_HOLDOUT_FRACTION,
    })
    
    if not result or result.get("error"):
        return {
            "passed": False,
            "error": result.get("error", "Unknown error"),
            "recommendation": "Could not validate - try again"
        }
    
    # Extract final holdout results
    holdout = result.get("final_holdout", result.get("test", {}))
    holdout_roi = holdout.get("roi", 0)
    holdout_rows = holdout.get("rows", 0)
    
    # Compare to training performance
    train_roi = result.get("train", {}).get("roi", 0)
    test_roi = result.get("test", {}).get("roi", 0)
    
    # Calculate if strategy holds up
    # More lenient than test - just needs to be profitable
    passed = holdout_roi > -0.02  # Allow small negative (within noise)
    strong_pass = holdout_roi > 0.01  # Clearly profitable
    
    # Calculate complexity-adjusted score
    num_filters = _count_filters(filters)
    adjusted_score = _calculate_adjusted_score(holdout_roi, num_filters, abs(train_roi - holdout_roi))
    
    verdict = {
        "passed": passed,
        "strong_pass": strong_pass,
        "holdout_roi": holdout_roi,
        "holdout_rows": holdout_rows,
        "train_roi": train_roi,
        "test_roi": test_roi,
        "adjusted_score": adjusted_score,
        "num_filters": num_filters,
        "degradation": test_roi - holdout_roi,  # How much worse vs test
    }
    
    # Generate recommendation
    if strong_pass:
        verdict["recommendation"] = "✅ VALIDATED - Strategy performs well on unseen data!"
        verdict["confidence"] = "high"
    elif passed:
        verdict["recommendation"] = "⚠️ MARGINAL - Strategy is near breakeven on unseen data. Use with caution."
        verdict["confidence"] = "medium"
    else:
        verdict["recommendation"] = "❌ FAILED - Strategy does not hold up on unseen data. Likely overfit."
        verdict["confidence"] = "low"
    
    # Add warning if significant degradation
    if verdict["degradation"] > 0.03:
        verdict["warning"] = f"⚠️ Significant degradation: {verdict['degradation']*100:.1f}% worse than test set"
    
    _log(f"Final holdout result: ROI {holdout_roi:.2%}, passed={passed}")
    
    return verdict

# ============================================================
# Exploration Phase (from v5)
# ============================================================

def _run_exploration(pl_column: str, progress_container=None) -> Dict:
    """Phase 1: Initial data exploration."""
    results = {}
    
    # Describe data
    if progress_container:
        progress_container.text("Describing data...")
    describe = _run_tool("query_data", {"query_type": "describe"})
    results["describe"] = describe
    
    # MODE distribution
    if progress_container:
        progress_container.text("Analyzing MODE...")
    mode_dist = _run_tool("query_data", {
        "query_type": "aggregate",
        "group_by": ["MODE"],
        "metrics": ["count", f"mean:{pl_column}"]
    })
    results["mode_distribution"] = mode_dist
    
    # MARKET distribution
    if progress_container:
        progress_container.text("Analyzing MARKET...")
    market_dist = _run_tool("query_data", {
        "query_type": "aggregate",
        "group_by": ["MARKET"],
        "metrics": ["count", f"mean:{pl_column}"]
    })
    results["market_distribution"] = market_dist
    
    # DRIFT distribution
    if progress_container:
        progress_container.text("Analyzing DRIFT...")
    drift_dist = _run_tool("query_data", {
        "query_type": "aggregate",
        "group_by": ["DRIFT IN / OUT"],
        "metrics": ["count", f"mean:{pl_column}"]
    })
    results["drift_distribution"] = drift_dist
    
    # MODE x MARKET x DRIFT
    if progress_container:
        progress_container.text("Analyzing combinations...")
    combo = _run_tool("query_data", {
        "query_type": "aggregate",
        "group_by": ["MODE", "MARKET", "DRIFT IN / OUT"],
        "metrics": ["count", f"mean:{pl_column}", f"sum:{pl_column}"]
    })
    results["mode_market_drift"] = combo
    
    return results

def _analyze_exploration(bible: Dict, exploration: Dict, pl_column: str) -> Dict:
    """Deep LLM analysis of exploration results using WORLD-CLASS QUANT thinking."""
    
    # Build rich context
    context = f"""## RESEARCH CONTEXT

### Target: {pl_column}

### Bible Rules (THE LAW)
{_safe_json(bible.get('research_rules', [])[:10], 1500)}

### Dataset Overview
{_safe_json(bible.get('dataset_overview', {}), 500)}

### Past Learnings (What We Already Know)
{_safe_json(bible.get('recent_learnings', []), 1000)}

### Strategy Repository Status
{_safe_json(bible.get('strategy_counts', {}), 200)}

---

## EXPLORATION RESULTS

### Data Summary
- Total rows: {exploration.get('describe', {}).get('total_rows', 'Unknown')}
- Columns: {len(exploration.get('describe', {}).get('columns', []))} features

### MODE Distribution (First Split)
{_safe_json(exploration.get('mode_distribution', {}), 1500)}

### MARKET Distribution  
{_safe_json(exploration.get('market_distribution', {}), 2000)}

### DRIFT Distribution
{_safe_json(exploration.get('drift_distribution', {}), 800)}

### MODE x MARKET x DRIFT Combinations (Key Patterns)
{_safe_json(exploration.get('mode_market_drift', {}), 3000)}
"""

    question = """## YOUR TASK: DEEP ANALYSIS

You are a SENIOR QUANT RESEARCHER. Analyze this data using the DEEP THINKING CYCLE.

### 1. OBSERVE (What do you see in the data?)
- Which MODEs have the best/worst mean PL?
- Which MARKETs show promise (positive mean PL)?
- How does DRIFT affect results?
- What combinations stand out?
- Note sample sizes - are they sufficient?

### 2. CONNECT (Link to what we know)
- How does this relate to any past learnings shown above?
- Are there patterns consistent with betting market theory?
- Any contradictions with previous findings?

### 3. HYPOTHESIZE (WHY might patterns exist?)
For each promising pattern, propose a SPECIFIC market inefficiency:
- Example: "DRIFT IN + Back bets might capture informed money entering late"
- Example: "FHGO0.5 might be underpriced because market underestimates first-half goals"
- If you can't explain WHY, be skeptical of the pattern

### 4. PLAN (Candidate research avenues, scored by expected value)
Generate 8-10 research avenues. For EACH:
- What filters define it?
- What's your hypothesis for WHY it might work?
- Expected rows (is sample size sufficient?)
- Confidence level (high/medium/low)
- What DRIFT direction seems best for this combination?

### 5. DECIDE (Rank and prioritize)
- Rank avenues by expected value
- Consider: probability of success × magnitude of edge × learning value
- Balance exploitation (refine promising) vs exploration (try new)

Respond with JSON:
{
    "situation_assessment": "2-3 paragraphs: What's the current state? What patterns emerge? What's surprising?",
    
    "key_observations": [
        "Observation 1 with specific numbers",
        "Observation 2 with specific numbers",
        "..."
    ],
    
    "mode_analysis": {
        "best_mode": "MODE name",
        "best_mode_pl": 0.00,
        "best_mode_hypothesis": "Why this MODE might have edge",
        "worst_mode": "MODE name",
        "worst_mode_pl": 0.00,
        "worst_mode_reason": "Why this MODE underperforms"
    },
    
    "market_analysis": {
        "promising_markets": [
            {"market": "X", "mean_pl": 0.00, "count": 1000, "hypothesis": "Why this might work"}
        ],
        "avoid_markets": [
            {"market": "Y", "mean_pl": -0.05, "count": 500, "reason": "Why to avoid"}
        ]
    },
    
    "drift_analysis": {
        "best_drift": "IN/OUT/SAME",
        "drift_impact": "How drift affects profitability",
        "drift_hypothesis": "Why drift might matter (informed money, etc.)"
    },
    
    "connections_to_past": [
        "Connection to learning #X: ...",
        "This confirms/contradicts previous finding that..."
    ],
    
    "market_inefficiency_hypotheses": [
        {
            "pattern": "MODE=X + MARKET=Y",
            "hypothesis": "Specific market inefficiency explanation",
            "testable_prediction": "If true, we should see..."
        }
    ],
    
    "prioritized_avenues": [
        {
            "rank": 1,
            "avenue": "MODE=X, MARKET=Y, DRIFT=Z",
            "base_filters": [
                {"col": "MODE", "op": "=", "value": "X"},
                {"col": "MARKET", "op": "=", "value": "Y"}
            ],
            "promising_drift": "IN or OUT or SAME",
            "market_inefficiency_hypothesis": "Specific explanation of WHY this edge exists",
            "expected_rows": "~500",
            "expected_value_score": 8,
            "confidence": "high/medium/low",
            "what_we_learn_if_fails": "Even if this fails, we learn..."
        }
    ],
    
    "next_steps_reasoning": "After testing these avenues, the logical next steps would be...",
    
    "red_flags": [
        "Concern 1: Small sample size for X",
        "Concern 2: Pattern might be noise because..."
    ]
}"""

    resp = _llm(context, question, max_tokens=4000)  # More tokens for deep thinking
    _log(f"LLM analysis response length: {len(resp) if resp else 0}")
    
    # Check for errors
    if not resp or resp.startswith('{"error"'):
        _log(f"LLM error: {resp}")
        return _generate_fallback_avenues(exploration, pl_column)
    
    parsed = _parse_json(resp)
    
    if not parsed:
        _log("Failed to parse LLM response, using fallback")
        return {"detailed_analysis": resp[:1000] if resp else "Analysis failed", "prioritized_avenues": _generate_fallback_avenues(exploration, pl_column).get("prioritized_avenues", [])}
    
    # Ensure we have avenues
    if not parsed.get("prioritized_avenues"):
        fallback = _generate_fallback_avenues(exploration, pl_column)
        parsed["prioritized_avenues"] = fallback.get("prioritized_avenues", [])
    
    # Add situation_assessment as detailed_analysis for backward compatibility
    if parsed.get("situation_assessment") and not parsed.get("detailed_analysis"):
        parsed["detailed_analysis"] = parsed["situation_assessment"]
    
    return parsed


def _generate_fallback_avenues(exploration: Dict, pl_column: str) -> Dict:
    """Generate avenues from exploration data without LLM."""
    avenues = []
    
    # Extract from mode distribution
    mode_dist = exploration.get("mode_distribution", {}).get("result", [])
    for m in mode_dist:
        mode = m.get("MODE")
        mean_pl = m.get(f"mean_{pl_column}", m.get("mean_BO 2.5 PL", 0))
        if mode and mean_pl and mean_pl > -0.01:  # Not too negative
            avenues.append({
                "rank": len(avenues) + 1,
                "avenue": f"MODE={mode}",
                "base_filters": [{"col": "MODE", "op": "=", "value": mode}],
                "market_inefficiency_hypothesis": f"MODE {mode} shows {mean_pl:.4f} mean PL - worth exploring",
                "confidence": "medium" if mean_pl > 0 else "low",
            })
    
    # Extract from market distribution
    market_dist = exploration.get("market_distribution", {}).get("result", [])
    for m in sorted(market_dist, key=lambda x: x.get(f"mean_{pl_column}", x.get("mean_BO 2.5 PL", 0)) or 0, reverse=True)[:5]:
        market = m.get("MARKET")
        mean_pl = m.get(f"mean_{pl_column}", m.get("mean_BO 2.5 PL", 0))
        count = m.get("_count", 0)
        if market and mean_pl and mean_pl > 0 and count >= 300:
            avenues.append({
                "rank": len(avenues) + 1,
                "avenue": f"MARKET={market}",
                "base_filters": [{"col": "MARKET", "op": "=", "value": market}],
                "market_inefficiency_hypothesis": f"MARKET {market} shows {mean_pl:.4f} mean PL with {count} rows",
                "confidence": "medium",
            })
    
    return {
        "detailed_analysis": "Fallback analysis: Generated avenues from exploration data distribution. LLM analysis was not available.",
        "prioritized_avenues": avenues[:8],
    }

# ============================================================
# Sweeps Phase (from v5)
# ============================================================

def _run_sweeps(pl_column: str, bible: Dict, progress_container=None) -> Dict:
    """Phase 2: Run bracket and subgroup sweeps."""
    results = {}
    gates = bible.get("gates", DEFAULT_GATES)
    
    # Bracket sweep
    if progress_container:
        progress_container.text("Running bracket sweep...")
    bracket = _run_tool("bracket_sweep", {
        "pl_column": pl_column,
        "sweep_cols": ["ACTUAL ODDS", "% DRIFT"],
        "n_bins": 8,
        "max_results": 50,
        "enforcement": gates
    })
    results["bracket_sweep"] = bracket
    
    # Subgroup scan
    if progress_container:
        progress_container.text("Running subgroup scan...")
    subgroup = _run_tool("subgroup_scan", {
        "pl_column": pl_column,
        "group_cols": ["MODE", "MARKET", "DRIFT IN / OUT", "LEAGUE"],
        "max_groups": 50,
        "enforcement": gates
    })
    results["subgroup_scan"] = subgroup
    
    return results

# ============================================================
# Filter Testing (from v5)
# ============================================================

def _test_filter_batch(base_filters: List[Dict], variations: List[Dict], pl_column: str, bible: Dict) -> List[Dict]:
    """Test base filters with multiple variations."""
    results = []
    gates = bible.get("gates", DEFAULT_GATES)
    
    # Always test base first
    _log(f"Testing base filters: {base_filters}")
    base_result = _run_tool("test_filter", {
        "filters": base_filters,
        "pl_column": pl_column,
        "enforcement": gates
    })
    base_result["variation"] = "base"
    base_result["filters_tested"] = base_filters
    results.append(base_result)
    
    # Test each variation
    for var in variations:
        combined = base_filters + [var]
        fhash = _filter_hash(combined)
        
        # Skip if already tested
        if "tested_filter_hashes" not in st.session_state:
            st.session_state.tested_filter_hashes = set()
        if fhash in st.session_state.tested_filter_hashes:
            continue
        st.session_state.tested_filter_hashes.add(fhash)
        
        # Create variation name
        if var.get("op") == "=":
            var_name = f"{var.get('col')} = {var.get('value')}"
        elif var.get("op") == "between":
            var_name = f"{var.get('col')} {var.get('min')}-{var.get('max')}"
        else:
            var_name = f"{var.get('col')} {var.get('op')} {var.get('value')}"
        
        _log(f"Testing variation: {var_name}")
        result = _run_tool("test_filter", {
            "filters": combined,
            "pl_column": pl_column,
            "enforcement": gates
        })
        result["variation"] = f"+ {var_name}"
        result["filters_tested"] = combined
        results.append(result)
    
    return results

def _generate_variations(exploration_analysis: Dict) -> List[Dict]:
    """Generate standard variations to test."""
    variations = [
        # Drift variations
        {"col": "DRIFT IN / OUT", "op": "=", "value": "IN"},
        {"col": "DRIFT IN / OUT", "op": "=", "value": "OUT"},
        {"col": "DRIFT IN / OUT", "op": "=", "value": "SAME"},
        # Odds variations
        {"col": "ACTUAL ODDS", "op": "between", "min": 1.5, "max": 2.5},
        {"col": "ACTUAL ODDS", "op": "between", "min": 2.0, "max": 3.5},
        {"col": "ACTUAL ODDS", "op": "between", "min": 1.2, "max": 2.0},
        # % DRIFT variations
        {"col": "% DRIFT", "op": ">", "value": 0},
        {"col": "% DRIFT", "op": "<", "value": 0},
    ]
    return variations

# ============================================================
# Avenue Exploration (from v5)
# ============================================================

def _explore_avenue(avenue: Dict, pl_column: str, bible: Dict, exploration_analysis: Dict) -> Dict:
    """Thoroughly explore one avenue with multiple variations."""
    base_filters = avenue.get("base_filters", [])
    avenue_name = avenue.get("avenue", "Unknown")
    
    _log(f"Exploring avenue: {avenue_name}")
    
    # CRITICAL: Validate filters don't use banned columns!
    is_valid, error_msg = _validate_avenue_filters(base_filters)
    if not is_valid:
        _log(f"⛔ SKIPPING AVENUE - {error_msg}")
        return {
            "avenue": avenue_name,
            "base_filters": base_filters,
            "variations_tested": 0,
            "results": [],
            "analysis": {
                "summary": f"SKIPPED: {error_msg}",
                "recommendation": "SKIP - uses banned column (data leakage)",
                "truly_passing": [],
                "near_misses": [],
            },
            "skipped_reason": error_msg,
        }
    
    # Generate variations
    variations = _generate_variations(exploration_analysis)
    
    # Run batch test
    results = _test_filter_batch(base_filters, variations, pl_column, bible)
    
    # Analyze results
    analysis = _analyze_avenue_results(avenue, results, bible)
    
    return {
        "avenue": avenue_name,
        "base_filters": base_filters,
        "variations_tested": len(results),
        "results": results,
        "analysis": analysis,
    }


def _deep_drill_down(base_filters: List[Dict], pl_column: str, bible: Dict) -> Dict:
    """
    COMPREHENSIVE AI-DRIVEN DEEP DRILL DOWN.
    
    Tests EVERYTHING systematically:
    1. ALL categorical values (DRIFT, LEAGUE, etc.)
    2. ALL numeric ranges (multiple odds columns, points, etc.)
    3. MANY combinations (cat+num, cat+cat, num+num)
    4. AI analysis between phases to guide exploration
    5. COMPLEXITY-ADJUSTED SCORING (simpler strategies preferred)
    """
    _log(f"🔬 DEEP DRILL DOWN on: {base_filters}")
    
    gates = bible.get("gates", DEFAULT_GATES)
    base_filter_count = _count_filters(base_filters)
    
    drill_results = {
        "base_filters": base_filters,
        "individual_tests": [],
        "combinations": [],
        "recommended_additions": [],
        "tests_run": 0,  # Track for hypothesis counting
    }
    
    st.markdown("### 🔬 Comprehensive Deep Drill Down")
    
    # Show hypothesis count so far
    hypothesis_count = st.session_state.get("hypothesis_count", 0)
    st.caption(f"📊 Total hypotheses tested this session: {hypothesis_count}")
    
    # ===== PHASE 1: Get Baseline =====
    st.markdown("**Phase 1: Establishing Baseline**")
    base_result = _run_tool("test_filter", {
        "filters": base_filters,
        "pl_column": pl_column,
        "enforcement": gates,
    })
    
    base_test_roi = base_result.get("test", {}).get("roi", 0) if base_result else 0
    base_test_rows = base_result.get("test", {}).get("rows", 0) if base_result else 0
    base_train_roi = base_result.get("train", {}).get("roi", 0) if base_result else 0
    base_val_roi = base_result.get("val", {}).get("roi", 0) if base_result else 0
    base_train_val_gap = abs(base_train_roi - base_val_roi)
    
    # Calculate baseline adjusted score
    base_adjusted_score = _calculate_adjusted_score(base_test_roi, base_filter_count, base_train_val_gap)
    
    st.markdown(f"📊 **Baseline:** Train {base_train_roi:.2%} → Test {base_test_roi:.2%} ({base_test_rows} rows)")
    st.markdown(f"📐 **Complexity:** {base_filter_count} filters | Adjusted Score: {base_adjusted_score:.2%}")
    
    # ===== DEFINE ALL FEATURES TO TEST =====
    
    # Categorical columns - test EVERY value
    CATEGORICAL_COLS = [
        "DRIFT IN / OUT",
        "LEAGUE", 
        "xG MARKET",
        "Home Form Rag",
        "Away Form Rag",
        "SEASON",
    ]
    
    # Numeric columns with multiple test approaches
    # For each: test brackets AND thresholds (>= X, <= X)
    NUMERIC_TESTS = {
        "ACTUAL ODDS": {
            "brackets": [(1.1, 1.3), (1.3, 1.5), (1.5, 1.8), (1.8, 2.2), (2.2, 3.0)],
            "thresholds_gte": [1.3, 1.5, 1.8],
            "thresholds_lte": [1.5, 1.8, 2.0],
        },
        "% DRIFT": {
            "brackets": [(0, 3), (3, 6), (6, 10), (10, 20)],
            "thresholds_gte": [3, 5, 8],
            "thresholds_lte": [3, 5, 8],
        },
        "STRIKE RATE": {
            "brackets": [(30, 50), (50, 60), (60, 70), (70, 90)],
            "thresholds_gte": [50, 60, 70],
            "thresholds_lte": [50, 60],
        },
        "Home Avg Points": {
            "brackets": [(0, 1.0), (1.0, 1.4), (1.4, 1.8), (1.8, 2.2), (2.2, 3.0)],
            "thresholds_gte": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "thresholds_lte": [1.2, 1.4, 1.6],
        },
        "Away Avg Points": {
            "brackets": [(0, 1.0), (1.0, 1.4), (1.4, 1.8), (1.8, 2.2)],
            "thresholds_gte": [1.0, 1.2, 1.4],
            "thresholds_lte": [1.2, 1.4],
        },
        "O2.5 Odds": {
            "brackets": [(1.3, 1.6), (1.6, 2.0), (2.0, 2.5), (2.5, 3.5)],
            "thresholds_gte": [1.8, 2.0, 2.5],
            "thresholds_lte": [1.8, 2.0, 2.5],
        },
        "BTTS Y Odds": {
            "brackets": [(1.4, 1.7), (1.7, 2.0), (2.0, 2.5)],
            "thresholds_gte": [1.6, 1.8, 2.0],
            "thresholds_lte": [1.8, 2.0],
        },
        "Points Diff": {
            "brackets": [(-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 3.0)],
            "thresholds_gte": [0, 0.5, 1.0],
            "thresholds_lte": [0, -0.5],
        },
    }
    
    all_tests = []
    improving_filters = []
    categorical_results = {}
    numeric_results = {}
    
    progress = st.progress(0)
    status = st.empty()
    
    total_tests = 100  # Rough estimate
    test_count = 0
    
    # ===== PHASE 2A: Test ALL Categorical Values =====
    st.markdown("**Phase 2A: Testing Categorical Features**")
    
    for col_name in CATEGORICAL_COLS:
        status.text(f"Testing {col_name}...")
        categorical_results[col_name] = []
        
        try:
            query_result = _run_tool("query_data", {
                "filters": base_filters,
                "group_by": [col_name],
                "metrics": ["count"],
            })
            
            groups = query_result.get("result", []) if query_result else []
            _log(f"Found {len(groups)} groups for {col_name}")
            
            # Debug: show if query failed or returned no groups
            if not groups:
                _log(f"WARNING: No groups returned for {col_name}. Query result: {query_result}")
                if query_result and query_result.get("error"):
                    st.warning(f"⚠️ Could not test {col_name}: {query_result.get('error')}")
                elif not query_result:
                    st.warning(f"⚠️ Query for {col_name} returned nothing")
            else:
                # Show what values we found
                _log(f"Groups for {col_name}: {[g.get(col_name) for g in groups[:5]]}")
            
            tested_values = 0
            for group in groups:
                value = group.get(col_name)
                count = group.get("_count", 0)
                
                # Lower threshold to 20 for categorical testing (we'll validate with gates later)
                if value is None or value == "" or count < 20:
                    continue
                
                tested_values += 1
                test_filters = base_filters + [{"col": col_name, "op": "=", "value": value}]
                
                result = _run_tool("test_filter", {
                    "filters": test_filters,
                    "pl_column": pl_column,
                    "enforcement": gates,
                })
                
                if result and not result.get("error"):
                    test_roi = result.get("test", {}).get("roi", 0)
                    test_rows = result.get("test", {}).get("rows", 0)
                    improvement = test_roi - base_test_roi
                    
                    test_data = {
                        "type": "categorical",
                        "column": col_name,
                        "value": value,
                        "display": f"{col_name} = {value}",
                        "train_roi": result.get("train", {}).get("roi", 0),
                        "val_roi": result.get("val", {}).get("roi", 0),
                        "test_roi": test_roi,
                        "test_rows": test_rows,
                        "improvement": improvement,
                        "gates_passed": result.get("gates_passed", False),
                        "filter": {"col": col_name, "op": "=", "value": value},
                    }
                    all_tests.append(test_data)
                    categorical_results[col_name].append(test_data)
                    
                    # Log significant findings
                    if improvement > 0.01:
                        _log(f"  {col_name}={value}: improvement={improvement:.2%}, test_rows={test_rows}")
                    
                    # Track improving filters (lower threshold: improvement > -0.005 to catch near-baseline)
                    if improvement > 0 and test_rows >= 30:
                        improving_filters.append(test_data)
                
                test_count += 1
                progress.progress(min(test_count / total_tests, 0.95))
            
            if tested_values == 0:
                st.caption(f"  ⚠️ No testable values for {col_name} (all groups too small)")
                        
        except Exception as e:
            _log(f"Error testing {col_name}: {e}")
            st.warning(f"⚠️ Error testing {col_name}: {str(e)[:100]}")
    
    # Show categorical results
    cat_improving = [f for f in all_tests if f["type"] == "categorical" and f["improvement"] > 0]
    if cat_improving:
        st.markdown(f"✅ **Categorical improvements found:** {len(cat_improving)}")
        for f in sorted(cat_improving, key=lambda x: x["improvement"], reverse=True)[:5]:
            st.markdown(f"  - {f['display']}: +{f['improvement']*100:.1f}% ({f['test_rows']} rows)")
    else:
        st.info("No categorical improvements found (this is unusual)")
    
    # ===== PHASE 2B: Test ALL Numeric Ranges =====
    st.markdown("**Phase 2B: Testing Numeric Features**")
    
    for col_name, tests in NUMERIC_TESTS.items():
        status.text(f"Testing {col_name}...")
        numeric_results[col_name] = []
        
        # Test brackets
        for min_val, max_val in tests.get("brackets", []):
            try:
                test_filters = base_filters + [{"col": col_name, "op": "between", "min": min_val, "max": max_val}]
                
                result = _run_tool("test_filter", {
                    "filters": test_filters,
                    "pl_column": pl_column,
                    "enforcement": gates,
                })
                
                if result and not result.get("error"):
                    test_rows = result.get("test", {}).get("rows", 0)
                    if test_rows >= 20:
                        test_data = {
                            "type": "numeric_bracket",
                            "column": col_name,
                            "range": f"{min_val}-{max_val}",
                            "display": f"{col_name} {min_val}-{max_val}",
                            "train_roi": result.get("train", {}).get("roi", 0),
                            "val_roi": result.get("val", {}).get("roi", 0),
                            "test_roi": result.get("test", {}).get("roi", 0),
                            "test_rows": test_rows,
                            "improvement": result.get("test", {}).get("roi", 0) - base_test_roi,
                            "gates_passed": result.get("gates_passed", False),
                            "filter": {"col": col_name, "op": "between", "min": min_val, "max": max_val},
                        }
                        all_tests.append(test_data)
                        numeric_results[col_name].append(test_data)
                        
                        if test_data["improvement"] > 0 and test_rows >= 30:
                            improving_filters.append(test_data)
                
                test_count += 1
                progress.progress(min(test_count / total_tests, 0.95))
                        
            except Exception as e:
                _log(f"Error testing {col_name} {min_val}-{max_val}: {e}")
        
        # Test >= thresholds
        for threshold in tests.get("thresholds_gte", []):
            try:
                test_filters = base_filters + [{"col": col_name, "op": ">=", "value": threshold}]
                
                result = _run_tool("test_filter", {
                    "filters": test_filters,
                    "pl_column": pl_column,
                    "enforcement": gates,
                })
                
                if result and not result.get("error"):
                    test_rows = result.get("test", {}).get("rows", 0)
                    if test_rows >= 30:
                        test_data = {
                            "type": "numeric_gte",
                            "column": col_name,
                            "threshold": threshold,
                            "display": f"{col_name} >= {threshold}",
                            "train_roi": result.get("train", {}).get("roi", 0),
                            "val_roi": result.get("val", {}).get("roi", 0),
                            "test_roi": result.get("test", {}).get("roi", 0),
                            "test_rows": test_rows,
                            "improvement": result.get("test", {}).get("roi", 0) - base_test_roi,
                            "gates_passed": result.get("gates_passed", False),
                            "filter": {"col": col_name, "op": ">=", "value": threshold},
                        }
                        all_tests.append(test_data)
                        
                        if test_data["improvement"] > 0:
                            improving_filters.append(test_data)
                
                test_count += 1
                        
            except Exception as e:
                _log(f"Error testing {col_name} >= {threshold}: {e}")
        
        # Test <= thresholds
        for threshold in tests.get("thresholds_lte", []):
            try:
                test_filters = base_filters + [{"col": col_name, "op": "<=", "value": threshold}]
                
                result = _run_tool("test_filter", {
                    "filters": test_filters,
                    "pl_column": pl_column,
                    "enforcement": gates,
                })
                
                if result and not result.get("error"):
                    test_rows = result.get("test", {}).get("rows", 0)
                    if test_rows >= 30:
                        test_data = {
                            "type": "numeric_lte",
                            "column": col_name,
                            "threshold": threshold,
                            "display": f"{col_name} <= {threshold}",
                            "train_roi": result.get("train", {}).get("roi", 0),
                            "val_roi": result.get("val", {}).get("roi", 0),
                            "test_roi": result.get("test", {}).get("roi", 0),
                            "test_rows": test_rows,
                            "improvement": result.get("test", {}).get("roi", 0) - base_test_roi,
                            "gates_passed": result.get("gates_passed", False),
                            "filter": {"col": col_name, "op": "<=", "value": threshold},
                        }
                        all_tests.append(test_data)
                        
                        if test_data["improvement"] > 0:
                            improving_filters.append(test_data)
                
                test_count += 1
                        
            except Exception as e:
                _log(f"Error testing {col_name} <= {threshold}: {e}")
    
    progress.progress(1.0)
    status.empty()
    
    # ===== PHASE 3: Display All Results =====
    st.markdown("**Phase 3: Analysis of All Results**")
    
    improving_filters.sort(key=lambda x: x["improvement"], reverse=True)
    
    if improving_filters:
        st.markdown(f"🎯 **Found {len(improving_filters)} filters that IMPROVE test ROI:**")
        
        for f in improving_filters[:12]:
            status_icon = "✅" if f.get("gates_passed") else "⚠️"
            st.markdown(
                f"- **{f['display']}**: Test {f['test_roi']:.2%} "
                f"(**+{f['improvement']*100:.1f}%**) "
                f"[{f['test_rows']} rows] {status_icon}"
            )
    else:
        st.warning("No individual filters improve test ROI over baseline")
        return drill_results
    
    drill_results["individual_tests"] = all_tests
    
    # ===== PHASE 4: AI Analysis & Combination Planning =====
    st.markdown("**Phase 4: AI Strategic Analysis**")
    
    top_cat = [f for f in improving_filters if f["type"] == "categorical"][:5]
    top_num = [f for f in improving_filters if f["type"] != "categorical"][:8]
    
    improving_summary = "\n".join([
        f"- {f['display']}: +{f['improvement']*100:.1f}%, {f['test_rows']} rows, gates={'PASS' if f['gates_passed'] else 'FAIL'}"
        for f in improving_filters[:15]
    ])
    
    cat_summary = "\n".join([f"- {f['display']}: +{f['improvement']*100:.1f}%" for f in top_cat]) if top_cat else "None found"
    num_summary = "\n".join([f"- {f['display']}: +{f['improvement']*100:.1f}%" for f in top_num]) if top_num else "None found"
    
    analysis_context = f"""
## Deep Drill Down Results for {pl_column}

### Base Strategy: {json.dumps(base_filters)}
Train ROI: {base_train_roi:.2%}, Test ROI: {base_test_roi:.2%} ({base_test_rows} rows)

### Top Improving Filters (sorted by improvement):
{improving_summary}

### Categorical Winners:
{cat_summary}

### Numeric Winners:
{num_summary}

### Summary Stats:
- Total tests run: {len(all_tests)}
- Filters showing improvement: {len(improving_filters)}
- Filters passing all gates: {len([f for f in improving_filters if f['gates_passed']])}
"""
    
    ai_question = """Analyze these drill down results and plan combination testing.

Think step by step:
1. What patterns do you see? Which feature types (categorical vs numeric) are strongest?
2. Are there logical combinations? (e.g., DRIFT IN + high home points = informed money on strong home teams)
3. What 2-way combinations should we test? Consider ALL of:
   - Best categorical + best numeric
   - Best categorical + second best numeric  
   - Two different numerics that make sense together
   - Second best categorical + best numeric
4. What 3-way combinations might work? (only if 2-ways show promise)

Provide your analysis and exactly 8-10 combinations to test.

Respond with JSON:
{
    "pattern_analysis": "2-3 sentences on what patterns you see",
    "key_insight": "The single most important finding",
    "combinations_to_test": [
        {
            "name": "Descriptive name for this combo",
            "filters": [
                {"col": "X", "op": "=", "value": "Y"},
                {"col": "Z", "op": ">=", "value": 1.5}
            ],
            "hypothesis": "Why this combination might work well together"
        }
    ]
}"""
    
    with st.spinner("🧠 AI analyzing patterns and planning combinations..."):
        raw_response = _llm(analysis_context, ai_question)
        ai_response = _parse_json(raw_response)
    
    combinations_to_test = []
    
    if ai_response and not ai_response.get("error"):
        st.markdown(f"🧠 **AI Analysis:** {ai_response.get('pattern_analysis', 'N/A')}")
        st.markdown(f"💡 **Key Insight:** {ai_response.get('key_insight', 'N/A')}")
        
        combinations_to_test = ai_response.get("combinations_to_test", [])
        st.markdown(f"📋 AI suggests **{len(combinations_to_test)}** combinations to test")
    
    # Fallback: generate combinations automatically
    if not combinations_to_test:
        st.info("Generating combinations automatically...")
        
        top_cat = [f for f in improving_filters if f["type"] == "categorical"][:4]
        top_num = [f for f in improving_filters if f["type"] != "categorical"][:6]
        
        # cat × num
        for cat in top_cat:
            for num in top_num:
                if cat["column"] != num["column"]:
                    combinations_to_test.append({
                        "name": f"{cat['display']} + {num['display']}",
                        "filters": [cat["filter"], num["filter"]],
                        "hypothesis": "Top categorical × top numeric",
                    })
        
        # num × num
        for i, num1 in enumerate(top_num):
            for num2 in top_num[i+1:]:
                if num1["column"] != num2["column"]:
                    combinations_to_test.append({
                        "name": f"{num1['display']} + {num2['display']}",
                        "filters": [num1["filter"], num2["filter"]],
                        "hypothesis": "Two numeric filters",
                    })
        
        # cat × cat
        for i, cat1 in enumerate(top_cat):
            for cat2 in top_cat[i+1:]:
                if cat1["column"] != cat2["column"]:
                    combinations_to_test.append({
                        "name": f"{cat1['display']} + {cat2['display']}",
                        "filters": [cat1["filter"], cat2["filter"]],
                        "hypothesis": "Two categorical filters",
                    })
    
    # ===== PHASE 5: Test ALL Combinations =====
    st.markdown(f"**Phase 5: Testing {len(combinations_to_test)} Combinations**")
    
    combination_results = []
    combo_progress = st.progress(0)
    
    for idx, combo in enumerate(combinations_to_test[:25]):  # Test up to 25 combinations
        combo_name = combo.get("name", f"Combo {idx+1}")
        combo_filters = combo.get("filters", [])
        
        if not combo_filters:
            continue
        
        try:
            full_filters = list(base_filters) + combo_filters
            
            result = _run_tool("test_filter", {
                "filters": full_filters,
                "pl_column": pl_column,
                "enforcement": gates,
            })
            
            if result and not result.get("error"):
                test_roi = result.get("test", {}).get("roi", 0)
                test_rows = result.get("test", {}).get("rows", 0)
                improvement = test_roi - base_test_roi
                
                combo_result = {
                    "name": combo_name,
                    "filters": full_filters,
                    "added_filters": combo_filters,
                    "hypothesis": combo.get("hypothesis", ""),
                    "train_roi": result.get("train", {}).get("roi", 0),
                    "val_roi": result.get("val", {}).get("roi", 0),
                    "test_roi": test_roi,
                    "test_rows": test_rows,
                    "improvement": improvement,
                    "gates_passed": result.get("gates_passed", False),
                }
                combination_results.append(combo_result)
        
        except Exception as e:
            _log(f"Error testing combo {combo_name}: {e}")
        
        combo_progress.progress((idx + 1) / min(len(combinations_to_test), 25))
    
    combo_progress.empty()
    
    combination_results.sort(key=lambda x: x["improvement"], reverse=True)
    drill_results["combinations"] = combination_results
    
    # Also set best_combinations for the queueing logic
    drill_results["best_combinations"] = [
        {
            "features": combo.get("name", "").split(" + ") if " + " in combo.get("name", "") else [combo.get("name", "")],
            "filters": combo.get("filters", []),
            "added_filters": combo.get("added_filters", []),
            "test_roi": combo.get("test_roi", 0),
            "gates_passed": combo.get("gates_passed", False),
        }
        for combo in combination_results[:5]
        if combo.get("improvement", 0) > 0
    ]
    
    improving_combos = [c for c in combination_results if c["improvement"] > 0]
    
    if improving_combos:
        st.markdown(f"🎯 **{len(improving_combos)} combinations IMPROVE over baseline:**")
        for combo in improving_combos[:10]:
            status_icon = "✅" if combo["gates_passed"] else "⚠️"
            st.markdown(
                f"- **{combo['name']}**: Test {combo['test_roi']:.2%} "
                f"(**+{combo['improvement']*100:.1f}%**) "
                f"[{combo['test_rows']} rows] {status_icon}"
            )
    else:
        st.warning("No combinations improved over individual filters")
    
    # ===== PHASE 6: Test 3-Way Combinations =====
    if improving_combos and len(improving_combos) >= 2:
        st.markdown("**Phase 6: Testing 3-Way Combinations**")
        
        best_combo = improving_combos[0]
        three_way_results = []
        
        for individual in improving_filters[:5]:
            individual_col = individual["filter"].get("col")
            combo_cols = [f.get("col") for f in best_combo["added_filters"]]
            
            if individual_col in combo_cols:
                continue
            
            try:
                full_filters = best_combo["filters"] + [individual["filter"]]
                
                result = _run_tool("test_filter", {
                    "filters": full_filters,
                    "pl_column": pl_column,
                    "enforcement": gates,
                })
                
                if result and not result.get("error"):
                    test_roi = result.get("test", {}).get("roi", 0)
                    test_rows = result.get("test", {}).get("rows", 0)
                    
                    if test_rows >= 30:
                        three_way_results.append({
                            "name": f"{best_combo['name']} + {individual['display']}",
                            "filters": full_filters,
                            "test_roi": test_roi,
                            "test_rows": test_rows,
                            "improvement": test_roi - base_test_roi,
                            "gates_passed": result.get("gates_passed", False),
                        })
            except:
                continue
        
        if three_way_results:
            three_way_results.sort(key=lambda x: x["improvement"], reverse=True)
            st.markdown("**Top 3-way combinations:**")
            for tw in three_way_results[:3]:
                status_icon = "✅" if tw["gates_passed"] else "⚠️"
                st.markdown(
                    f"- **{tw['name']}**: Test {tw['test_roi']:.2%} "
                    f"(**+{tw['improvement']*100:.1f}%**) [{tw['test_rows']} rows] {status_icon}"
                )
            
            combination_results.extend(three_way_results)
    
    # ===== PHASE 7: Final Recommendations (with COMPLEXITY SCORING) =====
    st.markdown("**Phase 7: Final Recommendations (Complexity-Adjusted)**")
    
    # Calculate adjusted scores for all results
    all_results = improving_filters + combination_results
    
    for result in all_results:
        # Count filters in this result
        if "filters" in result:
            num_filters = _count_filters(result["filters"])
        elif "filter" in result:
            num_filters = base_filter_count + 1  # Base + this one filter
        else:
            num_filters = base_filter_count
        
        # Calculate adjusted score
        train_val_gap = abs(result.get("train_roi", 0) - result.get("val_roi", 0))
        result["num_filters"] = num_filters
        result["adjusted_score"] = _calculate_adjusted_score(
            result.get("test_roi", 0), 
            num_filters, 
            train_val_gap
        )
        result["complexity_penalty"] = (max(0, num_filters - MIN_FILTERS_NO_PENALTY) * COMPLEXITY_PENALTY_PER_FILTER)
    
    # Sort by ADJUSTED score (not raw improvement) - this prefers simpler strategies!
    all_results.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
    
    if all_results:
        best = all_results[0]
        
        if best.get("improvement", 0) > 0 or best.get("adjusted_score", 0) > base_adjusted_score:
            # Show both raw and adjusted
            st.success(
                f"🏆 **Best Result (Complexity-Adjusted):** {best.get('name', best.get('display', 'Unknown'))} "
                f"with Test ROI **{best.get('test_roi', 0):.2%}** "
                f"(Adjusted: {best.get('adjusted_score', 0):.2%})"
            )
            st.caption(
                f"📐 {best.get('num_filters', '?')} filters | "
                f"Complexity penalty: -{best.get('complexity_penalty', 0)*100:.1f}% | "
                f"Raw improvement: +{best.get('improvement', 0)*100:.1f}%"
            )
            
            # Also show best by raw ROI if different
            raw_sorted = sorted(all_results, key=lambda x: x.get("test_roi", 0), reverse=True)
            if raw_sorted and raw_sorted[0] != best:
                raw_best = raw_sorted[0]
                st.info(
                    f"📈 **Highest raw ROI:** {raw_best.get('name', raw_best.get('display', 'Unknown'))} "
                    f"Test {raw_best.get('test_roi', 0):.2%} "
                    f"({raw_best.get('num_filters', '?')} filters, penalty: -{raw_best.get('complexity_penalty', 0)*100:.1f}%)"
                )
            
            # Show strategies passing gates
            passing_gates = [r for r in all_results if r.get("gates_passed") and r.get("adjusted_score", 0) > 0]
            if passing_gates and passing_gates[0] != best:
                best_passing = passing_gates[0]
                st.info(
                    f"✅ **Best passing all gates:** {best_passing.get('name', best_passing.get('display', 'Unknown'))} "
                    f"Adjusted: {best_passing.get('adjusted_score', 0):.2%} ({best_passing.get('num_filters', '?')} filters)"
                )
            
            # Add to recommendations (prefer simpler strategies WITH SUFFICIENT SAMPLE SIZE)
            MIN_TEST_ROWS_FOR_QUEUE = 50  # Don't queue strategies with tiny samples
            
            for result in all_results[:5]:
                test_rows = result.get("test_rows", 0)
                
                # Skip if sample too small (likely noise)
                if test_rows < MIN_TEST_ROWS_FOR_QUEUE:
                    _log(f"Skipping recommendation {result.get('name', '?')}: only {test_rows} test rows (min: {MIN_TEST_ROWS_FOR_QUEUE})")
                    continue
                    
                if result.get("adjusted_score", 0) > 0:
                    if "filters" in result and "added_filters" in result:
                        drill_results["recommended_additions"].append({
                            "filters": result.get("added_filters", result["filters"]),
                            "name": result.get("name", "Unknown"),
                            "test_roi": result.get("test_roi", 0),
                            "test_rows": test_rows,
                            "adjusted_score": result.get("adjusted_score", 0),
                            "improvement": result.get("improvement", 0),
                            "num_filters": result.get("num_filters", 0),
                            "gates_passed": result.get("gates_passed", False),
                        })
                    elif "filter" in result:
                        drill_results["recommended_additions"].append({
                            "filter": result["filter"],
                            "name": result.get("display", "Unknown"),
                            "test_roi": result.get("test_roi", 0),
                            "test_rows": test_rows,
                            "adjusted_score": result.get("adjusted_score", 0),
                            "improvement": result.get("improvement", 0),
                            "num_filters": result.get("num_filters", 0),
                            "gates_passed": result.get("gates_passed", False),
                        })
        else:
            st.info("📊 No improvements found - base strategy may already be optimal")
    
    # Show hypothesis count update
    final_hypothesis_count = st.session_state.get("hypothesis_count", 0)
    tests_this_drill = final_hypothesis_count - hypothesis_count
    
    st.markdown("---")
    st.markdown(f"""**Summary:**
- Individual tests: {len(all_tests)}
- Improving filters: {len(improving_filters)}
- Combinations tested: {len(combination_results)}
- Improving combinations: {len(improving_combos) if 'improving_combos' in dir() else 0}
- **Tests run in this drill down:** {tests_this_drill}
- **Total hypotheses tested this session:** {final_hypothesis_count}
""")
    
    # Warn if many hypotheses tested (multiple testing concern)
    if final_hypothesis_count > 100:
        st.warning(
            f"⚠️ **Multiple Testing Warning:** {final_hypothesis_count} hypotheses tested. "
            f"Consider that ~{int(final_hypothesis_count * 0.05)} could appear significant by chance alone (5% false discovery)."
        )
    
    _log(f"Deep drill down complete: {len(all_tests)} individual, {len(combination_results)} combos, {tests_this_drill} total tests")
    
    return drill_results

def _analyze_avenue_results(avenue: Dict, results: List[Dict], bible: Dict) -> Dict:
    """Deep analysis of results from exploring an avenue."""
    if not results:
        return {"summary": "No results", "best_variation": None, "recommendation": "skip"}
    
    all_results = []
    high_roi_insufficient_sample = []
    
    for r in results:
        train = r.get("train", {})
        val = r.get("val", {})
        test = r.get("test", {})
        gates_passed = r.get("gates_passed", False)
        gate_failures = r.get("gate_failures", [])
        
        train_roi = train.get("roi", 0)
        val_roi = val.get("roi", 0)
        test_roi = test.get("roi", 0)
        train_rows = train.get("rows", 0)
        val_rows = val.get("rows", 0)
        test_rows = test.get("rows", 0)
        
        train_val_gap = abs(train_roi - val_roi)
        
        # Quality checks
        train_positive = train_roi > 0
        val_positive = val_roi > -0.01
        test_positive = test_roi > 0
        
        # Check for high ROI but small sample
        sample_size_failure = train_rows < 300 or val_rows < 60 or test_rows < 60
        if test_roi > 0.05 and not gates_passed and sample_size_failure:
            high_roi_insufficient_sample.append({
                "variation": r.get("variation", "unknown"),
                "filters": r.get("filters_tested", []),
                "train_roi": train_roi, "train_rows": train_rows,
                "val_roi": val_roi, "val_rows": val_rows,
                "test_roi": test_roi, "test_rows": test_rows,
                "gate_failures": gate_failures,
            })
        
        # Quality score
        if train_positive and val_positive and test_positive:
            consistency_bonus = 1.0
        elif train_positive and test_positive:
            consistency_bonus = 0.6
        elif train_positive:
            consistency_bonus = 0.3
        else:
            consistency_bonus = 0.0
        
        gap_penalty = max(0, 1.0 - train_val_gap / 0.05)
        quality_score = test_roi * consistency_bonus * gap_penalty
        
        # Truly passing (stricter than gates)
        truly_passing = (
            gates_passed and 
            train_roi > 0 and
            val_roi > -0.02 and
            test_roi > 0 and
            train_val_gap < 0.06
        )
        
        all_results.append({
            "variation": r.get("variation", "unknown"),
            "filters": r.get("filters_tested", []),
            "train_roi": train_roi, "train_rows": train_rows,
            "val_roi": val_roi, "val_rows": val_rows,
            "test_roi": test_roi, "test_rows": test_rows,
            "train_val_gap": round(train_val_gap, 4),
            "quality_score": round(quality_score, 6),
            "gates_passed": gates_passed,
            "truly_passing": truly_passing,
            "gate_failures": gate_failures,
        })
    
    # Find best results
    truly_passing = [r for r in all_results if r["truly_passing"]]
    near_misses = [r for r in all_results if r["train_roi"] > 0.01 and r["test_roi"] > 0 and not r["truly_passing"]]
    interesting = [r for r in all_results if r["train_roi"] > 0.02]
    
    # Sort by quality
    all_results.sort(key=lambda x: x["quality_score"], reverse=True)
    
    # Summary
    summary = f"Tested {len(results)} variations: {len(truly_passing)} truly passing, {len(near_misses)} near-misses, {len(interesting)} interesting"
    
    best = all_results[0] if all_results else None
    
    # Recommendation
    if truly_passing:
        recommendation = "SUCCESS - found strategy with positive train/val/test!"
    elif near_misses:
        recommendation = "PROMISING - positive train+test but needs refinement"
    elif interesting:
        recommendation = "INVESTIGATE - positive train, explore further"
    else:
        recommendation = "SKIP - no promising signals"
    
    return {
        "summary": summary,
        "best_variation": best,
        "best_test_roi": best["test_roi"] if best else 0,
        "truly_passing": truly_passing,
        "near_misses": near_misses,
        "high_roi_insufficient_sample": high_roi_insufficient_sample,
        "recommendation": recommendation,
        "all_results_sorted": all_results[:10],
    }

# ============================================================
# Deep Analysis Functions (from v5)
# ============================================================

def _deep_analyze_iteration(avenue: Dict, avenue_results: Dict, accumulated_learnings: List, avenues_remaining: int) -> Dict:
    """LLM deep analysis of iteration results using WORLD-CLASS QUANT thinking."""
    
    # Build rich context with ALL accumulated knowledge
    learnings_summary = "\n".join([f"- Learning #{i+1}: {l.get('learning', '')}" for i, l in enumerate(accumulated_learnings[-10:])])
    
    context = f"""## ITERATION ANALYSIS

### Avenue Explored
{_safe_json(avenue, 600)}

### Results Summary
{_safe_json(avenue_results.get('analysis', {}), 2500)}

### Detailed Results (Top 5)
{_safe_json(avenue_results.get('analysis', {}).get('all_results_sorted', [])[:5], 2000)}

---

## ACCUMULATED KNOWLEDGE (CONNECT TO THIS)

### Past Learnings ({len(accumulated_learnings)} total)
{learnings_summary}

### Avenues Remaining: {avenues_remaining}

---

## PATTERNS TO LOOK FOR
- Did DRIFT direction matter?
- Did sample size affect reliability?
- Is there a sweet spot for ODDS ranges?
- Do certain MODE/MARKET combos consistently work?
"""

    question = """## YOUR TASK: DEEP ITERATION ANALYSIS

You are a SENIOR QUANT. Use the DEEP THINKING CYCLE to analyze these results.

### 1. OBSERVE (What happened in this test?)
- What were the key metrics (train/val/test ROI, sample sizes)?
- Did the strategy pass gates? Which ones failed?
- Any surprising results (positive or negative)?
- How does test ROI compare to train ROI (overfitting check)?

### 2. CONNECT (Link to accumulated learnings)
- Does this confirm any of the past learnings listed above?
- Does this contradict anything we thought we knew?
- Have we seen similar patterns in other tests?
- Is there a meta-pattern emerging?

### 3. HYPOTHESIZE (Why did we get these results?)
- If positive: What market inefficiency might explain this edge?
- If negative: Why might this hypothesis have been wrong?
- Is this result likely real or noise (consider sample size, p-value)?

### 4. PLAN (What should we do next?)
- Should we continue refining this avenue or move on?
- If continue: What specific refinements would help?
- If move on: What did we learn to apply elsewhere?
- Generate 3-5 specific next actions with expected value scores

### 5. DECIDE (Best next step)
- What's the single best action after this result?
- Full justification for the decision
- What would change your mind?

### 6. RECORD (Key insight to remember)
- What's the ONE key learning from this iteration?
- Should this be saved as a permanent learning?

Respond with JSON:
{
    "situation_assessment": "2 paragraphs: What happened? What does it mean?",
    
    "observation": {
        "key_metrics": {
            "train_roi": 0.00,
            "val_roi": 0.00,
            "test_roi": 0.00,
            "gates_passed": true,
            "sample_sufficient": true
        },
        "surprises": ["Surprise 1", "Surprise 2"],
        "overfitting_risk": "low/medium/high"
    },
    
    "connections": [
        {"learning_ref": "Learning #X", "connection": "This confirms/contradicts because..."},
        {"pattern": "Emerging pattern", "evidence": "This is the Nth time we've seen..."}
    ],
    
    "hypothesis_verdict": "supported/refuted/inconclusive",
    "hypothesis_reasoning": "Why the original hypothesis was right/wrong",
    "market_inefficiency_explanation": "If positive, what inefficiency explains this?",
    
    "refinement_ideas": [
        {
            "filter_to_add": {"col": "X", "op": "=", "value": "Y"},
            "reasoning": "Why this refinement might help",
            "expected_improvement": "What we expect to see"
        }
    ],
    
    "candidate_next_actions": [
        {
            "action": "Action description",
            "tool": "tool_name",
            "expected_value_score": 8,
            "reasoning": "Why this action"
        }
    ],
    
    "decision": {
        "should_continue_avenue": true,
        "next_action": "Specific action to take",
        "justification": "Full reasoning for this decision",
        "what_would_change_mind": "If we see X, we'd pivot to Y"
    },
    
    "key_learning": "One sentence: the key insight to remember",
    "save_as_permanent_learning": true,
    "learning_category": "drift_patterns/odds_patterns/mode_patterns/market_patterns",
    
    "confidence_in_direction": "high/medium/low",
    "next_recommendation": "Specific recommendation for next step"
}"""

    resp = _llm(context, question, max_tokens=3000)
    parsed = _parse_json(resp)
    
    if not parsed:
        return {
            "situation_assessment": resp[:500] if resp else "Analysis failed",
            "key_learning": "",
            "decision": {"should_continue_avenue": False, "next_action": "move to next avenue"}
        }
    
    # Backward compatibility
    if not parsed.get("detailed_reasoning"):
        parsed["detailed_reasoning"] = parsed.get("situation_assessment", "")
    if not parsed.get("should_continue_avenue"):
        parsed["should_continue_avenue"] = parsed.get("decision", {}).get("should_continue_avenue", False)
    
    return parsed


def _ai_decide_next_action(accumulated_learnings: List, strategies_found: List, 
                           avenues_explored: List, avenues_remaining: List,
                           near_misses: List, pl_column: str, iteration: int) -> Dict:
    """AI DECIDES what to do next - the core of the autonomous loop."""
    
    # Build comprehensive context
    learnings_summary = "\n".join([f"- #{i+1}: {l.get('learning', '')}" for i, l in enumerate(accumulated_learnings[-15:])])
    strategies_summary = "\n".join([f"- {s.get('result', {}).get('test_roi', 0):.2%} ROI: {s.get('filters', [])}" for s in strategies_found[-5:]])
    explored_summary = "\n".join([f"- {a.get('avenue', '')}: {a.get('result_summary', 'tested')}" for a in avenues_explored[-10:]])
    remaining_summary = "\n".join([f"- {a.get('avenue', '')}: {a.get('market_inefficiency_hypothesis', '')[:50]}" for a in avenues_remaining[:8]])
    near_miss_summary = "\n".join([f"- {nm.get('avenue', '')}: {nm.get('test_roi', 0):.2%} (gate failures: {nm.get('gate_failures', [])})" for nm in near_misses[-5:]])
    
    context = f"""## AUTONOMOUS RESEARCH STATE

### Current Position
- Iteration: {iteration}
- Target: {pl_column}
- Strategies found: {len(strategies_found)}
- Avenues explored: {len(avenues_explored)}
- Avenues remaining: {len(avenues_remaining)}
- Near misses: {len(near_misses)}

### Accumulated Learnings ({len(accumulated_learnings)} total)
{learnings_summary if learnings_summary else "No learnings yet"}

### Strategies Found
{strategies_summary if strategies_summary else "No strategies found yet"}

### Recently Explored
{explored_summary if explored_summary else "Nothing explored yet"}

### Remaining Avenues
{remaining_summary if remaining_summary else "No avenues queued"}

### Near Misses (worth refining)
{near_miss_summary if near_miss_summary else "No near misses"}

---

## AVAILABLE TOOLS
- **query_data**: Explore data distributions
- **bracket_sweep**: Test numeric ranges
- **subgroup_scan**: Test categorical combinations
- **test_filter**: Validate specific filter
- **forward_walk**: Walk-forward validation
- **monte_carlo_sim**: Confidence intervals
- **train_catboost**: ML feature discovery
- **train_logistic**: Interpretable ML
- **shap_explain**: Convert ML to filters
- **statistical_significance**: P-value calculation
- **time_decay_analysis**: Check alpha decay
"""

    question = """## YOUR TASK: DECIDE WHAT TO DO NEXT

You are a SENIOR QUANT RESEARCHER. Think deeply about the optimal next action.

### DECISION FRAMEWORK

**1. ASSESS CURRENT STATE**
- What have we learned so far?
- What's working? What's not?
- Are we exploiting (refining promising) or exploring (trying new)?

**2. CONSIDER OPTIONS**
Generate 5-8 candidate actions. For EACH:
- What tool/action?
- What's the expected outcome?
- What's the probability of finding an edge?
- What will we learn even if it fails?
- Expected Value Score (1-10)

**3. BALANCE PRIORITIES**
- Exploit near-misses (high probability, incremental gain)
- Explore new territory (lower probability, potentially bigger edge)
- Fill knowledge gaps (what don't we know yet?)
- Avoid what's been tried and failed

**4. MAKE THE DECISION**
- Choose the single best action
- Provide FULL justification
- State what would change your mind

Respond with JSON:
{
    "current_state_assessment": "2 paragraphs on where we are in the research",
    
    "what_we_know": [
        "Key learning 1",
        "Key learning 2"
    ],
    
    "knowledge_gaps": [
        "What we don't know yet 1",
        "What we should explore 2"
    ],
    
    "candidate_actions": [
        {
            "rank": 1,
            "action": "Specific action description",
            "tool": "tool_name",
            "params": {"key": "value"},
            "expected_outcome": "What we expect to find",
            "probability_of_success": 0.6,
            "potential_edge": "high/medium/low",
            "learning_value": "What we learn if it fails",
            "expected_value_score": 8,
            "reasoning": "Full reasoning for this action"
        }
    ],
    
    "exploit_vs_explore": {
        "recommendation": "exploit/explore/balanced",
        "reasoning": "Why this balance is right now"
    },
    
    "decision": {
        "chosen_action": "The action to take",
        "tool": "tool_name",
        "params": {"detailed": "parameters"},
        "justification": "3-4 sentences on why this is the best choice",
        "expected_outcome": "What we expect",
        "what_would_change_mind": "If we see X, we'd pivot to Y",
        "next_3_steps": ["Step 1", "Step 2", "Step 3"]
    },
    
    "if_this_fails": {
        "fallback_action": "What to do if chosen action doesn't work",
        "reasoning": "Why this is the right fallback"
    }
}"""

    resp = _llm(context, question, max_tokens=3500)
    parsed = _parse_json(resp)
    
    if not parsed:
        # Fallback: continue with remaining avenues or generate new ones
        return {
            "decision": {
                "chosen_action": "continue_exploration",
                "tool": "test_filter" if avenues_remaining else "subgroup_scan",
                "justification": "Fallback: LLM decision failed, continuing with default exploration"
            }
        }
    
    return parsed

def _deep_reflect(phase: str, findings: Dict, accumulated_learnings: List, avenues_remaining: List) -> Dict:
    """Deep reflection on overall progress."""
    context = f"""## Current Phase: {phase}

## Recent Findings
{_safe_json(findings, 3000)}

## Accumulated Learnings
{_safe_json(accumulated_learnings[-10:], 1500)}

## Avenues Remaining
{_safe_json(avenues_remaining[:8], 1000)}
"""

    question = """Reflect DEEPLY on the current state.

1. What have we learned from the latest tests?
2. What patterns are emerging?
3. Should we continue with current approach or pivot?
4. What's the most promising next step?
5. Are there any avenues we should skip or prioritize?

DO NOT recommend giving up unless we've truly exhausted all options.

Respond with JSON:
{
    "reflection": "Deep analysis of current state",
    "key_learnings": ["learning 1", "learning 2"],
    "emerging_patterns": ["pattern 1", "pattern 2"],
    "next_action": "continue|pivot|refine_promising|try_combinations",
    "priority_avenue": "which avenue to try next and why",
    "avenues_to_skip": ["avenue to skip and why"],
    "should_give_up": false,
    "give_up_reason": "only if should_give_up is true"
}"""

    resp = _llm(context, question)
    parsed = _parse_json(resp)
    return parsed if parsed else {"reflection": resp[:500], "next_action": "continue", "should_give_up": False}

# ============================================================
# Combination Scanning (from v5)
# ============================================================

def _run_combination_scan_for_base(base_filters: List[Dict], pl_column: str) -> Dict:
    """When a base filter shows promise, scan for best combinations."""
    _log(f"Running combination scan for: {base_filters}")
    return _run_tool("combination_scan", {
        "pl_column": pl_column,
        "base_filters": base_filters,
        "scan_cols": ["MODE", "MARKET", "DRIFT IN / OUT", "LEAGUE"],
        "max_combinations": 30,
    })

# ============================================================
# Validation Functions (v6 addition)
# ============================================================

def _validate_strategy(filters: List[Dict], pl_column: str) -> Dict:
    """Run full validation on a promising strategy."""
    _log(f"Validating strategy: {filters}")
    
    validation = {"filters": filters, "pl_column": pl_column}
    
    # Forward walk
    fw_result = _run_tool("forward_walk", {"filters": filters, "pl_column": pl_column, "n_windows": 6})
    validation["forward_walk"] = fw_result
    
    # Monte Carlo
    mc_result = _run_tool("monte_carlo_sim", {"filters": filters, "pl_column": pl_column, "n_simulations": 500})
    validation["monte_carlo"] = mc_result
    
    # Statistical significance
    sig_result = _run_tool("statistical_significance", {"filters": filters, "pl_column": pl_column})
    validation["statistical_significance"] = sig_result
    
    # Time decay
    decay_result = _run_tool("time_decay_analysis", {"filters": filters, "pl_column": pl_column})
    validation["time_decay"] = decay_result
    
    # Overall verdict
    fw_positive = (fw_result.get("windows_positive", 0) / max(fw_result.get("windows_total", 1), 1)) >= 0.6
    mc_positive = mc_result.get("ci_95_lower", -1) > 0
    sig_positive = sig_result.get("p_value", 1) < 0.10
    decay_ok = decay_result.get("verdict") != "DECAYING"
    
    validation["overall_verdict"] = {
        "forward_walk_pass": fw_positive,
        "monte_carlo_pass": mc_positive,
        "significance_pass": sig_positive,
        "decay_ok": decay_ok,
        "recommend_validate": fw_positive and mc_positive and sig_positive and decay_ok,
    }
    
    return validation

# ============================================================
# Checkpoint Functions (v6 addition)
# ============================================================

def _save_checkpoint():
    """Save current state to Supabase."""
    if not st.session_state.session_id:
        result = _run_tool("create_session", {"pl_column": st.session_state.target_pl_column})
        if result.get("created"):
            st.session_state.session_id = result.get("session_id")
    
    if not st.session_state.session_id:
        return
    
    state = {
        "iteration": st.session_state.agent_iteration,
        "phase": st.session_state.agent_phase,
        "avenues_to_explore": st.session_state.avenues_to_explore,
        "avenues_explored": st.session_state.avenues_explored,
        "strategies_found": st.session_state.strategies_found,
        "accumulated_learnings": st.session_state.accumulated_learnings,
        "near_misses": st.session_state.near_misses,
    }
    
    result = _run_tool("save_checkpoint", {
        "session_id": st.session_state.session_id,
        "iteration": st.session_state.agent_iteration,
        "phase": st.session_state.agent_phase,
        "state": state,
    })
    
    if result.get("saved"):
        st.session_state.last_checkpoint = datetime.utcnow().isoformat()
        _log(f"Checkpoint saved at iteration {st.session_state.agent_iteration}")

def _restore_checkpoint(session_id: str) -> bool:
    """Restore state from checkpoint."""
    result = _run_tool("load_checkpoint", {"session_id": session_id})
    
    if not result.get("found"):
        return False
    
    state = result.get("state", {})
    st.session_state.session_id = session_id
    st.session_state.agent_iteration = state.get("iteration", 0)
    st.session_state.agent_phase = state.get("phase", "idle")
    st.session_state.avenues_to_explore = state.get("avenues_to_explore", [])
    st.session_state.avenues_explored = state.get("avenues_explored", [])
    st.session_state.strategies_found = state.get("strategies_found", [])
    st.session_state.accumulated_learnings = state.get("accumulated_learnings", [])
    st.session_state.near_misses = state.get("near_misses", [])
    
    _log(f"Restored from checkpoint: iteration {st.session_state.agent_iteration}")
    return True

# ============================================================
# Main Agent (combined v5 + v6)
# ============================================================

def run_agent():
    """Main research agent loop."""
    if "target_pl_column" not in st.session_state:
        st.error("No target PL column selected.")
        return
    
    pl_column = st.session_state.target_pl_column
    
    # Initialize state
    if "log" not in st.session_state:
        st.session_state.log = []
    if "tested_filter_hashes" not in st.session_state:
        st.session_state.tested_filter_hashes = set()
    if "near_misses" not in st.session_state:
        st.session_state.near_misses = []
    if "accumulated_learnings" not in st.session_state:
        st.session_state.accumulated_learnings = []
    if "avenues_explored" not in st.session_state:
        st.session_state.avenues_explored = []
    if "strategies_found" not in st.session_state:
        st.session_state.strategies_found = []
    
    # Reset for new run (but DON'T reset research_log - we want to keep it!)
    st.session_state.log = []
    st.session_state.tested_filter_hashes = set()
    st.session_state.near_misses = []
    st.session_state.accumulated_learnings = []
    st.session_state.avenues_explored = []
    st.session_state.strategies_found = []
    
    # Initialize research log if not exists (don't reset it!)
    if "research_log" not in st.session_state:
        st.session_state.research_log = []
    
    # Log the start
    _research_log(f"Starting research for {pl_column}", "header")
    
    st.markdown(f"# 🤖 Research Agent v6: {pl_column}")
    st.caption("Deep Analysis Edition + Local Compute + 38 Tools + Persistence")
    
    # ========== PERSISTENT SUMMARY (always visible at top) ==========
    summary_container = st.container()
    with summary_container:
        st.markdown("---")
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        strategies_placeholder = sum_col1.empty()
        iteration_placeholder = sum_col2.empty()
        avenues_placeholder = sum_col3.empty()
        status_placeholder = sum_col4.empty()
        
        # Initialize placeholders
        strategies_placeholder.metric("📊 Strategies", 0)
        iteration_placeholder.metric("🔄 Iteration", 0)
        avenues_placeholder.metric("🎯 Avenues Left", "?")
        status_placeholder.metric("📍 Status", "Starting")
        st.markdown("---")
    
    # Store placeholders in session state for updates
    st.session_state._summary_placeholders = {
        "strategies": strategies_placeholder,
        "iteration": iteration_placeholder,
        "avenues": avenues_placeholder,
        "status": status_placeholder,
    }
    
    # ========== BIBLE ==========
    with st.status("📖 Loading Bible...", expanded=True) as status:
        bible = _load_bible()
        bible_source = bible.get("_source", "unknown")
        
        # Show source warning
        if bible_source == "google_sheets_direct":
            st.success(f"✅ Bible loaded from Google Sheets ({len(bible.get('research_rules', []))} rules, {len(bible.get('column_definitions', []))} column defs)")
            _research_log(f"Bible loaded from Google Sheets ({len(bible.get('research_rules', []))} rules)", "success")
        elif bible_source == "job_queue":
            st.warning("⚠️ Bible loaded from job queue (local_compute) - NOT Google Sheets!")
            _research_log("Bible loaded from job queue (NOT Google Sheets)", "warning")
        elif bible_source == "fallback":
            st.error("❌ Bible using FALLBACK - Google Sheets NOT connected! Check credentials.")
            _research_log("Bible using FALLBACK - check credentials!", "error")
        else:
            st.warning(f"⚠️ Bible source: {bible_source}")
            _research_log(f"Bible source: {bible_source}", "warning")
        
        st.markdown(_format_bible(bible))
        status.update(label="📖 Bible loaded", state="complete")
    
    with st.expander("📚 Full Bible Context", expanded=False):
        st.code(_safe_json(bible, 3000), language="json")
    
    _append("assistant", _format_bible(bible))
    
    # ========== EXPLORATION ==========
    _research_log("Phase 1: Exploration - Analyzing MODE, MARKET, DRIFT distributions...", "phase")
    with st.status("🔍 Phase 1: Exploration...", expanded=True) as status:
        progress = st.empty()
        exploration = _run_exploration(pl_column, progress)
        st.session_state.exploration_results = exploration
        status.update(label="🔍 Exploration complete", state="complete")
    
    _research_log("Exploration complete - analyzing results", "success")
    
    with st.expander("Exploration Results", expanded=False):
        st.code(_safe_json(exploration, 5000), language="json")
    
    # Check for pause
    if st.session_state.agent_phase == "paused":
        st.warning("⏸️ Research paused")
        _research_log("Research paused by user", "warning")
        return
    
    # ========== DEEP ANALYSIS ==========
    _research_log("Phase 1b: Deep Analysis - AI analyzing patterns...", "phase")
    st.markdown("### 🧠 Phase 1b: Deep Analysis")
    with st.status("🧠 Analyzing exploration...", expanded=True) as status:
        exploration_analysis = _analyze_exploration(bible, exploration, pl_column)
        st.session_state.exploration_analysis = exploration_analysis
        status.update(label="🧠 Analysis complete", state="complete")
    
    # Display analysis
    detailed_analysis = exploration_analysis.get('detailed_analysis', '')
    if detailed_analysis:
        st.markdown("#### 📝 Analysis Summary")
        st.markdown(detailed_analysis)
    
    key_obs = exploration_analysis.get("key_observations", [])
    if key_obs:
        st.markdown("#### 🔍 Key Observations")
        for obs in key_obs[:5]:
            st.markdown(f"- {obs}")
    
    mode_summary = exploration_analysis.get("mode_summary", {})
    if mode_summary:
        st.markdown("#### 📊 MODE Summary")
        st.markdown(f"- **Best MODE:** {mode_summary.get('best_mode', 'N/A')} - {mode_summary.get('best_mode_reason', '')}")
        st.markdown(f"- **Worst MODE:** {mode_summary.get('worst_mode', 'N/A')} - {mode_summary.get('worst_mode_reason', '')}")
    
    drift_summary = exploration_analysis.get("drift_summary", {})
    if drift_summary:
        st.markdown("#### 📈 DRIFT Summary")
        st.markdown(f"- **Best DRIFT:** {drift_summary.get('best_drift', 'N/A')} - {drift_summary.get('best_drift_reason', '')}")
    
    all_positive = exploration_analysis.get("all_positive_combinations", [])
    if all_positive:
        st.markdown(f"#### ✅ Found {len(all_positive)} Positive Combinations")
        for combo in all_positive[:8]:
            st.markdown(f"- **{combo.get('mode')} + {combo.get('market')}** → mean PL: `{combo.get('mean_pl', 0):.4f}`, best drift: **{combo.get('best_drift', '?')}**")
    
    avenues = exploration_analysis.get("prioritized_avenues", [])
    st.session_state.avenues_to_explore = avenues
    
    if avenues:
        st.markdown(f"#### 🎯 {len(avenues)} Prioritized Avenues")
        for av in avenues[:8]:
            st.markdown(f"**#{av.get('rank', '?')}: {av.get('avenue', '')}** (drift: {av.get('promising_drift', '?')}, confidence: {av.get('confidence', '?')})")
    
    with st.expander("📄 Full Analysis JSON", expanded=False):
        st.code(_safe_json(exploration_analysis, 6000), language="json")
    
    _add_learning(f"Exploration found {len(all_positive)} positive combinations, {len(avenues)} avenues")
    _save_checkpoint()
    
    # Check for pause
    if st.session_state.agent_phase == "paused":
        return
    
    # ========== SWEEPS ==========
    with st.status("🔬 Phase 2: Sweeps...", expanded=True) as status:
        progress = st.empty()
        sweeps = _run_sweeps(pl_column, bible, progress)
        st.session_state.sweep_results = sweeps
        top_brackets = len(sweeps.get("bracket_sweep", {}).get("top_brackets", []))
        top_subgroups = len(sweeps.get("subgroup_scan", {}).get("top_groups", []))
        status.update(label=f"🔬 Sweeps complete ({top_brackets}b, {top_subgroups}s)", state="complete")
    
    st.markdown("#### 📊 Sweep Results")
    st.markdown(f"Found **{top_brackets}** bracket patterns and **{top_subgroups}** subgroup patterns")
    
    # Show top findings
    top_brackets_list = sweeps.get("bracket_sweep", {}).get("top_brackets", [])
    if top_brackets_list:
        st.markdown("**Top Bracket Patterns:**")
        for i, br in enumerate(top_brackets_list[:3], 1):
            rule = br.get("rule", [])
            test_roi = br.get("test", {}).get("roi", 0)
            test_rows = br.get("test", {}).get("rows", 0)
            st.markdown(f"  {i}. `{rule}` → Test: {test_roi:.2%} ({test_rows} rows)")
    
    top_groups_list = sweeps.get("subgroup_scan", {}).get("top_groups", [])
    if top_groups_list:
        st.markdown("**Top Subgroup Patterns:**")
        for i, sg in enumerate(top_groups_list[:3], 1):
            group = sg.get("group", {})
            test_roi = sg.get("test", {}).get("roi", 0)
            test_rows = sg.get("test", {}).get("rows", 0)
            # Handle case where group might be string or dict
            if isinstance(group, dict):
                group_str = ", ".join([f"{k}={v}" for k, v in group.items() if v])
            else:
                group_str = str(group)
            st.markdown(f"  {i}. `{group_str}` → Test: {test_roi:.2%} ({test_rows} rows)")
    
    with st.expander("📄 Full Sweep Results", expanded=False):
        st.code(_safe_json(sweeps, 4000), language="json")
    
    _add_learning(f"Sweeps found {top_brackets} bracket patterns and {top_subgroups} subgroup patterns")
    _save_checkpoint()
    
    # Auto-generate avenues if LLM returned too few
    if len(avenues) < 3 and all_positive:
        st.markdown("#### 🔧 Auto-generating additional avenues...")
        for combo in all_positive:
            existing = [a.get("avenue", "") for a in avenues]
            avenue_name = f"MODE={combo.get('mode')}, MARKET={combo.get('market')}"
            if avenue_name in existing:
                continue
            avenues.append({
                "rank": len(avenues) + 1,
                "avenue": avenue_name,
                "base_filters": [
                    {"col": "MODE", "op": "=", "value": combo.get("mode")},
                    {"col": "MARKET", "op": "=", "value": combo.get("market")}
                ],
                "promising_drift": combo.get("best_drift", "?"),
                "market_inefficiency_hypothesis": f"Positive mean PL ({combo.get('mean_pl', 0):.4f})",
                "confidence": "medium",
            })
        st.session_state.avenues_to_explore = avenues
        st.markdown(f"Now have **{len(avenues)}** avenues to explore")
    
    # Check for pause
    if st.session_state.agent_phase == "paused":
        return
    
    # ========== AVENUE EXPLORATION ==========
    st.markdown("---")
    st.markdown("### 🧪 Phase 3: Avenue Exploration")
    st.markdown("*Testing each avenue with multiple variations*")
    
    avenues_remaining = list(avenues)
    success_found = False
    
    # Helper to update the persistent summary
    def _update_summary(status_text="Running"):
        placeholders = st.session_state.get("_summary_placeholders", {})
        if placeholders:
            try:
                placeholders.get("strategies", st.empty()).metric("📊 Strategies", len(st.session_state.strategies_found))
                placeholders.get("iteration", st.empty()).metric("🔄 Iteration", st.session_state.agent_iteration)
                placeholders.get("avenues", st.empty()).metric("🎯 Avenues Left", len(avenues_remaining))
                placeholders.get("status", st.empty()).metric("📍 Status", status_text)
            except Exception:
                pass  # Placeholders may have been cleared
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        # Update summary at start of each iteration
        _update_summary(f"Iter {iteration}")
        
        # Check for pause
        if st.session_state.agent_phase == "paused":
            _update_summary("Paused")
            st.warning("⏸️ Research paused")
            return
        
        st.session_state.agent_iteration = iteration
        
        # NEVER GIVE UP - Generate new avenues if exhausted
        if not avenues_remaining:
            st.markdown("#### 🔄 Generating New Research Directions...")
            
            # Strategy 1: Generate avenues from sweep results (subgroup patterns)
            if sweeps.get("subgroup_scan", {}).get("top_groups"):
                st.markdown("*Converting top subgroup patterns to avenues...*")
                for sg in sweeps.get("subgroup_scan", {}).get("top_groups", [])[:5]:
                    group = sg.get("group", [])
                    test_roi = sg.get("test", {}).get("roi", 0)
                    if test_roi > 0:
                        filters_desc = ", ".join([f"{g.get('col')}={g.get('value')}" for g in group if g.get('value')])
                        new_avenue = {
                            "avenue": f"Subgroup: {filters_desc}",
                            "base_filters": [{"col": g.get("col"), "op": "=", "value": g.get("value")} for g in group if g.get("value")],
                            "market_inefficiency_hypothesis": f"Subgroup scan found {test_roi:.2%} test ROI - exploring variations",
                            "source": "subgroup_scan",
                        }
                        if new_avenue not in st.session_state.avenues_explored:
                            avenues_remaining.append(new_avenue)
            
            # Strategy 2: Use ML tools to discover new patterns
            if not avenues_remaining:
                st.markdown("*Running ML discovery (CatBoost/SHAP)...*")
                ml_result = _run_tool("train_catboost", {"pl_column": pl_column})
                
                if ml_result and not ml_result.get("error"):
                    suggested = ml_result.get("suggested_filters", [])
                    feature_imp = ml_result.get("feature_importance", [])[:10]  # Get more, will filter
                    
                    # FILTER OUT BANNED COLUMNS from ML suggestions!
                    suggested = [sf for sf in suggested if not _is_banned_column(sf.get("col", ""))]
                    feature_imp = [fi for fi in feature_imp if not _is_banned_column(fi.get("feature", ""))]
                    
                    for sf in suggested[:3]:
                        new_avenue = {
                            "avenue": f"ML Discovery: {sf.get('col')} {sf.get('op')} {sf.get('value', sf.get('values', ''))}",
                            "base_filters": [sf],
                            "market_inefficiency_hypothesis": sf.get("reasoning", "ML model identified this pattern"),
                            "source": "ml_catboost",
                        }
                        if new_avenue not in st.session_state.avenues_explored:
                            avenues_remaining.append(new_avenue)
                    
                    # Also create avenues from top features (already filtered)
                    for fi in feature_imp[:5]:
                        feat = fi.get("feature")
                        if feat:
                            new_avenue = {
                                "avenue": f"Feature Exploration: {feat}",
                                "base_filters": [],
                                "explore_column": feat,
                                "market_inefficiency_hypothesis": f"ML ranked {feat} as important (score: {fi.get('importance', 0):.3f})",
                                "source": "ml_feature_importance",
                            }
                            if new_avenue not in st.session_state.avenues_explored:
                                avenues_remaining.append(new_avenue)
            
            # Strategy 3: Try SHAP explanations
            if not avenues_remaining:
                st.markdown("*Running SHAP analysis...*")
                shap_result = _run_tool("shap_explain", {"pl_column": pl_column})
                
                if shap_result and not shap_result.get("error"):
                    # FILTER OUT BANNED COLUMNS from SHAP suggestions!
                    shap_filters = [sf for sf in shap_result.get("suggested_filters", []) 
                                   if not _is_banned_column(sf.get("col", ""))]
                    
                    for sf in shap_filters[:3]:
                        new_avenue = {
                            "avenue": f"SHAP: {sf.get('col')} {sf.get('op')} {sf.get('value')}",
                            "base_filters": [sf],
                            "market_inefficiency_hypothesis": sf.get("reasoning", "SHAP identified this pattern"),
                            "source": "shap_explain",
                        }
                        if new_avenue not in st.session_state.avenues_explored:
                            avenues_remaining.append(new_avenue)
            
            # Strategy 4: Refine near-misses
            if not avenues_remaining and st.session_state.near_misses:
                st.markdown("*Refining near-miss strategies...*")
                for nm in st.session_state.near_misses[:3]:
                    new_avenue = {
                        "avenue": f"Near-Miss Refinement: {nm.get('avenue', 'Unknown')}",
                        "base_filters": nm.get("filters", []),
                        "market_inefficiency_hypothesis": f"Near-miss with {nm.get('test_roi', 0):.2%} test ROI - trying variations",
                        "source": "near_miss_refinement",
                    }
                    if new_avenue not in st.session_state.avenues_explored:
                        avenues_remaining.append(new_avenue)
            
            # Strategy 5: Try different PL columns (cross-market)
            if not avenues_remaining:
                other_pl_cols = [c for c in OUTCOME_COLUMNS if c != pl_column][:2]
                for other_pl in other_pl_cols:
                    new_avenue = {
                        "avenue": f"Cross-Market: {other_pl}",
                        "base_filters": [],
                        "explore_pl_column": other_pl,
                        "market_inefficiency_hypothesis": f"Exploring {other_pl} market for transferable insights",
                        "source": "cross_market",
                    }
                    if new_avenue not in st.session_state.avenues_explored:
                        avenues_remaining.append(new_avenue)
            
            # If still no avenues, run univariate scan for new ideas
            if not avenues_remaining:
                st.markdown("*Running univariate scan for new ideas...*")
                uni_result = _run_tool("univariate_scan", {"pl_column": pl_column})
                
                if uni_result and not uni_result.get("error"):
                    for bf in uni_result.get("best_filters", [])[:5]:
                        if bf.get("roi", 0) > 0:
                            new_avenue = {
                                "avenue": f"Univariate: {bf.get('col')} = {bf.get('value')}",
                                "base_filters": [{"col": bf.get("col"), "op": "=", "value": bf.get("value")}],
                                "market_inefficiency_hypothesis": f"Best single filter for {bf.get('col')}: {bf.get('roi'):.2%} ROI",
                                "source": "univariate_scan",
                            }
                            if new_avenue not in st.session_state.avenues_explored:
                                avenues_remaining.append(new_avenue)
            
            if avenues_remaining:
                st.success(f"✅ Generated {len(avenues_remaining)} new avenues to explore!")
            else:
                # Only stop if we've REALLY exhausted everything and found strategies
                if st.session_state.strategies_found:
                    st.success("✅ Research complete - found strategies and exhausted all avenues!")
                    break
                else:
                    st.warning("⚠️ Could not generate new avenues - consider manual guidance")
                    st.session_state.agent_phase = "paused"
                    return
        
        # Pick next avenue
        current_avenue = avenues_remaining.pop(0)
        st.session_state.avenues_explored.append(current_avenue)
        
        avenue_name = current_avenue.get('avenue', 'Unknown')
        _research_log(f"Iteration {iteration}: Testing {avenue_name}", "iteration")
        
        st.markdown(f"#### Iteration {iteration}: {avenue_name}")
        st.markdown(f"*{current_avenue.get('market_inefficiency_hypothesis', current_avenue.get('why_promising', ''))}*")
        
        # Explore avenue
        with st.status(f"Testing avenue {iteration}...", expanded=True) as status:
            avenue_results = _explore_avenue(current_avenue, pl_column, bible, exploration_analysis)
            status.update(label=f"Avenue {iteration} complete", state="complete")
        
        # Display results
        analysis = avenue_results.get("analysis", {})
        best_roi = analysis.get('best_test_roi', 0)
        recommendation = analysis.get('recommendation', 'N/A')
        
        _research_log(f"  → Best Test ROI: {best_roi:.2%}, Recommendation: {recommendation}", "info")
        
        st.markdown(f"**Results:** {analysis.get('summary', 'N/A')}")
        st.markdown(f"**Best Test ROI:** {best_roi:.4f}")
        st.markdown(f"**Recommendation:** {recommendation}")
        
        with st.expander(f"Avenue {iteration} Details", expanded=False):
            st.code(_safe_json(avenue_results, 3000), language="json")
        
        # Deep analysis
        st.markdown("**🧠 Analysis:**")
        iteration_analysis = _deep_analyze_iteration(
            current_avenue,
            avenue_results,
            st.session_state.accumulated_learnings,
            len(avenues_remaining)
        )
        
        detailed = iteration_analysis.get("detailed_reasoning", "")
        if detailed:
            st.markdown(detailed[:600] + "..." if len(detailed) > 600 else detailed)
        
        st.markdown(f"*Key learning: {iteration_analysis.get('key_learning', 'N/A')}*")
        _add_learning(iteration_analysis.get("key_learning", f"Iteration {iteration} complete"))
        
        # ====== AI DECIDES NEXT STEPS ======
        # Use AI's recommendations to adjust strategy
        
        # 1. If AI suggests refinements, add them as new avenues
        refinement_ideas = iteration_analysis.get("refinement_ideas", [])
        if refinement_ideas and iteration_analysis.get("should_continue_avenue", False):
            st.markdown("**🔧 AI suggests refinements:**")
            for idea in refinement_ideas[:3]:
                filter_to_add = idea.get("filter_to_add", {})
                reasoning = idea.get("reasoning", "")
                if filter_to_add:
                    new_avenue = {
                        "avenue": f"Refinement: {current_avenue.get('avenue', '')} + {filter_to_add.get('col')}={filter_to_add.get('value')}",
                        "base_filters": current_avenue.get("base_filters", []) + [filter_to_add],
                        "market_inefficiency_hypothesis": reasoning,
                        "source": "ai_refinement",
                    }
                    avenues_remaining.insert(0, new_avenue)  # Add to front of queue
                    st.markdown(f"- Added: {filter_to_add.get('col')}={filter_to_add.get('value')} ({reasoning[:50]}...)")
        
        # 2. Check AI's confidence and adjust
        confidence = iteration_analysis.get("confidence_in_direction", "medium")
        next_rec = iteration_analysis.get("next_recommendation", "")
        if next_rec:
            st.markdown(f"*AI recommends: {next_rec}*")
        
        # 3. If AI says don't continue with this avenue, remove similar ones
        if not iteration_analysis.get("should_continue_avenue", True):
            current_base = current_avenue.get("avenue", "").split(",")[0] if current_avenue.get("avenue") else ""
            if current_base:
                removed = 0
                new_remaining = []
                for av in avenues_remaining:
                    if current_base in av.get("avenue", ""):
                        removed += 1
                    else:
                        new_remaining.append(av)
                if removed > 0:
                    avenues_remaining[:] = new_remaining
                    st.markdown(f"*AI: Skipping {removed} similar avenues - this direction isn't promising*")
        
        # Handle results
        recommendation = analysis.get("recommendation", "").upper()
        truly_passing = analysis.get("truly_passing", [])
        
        if "SUCCESS" in recommendation and truly_passing:
            st.success("🎉 **Strategy Found!**")
            
            best_result = truly_passing[0]
            
            # LOG STRATEGY FOUND
            filter_desc = ", ".join([f"{f.get('col')}={f.get('value', f.get('min', ''))}" for f in best_result.get("filters", [])[:3]])
            _research_log(f"Test ROI: {best_result.get('test_roi', 0):.2%} | Filters: {filter_desc}", "strategy")
            
            # Display strategy
            st.markdown("#### 📋 Best Validated Strategy")
            st.markdown(f"**Filters:**")
            for f in best_result.get("filters", []):
                st.markdown(f"- {f.get('col')} {f.get('op')} {f.get('value', f.get('min', ''))}{'-' + str(f.get('max', '')) if f.get('max') else ''}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Train ROI", f"{best_result.get('train_roi', 0):.2%}")
            col2.metric("Val ROI", f"{best_result.get('val_roi', 0):.2%}")
            col3.metric("Test ROI", f"{best_result.get('test_roi', 0):.2%}")
            
            col1.metric("Train Rows", best_result.get("train_rows", 0))
            col2.metric("Val Rows", best_result.get("val_rows", 0))
            col3.metric("Test Rows", best_result.get("test_rows", 0))
            
            st.code(json.dumps(best_result.get("filters", []), indent=2), language="json")
            
            # Save to Supabase (use nested format that local_compute expects)
            save_result = _run_tool("save_strategy", {
                "filters": best_result.get("filters", []),
                "pl_column": pl_column,
                # Use nested format for compatibility with local_compute
                "train": {"roi": best_result.get("train_roi", 0), "rows": best_result.get("train_rows", 0)},
                "val": {"roi": best_result.get("val_roi", 0), "rows": best_result.get("val_rows", 0)},
                "test": {"roi": best_result.get("test_roi", 0), "rows": best_result.get("test_rows", 0)},
                # Also include flat values as fallback
                "train_roi": best_result.get("train_roi", 0),
                "train_rows": best_result.get("train_rows", 0),
                "val_roi": best_result.get("val_roi", 0),
                "val_rows": best_result.get("val_rows", 0),
                "test_roi": best_result.get("test_roi", 0),
                "test_rows": best_result.get("test_rows", 0),
                "hypothesis": current_avenue.get("market_inefficiency_hypothesis", ""),
                "status": "draft"
            })
            
            if save_result.get("error"):
                st.error(f"❌ Failed to save strategy: {save_result.get('error')}")
            elif save_result.get("saved") or save_result.get("filter_hash"):
                st.success(f"💾 Strategy saved! Hash: {save_result.get('filter_hash', '?')[:8]}...")
                st.session_state.strategies_found.append({
                    "filters": best_result.get("filters", []),
                    "result": best_result,
                    "filter_hash": save_result.get("filter_hash"),
                })
            
            # ===== VALIDATION OPTIONS =====
            with st.expander("🔬 Run Validation", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Standard Validation**")
                    if st.button(f"Validate Strategy {iteration}", key=f"validate_{iteration}"):
                        validation = _validate_strategy(best_result.get("filters", []), pl_column)
                        st.json(validation)
                        
                        if validation.get("overall_verdict", {}).get("recommend_validate"):
                            st.success("🎉 **STRATEGY VALIDATED!**")
                            _run_tool("promote_strategy", {"filter_hash": save_result.get("filter_hash", "")})
                        else:
                            st.warning("⚠️ Did not pass full validation")
                
                with col2:
                    st.markdown("**🔒 Final Holdout Test**")
                    st.caption("Tests on data NEVER seen during exploration")
                    if st.button(f"Final Holdout Test {iteration}", key=f"holdout_{iteration}"):
                        with st.spinner("Testing on final holdout (unseen data)..."):
                            holdout_result = _validate_on_final_holdout(
                                best_result.get("filters", []),
                                pl_column,
                                bible
                            )
                        
                        if holdout_result.get("skipped"):
                            st.info(f"Skipped: {holdout_result.get('reason')}")
                        elif holdout_result.get("error"):
                            st.error(f"Error: {holdout_result.get('error')}")
                        else:
                            # Show results
                            st.markdown(f"**Holdout ROI:** {holdout_result.get('holdout_roi', 0):.2%}")
                            st.markdown(f"**Holdout Rows:** {holdout_result.get('holdout_rows', 0)}")
                            st.markdown(f"**Adjusted Score:** {holdout_result.get('adjusted_score', 0):.2%}")
                            
                            if holdout_result.get("warning"):
                                st.warning(holdout_result["warning"])
                            
                            # Recommendation
                            if holdout_result.get("strong_pass"):
                                st.success(holdout_result.get("recommendation", "VALIDATED!"))
                                _run_tool("promote_strategy", {"filter_hash": save_result.get("filter_hash", "")})
                            elif holdout_result.get("passed"):
                                st.warning(holdout_result.get("recommendation", "MARGINAL"))
                            else:
                                st.error(holdout_result.get("recommendation", "FAILED"))
            
            # ====== DEEP DRILL DOWN ======
            # When we find a strategy, drill down to find optimal parameters!
            st.markdown("---")
            st.markdown("#### 🔬 Deep Drill Down: Finding Optimal Parameters")
            st.caption("Analyzing every feature to find what improves this strategy...")
            
            with st.spinner("Running deep drill down analysis..."):
                drill_results = _deep_drill_down(
                    best_result.get("filters", []),
                    pl_column,
                    bible
                )
            
            # Add recommended filters as new avenues to explore
            for rec in drill_results.get("recommended_additions", [])[:3]:
                # Extract the actual filter(s) from the recommendation
                rec_filter = rec.get("filter")  # Single filter case
                rec_filters = rec.get("filters", [])  # Multiple filters case
                rec_name = rec.get("name", "Unknown refinement")
                
                if rec_filter:
                    # Single filter - add to base filters
                    new_filters = best_result.get("filters", []) + [rec_filter]
                elif rec_filters:
                    # Multiple filters from combination
                    new_filters = best_result.get("filters", []) + rec_filters
                else:
                    continue  # Skip if no valid filters
                
                new_avenue = {
                    "avenue": f"Drill Down: {current_avenue.get('avenue', '')} + {rec_name}",
                    "base_filters": new_filters,
                    "market_inefficiency_hypothesis": f"Tested refinement showing {rec.get('test_roi', 0):.1%} test ROI",
                    "source": "deep_drill_down",
                    "expected_improvement": rec.get("improvement", 0),
                }
                avenues_remaining.insert(0, new_avenue)
                st.markdown(f"✅ Queued for testing: **{rec_name}** (expected +{rec.get('improvement', 0):.1%})")
            
            # Also save drill down results to session state for review
            if "drill_down_history" not in st.session_state:
                st.session_state.drill_down_history = []
            st.session_state.drill_down_history.append({
                "timestamp": datetime.now().isoformat(),
                "base_filters": best_result.get("filters", []),
                "improving_filters": [f for f in drill_results.get("individual_tests", []) if f.get("improvement", 0) > 0],
                "combinations": drill_results.get("combinations", []),
                "recommended": drill_results.get("recommended_additions", []),
            })
            
            # If we found good combinations, add those too
            for combo in drill_results.get("best_combinations", [])[:2]:
                if combo.get("gates_passed") and combo.get("test_roi", 0) > best_result.get("test_roi", 0):
                    combo_features = combo.get("features", [])
                    if isinstance(combo_features, str):
                        combo_features = [combo_features]
                    new_avenue = {
                        "avenue": f"Combo: {' + '.join(combo_features)}",
                        "base_filters": combo.get("filters", []),
                        "market_inefficiency_hypothesis": f"Combined top features show {combo['test_roi']:.1%} test ROI",
                        "source": "deep_drill_down_combo",
                    }
                    avenues_remaining.insert(0, new_avenue)
                    st.markdown(f"✅ Queued promising combination: **{' + '.join(combo_features)}** ({combo['test_roi']:.1%})")
            
            st.markdown("---")
            
            # Check if strong enough - but KEEP SEARCHING for more!
            is_strong = (
                best_result.get("train_roi", 0) > 0.01 and
                best_result.get("val_roi", 0) > -0.01 and
                best_result.get("test_roi", 0) > 0.01 and
                abs(best_result.get("train_roi", 0) - best_result.get("val_roi", 0)) < 0.03
            )
            
            if is_strong:
                st.success(f"🎯 Found STRONG strategy #{len(st.session_state.strategies_found)}!")
                st.info("💪 Continuing to search for MORE strategies...")
                # DON'T STOP - keep searching for more strategies!
            else:
                st.warning("📊 Strategy found but weak - continuing to explore...")
        
        # Show high ROI insufficient sample
        high_roi = analysis.get("high_roi_insufficient_sample", [])
        if high_roi:
            st.markdown("#### 🔍 High ROI But Insufficient Sample")
            for hr in high_roi[:3]:
                st.markdown(f"- **{hr.get('variation')}**: Test ROI {hr.get('test_roi', 0):.2%} ({hr.get('test_rows', 0)} rows)")
                st.markdown(f"  Gate failures: {hr.get('gate_failures', [])}")
        
        # Add near misses
        for nm in analysis.get("near_misses", []):
            st.session_state.near_misses.append({
                "iteration": iteration,
                "filters": nm.get("filters", []),
                "avenue": current_avenue.get("avenue"),
                "test_roi": nm.get("test_roi", 0),
            })
        
        # Save findings
        st.session_state.agent_findings.append({
            "iteration": iteration,
            "avenue": current_avenue,
            "results": avenue_results,
            "analysis": iteration_analysis
        })
        
        # Deep reflection every 3 iterations - AI reassesses strategy
        if iteration % 3 == 0 and avenues_remaining:
            st.markdown("**🤔 Mid-point Reflection...**")
            reflection = _deep_reflect(
                f"After iteration {iteration}",
                avenue_results,
                st.session_state.accumulated_learnings,
                avenues_remaining
            )
            st.markdown(f"*{reflection.get('reflection', '')[:300]}*")
            
            # ====== AI DECIDES OVERALL DIRECTION ======
            
            # 1. Handle avenues to skip
            avenues_to_skip = reflection.get("avenues_to_skip", [])
            if avenues_to_skip:
                st.markdown("**AI decides to skip:**")
                for skip_reason in avenues_to_skip[:3]:
                    st.markdown(f"- {skip_reason}")
                # Remove skipped avenues
                skip_keywords = [s.split()[0].lower() for s in avenues_to_skip if isinstance(s, str)]
                if skip_keywords:
                    new_remaining = [av for av in avenues_remaining 
                                    if not any(kw in av.get("avenue", "").lower() for kw in skip_keywords)]
                    skipped = len(avenues_remaining) - len(new_remaining)
                    if skipped > 0:
                        avenues_remaining[:] = new_remaining
                        st.markdown(f"*Removed {skipped} avenues*")
            
            # 2. Prioritize recommended avenue
            priority_avenue = reflection.get("priority_avenue", "")
            if priority_avenue and avenues_remaining:
                # Try to find and move to front
                priority_lower = priority_avenue.lower()
                for i, av in enumerate(avenues_remaining):
                    if priority_lower in av.get("avenue", "").lower():
                        if i > 0:
                            avenues_remaining.insert(0, avenues_remaining.pop(i))
                            st.markdown(f"*AI prioritized: {av.get('avenue', '')[:50]}*")
                        break
            
            # 3. Handle next_action recommendation
            next_action = reflection.get("next_action", "continue")
            if next_action == "pivot":
                st.warning("🔄 AI recommends pivoting to new approach")
                # Generate new avenues with ML tools
                ml_result = _run_tool("train_catboost", {"pl_column": pl_column})
                if ml_result and not ml_result.get("error"):
                    for sf in ml_result.get("suggested_filters", [])[:2]:
                        new_avenue = {
                            "avenue": f"ML Pivot: {sf.get('col')} {sf.get('op')} {sf.get('value')}",
                            "base_filters": [sf],
                            "market_inefficiency_hypothesis": "ML-driven pivot based on feature importance",
                            "source": "ai_pivot",
                        }
                        avenues_remaining.insert(0, new_avenue)
            elif next_action == "refine_promising":
                st.info("🎯 AI recommends refining promising strategies")
            elif next_action == "try_combinations":
                st.info("🔗 AI recommends testing filter combinations")
            
            # 4. Key learnings from reflection
            key_learnings = reflection.get("key_learnings", [])
            for kl in key_learnings[:3]:
                _add_learning(kl)
        
        _save_checkpoint()
        st.markdown("---")
    
    # ========== FINAL ANALYSIS ==========
    st.markdown("### 📋 Final Analysis")
    
    if st.session_state.strategies_found:
        st.success(f"✅ Found {len(st.session_state.strategies_found)} strategies!")
        for i, strat in enumerate(st.session_state.strategies_found, 1):
            st.markdown(f"**Strategy {i}:** Test ROI {strat['result'].get('test_roi', 0):.2%}")
            st.code(json.dumps(strat["filters"], indent=2), language="json")
    
    if st.session_state.near_misses:
        st.markdown(f"**{len(st.session_state.near_misses)} near-misses found** - may warrant investigation")
        for nm in st.session_state.near_misses[:5]:
            st.markdown(f"- {nm.get('avenue', 'Unknown')}: Test ROI {nm.get('test_roi', 0):.2%}")
    
    if st.session_state.accumulated_learnings:
        st.markdown("**Key Learnings:**")
        for learning in st.session_state.accumulated_learnings[-10:]:
            st.markdown(f"- {learning.get('learning', '')}")
    
    st.session_state.agent_phase = "complete"

# ============================================================
# Ask Agent Function (v6)
# ============================================================

def _ask_agent(question: str) -> str:
    """Answer user question about research progress."""
    context = f"""## Current Research State
- PL Column: {st.session_state.target_pl_column}
- Phase: {st.session_state.agent_phase}
- Iteration: {st.session_state.agent_iteration}
- Strategies found: {len(st.session_state.strategies_found)}
- Near misses: {len(st.session_state.near_misses)}
- Avenues explored: {len(st.session_state.avenues_explored)}
- Avenues remaining: {len(st.session_state.avenues_to_explore)}

## Recent Findings
{_safe_json(st.session_state.agent_findings[-2:] if st.session_state.agent_findings else {}, 2000)}

## Accumulated Learnings
{_safe_json(st.session_state.accumulated_learnings[-5:], 1000)}
"""

    question_prompt = f"""User question: {question}

Answer helpfully and concisely. Explain your reasoning if asked about decisions.
If the user gives a direction (e.g., "try BTTS more"), acknowledge and incorporate it.
"""

    return _llm(context, question_prompt, max_tokens=1000)

# ============================================================
# UI
# ============================================================

st.title("⚽ Football Research Agent v6")
st.caption("Deep Analysis Edition + Local Compute + 38 Tools + Persistence + GPT-5")

# Check server
server_status = _check_server()
if server_status.get("status") != "healthy":
    st.error("⚠️ Cannot connect to Supabase!")
    st.markdown("""
    Check your Streamlit secrets have:
    ```
    SUPABASE_URL = "https://xxx.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY = "xxx"
    DATA_STORAGE_BUCKET = "football-data"
    DATA_STORAGE_PATH = "football_ai_NNIA.csv"
    ```
    
    And run local_compute.py on your Mac:
    ```bash
    cd ~/football-agent-v6
    export $(grep -v '^#' .env | xargs)
    python3 local_compute.py
    ```
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Server status (Supabase + local worker)
    st.success(f"🟢 Supabase Connected")
    st.caption("Jobs queued to Supabase, run local_compute.py to process")
    
    # SESSION ID - Important for recovery!
    if st.session_state.session_id:
        st.divider()
        st.markdown("**📋 Session ID:**")
        st.code(st.session_state.session_id, language=None)
        st.caption("⚠️ SAVE THIS to resume if Streamlit reboots!")
    
    st.divider()
    
    # Market selection
    pl_col = st.selectbox("Target Market", OUTCOME_COLUMNS, index=0)
    st.session_state.target_pl_column = pl_col
    
    # Control buttons
    phase = st.session_state.agent_phase
    
    if phase == "idle":
        if st.button("🚀 Start Research", type="primary", use_container_width=True):
            st.session_state.agent_phase = "running"
            st.session_state.agent_iteration = 0
            st.session_state.agent_findings = []
            st.session_state.past_failures = []
            st.session_state.near_misses = []
            st.session_state.exploration_results = {}
            st.session_state.sweep_results = {}
            st.session_state.exploration_analysis = {}
            st.session_state.tested_filter_hashes = set()
            st.session_state.bible = None
            st.session_state.accumulated_learnings = []
            st.session_state.avenues_to_explore = []
            st.session_state.avenues_explored = []
            st.session_state.strategies_found = []
            st.session_state.session_id = None
            st.session_state.run_requested = True
            # Reset hypothesis tracking for new session
            st.session_state.hypothesis_count = 0
            st.session_state.tests_per_iteration = []
            st.session_state.final_holdout_tested = False
            st.session_state.strategies_pending_final_validation = []
            st.rerun()
    
    elif phase == "running":
        col1, col2 = st.columns(2)
        if col1.button("⏸️ Pause", use_container_width=True):
            st.session_state.agent_phase = "paused"
            st.rerun()
        if col2.button("🛑 Stop", use_container_width=True):
            st.session_state.agent_phase = "idle"
            st.rerun()
    
    elif phase == "paused":
        col1, col2 = st.columns(2)
        if col1.button("▶️ Resume", use_container_width=True):
            st.session_state.agent_phase = "running"
            st.session_state.run_requested = True
            st.rerun()
        if col2.button("🛑 Stop", use_container_width=True):
            st.session_state.agent_phase = "idle"
            st.rerun()
    
    elif phase == "complete":
        if st.button("🔄 New Research", use_container_width=True):
            st.session_state.agent_phase = "idle"
            st.rerun()
    
    st.divider()
    
    # Status
    st.markdown("### 📊 Status")
    phase_display = {"idle": "⚪ Idle", "running": "🟢 Running", "paused": "🟡 Paused", "complete": "✅ Complete"}
    st.markdown(f"**Phase:** {phase_display.get(phase, phase)}")
    st.markdown(f"**Iteration:** {st.session_state.agent_iteration}")
    st.metric("Strategies", len(st.session_state.strategies_found))
    st.metric("Avenues Explored", len(st.session_state.avenues_explored))
    st.metric("Remaining", len(st.session_state.avenues_to_explore))
    st.metric("Near-misses", len(st.session_state.near_misses))
    
    # Hypothesis tracking
    hypothesis_count = st.session_state.get("hypothesis_count", 0)
    st.metric("Hypotheses Tested", hypothesis_count)
    
    # Multiple testing warning
    if hypothesis_count > 100:
        expected_false_positives = int(hypothesis_count * 0.05)
        st.warning(f"⚠️ ~{expected_false_positives} may be false positives")
    
    if st.session_state.last_checkpoint:
        st.caption(f"Last checkpoint: {st.session_state.last_checkpoint}")
    
    st.divider()
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("📋 Query Strategies", use_container_width=True):
        result = _run_tool("query_strategies", {"limit": 10})
        st.json(result)
    
    if st.button("💡 Query Learnings", use_container_width=True):
        result = _run_tool("query_learnings", {"limit": 10})
        st.json(result)
    
    # Drill down history viewer
    drill_history = st.session_state.get("drill_down_history", [])
    if drill_history:
        with st.expander(f"🔬 Drill Down History ({len(drill_history)} runs)", expanded=False):
            for i, dd in enumerate(reversed(drill_history[-5:])):  # Show last 5
                st.markdown(f"**Run {len(drill_history) - i}** - {dd.get('timestamp', '?')[:16]}")
                
                improving = dd.get("improving_filters", [])
                if improving:
                    st.markdown(f"  📈 {len(improving)} improving filters found")
                    # Show top 3
                    for f in sorted(improving, key=lambda x: x.get("improvement", 0), reverse=True)[:3]:
                        st.caption(f"    • {f.get('display', '?')}: +{f.get('improvement', 0)*100:.1f}%")
                
                combos = dd.get("combinations", [])
                if combos:
                    good_combos = [c for c in combos if c.get("improvement", 0) > 0]
                    if good_combos:
                        st.markdown(f"  🎯 {len(good_combos)} improving combinations")
                
                st.markdown("---")
    
    st.divider()
    
    # Clear button
    if st.button("🗑️ Clear All", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    
    # Log
    if st.session_state.log:
        with st.expander("📋 Log", expanded=False):
            for line in st.session_state.log[-30:]:
                st.text(line)
    
    st.caption("v6.0 - GPT-5 + Local Compute")

# Main content
if st.session_state.agent_phase == "paused":
    st.warning("⏸️ Research is paused. Use sidebar to resume or ask questions below.")
    
    # ALWAYS show research log
    _display_research_log()
    
    # Ask Agent feature
    st.markdown("### 💬 Ask Agent")
    question = st.text_input("Ask about the research progress:", key="ask_input")
    if st.button("Ask", key="ask_button"):
        if question:
            with st.spinner("Thinking..."):
                answer = _ask_agent(question)
            st.markdown("**Answer:**")
            st.markdown(answer)
    
    # Show current findings
    if st.session_state.strategies_found:
        with st.expander(f"🏆 Strategies Found ({len(st.session_state.strategies_found)})", expanded=True):
            for i, strat in enumerate(st.session_state.strategies_found, 1):
                st.markdown(f"**Strategy {i}:** Test ROI {strat['result'].get('test_roi', 0):.2%}")
                st.code(json.dumps(strat["filters"], indent=2), language="json")

elif st.session_state.run_requested:
    st.session_state.run_requested = False
    run_agent()

elif st.session_state.agent_phase == "idle":
    st.info("👆 Click **Start Research** in the sidebar to begin")
    
    # ALWAYS show research log if it exists
    _display_research_log()
    
    # Resume option - MORE PROMINENT
    st.markdown("### 📂 Resume from Previous Session")
    st.markdown("*Sessions are saved automatically every iteration. If Streamlit reboots, you can resume!*")
    
    # Show current session ID if we have one
    if st.session_state.session_id:
        st.success(f"📋 Current Session ID: `{st.session_state.session_id}`")
        st.caption("Copy this ID to resume later!")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        session_id = st.text_input("Enter Session ID to resume:", placeholder="e.g., abc123-def456-...")
    with col2:
        st.write("")  # Spacing
        st.write("")
        load_btn = st.button("🔄 Load", use_container_width=True)
    
    if load_btn and session_id:
        with st.spinner("Loading checkpoint..."):
            if _restore_checkpoint(session_id):
                st.success(f"✅ Checkpoint loaded! Iteration {st.session_state.agent_iteration}, {len(st.session_state.strategies_found)} strategies found")
                st.rerun()
            else:
                st.error("❌ Checkpoint not found - check the session ID")
    
    with st.expander("🆕 What's New in v6"):
        st.markdown("""
**v6 - Comprehensive Edition:**

**From v5:**
1. **BATCH TESTING**: Each avenue tested with 8-10 variations
2. **DEEP ANALYSIS**: GPT-5 analyzes after EVERY phase
3. **NO PREMATURE CONCLUSIONS**: Explores ALL avenues
4. **COMBINATION SCANNING**: Auto-scans when base shows promise
5. **LEARNING ACCUMULATION**: Tracks insights across iterations
6. **AVENUE TRACKING**: Shows explored vs remaining

**New in v6:**
7. **LOCAL COMPUTE**: FastAPI server (no Modal costs!)
8. **38 TOOLS**: Including ML (CatBoost, XGBoost, SHAP)
9. **SUPABASE PERSISTENCE**: Strategies & learnings saved
10. **AUTO-PROMOTION**: Lifecycle management
11. **PAUSE/RESUME**: Control research flow
12. **ASK AGENT**: Query when paused
13. **CRASH RECOVERY**: Checkpoint restore
14. **GPT-5**: Latest model for analysis
""")

elif st.session_state.agent_phase == "complete":
    st.success("✅ Research complete!")
    
    # ALWAYS show research log
    _display_research_log()
    
    if st.session_state.strategies_found:
        st.markdown("### 🏆 Strategies Found")
        for i, strat in enumerate(st.session_state.strategies_found, 1):
            st.markdown(f"**Strategy {i}:** Test ROI {strat['result'].get('test_roi', 0):.2%}")
            st.code(json.dumps(strat["filters"], indent=2), language="json")
    
    if st.session_state.near_misses:
        with st.expander(f"🎯 Near Misses ({len(st.session_state.near_misses)})"):
            for nm in st.session_state.near_misses[:10]:
                st.markdown(f"- {nm.get('avenue')}: Test ROI {nm.get('test_roi', 0):.2%}")

else:
    # Show messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
