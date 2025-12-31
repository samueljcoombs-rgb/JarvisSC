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
from typing import Any, Dict, List, Optional, Set
import requests
import streamlit as st

st.set_page_config(page_title="Football Research Agent v6", page_icon="⚽", layout="wide")

# ============================================================
# Configuration
# ============================================================

MAX_ITERATIONS = 15
MAX_MESSAGES = 200
JOB_TIMEOUT = 300

# Default gates
DEFAULT_GATES = {
    "min_train_rows": 300,
    "min_val_rows": 60,
    "min_test_rows": 60,
    "max_train_val_gap_roi": 0.40,
    "max_test_drawdown": -50,
    "max_rolling_dd": -50,
    "max_test_losing_streak_bets": 50,
}

# Outcome columns (NEVER use as features)
OUTCOME_COLUMNS = ["BO 2.5 PL", "BTTS PL", "SHG PL", "SHG 2+ PL", "LU1.5 PL", "LFGHU0.5 PL", "BO1.5 FHG PL", "PL"]

# ============================================================
# Helpers
# ============================================================

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
    return "gpt-5"  # ALWAYS GPT-5

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
    """Load research context from local compute."""
    try:
        if "bible" not in st.session_state:
            st.session_state.bible = None
        if st.session_state.bible:
            return st.session_state.bible
        
        _log("Loading Bible...")
        bible = _run_tool("get_research_context", {"pl_column": st.session_state.target_pl_column})
        
        # If no Supabase, create minimal bible
        if bible.get("error"):
            bible = {
                "dataset_overview": {"primary_goal": "Find profitable betting strategies"},
                "gates": DEFAULT_GATES,
                "derived": {"outcome_columns": OUTCOME_COLUMNS},
                "research_rules": [
                    {"rule": "Never use PL columns as features"},
                    {"rule": "Split by time: train older, test newer"},
                    {"rule": "Minimum 300 train rows"},
                ],
            }
        
        st.session_state.bible = bible
        return bible
    except Exception as e:
        _log(f"Error loading bible: {e}")
        return {"gates": DEFAULT_GATES}

def _format_bible(bible: Dict) -> str:
    overview = bible.get("dataset_overview") or {}
    gates = bible.get("gates") or DEFAULT_GATES
    derived = bible.get("derived") or {}
    
    outcome_cols = derived.get('outcome_columns', OUTCOME_COLUMNS)
    outcome_str = ', '.join(str(c) for c in outcome_cols) if isinstance(outcome_cols, list) else str(outcome_cols)
    
    return f"""## 📖 Bible Loaded
**Goal:** {overview.get('primary_goal', 'Find profitable strategies')}
**Gates:** min_train={gates.get('min_train_rows', 300)}, min_val={gates.get('min_val_rows', 60)}, min_test={gates.get('min_test_rows', 60)}, max_gap={gates.get('max_train_val_gap_roi', 0.4)}, max_dd={gates.get('max_test_drawdown', -50)}
**Outcome Columns (NEVER features):** {outcome_str}"""

# ============================================================
# LLM Functions (GPT-5)
# ============================================================

SYSTEM_PROMPT = """You are an expert football betting research agent using a WORLD-CLASS quantitative approach. Your goal is to find PROFITABLE and STABLE strategies through systematic, thoughtful exploration.

## Your Mindset - SENIOR QUANT RESEARCHER
- 70% thinking, 30% testing
- Think like a quant researcher at a top hedge fund
- ANALYZE DEEPLY before deciding next steps
- Build a mental model of what's working and why
- Never give up until you've truly explored all angles
- Be SKEPTICAL of your own results - assume noise until proven
- Apply domain knowledge about football and betting markets

## DUAL-TRACK RESEARCH APPROACH

You have TWO complementary research tracks, but the FINAL OUTPUT is ALWAYS filter rules.

### TRACK A: Rule-Based Filters (PRIMARY - This is the output)
- MODE/MARKET/DRIFT exploration
- Bracket sweeps on numeric columns
- Subgroup scans on categorical columns
- Filter combinations
- **THIS IS WHAT WE DEPLOY** - Simple, interpretable, executable rules

### TRACK B: ML Models (DISCOVERY HELPER - Finds what to test)
- CatBoost, XGBoost, LightGBM, Logistic Regression
- SHAP explanations to understand WHY
- Feature importance to find what matters
- **PURPOSE: Find patterns to convert into filter rules**

### THE KEY INSIGHT:
```
ML is for DISCOVERY → Filter Rules are the OUTPUT

ML says "ODDS 1.8-2.2 important" 
    → Create filter: ODDS between 1.8-2.2
    → Validate with test_filter
    → Deploy the FILTER (not the ML model)
```

### WHY USE ML AT ALL?
1. **Finds patterns humans miss** - ML might discover "% DRIFT > 5" matters
2. **SHAP converts ML → Filters** - Explicit filter suggestions from black box
3. **Confirms filter ideas** - If ML agrees with your filter, more confidence
4. **Prioritizes exploration** - Feature importance tells you what to bracket_sweep

### HYBRID WORKFLOW (ML Discovery → Filter Output)
```
Step 1: train_catboost or train_logistic
        → Get feature_importance list
        
Step 2: shap_explain 
        → Get suggested_filters from SHAP
        
Step 3: test_filter on each suggested filter
        → Validate with train/val/test
        
Step 4: forward_walk on passing filters
        → Ensure stability over time
        
Step 5: save_strategy 
        → Save the FILTER RULES (not the ML model)
```

### EXAMPLE:
```
1. train_catboost returns: feature_importance = ["ACTUAL ODDS", "% DRIFT", "MODE"]
2. shap_explain returns: suggested_filters = [
     {"col": "ACTUAL ODDS", "op": "between", "min": 1.8, "max": 2.2},
     {"col": "% DRIFT", "op": ">=", "value": 5}
   ]
3. test_filter on these → Train: +3%, Val: +2%, Test: +1.5%
4. forward_walk → 5/6 windows positive
5. FINAL OUTPUT: Filter rules that can be executed without ML
```

**Remember: The bettor will use FILTER RULES, not ML models. ML just helps us find good rules faster.**

## Deep Thinking Cycle (USE THIS EVERY ITERATION)
1. OBSERVE: What just happened? Note surprises.
2. CONNECT: How does this relate to past learnings?
3. HYPOTHESIZE: Why might this work? What market inefficiency?
4. PLAN: Generate 3-5 candidate actions, score by expected value
5. DECIDE: Pick best action with full justification
6. RECORD: Save key insights

## Critical Rules
1. NEVER use PL columns as features (data leakage!)
2. Split by TIME: train older, test newer (never random!)
3. Explain WHY a filter exploits market inefficiency
4. Simple > complex - prefer fewer filters
5. Sample size matters - need 300+ train rows, 60+ test rows
6. p-value < 0.10 for statistical significance
7. Forward walk must show >60% positive periods

## COMPLETE TOOL ARSENAL (29 Tools)

### 🔍 EXPLORATION TOOLS (Understand the data)
- **query_data**: Run aggregations (group_by, metrics). Start here!
  - Use: `query_type="aggregate", group_by=["MODE"], metrics=["count", "mean:BO 2.5 PL"]`
- **feature_importance**: Find which numeric columns correlate with profit
- **univariate_scan**: For each column, find single best filter value
- **correlation_check**: Check for feature leakage before using a column

### 🎯 SWEEP TOOLS (Systematic search)
- **bracket_sweep**: Test numeric ranges (ACTUAL ODDS, % DRIFT, etc.)
  - Returns top brackets with train/val/test splits
- **subgroup_scan**: Test categorical combos (MODE, MARKET, DRIFT IN/OUT)
  - Returns top groups passing gates
- **combination_scan**: Test multi-filter combinations
  - Use when you have a promising base to build on

### ✅ TESTING & VALIDATION TOOLS
- **test_filter**: Test specific filter combo with train/val/test splits
  - The workhorse - use this to validate any hypothesis
- **forward_walk**: Walk-forward validation (6 windows)
  - REQUIRED for any strategy claiming to be "validated"
- **monte_carlo_sim**: Bootstrap simulation for confidence intervals
  - Answers: "What's the probability this is profitable?"
- **regime_check**: Check stability across time periods (month/quarter)
- **statistical_significance**: Calculate p-value
  - p < 0.05 = significant, p < 0.01 = highly significant
- **time_decay_analysis**: Check if edge decays over time (alpha decay)
  - Critical! Edges can disappear

### 🤖 ML TOOLS (Pattern Discovery)
- **train_catboost**: CatBoost model - BEST for categorical features
  - Handles MODE, MARKET, LEAGUE natively
  - Returns feature importance + suggested filters
- **train_xgboost**: XGBoost model - robust general purpose
  - Numeric features only, very reliable
- **train_lightgbm**: LightGBM model - fast, handles categoricals
  - Good for large datasets
- **train_logistic**: Logistic Regression - INTERPRETABLE baseline
  - Returns coefficients you can understand
  - Great for "WHY does this work?"
- **shap_explain**: SHAP explanations - convert ML to filters!
  - THIS IS KEY for dual-track research
  - Tells you exactly which features matter and how
- **hyperopt_pl_lab**: Optuna hyperparameter tuning
  - Use for final optimization

### 💾 MEMORY TOOLS (Supabase Persistence)
- **save_strategy**: Store strategy with full stats
  - Save every promising strategy!
- **query_strategies**: Find strategies by status/pl_column
- **promote_strategy**: Move strategy through lifecycle
  - draft → candidate → promising → validated
- **save_learning**: Store insight/learning
  - "DRIFT IN works better for Back bets"
- **query_learnings**: Search past learnings
  - "What do I know about DRIFT?"
- **save_checkpoint**: Save current state (crash recovery)
- **load_checkpoint**: Resume from checkpoint
- **get_research_context**: Load Bible (rules, gates, learnings)

### 🔧 SESSION TOOLS
- **create_session**: Start new research session
- **pl_lab**: Full ML pipeline with distillation

## STRATEGY LIFECYCLE (Auto-Promotion)

DRAFT → test_roi > 0 + gates_passed
CANDIDATE → forward_walk > 60% + monte_carlo > 65%
PROMISING → no alpha decay + p-value < 0.10 + sharpe > 0.5
VALIDATED → Ready for live trading

## WHEN TO USE WHICH TOOL

| Situation | Best Tool |
|-----------|-----------|
| "What does the data look like?" | query_data |
| "Which columns matter?" | feature_importance, train_logistic |
| "Find me good categorical combos" | subgroup_scan |
| "Find me good numeric ranges" | bracket_sweep |
| "Test this specific filter" | test_filter |
| "Is this statistically significant?" | statistical_significance |
| "Will this work going forward?" | forward_walk |
| "Find complex patterns" | train_catboost, shap_explain |
| "Why does the ML model predict?" | shap_explain |
| "Is the edge decaying?" | time_decay_analysis |
| "What's the confidence interval?" | monte_carlo_sim |
| "Save this finding" | save_learning, save_strategy |

## EXAMPLE HYBRID WORKFLOW

1. Start with subgroup_scan → Find MODE=Quick League looks good
2. Run train_catboost → Confirms MODE important, also finds DRIFT IN
3. Run shap_explain → Shows ACTUAL ODDS 1.8-2.2 matters
4. Create combined filter: MODE=QL + DRIFT=IN + ODDS 1.8-2.2
5. test_filter → Train: +4%, Val: +3%, Test: +2%
6. forward_walk → 5/6 windows positive
7. statistical_significance → p-value = 0.03
8. save_strategy with status="validated"

This is a VALIDATED strategy because:
- Rule-based filter (interpretable) ✓
- ML confirmation (pattern real) ✓
- Statistical significance (not noise) ✓
- Forward walk (works over time) ✓
"""

def _llm(context: str, question: str, max_tokens: int = 3000) -> str:
    """Call GPT-5 for analysis."""
    client = _get_client()
    if not client:
        return '{"error": "OpenAI client not available"}'
    
    try:
        model = _get_model()
        
        # GPT-5 and o1 models use max_completion_tokens, older models use max_tokens
        if model.startswith("gpt-5") or model.startswith("o1"):
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{context}\n\n{question}"}
                ]
            )
        else:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{context}\n\n{question}"}
                ]
            )
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'

def _parse_json(resp: str) -> Optional[Dict]:
    """Parse JSON from LLM response."""
    try:
        return json.loads(resp)
    except:
        pass
    try:
        start = resp.find('{')
        end = resp.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(resp[start:end])
    except:
        pass
    return None

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
    """Deep LLM analysis of exploration results."""
    context = f"""## Bible Context
{_safe_json(bible.get('dataset_overview', {}), 1000)}

## Exploration Results for {pl_column}

### Data Overview
{_safe_json(exploration.get('describe', {}), 1500)}

### MODE Distribution
{_safe_json(exploration.get('mode_distribution', {}), 1500)}

### MARKET Distribution  
{_safe_json(exploration.get('market_distribution', {}), 1500)}

### DRIFT Distribution
{_safe_json(exploration.get('drift_distribution', {}), 1000)}

### MODE x MARKET x DRIFT Combinations
{_safe_json(exploration.get('mode_market_drift', {}), 3000)}
"""

    question = """Analyze this exploration data DEEPLY. Think step by step:

1. Which MODEs have positive mean PL? Why might that be?
2. Which MARKETs show promise? What market inefficiency could explain it?
3. How does DRIFT (IN/OUT/SAME) affect profitability? Why?
4. What MODE + MARKET + DRIFT combinations look most promising?
5. Any patterns that surprise you? Red flags?

Then generate prioritized research avenues.

Respond with JSON:
{
    "detailed_analysis": "Your deep thinking about the data (2-3 paragraphs)",
    "key_observations": ["observation 1", "observation 2", ...],
    "mode_summary": {
        "best_mode": "MODE name",
        "best_mode_reason": "why it's best",
        "worst_mode": "MODE name", 
        "worst_mode_reason": "why it's worst"
    },
    "drift_summary": {
        "best_drift": "IN/OUT/SAME",
        "best_drift_reason": "why",
        "drift_depends_on": "what drift effectiveness depends on"
    },
    "all_positive_combinations": [
        {"mode": "X", "market": "Y", "best_drift": "Z", "mean_pl": 0.05, "count": 100}
    ],
    "sample_size_concerns": [
        {"combination": "X+Y", "count": 50, "concern": "too small"}
    ],
    "prioritized_avenues": [
        {
            "rank": 1,
            "avenue": "MODE=X, MARKET=Y",
            "base_filters": [{"col": "MODE", "op": "=", "value": "X"}, {"col": "MARKET", "op": "=", "value": "Y"}],
            "promising_drift": "IN",
            "market_inefficiency_hypothesis": "Why this might be profitable",
            "expected_rows": "~500",
            "confidence": "high/medium/low"
        }
    ]
}"""

    resp = _llm(context, question)
    _log(f"LLM analysis response length: {len(resp) if resp else 0}")
    
    # Check for errors
    if not resp or resp.startswith('{"error"'):
        _log(f"LLM error: {resp}")
        # Generate basic avenues from exploration data without LLM
        return _generate_fallback_avenues(exploration, pl_column)
    
    parsed = _parse_json(resp)
    
    if not parsed:
        _log("Failed to parse LLM response, using fallback")
        return {"detailed_analysis": resp[:1000] if resp else "Analysis failed", "prioritized_avenues": _generate_fallback_avenues(exploration, pl_column).get("prioritized_avenues", [])}
    
    # Ensure we have avenues
    if not parsed.get("prioritized_avenues"):
        fallback = _generate_fallback_avenues(exploration, pl_column)
        parsed["prioritized_avenues"] = fallback.get("prioritized_avenues", [])
    
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
    """LLM deep analysis of iteration results."""
    context = f"""## Avenue Explored
{_safe_json(avenue, 500)}

## Results Summary
{_safe_json(avenue_results.get('analysis', {}), 2000)}

## Best Results
{_safe_json(avenue_results.get('analysis', {}).get('all_results_sorted', [])[:5], 2000)}

## Accumulated Learnings
{_safe_json(accumulated_learnings[-5:], 1000)}

## Avenues Remaining: {avenues_remaining}
"""

    question = """Analyze these results DEEPLY using the thinking cycle:

1. OBSERVE: What happened? Any surprises?
2. CONNECT: How does this relate to past learnings?
3. HYPOTHESIZE: What market inefficiency might explain any positive results?
4. Should we continue with this avenue or move on?

Respond with JSON:
{
    "detailed_reasoning": "Your step-by-step thinking (2-3 paragraphs)",
    "hypothesis_verdict": "supported/refuted/inconclusive",
    "key_learning": "One key insight to remember",
    "patterns_noticed": ["pattern 1", "pattern 2"],
    "refinement_ideas": [
        {"filter_to_add": {"col": "X", "op": "=", "value": "Y"}, "reasoning": "why"}
    ],
    "should_continue_avenue": true/false,
    "confidence_in_direction": "high/medium/low",
    "next_recommendation": "what to do next"
}"""

    resp = _llm(context, question)
    parsed = _parse_json(resp)
    return parsed if parsed else {"detailed_reasoning": resp[:500], "key_learning": "", "should_continue_avenue": False}

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
    
    # Reset for new run
    st.session_state.log = []
    st.session_state.tested_filter_hashes = set()
    st.session_state.near_misses = []
    st.session_state.accumulated_learnings = []
    st.session_state.avenues_explored = []
    st.session_state.strategies_found = []
    
    st.markdown(f"# 🤖 Research Agent v6: {pl_column}")
    st.caption("Deep Analysis Edition + Local Compute + 38 Tools + Persistence")
    
    # ========== BIBLE ==========
    with st.status("📖 Loading Bible...", expanded=True) as status:
        bible = _load_bible()
        st.markdown(_format_bible(bible))
        status.update(label="📖 Bible loaded", state="complete")
    
    with st.expander("📚 Full Bible Context", expanded=False):
        st.code(_safe_json(bible, 3000), language="json")
    
    _append("assistant", _format_bible(bible))
    
    # ========== EXPLORATION ==========
    with st.status("🔍 Phase 1: Exploration...", expanded=True) as status:
        progress = st.empty()
        exploration = _run_exploration(pl_column, progress)
        st.session_state.exploration_results = exploration
        status.update(label="🔍 Exploration complete", state="complete")
    
    with st.expander("Exploration Results", expanded=False):
        st.code(_safe_json(exploration, 5000), language="json")
    
    # Check for pause
    if st.session_state.agent_phase == "paused":
        st.warning("⏸️ Research paused")
        return
    
    # ========== DEEP ANALYSIS ==========
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
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        # Check for pause
        if st.session_state.agent_phase == "paused":
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
                    feature_imp = ml_result.get("feature_importance", [])[:5]
                    
                    for sf in suggested[:3]:
                        new_avenue = {
                            "avenue": f"ML Discovery: {sf.get('col')} {sf.get('op')} {sf.get('value', sf.get('values', ''))}",
                            "base_filters": [sf],
                            "market_inefficiency_hypothesis": sf.get("reasoning", "ML model identified this pattern"),
                            "source": "ml_catboost",
                        }
                        if new_avenue not in st.session_state.avenues_explored:
                            avenues_remaining.append(new_avenue)
                    
                    # Also create avenues from top features
                    for fi in feature_imp:
                        feat = fi.get("feature")
                        if feat and not any(pl in feat.lower() for pl in ["pl", "profit", "result"]):
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
                    for sf in shap_result.get("suggested_filters", [])[:3]:
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
        
        st.markdown(f"#### Iteration {iteration}: {current_avenue.get('avenue', 'Unknown')}")
        st.markdown(f"*{current_avenue.get('market_inefficiency_hypothesis', current_avenue.get('why_promising', ''))}*")
        
        # Explore avenue
        with st.status(f"Testing avenue {iteration}...", expanded=True) as status:
            avenue_results = _explore_avenue(current_avenue, pl_column, bible, exploration_analysis)
            status.update(label=f"Avenue {iteration} complete", state="complete")
        
        # Display results
        analysis = avenue_results.get("analysis", {})
        st.markdown(f"**Results:** {analysis.get('summary', 'N/A')}")
        st.markdown(f"**Best Test ROI:** {analysis.get('best_test_roi', 0):.4f}")
        st.markdown(f"**Recommendation:** {analysis.get('recommendation', 'N/A')}")
        
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
        
        # Handle results
        recommendation = analysis.get("recommendation", "").upper()
        truly_passing = analysis.get("truly_passing", [])
        
        if "SUCCESS" in recommendation and truly_passing:
            st.success("🎉 **Strategy Found!**")
            
            best_result = truly_passing[0]
            
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
            
            # Save to Supabase
            save_result = _run_tool("save_strategy", {
                "filters": best_result.get("filters", []),
                "pl_column": pl_column,
                "train_roi": best_result.get("train_roi", 0),
                "train_rows": best_result.get("train_rows", 0),
                "val_roi": best_result.get("val_roi", 0),
                "val_rows": best_result.get("val_rows", 0),
                "test_roi": best_result.get("test_roi", 0),
                "test_rows": best_result.get("test_rows", 0),
                "hypothesis": current_avenue.get("market_inefficiency_hypothesis", ""),
                "status": "draft"
            })
            
            if save_result.get("saved"):
                st.info(f"💾 Strategy saved (quality: {save_result.get('quality_score', '?')})")
                st.session_state.strategies_found.append({
                    "filters": best_result.get("filters", []),
                    "result": best_result,
                    "filter_hash": save_result.get("filter_hash"),
                })
            
            # Advanced validation option
            with st.expander("🔬 Run Advanced Validation", expanded=False):
                if st.button(f"Validate Strategy {iteration}", key=f"validate_{iteration}"):
                    validation = _validate_strategy(best_result.get("filters", []), pl_column)
                    st.json(validation)
                    
                    if validation.get("overall_verdict", {}).get("recommend_validate"):
                        st.success("🎉 **STRATEGY VALIDATED!**")
                        # Promote
                        _run_tool("promote_strategy", {"filter_hash": save_result.get("filter_hash", "")})
                    else:
                        st.warning("⚠️ Did not pass full validation")
            
            # Check if strong enough to stop
            is_strong = (
                best_result.get("train_roi", 0) > 0.01 and
                best_result.get("val_roi", 0) > -0.01 and
                best_result.get("test_roi", 0) > 0.01 and
                abs(best_result.get("train_roi", 0) - best_result.get("val_roi", 0)) < 0.03
            )
            
            if is_strong:
                st.success("🎯 Found STRONG strategy - stopping search!")
                st.session_state.agent_phase = "complete"
                return
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
        
        # Deep reflection every 3 iterations
        if iteration % 3 == 0 and avenues_remaining:
            st.markdown("**🤔 Mid-point Reflection...**")
            reflection = _deep_reflect(
                f"After iteration {iteration}",
                avenue_results,
                st.session_state.accumulated_learnings,
                avenues_remaining
            )
            st.markdown(f"*{reflection.get('reflection', '')[:300]}*")
        
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
    
    # Resume option
    st.markdown("### 📂 Or Resume from Checkpoint")
    session_id = st.text_input("Session ID:")
    if st.button("Load Checkpoint"):
        if session_id:
            if _restore_checkpoint(session_id):
                st.success("Checkpoint loaded!")
                st.rerun()
            else:
                st.error("Checkpoint not found")
    
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
