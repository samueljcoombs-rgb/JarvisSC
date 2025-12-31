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

LOCAL_COMPUTE_URL = "http://localhost:8000"
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
# Local Compute Client
# ============================================================

def _check_server() -> Dict:
    """Check if local compute server is running."""
    try:
        response = requests.get(f"{LOCAL_COMPUTE_URL}/health", timeout=5)
        return response.json()
    except Exception:
        return {"status": "offline"}

def _run_tool(name: str, args: Optional[Dict] = None, timeout: int = JOB_TIMEOUT) -> Any:
    """Call local compute server."""
    args = args or {}
    url = f"{LOCAL_COMPUTE_URL}/run_task"
    payload = {
        "task_type": name,
        "params": args,
        "job_id": str(uuid.uuid4())
    }
    
    try:
        _log(f"Calling {name}...")
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "error":
            _log(f"Error in {name}: {data.get('error')}")
            return {"error": data.get("error")}
        
        return data.get("result", data)
    
    except requests.exceptions.ConnectionError:
        _log(f"Connection error - is local_compute.py running?")
        return {"error": "Cannot connect to local compute server. Run: python3 local_compute.py"}
    except requests.exceptions.Timeout:
        _log(f"Timeout calling {name}")
        return {"error": f"Timeout after {timeout}s"}
    except Exception as e:
        _log(f"Error calling {name}: {e}")
        return {"error": str(e)}

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
- Think like a quant researcher, not a code executor
- ANALYZE DEEPLY before deciding next steps
- Build a mental model of what's working and why
- Never give up until you've truly explored all angles
- Be SKEPTICAL of your own results - assume noise until proven

## Deep Thinking Cycle (USE THIS)
1. OBSERVE: What just happened? Note surprises.
2. CONNECT: How does this relate to past learnings?
3. HYPOTHESIZE: Why might this work? What market inefficiency?
4. PLAN: Generate 3-5 candidate actions, score by expected value
5. DECIDE: Pick best action with full justification
6. RECORD: Save key insights

## Critical Rules
1. NEVER use PL columns as features (data leakage!)
2. Split by TIME: train older, test newer
3. Explain WHY a filter exploits market inefficiency
4. Simple > complex - prefer fewer filters
5. Sample size matters - need 300+ train rows
6. p-value < 0.10 for statistical significance
7. Forward walk must show >60% positive periods

## Available Tools (38 total - USE WISELY)

### Exploration Tools
- **query_data**: Run aggregations (group_by, metrics). Good for initial exploration.
- **feature_importance**: Find which columns correlate with profit.
- **univariate_scan**: For each column, find the single best filter value.

### Sweep Tools
- **bracket_sweep**: Test numeric column ranges systematically (ACTUAL ODDS, % DRIFT)
- **subgroup_scan**: Test categorical combinations (MODE, MARKET, DRIFT)
- **combination_scan**: Test multi-filter combinations - USE when you have promising base

### Testing Tools
- **test_filter**: Test specific filter combination with train/val/test splits
- **forward_walk**: Walk-forward validation (6 windows)
- **monte_carlo_sim**: Bootstrap simulation for confidence intervals
- **correlation_check**: Check for feature leakage
- **regime_check**: Check stability across time periods

### ML Tools
- **train_catboost**: Train CatBoost model - handles categoricals natively
- **train_xgboost**: Train XGBoost model
- **shap_explain**: SHAP explanations - convert ML insights to filters
- **hyperopt_model**: Optuna hyperparameter tuning

### Analysis Tools
- **statistical_significance**: Calculate p-value for strategy
- **time_decay_analysis**: Check if edge decays over time
- **cross_market_test**: Test if strategy works across multiple PL columns
- **league_breakdown**: Break down performance by league
- **odds_band_optimizer**: Find optimal odds range

### Memory Tools (Supabase)
- **save_strategy**: Store strategy with full stats
- **query_strategies**: Find similar strategies
- **save_learning**: Store insight/learning
- **query_learnings**: "What do I know about X?"
"""

def _llm(context: str, question: str, max_tokens: int = 3000) -> str:
    """Call GPT-5 for analysis."""
    client = _get_client()
    if not client:
        return '{"error": "OpenAI client not available"}'
    
    try:
        response = client.chat.completions.create(
            model=_get_model(),
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
    parsed = _parse_json(resp)
    
    if not parsed:
        return {"detailed_analysis": resp[:1000], "prioritized_avenues": []}
    
    return parsed

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
            group_str = ", ".join([f"{k}={v}" for k, v in group.items() if v])
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
        
        if not avenues_remaining:
            st.warning("All avenues explored!")
            break
        
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
    st.error("⚠️ Local compute server is not running!")
    st.markdown("""
    Start the server:
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
    
    # Server status
    st.success(f"🟢 Server Online ({server_status.get('data_rows', 0):,} rows)")
    st.caption(f"Tools: {server_status.get('total_tools', 38)}")
    
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
