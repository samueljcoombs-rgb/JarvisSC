"""
Football Researcher v5 - Deep Analysis Edition

KEY CHANGES from v4:
1. BATCH TESTING: Test 5-10 variations per avenue, not just 1
2. DEEP ANALYSIS: LLM analyzes results after EVERY phase
3. NO PREMATURE CONCLUSIONS: Must explore ALL avenues before giving up
4. COMBINATION SCANNING: When base filter shows promise, auto-scan for additive filters
5. LEARNING ACCUMULATION: Track insights across iterations
6. AVENUE TRACKING: Explicitly track which avenues explored vs remaining
"""

from __future__ import annotations
import os, json, uuid, re, time, hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import streamlit as st
from openai import OpenAI
from modules import football_tools as tools

st.set_page_config(page_title="Football Research Agent v5", page_icon="⚽", layout="wide")

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

@st.cache_resource
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()
    return OpenAI(api_key=api_key)

def _get_model() -> str:
    return os.getenv("PREFERRED_OPENAI_MODEL") or st.secrets.get("PREFERRED_OPENAI_MODEL", "gpt-4o")

MAX_ITERATIONS = 15  # More iterations for thorough exploration
MAX_MESSAGES = 200
JOB_TIMEOUT = 300

# ============================================================
# Session State
# ============================================================

def _init_state():
    defaults = {
        "messages": [], "session_id": str(uuid.uuid4()), "bible": None,
        "agent_phase": "idle", "agent_iteration": 0, "agent_findings": [],
        "target_pl_column": "BO 2.5 PL", "exploration_results": {}, "sweep_results": {},
        "exploration_analysis": {},
        "past_failures": [], "near_misses": [], "tested_filter_hashes": set(),
        "run_requested": False, "log": [],
        # NEW v5: Track avenues and learnings
        "avenues_to_explore": [],
        "avenues_explored": [],
        "accumulated_learnings": [],
        "promising_bases": [],  # Base filters that showed promise
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def _log(msg: str):
    st.session_state.log.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")

def _append(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

def _add_learning(learning: str):
    """Accumulate learnings across iterations."""
    st.session_state.accumulated_learnings.append({
        "iteration": st.session_state.agent_iteration,
        "learning": learning,
        "ts": datetime.utcnow().isoformat()
    })

# ============================================================
# Tool Runner
# ============================================================

def _run_tool(name: str, args: Optional[Dict] = None) -> Any:
    args = args or {}
    aliases = {
        "bracket_sweep": "start_bracket_sweep", 
        "subgroup_scan": "start_subgroup_scan",
        "query_data": "start_query_data", 
        "test_filter": "start_test_filter", 
        "regime_check": "start_regime_check",
        "combination_scan": "start_combination_scan",
        "forward_walk": "start_forward_walk",
        "monte_carlo_sim": "start_monte_carlo_sim",
        "correlation_check": "start_correlation_check",
        "feature_importance": "start_feature_importance",
        "univariate_scan": "start_univariate_scan",
    }
    fn = getattr(tools, aliases.get(name, name), None)
    if not callable(fn):
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _wait_for_job(job_id: str, timeout: int = JOB_TIMEOUT) -> Dict:
    start = time.time()
    while time.time() - start < timeout:
        job = _run_tool("get_job", {"job_id": job_id})
        status = (job.get("status") or "").lower()
        _log(f"Job {job_id[:8]}: {status}")
        if status == "done":
            result_path = job.get("result_path")
            if result_path:
                return _run_tool("download_result", {"result_path": result_path}).get("result", job)
            return job
        elif status == "error":
            return {"error": job.get("error_message", "Job failed")}
        time.sleep(3)
    return {"error": "Timeout"}

# ============================================================
# Bible
# ============================================================

def _load_bible() -> Dict:
    if st.session_state.bible:
        return st.session_state.bible
    _log("Loading Bible...")
    bible = _run_tool("get_research_context", {"limit_notes": 50})
    st.session_state.bible = bible
    return bible

def _format_bible(bible: Dict) -> str:
    overview = bible.get("dataset_overview") or {}
    gates = bible.get("gates") or {}
    derived = bible.get("derived") or {}
    rules = bible.get("research_rules") or []
    notes = bible.get("research_notes") or []
    col_defs = bible.get("column_definitions") or []
    
    outcome_cols = derived.get('outcome_columns', [])
    outcome_str = ', '.join(str(c) for c in outcome_cols) if isinstance(outcome_cols, list) else str(outcome_cols)
    
    rules_count = len(rules) if isinstance(rules, list) else 1
    notes_count = len(notes) if isinstance(notes, list) else 0
    cols_count = len(col_defs) if isinstance(col_defs, list) else 0
    
    return f"""## 📖 Bible Loaded
**Goal:** {overview.get('primary_goal', 'Find profitable strategies')}
**Gates:** min_train={gates.get('min_train_rows', 300)}, min_val={gates.get('min_val_rows', 60)}, min_test={gates.get('min_test_rows', 60)}, max_gap={gates.get('max_train_val_gap_roi', 0.4)}, max_dd={gates.get('max_test_drawdown', -50)}
**Outcome Columns (NEVER features):** {outcome_str}
**Loaded:** {rules_count} rules, {cols_count} column defs, {notes_count} past notes"""

# ============================================================
# LLM
# ============================================================

SYSTEM_PROMPT = """You are an expert football betting research agent. Your goal is to find PROFITABLE and STABLE strategies through systematic, thoughtful exploration.

## Your Mindset
- Think like a quant researcher, not a code executor
- ANALYZE DEEPLY before deciding next steps
- Build a mental model of what's working and why
- Never give up until you've truly explored all angles

## Critical Rules
1. NEVER use PL columns as features (data leakage!)
2. Split by TIME: train older, test newer
3. Explain WHY a filter exploits market inefficiency
4. Simple > complex - prefer fewer filters
5. Sample size matters - need 300+ train rows

## Available Tools (USE WISELY)

### Exploration Tools
- **query_data**: Run aggregations (group_by, metrics). Good for initial exploration.
  - query_type: "aggregate" or "describe"
  - group_by: ["MODE", "MARKET"] etc.
  - metrics: ["count", "sum:PL", "mean:PL"]

- **feature_importance**: Find which columns correlate with profit. 
  - Returns correlation + high/low split analysis for each numeric column
  - USE THIS to identify promising features to filter on

- **univariate_scan**: For each column, find the single best filter value.
  - Returns best filter for each column independently
  - USE THIS to find starting points for multi-filter strategies

### Sweep Tools
- **bracket_sweep**: Test numeric column ranges systematically
  - sweep_cols: ["ACTUAL ODDS", "% DRIFT"]
  - Returns best brackets with train/val/test splits

- **subgroup_scan**: Test categorical combinations
  - group_cols: ["MODE", "MARKET", "DRIFT IN / OUT"]
  - Returns best subgroups

- **combination_scan**: Test multi-filter combinations
  - base_filters: Starting filters to build on
  - scan_cols: Columns to add filters from
  - USE THIS when you have a promising base and want to find what to add

### Testing Tools
- **test_filter**: Test a specific filter combination
  - filters: [{"col": "MODE", "op": "=", "value": "XG"}, ...]
  - Returns train/val/test splits with ROI and gates

- **forward_walk**: Walk-forward validation (6 windows default)
  - Tests if strategy works across different time periods

- **monte_carlo_sim**: Bootstrap simulation for confidence intervals
  - Returns probability of positive ROI

- **correlation_check**: Check for data leakage
  - Warns if filters correlate too highly with outcomes

- **regime_check**: Test across different time regimes
  - Good for checking if strategy is stable over time

## Column Names (use EXACTLY)
- MODE: "XG", "Quick League", "Quick Team"
- MARKET: "O2.5 Back", "FHGO1.5 Back", "BTTS Yes Back", "Away Win Back", etc.
- DRIFT IN / OUT: "IN", "OUT", "SAME"
- ACTUAL ODDS: numeric (1.01 to 34.0)
- % DRIFT: numeric (can be negative or positive, typically -10 to +20)

## Filter Format
- Equality: {"col": "MODE", "op": "=", "value": "XG"}
- Range: {"col": "ACTUAL ODDS", "op": "between", "min": 1.8, "max": 2.5}
- Comparison: {"col": "% DRIFT", "op": ">", "value": 0}

## How to Think

When analyzing results, ask yourself:
1. What patterns do I see? Which combinations are profitable?
2. WHY might this be profitable? What market inefficiency does it exploit?
3. Is this likely to persist, or is it random noise?
4. What should I explore next based on these findings?
5. How can I REFINE a promising filter to pass gates?

## Analysis Guidelines

When you analyze results, write DETAILED analysis:
- Don't just list numbers, INTERPRET them
- Compare different combinations
- Note what's surprising or unexpected
- Explain your reasoning for next steps
- Build on previous learnings

Remember: You're building understanding, not just testing random combinations."""

def _llm(context: str, question: str, max_tokens: int = 3000) -> str:
    try:
        client = _get_client()
        resp = client.chat.completions.create(model=_get_model(),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{context}\n\n---\n{question}"}],
            max_tokens=max_tokens, temperature=0.7)
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Error: {e}]"

def _parse_json(resp: str) -> Optional[Dict]:
    try:
        resp = re.sub(r'```json\s*', '', resp.strip())
        resp = re.sub(r'```\s*', '', resp)
        match = re.search(r'\{[\s\S]*\}', resp)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None

# ============================================================
# Phase 1: Exploration
# ============================================================

def _run_exploration(pl_column: str, progress_container=None) -> Dict:
    results = {}
    
    # Get data distribution first
    if progress_container:
        progress_container.text("🔍 Getting data distribution...")
    _log("Getting data distribution...")
    job = _run_tool("query_data", {"query_type": "describe"})
    if job.get("job_id"):
        results["data_distribution"] = _wait_for_job(job["job_id"], timeout=120)
    
    queries = [
        ("by_mode", {"query_type": "aggregate", "group_by": ["MODE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_mode_market", {"query_type": "aggregate", "group_by": ["MODE", "MARKET"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_mode_market_drift", {"query_type": "aggregate", "group_by": ["MODE", "MARKET", "DRIFT IN / OUT"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_drift", {"query_type": "aggregate", "group_by": ["DRIFT IN / OUT"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_league", {"query_type": "aggregate", "group_by": ["LEAGUE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"], "limit": 20}),
        ("by_league_mode", {"query_type": "aggregate", "group_by": ["LEAGUE", "MODE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"], "limit": 30}),
    ]
    for idx, (name, params) in enumerate(queries):
        _log(f"Exploring {name}...")
        if progress_container:
            progress_container.text(f"🔍 Exploring {name}... ({idx+1}/{len(queries)})")
        job = _run_tool("query_data", params)
        if job.get("job_id"):
            results[name] = _wait_for_job(job["job_id"], timeout=120)
        else:
            results[name] = {"error": job.get("error", "Failed to submit")}
    if progress_container:
        progress_container.text("✅ Exploration complete")
    return results


def _analyze_exploration(bible: Dict, exploration: Dict, pl_column: str) -> Dict:
    """Deep analysis of exploration - identify ALL promising avenues with detailed reasoning."""
    
    mode_market = exploration.get("by_mode_market", {}).get("result", {}).get("groups", [])
    mode_market_drift = exploration.get("by_mode_market_drift", {}).get("result", {}).get("groups", [])
    data_dist = exploration.get("data_distribution", {}).get("result", {})
    by_mode = exploration.get("by_mode", {}).get("result", {}).get("groups", [])
    by_drift = exploration.get("by_drift", {}).get("result", {}).get("groups", [])
    
    # Extract numeric column ranges
    numeric_ranges = {}
    for col in data_dist.get("columns", []):
        if col.get("dtype") in ["float64", "int64"] and "min" in col:
            numeric_ranges[col["name"]] = {"min": col["min"], "max": col["max"], "mean": col.get("mean")}
    
    context = f"""## Deep Exploration Analysis for {pl_column}

You have exploration results from a football betting dataset. Your task is to DEEPLY ANALYZE these results and identify promising strategies.

## Numeric Column Ranges (CRITICAL - use these actual values!)
{_safe_json(numeric_ranges, 1500)}

## Overall by MODE
{_safe_json(by_mode, 500)}

## Overall by DRIFT
{_safe_json(by_drift, 500)}

## MODE + MARKET Results (sorted by mean PL, showing top 35)
{_safe_json(sorted(mode_market, key=lambda x: x.get(f'mean_{pl_column}', 0), reverse=True)[:35], 4000)}

## MODE + MARKET + DRIFT Results (sorted by mean PL, showing top 50)
{_safe_json(sorted(mode_market_drift, key=lambda x: x.get(f'mean_{pl_column}', 0), reverse=True)[:50], 5000)}

## Gates to eventually pass
{_safe_json(bible.get('gates', {}), 300)}
"""
    
    question = """Perform DEEP, THOUGHTFUL analysis of these exploration results.

THINK CAREFULLY about:
1. Which MODE performs best overall? Why might that be?
2. Which MARKET types are profitable? What does this tell us about market inefficiency?
3. How does DRIFT direction affect profitability? Is there a pattern?
4. Which specific MODE + MARKET + DRIFT combinations look most promising?
5. Are there any SURPRISING findings? Things that don't fit the pattern?
6. What's the sample size for each? (Don't trust small samples)

WRITE YOUR ANALYSIS IN DETAIL - explain your reasoning, not just list numbers.

Then identify 8-12 PRIORITIZED avenues to explore, with:
- Clear base filters
- Why you think it's promising (market inefficiency hypothesis)
- What additional filters might help
- Expected row count

Respond with JSON:
{
    "detailed_analysis": "Write 300-500 words analyzing the patterns you see, what they might mean, and your hypotheses about WHY certain combinations are profitable...",
    
    "key_observations": [
        "Observation 1 with reasoning",
        "Observation 2 with reasoning"
    ],
    
    "numeric_ranges": {
        "% DRIFT": {"min": X, "max": Y, "typical_range": "..."},
        "ACTUAL ODDS": {"min": X, "max": Y, "typical_range": "..."}
    },
    
    "all_positive_combinations": [
        {"mode": "...", "market": "...", "drift": "...", "mean_pl": X, "count": N}
    ],
    
    "prioritized_avenues": [
        {
            "rank": 1,
            "avenue": "MODE=XG, MARKET=FHGO1.5 Back",
            "base_filters": [{"col": "MODE", "op": "=", "value": "XG"}, {"col": "MARKET", "op": "=", "value": "FHGO1.5 Back"}],
            "promising_drift": "SAME",
            "market_inefficiency_hypothesis": "WHY this might be profitable - what market inefficiency does it exploit?",
            "suggested_refinements": ["Try adding ACTUAL ODDS 1.5-2.5", "Try filtering by % DRIFT > 0"],
            "expected_rows": "~2000 based on count",
            "confidence": "high/medium/low",
            "reasoning": "Detailed reasoning for why this is worth exploring"
        }
    ],
    
    "recommended_next_tools": [
        {"tool": "feature_importance", "reason": "To find which numeric features correlate with profit"},
        {"tool": "univariate_scan", "reason": "To find best single filters"}
    ]
}"""
    
    resp = _llm(context, question, max_tokens=4000)
    parsed = _parse_json(resp)
    if parsed:
        parsed["raw_analysis"] = resp  # Keep the raw response for display
    return parsed if parsed else {"detailed_analysis": resp[:1000], "prioritized_avenues": [], "all_positive_combinations": []}


# ============================================================
# Phase 2: Sweeps (Relaxed)
# ============================================================

def _run_sweeps(pl_column: str, bible: Dict, progress_container=None) -> Dict:
    results = {}
    
    relaxed_enforcement = {
        "min_train_rows": 100,
        "min_val_rows": 30,
        "min_test_rows": 30,
        "max_train_val_gap_roi": 1.0,
        "max_test_drawdown": -100,
    }
    
    if progress_container:
        progress_container.text("🔬 Running bracket_sweep...")
    _log("Running bracket_sweep...")
    job = _run_tool("bracket_sweep", {"pl_column": pl_column, "sweep_cols": ["ACTUAL ODDS", "% DRIFT"], "n_bins": 8, "enforcement": relaxed_enforcement})
    results["bracket_sweep"] = _wait_for_job(job["job_id"], timeout=180) if job.get("job_id") else {"error": job.get("error", "Failed")}
    
    if progress_container:
        progress_container.text("🔬 Running subgroup_scan...")
    _log("Running subgroup_scan...")
    job = _run_tool("subgroup_scan", {"pl_column": pl_column, "group_cols": ["MODE", "MARKET", "DRIFT IN / OUT", "LEAGUE"], "enforcement": relaxed_enforcement})
    results["subgroup_scan"] = _wait_for_job(job["job_id"], timeout=180) if job.get("job_id") else {"error": job.get("error", "Failed")}
    
    if progress_container:
        progress_container.text("✅ Sweeps complete")
    return results


# ============================================================
# BATCH TESTING: Test multiple variations at once
# ============================================================

def _test_filter_batch(base_filters: List[Dict], variations: List[Dict], pl_column: str, bible: Dict) -> List[Dict]:
    """
    Test multiple filter variations in batch.
    
    Args:
        base_filters: The base filter set
        variations: List of additional filter dicts to test one at a time
        pl_column: Target PL column
        bible: Bible with gates
    
    Returns:
        List of results for each variation
    """
    results = []
    enforcement = bible.get("gates", {})
    
    # First test the base alone
    _log(f"Testing base: {base_filters}")
    job = _run_tool("test_filter", {"filters": base_filters, "pl_column": pl_column, "enforcement": enforcement})
    if job.get("job_id"):
        base_result = _wait_for_job(job["job_id"], timeout=JOB_TIMEOUT)
        base_result["variation"] = "BASE"
        base_result["filters_tested"] = base_filters
        results.append(base_result)
    
    # Then test each variation
    for var in variations:
        combined = base_filters + [var]
        fhash = _filter_hash(combined)
        if fhash in st.session_state.tested_filter_hashes:
            continue
        st.session_state.tested_filter_hashes.add(fhash)
        
        _log(f"Testing variation: {var}")
        job = _run_tool("test_filter", {"filters": combined, "pl_column": pl_column, "enforcement": enforcement})
        if job.get("job_id"):
            result = _wait_for_job(job["job_id"], timeout=JOB_TIMEOUT)
            result["variation"] = var
            result["filters_tested"] = combined
            results.append(result)
    
    return results


def _generate_variations(exploration_analysis: Dict) -> List[Dict]:
    """Generate standard variations to test."""
    numeric_ranges = exploration_analysis.get("numeric_ranges", {})
    
    variations = [
        # Drift variations
        {"col": "DRIFT IN / OUT", "op": "=", "value": "IN"},
        {"col": "DRIFT IN / OUT", "op": "=", "value": "OUT"},
        {"col": "DRIFT IN / OUT", "op": "=", "value": "SAME"},
    ]
    
    # Odds variations based on actual data
    odds_range = numeric_ranges.get("ACTUAL ODDS", {})
    if odds_range:
        variations.extend([
            {"col": "ACTUAL ODDS", "op": "between", "min": 1.5, "max": 2.5},
            {"col": "ACTUAL ODDS", "op": "between", "min": 2.0, "max": 3.5},
            {"col": "ACTUAL ODDS", "op": "between", "min": 1.2, "max": 2.0},
        ])
    
    # % DRIFT variations
    drift_range = numeric_ranges.get("% DRIFT", {})
    if drift_range:
        variations.extend([
            {"col": "% DRIFT", "op": ">", "value": 0},
            {"col": "% DRIFT", "op": "<", "value": 0},
        ])
    
    return variations


# ============================================================
# Avenue Exploration
# ============================================================

def _explore_avenue(avenue: Dict, pl_column: str, bible: Dict, exploration_analysis: Dict) -> Dict:
    """
    Thoroughly explore one avenue with multiple variations.
    
    Returns detailed results and analysis.
    """
    base_filters = avenue.get("base_filters", [])
    avenue_name = avenue.get("avenue", "Unknown")
    
    _log(f"Exploring avenue: {avenue_name}")
    
    # Generate variations
    variations = _generate_variations(exploration_analysis)
    
    # Add avenue-specific suggested refinements
    for refinement in avenue.get("suggested_refinements", [])[:3]:
        # Try to parse refinement into a filter
        # This is a simplified version - in practice might need more parsing
        pass
    
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
    
    # Collect all results
    all_results = []
    for r in results:
        train = r.get("train", {})
        val = r.get("val", {})
        test = r.get("test", {})
        gates_passed = r.get("gates_passed", False)
        
        all_results.append({
            "variation": r.get("variation", "unknown"),
            "filters": r.get("filters_tested", []),
            "train_roi": train.get("roi", 0),
            "train_rows": train.get("rows", 0),
            "val_roi": val.get("roi", 0),
            "val_rows": val.get("rows", 0),
            "test_roi": test.get("roi", 0),
            "test_rows": test.get("rows", 0),
            "gates_passed": gates_passed,
            "gate_failures": r.get("gate_failures", []),
        })
    
    # Find best and categorize
    passing = [r for r in all_results if r["gates_passed"] and r["test_roi"] > 0]
    near_misses = [r for r in all_results if r["test_roi"] > 0 or r["train_roi"] > 0.02]
    
    best = max(all_results, key=lambda x: x["test_roi"]) if all_results else None
    best_train = max(all_results, key=lambda x: x["train_roi"]) if all_results else None
    
    # Determine recommendation
    if passing:
        recommendation = "SUCCESS - found passing strategy!"
    elif near_misses and any(r["train_roi"] > 0.03 for r in near_misses):
        recommendation = "PROMISING - refine further"
    elif best and best["train_roi"] > 0:
        recommendation = "INVESTIGATE - shows potential"
    else:
        recommendation = "MOVE_ON - limited promise"
    
    return {
        "summary": f"Tested {len(results)} variations: {len(passing)} passing, {len(near_misses)} promising",
        "best_test_roi": best["test_roi"] if best else 0,
        "best_train_roi": best_train["train_roi"] if best_train else 0,
        "best_variation": best["variation"] if best else None,
        "passing_count": len(passing),
        "near_miss_count": len(near_misses),
        "recommendation": recommendation,
        "all_results": all_results,
        "passing_filters": [p["filters"] for p in passing],
        "near_miss_filters": [n["filters"] for n in near_misses[:3]],
    }


def _deep_analyze_iteration(avenue: Dict, avenue_results: Dict, accumulated_learnings: List, avenues_remaining: int) -> Dict:
    """LLM does deep analysis after each avenue exploration."""
    
    analysis = avenue_results.get("analysis", {})
    all_results = analysis.get("all_results", [])
    
    context = f"""## Analysis After Exploring: {avenue.get('avenue', 'Unknown')}

## Results from Testing
{_safe_json(all_results, 3000)}

## Accumulated Learnings So Far
{_safe_json(accumulated_learnings[-10:], 1500)}

## Avenues Remaining: {avenues_remaining}

## Original Hypothesis
{avenue.get('market_inefficiency_hypothesis', avenue.get('why_promising', 'Not specified'))}
"""
    
    question = """Analyze these results DEEPLY. Think about:

1. What do these results tell us? Did the hypothesis hold?
2. Why might train ROI differ from test ROI?
3. Which variation performed best and why?
4. What does this teach us about the market?
5. Should we refine this avenue or move on?
6. What specific refinement would you try if you could?

Write a thoughtful analysis (200-400 words), then provide structured output.

Respond with JSON:
{
    "detailed_reasoning": "Your detailed analysis of what these results mean...",
    "hypothesis_verdict": "supported/partially_supported/rejected",
    "key_learning": "One key insight from this exploration",
    "patterns_noticed": ["pattern 1", "pattern 2"],
    "refinement_ideas": [
        {"filter_to_add": {...}, "reasoning": "why this might help"}
    ],
    "should_continue_avenue": true/false,
    "confidence_in_direction": "high/medium/low",
    "next_recommendation": "What to do next and why"
}"""
    
    resp = _llm(context, question, max_tokens=2500)
    parsed = _parse_json(resp)
    return parsed if parsed else {"detailed_reasoning": resp[:500], "key_learning": "Analysis failed to parse"}


# ============================================================
# Deep Analysis Between Phases
# ============================================================

def _deep_reflect(phase: str, findings: Dict, accumulated_learnings: List, avenues_remaining: List) -> Dict:
    """LLM reflects deeply on current state and decides next steps."""
    
    context = f"""## Current Phase: {phase}

## Latest Findings
{_safe_json(findings, 3000)}

## Accumulated Learnings So Far
{_safe_json(accumulated_learnings[-10:], 1500)}

## Avenues Remaining to Explore
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
# Combination Scanning (when base shows promise)
# ============================================================

def _run_combination_scan_for_base(base_filters: List[Dict], pl_column: str) -> Dict:
    """When a base filter shows promise, scan for best combinations."""
    
    _log(f"Running combination scan for: {base_filters}")
    job = _run_tool("combination_scan", {
        "pl_column": pl_column,
        "base_filters": base_filters,
        "scan_cols": ["MODE", "MARKET", "DRIFT IN / OUT", "LEAGUE"],
        "max_combinations": 30,
    })
    
    if job.get("job_id"):
        return _wait_for_job(job["job_id"], timeout=300)
    return {"error": job.get("error", "Failed")}


# ============================================================
# Main Agent
# ============================================================

def run_agent():
    pl_column = st.session_state.target_pl_column
    st.session_state.log = []
    st.session_state.tested_filter_hashes = set()
    st.session_state.near_misses = []
    st.session_state.accumulated_learnings = []
    st.session_state.avenues_explored = []
    
    st.markdown(f"# 🤖 Research Agent v5: {pl_column}")
    st.caption("Deep Analysis Edition - Thorough exploration before conclusions")
    
    # ========== BIBLE ==========
    with st.status("📖 Loading Bible...", expanded=True) as status:
        bible = _load_bible()
        st.markdown(_format_bible(bible))
        status.update(label="📖 Bible loaded", state="complete")
    
    with st.expander("📚 Full Bible Context", expanded=False):
        st.markdown("**Dataset Overview:**")
        st.code(_safe_json(bible.get("dataset_overview", {}), 1500), language="json")
        st.markdown("**Research Rules:**")
        st.code(_safe_json(bible.get("research_rules", []), 1500), language="json")
        st.markdown("**Gates:**")
        st.code(_safe_json(bible.get("gates", {}), 500), language="json")
    
    _append("assistant", _format_bible(bible))
    
    # ========== EXPLORATION ==========
    with st.status("🔍 Phase 1: Exploration...", expanded=True) as status:
        progress = st.empty()
        exploration = _run_exploration(pl_column, progress)
        st.session_state.exploration_results = exploration
        status.update(label="🔍 Exploration complete", state="complete")
    
    with st.expander("Exploration Results", expanded=False):
        st.code(_safe_json(exploration, 5000), language="json")
    
    # ========== DEEP ANALYSIS ==========
    st.markdown("### 🧠 Phase 1b: Deep Analysis")
    with st.status("🧠 Analyzing exploration...", expanded=True) as status:
        exploration_analysis = _analyze_exploration(bible, exploration, pl_column)
        st.session_state.exploration_analysis = exploration_analysis
        status.update(label="🧠 Analysis complete", state="complete")
    
    # Display analysis
    st.markdown(f"**Deep Analysis:** {exploration_analysis.get('deep_analysis', 'N/A')[:500]}")
    
    # Show all positive combinations
    all_positive = exploration_analysis.get("all_positive_combinations", [])
    if all_positive:
        st.markdown(f"**Found {len(all_positive)} positive MODE+MARKET combinations:**")
        for combo in all_positive[:10]:
            st.markdown(f"- {combo.get('mode')} + {combo.get('market')} (mean PL: {combo.get('mean_pl', 0):.4f}, count: {combo.get('count', 0)}, best drift: {combo.get('best_drift', '?')})")
    
    # Show prioritized avenues
    avenues = exploration_analysis.get("prioritized_avenues", [])
    st.session_state.avenues_to_explore = avenues
    
    if avenues:
        st.markdown(f"**{len(avenues)} Prioritized Avenues to Explore:**")
        for av in avenues[:8]:
            st.markdown(f"- **#{av.get('rank', '?')}**: {av.get('avenue')} - {av.get('why_promising', '')[:80]}")
    
    with st.expander("Full Analysis", expanded=False):
        st.code(_safe_json(exploration_analysis, 4000), language="json")
    
    _add_learning(f"Exploration found {len(all_positive)} positive combinations, {len(avenues)} avenues to explore")
    
    # ========== SWEEPS ==========
    with st.status("🔬 Phase 2: Segment Sweeps...", expanded=True) as status:
        progress = st.empty()
        sweeps = _run_sweeps(pl_column, bible, progress)
        st.session_state.sweep_results = sweeps
        top_brackets = len(sweeps.get("bracket_sweep", {}).get("top_brackets", []))
        top_subgroups = len(sweeps.get("subgroup_scan", {}).get("top_groups", []))
        status.update(label=f"🔬 Sweeps complete ({top_brackets}b, {top_subgroups}s)", state="complete")
    
    st.markdown(f"**Found {top_brackets} bracket patterns, {top_subgroups} subgroup patterns**")
    with st.expander("Sweep Results", expanded=False):
        st.code(_safe_json(sweeps, 3000), language="json")
    
    # ========== AVENUE EXPLORATION ==========
    st.markdown("---")
    st.markdown("### 🧪 Phase 3: Avenue Exploration")
    st.markdown("*Testing each avenue with multiple variations*")
    
    avenues_remaining = list(avenues)
    success_found = False
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        st.session_state.agent_iteration = iteration
        
        if not avenues_remaining:
            st.warning("All avenues explored!")
            break
        
        # Pick next avenue
        current_avenue = avenues_remaining.pop(0)
        st.session_state.avenues_explored.append(current_avenue)
        
        st.markdown(f"#### Iteration {iteration}: {current_avenue.get('avenue', 'Unknown')}")
        st.markdown(f"*{current_avenue.get('why_promising', '')}*")
        
        # Explore avenue with batch testing
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
        
        # NEW: Deep analysis of this iteration
        st.markdown("**🧠 Analysis:**")
        iteration_analysis = _deep_analyze_iteration(
            current_avenue, 
            avenue_results, 
            st.session_state.accumulated_learnings,
            len(avenues_remaining)
        )
        
        # Display the detailed reasoning
        detailed = iteration_analysis.get("detailed_reasoning", "")
        if detailed:
            st.markdown(detailed[:600] + "..." if len(detailed) > 600 else detailed)
        
        st.markdown(f"*Key learning: {iteration_analysis.get('key_learning', 'N/A')}*")
        
        # Add to accumulated learnings
        _add_learning(iteration_analysis.get("key_learning", f"Explored {current_avenue.get('avenue')}"))
        
        with st.expander("Full Analysis", expanded=False):
            st.code(_safe_json(iteration_analysis, 2000), language="json")
        
        # Handle recommendation
        recommendation = analysis.get("recommendation", "")
        
        if "SUCCESS" in recommendation:
            st.markdown("# 🎉 Strategy Found!")
            success_found = True
            
            # Get passing filters - show them prominently!
            passing_filters = analysis.get("passing_filters", [[]])
            best_result = None
            for r in analysis.get("all_results", []):
                if r.get("gates_passed") and r.get("test_roi", 0) > 0:
                    best_result = r
                    break
            
            # SHOW THE STRATEGY CLEARLY
            st.markdown("## 📋 Winning Strategy")
            st.markdown("### Filters:")
            if passing_filters and passing_filters[0]:
                for f in passing_filters[0]:
                    if f.get("op") == "=":
                        st.markdown(f"- **{f.get('col')}** = `{f.get('value')}`")
                    elif f.get("op") == "between":
                        st.markdown(f"- **{f.get('col')}** between `{f.get('min')}` and `{f.get('max')}`")
                    else:
                        st.markdown(f"- **{f.get('col')}** {f.get('op')} `{f.get('value')}`")
            
            # Show performance
            if best_result:
                st.markdown("### Performance:")
                col1, col2, col3 = st.columns(3)
                col1.metric("Train ROI", f"{best_result.get('train_roi', 0):.2%}")
                col2.metric("Val ROI", f"{best_result.get('val_roi', 0):.2%}")
                col3.metric("Test ROI", f"{best_result.get('test_roi', 0):.2%}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Train Rows", best_result.get('train_rows', 0))
                col2.metric("Val Rows", best_result.get('val_rows', 0))
                col3.metric("Test Rows", best_result.get('test_rows', 0))
            
            # Copy-pasteable format
            st.markdown("### Copy-Paste Format:")
            filter_str = json.dumps(passing_filters[0] if passing_filters else [], indent=2)
            st.code(filter_str, language="json")
            
            st.balloons()
            
            # Optional: Advanced validation (don't block on it)
            with st.expander("🔬 Run Advanced Validation (optional)", expanded=False):
                if st.button("Run Forward Walk & Monte Carlo"):
                    with st.spinner("Running forward walk..."):
                        job = _run_tool("forward_walk", {"filters": passing_filters[0] if passing_filters else [], "pl_column": pl_column, "n_windows": 6})
                        if job.get("job_id"):
                            fw_result = _wait_for_job(job["job_id"], timeout=180)
                            st.markdown(f"**Forward Walk:** {fw_result.get('verdict', 'N/A')}")
                            st.code(_safe_json(fw_result, 2000), language="json")
                    
                    with st.spinner("Running monte carlo..."):
                        job = _run_tool("monte_carlo_sim", {"filters": passing_filters[0] if passing_filters else [], "pl_column": pl_column, "n_simulations": 500})
                        if job.get("job_id"):
                            mc_result = _wait_for_job(job["job_id"], timeout=180)
                            prob = mc_result.get('probability', {}).get('positive_roi', 0)
                            st.markdown(f"**Monte Carlo:** {prob:.1%} probability positive")
                            st.code(_safe_json(mc_result, 2000), language="json")
            
            # Log to Bible
            _run_tool("append_research_note", {
                "note": json.dumps({
                    "type": "SUCCESS",
                    "pl_column": pl_column,
                    "filters": passing_filters[0] if passing_filters else [],
                    "test_roi": best_result.get("test_roi") if best_result else 0,
                    "train_roi": best_result.get("train_roi") if best_result else 0,
                }),
                "tags": f"success,{pl_column.replace(' ', '_')}"
            })
            
            st.session_state.agent_phase = "complete"
            return
        
        elif "PROMISING" in recommendation or "INVESTIGATE" in recommendation:
            # Add near-misses for potential refinement
            near_miss_filters = analysis.get("near_miss_filters", [])
            for nmf in near_miss_filters:
                st.session_state.near_misses.append({
                    "iteration": iteration,
                    "filters": nmf,
                    "avenue": current_avenue.get("avenue"),
                    "train_roi": analysis.get("best_train_roi", 0),
                })
            
            # If LLM analysis says continue, offer to run combination scan
            if iteration_analysis.get("should_continue_avenue") and analysis.get("best_train_roi", 0) > 0.02:
                st.markdown("**Running combination scan for refinement...**")
                combo_result = _run_combination_scan_for_base(
                    current_avenue.get("base_filters", []), 
                    pl_column
                )
                if combo_result.get("promising_combinations"):
                    st.markdown(f"Found {len(combo_result.get('promising_combinations', []))} promising combinations!")
                    with st.expander("Combination Scan Results"):
                        st.code(_safe_json(combo_result, 2000), language="json")
        
        st.session_state.agent_findings.append({"iteration": iteration, "avenue": current_avenue, "results": avenue_results, "analysis": iteration_analysis})
        
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
            
            # Reorder avenues if needed
            if reflection.get("priority_avenue"):
                st.markdown(f"**Prioritizing:** {reflection.get('priority_avenue')}")
        
        st.markdown("---")
    
    # ========== FINAL ANALYSIS ==========
    if not success_found:
        st.markdown("### 📋 Final Analysis")
        
        # Check if we have near-misses worth more investigation
        if st.session_state.near_misses:
            st.markdown(f"**{len(st.session_state.near_misses)} near-misses found** - may warrant further investigation")
            for nm in st.session_state.near_misses[:5]:
                st.markdown(f"- {nm.get('avenue', 'Unknown')}: `{nm.get('filters')}`")
        
        # Summarize learnings
        if st.session_state.accumulated_learnings:
            st.markdown("**Key Learnings:**")
            for learning in st.session_state.accumulated_learnings[-10:]:
                st.markdown(f"- {learning.get('learning', '')}")
        
        st.markdown("**Recommendation:** Review near-misses manually or try different PL column")
    
    st.session_state.agent_phase = "complete"


# ============================================================
# UI
# ============================================================

st.title("⚽ Football Research Agent v5")
st.caption("Deep Analysis Edition - Tests multiple variations per avenue, never gives up prematurely")

with st.sidebar:
    st.header("🎛️ Controls")
    pl_col = st.selectbox("Market", ["BO 2.5 PL", "BTTS PL", "SHG PL", "SHG 2+ PL", "LU1.5 PL", "LFGHU0.5 PL", "BO1.5 FHG PL"])
    st.session_state.target_pl_column = pl_col
    
    if st.button("🚀 Start Research", type="primary"):
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
        st.session_state.run_requested = True
        st.rerun()
    
    st.divider()
    if st.session_state.agent_phase == "running":
        st.warning(f"🔄 Iteration {st.session_state.agent_iteration}")
        st.metric("Avenues Explored", len(st.session_state.avenues_explored))
        st.metric("Remaining", len(st.session_state.avenues_to_explore))
    elif st.session_state.agent_phase == "complete":
        st.success("✅ Done")
        st.metric("Near-misses", len(st.session_state.near_misses))
        st.metric("Learnings", len(st.session_state.accumulated_learnings))
    
    st.divider()
    if st.button("🗑️ Clear"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    
    if st.session_state.log:
        with st.expander("📋 Log"):
            for line in st.session_state.log[-30:]:
                st.text(line)
    st.caption("v5.0 - Deep Analysis")

if st.session_state.run_requested:
    st.session_state.run_requested = False
    run_agent()
elif st.session_state.agent_phase == "idle":
    st.info("👆 Click **Start Research**")
    with st.expander("🆕 What's New in v5"):
        st.markdown("""
**v5 - Deep Analysis Edition:**

1. **BATCH TESTING**: Each avenue tested with 8-10 variations (drift, odds ranges, etc.)
2. **DEEP ANALYSIS**: LLM analyzes results after EVERY phase
3. **NO PREMATURE CONCLUSIONS**: Must explore ALL avenues before giving up
4. **COMBINATION SCANNING**: When base filter shows promise, auto-scan for additive filters
5. **LEARNING ACCUMULATION**: Tracks insights across iterations
6. **AVENUE TRACKING**: Explicitly shows which avenues explored vs remaining
7. **MID-POINT REFLECTION**: Every 3 iterations, LLM reflects and re-prioritizes
""")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
