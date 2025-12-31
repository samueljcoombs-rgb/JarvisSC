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
    try:
        if "log" not in st.session_state:
            st.session_state.log = []
        st.session_state.log.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")
    except Exception:
        pass  # Silently fail if session state not ready

def _append(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

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
        pass  # Silently fail if session state not ready

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
    
    # PRE-PROCESS: For each MODE+MARKET, find the best DRIFT
    # This helps the LLM and ensures we don't lose this info
    best_drift_by_mode_market = {}
    for item in mode_market_drift:
        mode = item.get("MODE", "")
        market = item.get("MARKET", "")
        drift = item.get("DRIFT IN / OUT", "")
        mean_pl = item.get(f"mean_{pl_column}", 0)
        count = item.get("count", 0)
        
        key = f"{mode}|{market}"
        if key not in best_drift_by_mode_market or mean_pl > best_drift_by_mode_market[key]["mean_pl"]:
            best_drift_by_mode_market[key] = {
                "drift": drift,
                "mean_pl": mean_pl,
                "count": count
            }
    
    # Create enriched MODE+MARKET data with best drift info
    enriched_mode_market = []
    for item in mode_market:
        mode = item.get("MODE", "")
        market = item.get("MARKET", "")
        mean_pl = item.get(f"mean_{pl_column}", 0)
        count = item.get("count", 0)
        
        key = f"{mode}|{market}"
        best_drift_info = best_drift_by_mode_market.get(key, {})
        
        enriched_mode_market.append({
            "MODE": mode,
            "MARKET": market,
            "mean_pl": mean_pl,
            "count": count,
            "best_drift": best_drift_info.get("drift", "unknown"),
            "best_drift_mean_pl": best_drift_info.get("mean_pl", 0),
            "best_drift_count": best_drift_info.get("count", 0)
        })
    
    # Sort by mean PL
    enriched_mode_market.sort(key=lambda x: x["mean_pl"], reverse=True)
    
    context = f"""## Deep Exploration Analysis for {pl_column}

You have exploration results from a football betting dataset. Your task is to DEEPLY ANALYZE these results and identify promising strategies.

## Numeric Column Ranges (CRITICAL - use these actual values!)
{_safe_json(numeric_ranges, 1500)}

## Overall by MODE (how each mode performs across all markets)
{_safe_json(by_mode, 600)}

## Overall by DRIFT (how each drift direction performs)
{_safe_json(by_drift, 600)}

## MODE + MARKET Results WITH BEST DRIFT (sorted by mean PL, top 40)
This shows each MODE+MARKET combination with its best performing DRIFT direction:
{_safe_json(enriched_mode_market[:40], 5000)}

## Detailed MODE + MARKET + DRIFT breakdown (top 60 combinations)
{_safe_json(sorted(mode_market_drift, key=lambda x: x.get(f'mean_{pl_column}', 0), reverse=True)[:60], 6000)}

## Gates to eventually pass
{_safe_json(bible.get('gates', {}), 300)}
"""
    
    question = """Perform DEEP, THOUGHTFUL analysis of these exploration results.

## YOUR ANALYSIS MUST INCLUDE:

### 1. MODE Analysis (2-3 sentences each)
- Which MODE performs best overall and WHY might that be?
- Are there MODEs to avoid? Why?

### 2. MARKET Analysis (2-3 sentences each)
- Which MARKET types show profit potential?
- What might this tell us about bookmaker pricing inefficiencies?

### 3. DRIFT Analysis (2-3 sentences each)  
- How does DRIFT direction (IN/OUT/SAME) affect profitability?
- Is there a consistent pattern or does it depend on MODE/MARKET?

### 4. Top Combinations (be specific!)
- List the top 5-8 MODE+MARKET+DRIFT combinations with their exact numbers
- For each, explain WHY it might be profitable (market inefficiency hypothesis)

### 5. Sample Size Concerns
- Which combinations have good sample sizes (>500)?
- Which are promising but need more data (<200)?

### 6. Surprises or Anomalies
- Anything unexpected in the data?
- Any contradictions to investigate?

Respond with JSON:
{
    "detailed_analysis": "Write 400-600 words with your full analysis covering all 6 points above. Be specific with numbers and reasoning...",
    
    "key_observations": [
        "Observation 1 with specific numbers and reasoning",
        "Observation 2 with specific numbers and reasoning",
        "Observation 3 with specific numbers and reasoning"
    ],
    
    "mode_summary": {
        "best_mode": "...",
        "best_mode_reason": "...",
        "worst_mode": "...",
        "worst_mode_reason": "..."
    },
    
    "drift_summary": {
        "best_drift": "IN/OUT/SAME",
        "best_drift_reason": "...",
        "drift_depends_on": "explanation of when different drifts work"
    },
    
    "numeric_ranges": {
        "% DRIFT": {"min": X, "max": Y, "typical_range": "..."},
        "ACTUAL ODDS": {"min": X, "max": Y, "typical_range": "..."}
    },
    
    "all_positive_combinations": [
        {"mode": "Quick League", "market": "O2.5 Back", "best_drift": "SAME", "mean_pl": 0.101, "count": 235}
    ],
    
    "prioritized_avenues": [
        {
            "rank": 1,
            "avenue": "MODE=Quick League, MARKET=O2.5 Back",
            "base_filters": [{"col": "MODE", "op": "=", "value": "Quick League"}, {"col": "MARKET", "op": "=", "value": "O2.5 Back"}],
            "promising_drift": "SAME",
            "market_inefficiency_hypothesis": "WHY this might be profitable - be specific about the market inefficiency",
            "suggested_refinements": ["Try adding ACTUAL ODDS 1.5-2.5", "Try filtering by % DRIFT > 0"],
            "expected_rows": "~2000 based on count",
            "confidence": "high/medium/low",
            "reasoning": "Detailed reasoning for why this is worth exploring"
        }
    ],
    
    "sample_size_concerns": [
        {"combination": "...", "count": N, "concern": "..."}
    ],
    
    "recommended_next_tools": [
        {"tool": "feature_importance", "reason": "To find which numeric features correlate with profit"},
        {"tool": "univariate_scan", "reason": "To find best single filters"}
    ]
}"""
    
    resp = _llm(context, question, max_tokens=5000)
    parsed = _parse_json(resp)
    
    # If LLM didn't extract best_drift properly, fill it in from our pre-processing
    if parsed and "all_positive_combinations" in parsed:
        for combo in parsed["all_positive_combinations"]:
            if combo.get("best_drift") in [None, "?", "", "unknown"]:
                key = f"{combo.get('mode', '')}|{combo.get('market', '')}"
                if key in best_drift_by_mode_market:
                    combo["best_drift"] = best_drift_by_mode_market[key]["drift"]
    
    if parsed:
        parsed["raw_analysis"] = resp  # Keep the raw response for display
        parsed["_enriched_mode_market"] = enriched_mode_market[:20]  # Store for reference
    
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
        base_result["variation"] = "BASE (no additional filters)"
        base_result["filters_tested"] = base_filters
        results.append(base_result)
    
    # Then test each variation
    for var in variations:
        combined = base_filters + [var]
        fhash = _filter_hash(combined)
        
        # Safe access to tested_filter_hashes
        if "tested_filter_hashes" not in st.session_state:
            st.session_state.tested_filter_hashes = set()
        if fhash in st.session_state.tested_filter_hashes:
            continue
        st.session_state.tested_filter_hashes.add(fhash)
        
        # Create descriptive variation name
        if var.get("op") == "=":
            var_name = f"{var.get('col')} = {var.get('value')}"
        elif var.get("op") == "between":
            var_name = f"{var.get('col')} {var.get('min')}-{var.get('max')}"
        else:
            var_name = f"{var.get('col')} {var.get('op')} {var.get('value')}"
        
        _log(f"Testing variation: {var_name}")
        job = _run_tool("test_filter", {"filters": combined, "pl_column": pl_column, "enforcement": enforcement})
        if job.get("job_id"):
            result = _wait_for_job(job["job_id"], timeout=JOB_TIMEOUT)
            result["variation"] = f"+ {var_name}"
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
    
    # Collect all results with quality metrics
    all_results = []
    high_roi_insufficient_sample = []  # Track high ROI but failed sample gates
    
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
        
        # CRITICAL: A good strategy needs positive train AND val
        # Test positive with negative train/val is likely NOISE
        train_positive = train_roi > 0
        val_positive = val_roi > -0.01  # Allow slightly negative val
        test_positive = test_roi > 0
        
        # Check if this is a high ROI strategy that failed due to sample size
        sample_size_failure = any("rows" in str(f).lower() or "sample" in str(f).lower() for f in gate_failures)
        if not sample_size_failure:
            # Check if rows are below thresholds
            sample_size_failure = train_rows < 300 or val_rows < 60 or test_rows < 60
        
        if test_roi > 0.05 and not gates_passed and sample_size_failure:
            high_roi_insufficient_sample.append({
                "variation": r.get("variation", "unknown"),
                "filters": r.get("filters_tested", []),
                "train_roi": train_roi,
                "train_rows": train_rows,
                "val_roi": val_roi,
                "val_rows": val_rows,
                "test_roi": test_roi,
                "test_rows": test_rows,
                "gate_failures": gate_failures,
                "note": "High ROI but insufficient sample size - needs more data"
            })
        
        # Quality score considers ALL splits, not just test
        if train_positive and val_positive and test_positive:
            # Best case: all positive
            consistency_bonus = 1.0
        elif train_positive and test_positive:
            # Train and test positive, val weak
            consistency_bonus = 0.6
        elif train_positive:
            # Only train positive - might be overfitting
            consistency_bonus = 0.3
        else:
            # Train negative - this is noise, not signal
            consistency_bonus = 0.0
        
        # Penalize large train/val gaps (overfitting signal)
        gap_penalty = max(0, 1.0 - train_val_gap / 0.05)  # Penalize gaps > 5%
        
        # Quality score
        quality_score = test_roi * consistency_bonus * gap_penalty
        
        # Determine if truly passing (stricter than just gates)
        truly_passing = (
            gates_passed and 
            train_roi > 0 and  # Train MUST be positive
            val_roi > -0.02 and  # Val can't be too negative
            test_roi > 0 and  # Test must be positive
            train_val_gap < 0.06  # Gap must be reasonable
        )
        
        all_results.append({
            "variation": r.get("variation", "unknown"),
            "filters": r.get("filters_tested", []),
            "train_roi": train_roi,
            "train_rows": train_rows,
            "val_roi": val_roi,
            "val_rows": val_rows,
            "test_roi": test_roi,
            "test_rows": test_rows,
            "train_val_gap": round(train_val_gap, 4),
            "quality_score": round(quality_score, 6),
            "gates_passed": gates_passed,
            "truly_passing": truly_passing,
            "gate_failures": gate_failures,
            "train_positive": train_positive,
            "val_positive": val_positive,
            "test_positive": test_positive,
        })
    
    # Find TRULY passing strategies (not just technically passing gates)
    truly_passing = [r for r in all_results if r["truly_passing"]]
    
    # Near misses: positive train, positive test, but something off
    near_misses = [r for r in all_results if 
                   r["train_roi"] > 0.01 and r["test_roi"] > 0 and not r["truly_passing"]]
    
    # Interesting: positive train even if test failed
    interesting = [r for r in all_results if r["train_roi"] > 0.02]
    
    # Sort by quality score
    truly_passing.sort(key=lambda x: x["quality_score"], reverse=True)
    near_misses.sort(key=lambda x: x["quality_score"], reverse=True)
    high_roi_insufficient_sample.sort(key=lambda x: x["test_roi"], reverse=True)
    
    best_quality = max(all_results, key=lambda x: x["quality_score"]) if all_results else None
    best_test = max(all_results, key=lambda x: x["test_roi"]) if all_results else None
    best_train = max(all_results, key=lambda x: x["train_roi"]) if all_results else None
    
    # Determine recommendation based on STRICT criteria
    if truly_passing:
        recommendation = "SUCCESS - found strategy with positive train/val/test!"
    elif near_misses:
        recommendation = "PROMISING - positive train+test but needs refinement"
    elif high_roi_insufficient_sample:
        recommendation = "INVESTIGATE - high ROI found but needs more data"
    elif interesting:
        recommendation = "INVESTIGATE - positive train, explore further"
    else:
        recommendation = "MOVE_ON - no consistent signal found"
    
    return {
        "summary": f"Tested {len(all_results)} variations: {len(truly_passing)} truly passing, {len(near_misses)} near-misses, {len(interesting)} interesting",
        "best_test_roi": best_test["test_roi"] if best_test else 0,
        "best_train_roi": best_train["train_roi"] if best_train else 0,
        "best_quality_score": best_quality["quality_score"] if best_quality else 0,
        "best_variation": best_quality["variation"] if best_quality else None,
        "passing_count": len(truly_passing),
        "near_miss_count": len(near_misses),
        "interesting_count": len(interesting),
        "high_roi_insufficient_sample": high_roi_insufficient_sample,  # NEW: Track these!
        "recommendation": recommendation,
        "all_results": all_results,
        "passing_filters": [p["filters"] for p in truly_passing],
        "near_miss_filters": [n["filters"] for n in near_misses[:5]],
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
    
    # Display DETAILED analysis - not just a snippet
    detailed_analysis = exploration_analysis.get('detailed_analysis', '')
    if detailed_analysis:
        st.markdown("#### 📝 Analysis Summary")
        st.markdown(detailed_analysis)
    
    # Show key observations
    key_obs = exploration_analysis.get("key_observations", [])
    if key_obs:
        st.markdown("#### 🔍 Key Observations")
        for i, obs in enumerate(key_obs, 1):
            st.markdown(f"{i}. {obs}")
    
    # Show mode summary
    mode_summary = exploration_analysis.get("mode_summary", {})
    if mode_summary:
        st.markdown("#### 📊 MODE Summary")
        st.markdown(f"- **Best MODE:** {mode_summary.get('best_mode', 'N/A')} - {mode_summary.get('best_mode_reason', '')}")
        st.markdown(f"- **Worst MODE:** {mode_summary.get('worst_mode', 'N/A')} - {mode_summary.get('worst_mode_reason', '')}")
    
    # Show drift summary
    drift_summary = exploration_analysis.get("drift_summary", {})
    if drift_summary:
        st.markdown("#### 📈 DRIFT Summary")
        st.markdown(f"- **Best DRIFT:** {drift_summary.get('best_drift', 'N/A')} - {drift_summary.get('best_drift_reason', '')}")
        st.markdown(f"- **Pattern:** {drift_summary.get('drift_depends_on', 'N/A')}")
    
    # Show all positive combinations with BEST DRIFT
    all_positive = exploration_analysis.get("all_positive_combinations", [])
    if all_positive:
        st.markdown(f"#### ✅ Found {len(all_positive)} Positive MODE+MARKET Combinations")
        for combo in all_positive[:10]:
            drift_text = combo.get('best_drift', '?')
            st.markdown(f"- **{combo.get('mode')} + {combo.get('market')}** → mean PL: `{combo.get('mean_pl', 0):.4f}`, count: `{combo.get('count', 0)}`, best drift: **{drift_text}**")
    
    # Show sample size concerns
    sample_concerns = exploration_analysis.get("sample_size_concerns", [])
    if sample_concerns:
        st.markdown("#### ⚠️ Sample Size Concerns")
        for concern in sample_concerns[:5]:
            st.markdown(f"- {concern.get('combination', 'N/A')}: {concern.get('count', 0)} rows - {concern.get('concern', '')}")
    
    # Show prioritized avenues
    avenues = exploration_analysis.get("prioritized_avenues", [])
    st.session_state.avenues_to_explore = avenues
    
    if avenues:
        st.markdown(f"#### 🎯 {len(avenues)} Prioritized Avenues to Explore")
        for av in avenues[:8]:
            rank = av.get('rank', '?')
            avenue = av.get('avenue', '')
            drift = av.get('promising_drift', '?')
            hypothesis = av.get('market_inefficiency_hypothesis', av.get('why_promising', ''))[:100]
            confidence = av.get('confidence', '?')
            st.markdown(f"**#{rank}: {avenue}** (drift: {drift}, confidence: {confidence})")
            st.markdown(f"   *{hypothesis}...*")
    
    with st.expander("📄 Full Analysis JSON", expanded=False):
        st.code(_safe_json(exploration_analysis, 6000), language="json")
    
    _add_learning(f"Exploration found {len(all_positive)} positive combinations, {len(avenues)} avenues to explore")
    
    # ========== SWEEPS ==========
    with st.status("🔬 Phase 2: Segment Sweeps...", expanded=True) as status:
        progress = st.empty()
        sweeps = _run_sweeps(pl_column, bible, progress)
        st.session_state.sweep_results = sweeps
        top_brackets = len(sweeps.get("bracket_sweep", {}).get("top_brackets", []))
        top_subgroups = len(sweeps.get("subgroup_scan", {}).get("top_groups", []))
        status.update(label=f"🔬 Sweeps complete ({top_brackets}b, {top_subgroups}s)", state="complete")
    
    st.markdown("#### 📊 Sweep Results")
    st.markdown(f"Found **{top_brackets}** bracket patterns and **{top_subgroups}** subgroup patterns")
    
    # Show top bracket findings
    top_bracket_results = sweeps.get("bracket_sweep", {}).get("top_brackets", [])
    if top_bracket_results:
        st.markdown("**Top Bracket Patterns:**")
        for i, br in enumerate(top_bracket_results[:3], 1):
            rule = br.get("rule", [])
            test_roi = br.get("test", {}).get("roi", 0)
            test_rows = br.get("test", {}).get("rows", 0)
            train_roi = br.get("train", {}).get("roi", 0)
            rule_str = ", ".join([f"{r.get('col')} {r.get('op')} {r.get('min', r.get('value', ''))}-{r.get('max', '')}" for r in rule])
            st.markdown(f"  {i}. `{rule_str}` → Train: {train_roi:.2%}, Test: {test_roi:.2%} ({test_rows} rows)")
    
    # Show top subgroup findings
    top_subgroup_results = sweeps.get("subgroup_scan", {}).get("top_groups", [])
    if top_subgroup_results:
        st.markdown("**Top Subgroup Patterns:**")
        for i, sg in enumerate(top_subgroup_results[:3], 1):
            test_roi = sg.get("test", {}).get("roi", 0)
            test_rows = sg.get("test", {}).get("rows", 0)
            train_roi = sg.get("train", {}).get("roi", 0)
            group_key = sg.get("group_key", {})
            key_str = ", ".join([f"{k}={v}" for k, v in group_key.items() if k not in ["count", "mean", "sum"]])
            st.markdown(f"  {i}. `{key_str}` → Train: {train_roi:.2%}, Test: {test_roi:.2%} ({test_rows} rows)")
    
    with st.expander("📄 Full Sweep Results", expanded=False):
        st.code(_safe_json(sweeps, 4000), language="json")
    
    _add_learning(f"Sweeps found {top_brackets} bracket patterns and {top_subgroups} subgroup patterns")
    
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
            
            # Get TRULY passing strategies (stricter than just gates_passed)
            all_results = analysis.get("all_results", [])
            truly_passing_strategies = [r for r in all_results if r.get("truly_passing", False)]
            truly_passing_strategies.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
            
            # Also get "near miss" strategies for display
            near_miss_strategies = [r for r in all_results 
                                    if r.get("train_roi", 0) > 0 
                                    and r.get("test_roi", 0) > 0 
                                    and not r.get("truly_passing", False)]
            near_miss_strategies.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
            
            if not truly_passing_strategies:
                st.warning("⚠️ No strategies passed strict quality checks (positive train + val + test with consistency).")
                st.markdown("However, found some **near-miss** strategies:")
                best_result = near_miss_strategies[0] if near_miss_strategies else None
                passing_strategies = near_miss_strategies
                is_truly_passing = False
            else:
                best_result = truly_passing_strategies[0]
                passing_strategies = truly_passing_strategies
                is_truly_passing = True
            
            # SHOW THE BEST STRATEGY CLEARLY
            if is_truly_passing:
                st.markdown("## 📋 Best Validated Strategy")
            else:
                st.markdown("## 📋 Best Near-Miss Strategy (needs refinement)")
            
            if best_result and best_result.get("filters"):
                st.markdown("### Filters:")
                for f in best_result.get("filters", []):
                    if f.get("op") == "=":
                        st.markdown(f"- **{f.get('col')}** = `{f.get('value')}`")
                    elif f.get("op") == "between":
                        st.markdown(f"- **{f.get('col')}** between `{f.get('min')}` and `{f.get('max')}`")
                    else:
                        st.markdown(f"- **{f.get('col')}** {f.get('op')} `{f.get('value')}`")
                
                # Show performance
                st.markdown("### Performance:")
                col1, col2, col3 = st.columns(3)
                col1.metric("Train ROI", f"{best_result.get('train_roi', 0):.2%}")
                col2.metric("Val ROI", f"{best_result.get('val_roi', 0):.2%}")
                col3.metric("Test ROI", f"{best_result.get('test_roi', 0):.2%}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Train Rows", best_result.get('train_rows', 0))
                col2.metric("Val Rows", best_result.get('val_rows', 0))
                col3.metric("Test Rows", best_result.get('test_rows', 0))
                
                # Quality assessment - BE HONEST
                train_roi = best_result.get('train_roi', 0)
                val_roi = best_result.get('val_roi', 0)
                test_roi = best_result.get('test_roi', 0)
                train_val_gap = abs(train_roi - val_roi)
                test_rows = best_result.get('test_rows', 0)
                
                st.markdown("### Quality Assessment:")
                
                issues = []
                goods = []
                
                # Check train
                if train_roi < 0:
                    issues.append(f"⚠️ **Train ROI is NEGATIVE** ({train_roi:.2%}) - strategy may be noise")
                elif train_roi < 0.01:
                    issues.append(f"⚠️ Train ROI is weak ({train_roi:.2%})")
                else:
                    goods.append(f"✅ Train ROI positive ({train_roi:.2%})")
                
                # Check val
                if val_roi < -0.02:
                    issues.append(f"⚠️ **Val ROI is negative** ({val_roi:.2%}) - overfitting risk")
                elif val_roi < 0:
                    issues.append(f"⚠️ Val ROI slightly negative ({val_roi:.2%})")
                else:
                    goods.append(f"✅ Val ROI acceptable ({val_roi:.2%})")
                
                # Check train/val gap
                if train_val_gap > 0.05:
                    issues.append(f"⚠️ Train/Val gap is {train_val_gap:.1%} - possible overfitting")
                else:
                    goods.append(f"✅ Train/Val gap reasonable ({train_val_gap:.1%})")
                
                # Check test
                if test_roi < 0.01:
                    issues.append(f"⚠️ Test ROI is low ({test_roi:.2%}) - might be noise")
                else:
                    goods.append(f"✅ Test ROI positive ({test_roi:.2%})")
                
                # Check sample size
                if test_rows < 200:
                    issues.append(f"⚠️ Only {test_rows} test rows - small sample")
                else:
                    goods.append(f"✅ Decent sample size ({test_rows} test rows)")
                
                # Display assessment
                for good in goods:
                    st.markdown(good)
                for issue in issues:
                    st.markdown(issue)
                
                # Overall verdict
                if len(issues) == 0:
                    st.success("🎯 **STRONG STRATEGY** - Consistent across all splits!")
                elif len(issues) <= 2 and train_roi > 0:
                    st.info("📊 **MODERATE STRATEGY** - Some concerns but worth monitoring")
                else:
                    st.warning("⚠️ **WEAK STRATEGY** - Multiple red flags, likely noise")
                
                # Copy-pasteable format
                st.markdown("### Copy-Paste Format:")
                filter_str = json.dumps(best_result.get("filters", []), indent=2)
                st.code(filter_str, language="json")
            
            # Show ALL strategies with quality indicators
            if len(passing_strategies) > 1:
                title = f"## 📊 All {len(passing_strategies)} Strategies"
                if not is_truly_passing:
                    title += " (Near-Misses)"
                st.markdown(title)
                
                for i, strat in enumerate(passing_strategies, 1):
                    train_roi = strat.get('train_roi', 0)
                    val_roi = strat.get('val_roi', 0)
                    test_roi = strat.get('test_roi', 0)
                    
                    # Quality indicator
                    if train_roi > 0 and val_roi > -0.01 and test_roi > 0:
                        quality_icon = "🟢"  # Good
                    elif train_roi > 0 and test_roi > 0:
                        quality_icon = "🟡"  # Okay
                    else:
                        quality_icon = "🔴"  # Weak
                    
                    with st.expander(f"{quality_icon} Strategy {i}: Test {test_roi:.2%} | Train {train_roi:.2%} ({strat.get('variation', 'unknown')})"):
                        st.markdown(f"**Variation:** {strat.get('variation', 'N/A')}")
                        
                        # Show with color coding
                        train_color = "green" if train_roi > 0 else "red"
                        val_color = "green" if val_roi > -0.01 else "red"
                        test_color = "green" if test_roi > 0 else "red"
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Train ROI", f"{train_roi:.2%}", delta=f"{strat.get('train_rows', 0)} rows")
                        col2.metric("Val ROI", f"{val_roi:.2%}", delta=f"{strat.get('val_rows', 0)} rows")
                        col3.metric("Test ROI", f"{test_roi:.2%}", delta=f"{strat.get('test_rows', 0)} rows")
                        
                        # Quality notes
                        if train_roi < 0:
                            st.error("❌ Train ROI negative - likely noise")
                        if val_roi < -0.02:
                            st.warning("⚠️ Val ROI significantly negative")
                        
                        st.code(json.dumps(strat.get("filters", []), indent=2), language="json")
            
            # Show HIGH ROI strategies that failed due to sample size
            high_roi_insufficient = analysis.get("high_roi_insufficient_sample", [])
            if high_roi_insufficient:
                st.markdown("---")
                st.markdown("## 🔬 High ROI But Insufficient Sample Size")
                st.info("These strategies showed high ROI but failed sample size gates. They might be worth exploring with more data or in combination with other filters.")
                
                for i, strat in enumerate(high_roi_insufficient, 1):
                    test_roi = strat.get('test_roi', 0)
                    train_roi = strat.get('train_roi', 0)
                    test_rows = strat.get('test_rows', 0)
                    
                    with st.expander(f"⚡ High ROI #{i}: Test {test_roi:.2%} | {test_rows} rows ({strat.get('variation', 'unknown')})"):
                        st.markdown(f"**Variation:** {strat.get('variation', 'N/A')}")
                        st.markdown(f"**Test ROI:** {test_roi:.2%} ({test_rows} rows) - **HIGH but small sample!**")
                        st.markdown(f"**Train ROI:** {train_roi:.2%} ({strat.get('train_rows', 0)} rows)")
                        st.markdown(f"**Val ROI:** {strat.get('val_roi', 0):.2%} ({strat.get('val_rows', 0)} rows)")
                        
                        st.warning(f"⚠️ Gate failures: {strat.get('gate_failures', ['Sample size too small'])}")
                        st.markdown("**Consider:** Can you get more data for this filter combination? Or combine with other filters to increase sample size?")
                        
                        st.code(json.dumps(strat.get("filters", []), indent=2), language="json")
            
            st.balloons()
            
            # Optional: Advanced validation (don't block on it)
            with st.expander("🔬 Run Advanced Validation (optional)", expanded=False):
                if st.button("Run Forward Walk & Monte Carlo"):
                    filters_to_test = best_result.get("filters", []) if best_result else []
                    with st.spinner("Running forward walk..."):
                        job = _run_tool("forward_walk", {"filters": filters_to_test, "pl_column": pl_column, "n_windows": 6})
                        if job.get("job_id"):
                            fw_result = _wait_for_job(job["job_id"], timeout=180)
                            st.markdown(f"**Forward Walk:** {fw_result.get('verdict', 'N/A')}")
                            st.code(_safe_json(fw_result, 2000), language="json")
                    
                    with st.spinner("Running monte carlo..."):
                        job = _run_tool("monte_carlo_sim", {"filters": filters_to_test, "pl_column": pl_column, "n_simulations": 500})
                        if job.get("job_id"):
                            mc_result = _wait_for_job(job["job_id"], timeout=180)
                            prob = mc_result.get('probability', {}).get('positive_roi', 0)
                            st.markdown(f"**Monte Carlo:** {prob:.1%} probability positive")
                            st.code(_safe_json(mc_result, 2000), language="json")
            
            # Log to Bible
            if best_result:
                _run_tool("append_research_note", {
                    "note": json.dumps({
                        "type": "SUCCESS",
                        "pl_column": pl_column,
                        "filters": best_result.get("filters", []),
                        "test_roi": best_result.get("test_roi", 0),
                        "train_roi": best_result.get("train_roi", 0),
                        "variation": best_result.get("variation"),
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
