"""
Football Researcher v4 - Full Enhanced Edition

Features:
1. MARKET in exploration
2. Bracket sweep + subgroup scan phase before hypothesizing
3. Better memory logging per iteration
4. Smarter hypothesis formation based on sweep results
5. Near-miss refinement phase
6. Monthly stability check
7. Prevent duplicate tests
8. Statistical significance display
9. Rolling window drawdown (2-month max)
10. New v4 tools: combination_scan, forward_walk, monte_carlo_sim, correlation_check
"""

from __future__ import annotations
import os, json, uuid, re, time, hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import streamlit as st
from openai import OpenAI
from modules import football_tools as tools

st.set_page_config(page_title="Football Research Agent v4", page_icon="⚽", layout="wide")

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

MAX_ITERATIONS = 12
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
        "past_failures": [], "near_misses": [], "tested_filter_hashes": set(),
        "run_requested": False, "log": [],
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
    
    # Count what we loaded
    rules_count = len(rules) if isinstance(rules, list) else 1
    notes_count = len(notes) if isinstance(notes, list) else 0
    cols_count = len(col_defs) if isinstance(col_defs, list) else 0
    
    return f"""## 📖 Bible Loaded
**Goal:** {overview.get('primary_goal', 'Find profitable strategies')}
**Gates (v4):** min_train={gates.get('min_train_rows', 300)}, min_val={gates.get('min_val_rows', 60)}, min_test={gates.get('min_test_rows', 60)}, max_gap={gates.get('max_train_val_gap_roi', 0.4)}, max_rolling_dd={gates.get('max_rolling_dd', -50)}
**Outcome Columns (NEVER features):** {outcome_str}
**Loaded:** {rules_count} rules, {cols_count} column defs, {notes_count} past notes"""

# ============================================================
# LLM
# ============================================================

SYSTEM_PROMPT = """You are an expert football betting research agent finding PROFITABLE and STABLE strategies.

Rules:
1. NEVER use PL columns as features (data leakage!)
2. Split by TIME: train older, test newer
3. Explain WHY a filter exploits market inefficiency
4. Simple > complex
5. Check monthly stability

Columns: MODE, MARKET, LEAGUE, ACTUAL ODDS, % DRIFT, DRIFT IN / OUT

Filter format:
- {"col": "MODE", "op": "=", "value": "XG"}
- {"col": "ACTUAL ODDS", "op": "between", "min": 1.8, "max": 2.5}

Gates v4: rolling 2-month max dd=-50, >40% months profitable, statistical significance.

Available analysis tools:
- test_filter: Quick train/val/test split testing
- combination_scan: Test multi-filter combinations
- forward_walk: Walk-forward validation
- monte_carlo_sim: Bootstrap confidence intervals
- correlation_check: Detect data leakage"""

def _llm(context: str, question: str) -> str:
    try:
        client = _get_client()
        resp = client.chat.completions.create(model=_get_model(),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{context}\n\n---\n{question}"}],
            max_tokens=2500, temperature=0.7)
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
# Phase 1: Exploration (with MARKET)
# ============================================================

def _run_exploration(pl_column: str, progress_container=None) -> Dict:
    """
    Explore data systematically:
    - MODE alone (see which prediction type works)
    - MODE + MARKET (the key combination - which markets work for which modes)
    - MODE + MARKET + DRIFT (full picture)
    - LEAGUE (for geographic patterns)
    """
    results = {}
    queries = [
        # Core: MODE alone
        ("by_mode", {"query_type": "aggregate", "group_by": ["MODE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        
        # KEY: MODE + MARKET together (this is what you want!)
        ("by_mode_market", {"query_type": "aggregate", "group_by": ["MODE", "MARKET"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        
        # Full picture: MODE + MARKET + DRIFT
        ("by_mode_market_drift", {"query_type": "aggregate", "group_by": ["MODE", "MARKET", "DRIFT IN / OUT"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        
        # Drift patterns
        ("by_drift", {"query_type": "aggregate", "group_by": ["DRIFT IN / OUT"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        
        # Geographic patterns
        ("by_league", {"query_type": "aggregate", "group_by": ["LEAGUE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"], "limit": 20}),
        
        # League + Mode (does XG work better in certain leagues?)
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

# ============================================================
# Phase 2: Sweeps
# ============================================================

def _run_sweeps(pl_column: str, bible: Dict, progress_container=None) -> Dict:
    results = {}
    enforcement = bible.get("gates", {})
    
    if progress_container:
        progress_container.text("🔬 Running bracket_sweep...")
    _log("Running bracket_sweep...")
    job = _run_tool("bracket_sweep", {"pl_column": pl_column, "sweep_cols": ["ACTUAL ODDS", "% DRIFT"], "n_bins": 8, "enforcement": enforcement})
    results["bracket_sweep"] = _wait_for_job(job["job_id"], timeout=180) if job.get("job_id") else {"error": job.get("error", "Failed")}
    
    if progress_container:
        progress_container.text("🔬 Running subgroup_scan...")
    _log("Running subgroup_scan...")
    job = _run_tool("subgroup_scan", {"pl_column": pl_column, "group_cols": ["MODE", "MARKET", "DRIFT IN / OUT", "LEAGUE"], "enforcement": enforcement})
    results["subgroup_scan"] = _wait_for_job(job["job_id"], timeout=180) if job.get("job_id") else {"error": job.get("error", "Failed")}
    
    if progress_container:
        progress_container.text("✅ Sweeps complete")
    return results

# ============================================================
# Hypothesis Formation
# ============================================================

def _form_hypothesis(bible: Dict, exploration: Dict, sweeps: Dict, failures: List, near_misses: List, tested_hashes: Set) -> Dict:
    promising_brackets = sweeps.get("bracket_sweep", {}).get("promising", [])[:5]
    promising_subgroups = sweeps.get("subgroup_scan", {}).get("promising", [])[:5]
    context = f"""Gates: {_safe_json(bible.get('gates', {}), 300)}
Exploration: {_safe_json(exploration, 1500)}
Promising brackets: {_safe_json(promising_brackets, 800)}
Promising subgroups: {_safe_json(promising_subgroups, 800)}
Failures: {_safe_json(failures[-3:], 600) if failures else 'None'}
Near-misses: {_safe_json(near_misses[-2:], 500) if near_misses else 'None'}"""
    question = """Form hypothesis based on promising segments. JSON only:
{"hypothesis": "...", "market_inefficiency": "WHY this works", "filters": [...], "based_on": "sweep|near_miss|new", "confidence": "low/medium/high"}"""
    resp = _llm(context, question)
    parsed = _parse_json(resp)
    if parsed and parsed.get("filters"):
        fhash = _filter_hash(parsed["filters"])
        if fhash in tested_hashes:
            parsed["duplicate"] = True
        return parsed
    return {"hypothesis": resp[:300], "filters": [], "parse_error": True}

def _refine_near_miss(bible: Dict, near_miss: Dict, exploration: Dict) -> Dict:
    context = f"""Near-miss to refine (positive ROI but failed gates):
Filters: {near_miss.get('filters')}
Result: {_safe_json(near_miss.get('result', {}), 1000)}
Gate failures: {near_miss.get('gate_failures')}
Exploration: {_safe_json(exploration, 1000)}"""
    question = """Refine to reduce drawdown. JSON: {"refinement": "what changed", "filters": [...]}"""
    resp = _llm(context, question)
    parsed = _parse_json(resp)
    if parsed and parsed.get("filters"):
        return {"hypothesis": f"Refined: {parsed.get('refinement', '')}", "filters": parsed["filters"], "based_on": "near_miss_refinement", "confidence": "medium"}
    return {"filters": [], "parse_error": True}

# ============================================================
# Analysis
# ============================================================

def _analyze_result(bible: Dict, hypothesis: Dict, result: Dict, iteration: int) -> Dict:
    gates = bible.get('gates', {})
    test = result.get("test", {})
    gate_result = result.get("gates_passed", result.get("gates", {}))
    context = f"""Gates: {gates}
Hypothesis: {_safe_json(hypothesis, 400)}
Test ROI: {test.get('roi')}, Rows: {test.get('rows')}
Rolling DD: {result.get('test_rolling_dd', {}).get('max_rolling_dd')}
Stability: {result.get('monthly_stability', {})}
Significance: {result.get('statistical_significance', {})}
Gates passed: {gate_result if isinstance(gate_result, bool) else gate_result.get('passed')}, Failures: {result.get('gate_failures', [])}
Near-miss: {result.get('near_miss')}
Iteration: {iteration}/{MAX_ITERATIONS}"""
    question = """Analyze. JSON: {"analysis": "...", "passed_all_gates": true/false, "is_near_miss": true/false, "decision": "success|refine_near_miss|new_hypothesis|conclude_no_edge", "learning": "..."}"""
    resp = _llm(context, question)
    parsed = _parse_json(resp)
    return parsed if parsed else {"analysis": resp[:300], "decision": "new_hypothesis", "passed_all_gates": False}

def _format_conclusion(bible: Dict, findings: List, success: bool) -> str:
    context = f"Findings: {_safe_json(findings, 5000)}"
    question = "Format winning strategy with criteria, performance, stability." if success else "Summarize why no edge, what tried, recommendations."
    return _llm(context, question)

# ============================================================
# Testing
# ============================================================

def _test_hypothesis(hypothesis: Dict, pl_column: str, bible: Dict) -> Dict:
    filters = hypothesis.get("filters", [])
    if not filters:
        return {"error": "No filters"}
    fhash = _filter_hash(filters)
    if fhash in st.session_state.tested_filter_hashes:
        return {"error": "Duplicate", "skipped": True}
    st.session_state.tested_filter_hashes.add(fhash)
    _log(f"Testing (hash {fhash}): {filters}")
    job = _run_tool("test_filter", {"filters": filters, "pl_column": pl_column, "enforcement": bible.get("gates", {})})
    if not job.get("job_id"):
        return {"error": job.get("error", "Submit failed")}
    return _wait_for_job(job["job_id"], timeout=JOB_TIMEOUT)

# ============================================================
# Advanced Validation (v4)
# ============================================================

def _run_advanced_validation(filters: List[Dict], pl_column: str, bible: Dict) -> Dict:
    """Run additional validation: forward_walk, monte_carlo_sim, correlation_check."""
    results = {}
    
    _log("Running forward_walk...")
    job = _run_tool("forward_walk", {"filters": filters, "pl_column": pl_column, "n_windows": 6})
    if job.get("job_id"):
        results["forward_walk"] = _wait_for_job(job["job_id"], timeout=180)
    
    _log("Running monte_carlo_sim...")
    job = _run_tool("monte_carlo_sim", {"filters": filters, "pl_column": pl_column, "n_simulations": 500})
    if job.get("job_id"):
        results["monte_carlo"] = _wait_for_job(job["job_id"], timeout=180)
    
    _log("Running correlation_check...")
    job = _run_tool("correlation_check", {"filters": filters})
    if job.get("job_id"):
        results["correlation"] = _wait_for_job(job["job_id"], timeout=60)
    
    return results

# ============================================================
# Main Agent
# ============================================================

def run_agent():
    pl_column = st.session_state.target_pl_column
    st.session_state.log = []
    st.session_state.tested_filter_hashes = set()
    st.session_state.near_misses = []
    
    st.markdown(f"# 🤖 Research Agent v4: {pl_column}")
    
    # Bible
    with st.status("📖 Loading Bible...", expanded=True) as status:
        bible = _load_bible()
        st.markdown(_format_bible(bible))
        status.update(label="📖 Bible loaded", state="complete")
    
    # Show full Bible context outside status
    with st.expander("📚 Full Bible Context", expanded=False):
        st.markdown("**Research Rules:**")
        rules = bible.get("research_rules", [])
        st.code(_safe_json(rules if rules else "No rules loaded", 1500), language="json")
        
        st.markdown("**Dataset Overview:**")
        overview = bible.get("dataset_overview", {})
        st.code(_safe_json(overview if overview else "No overview", 1500), language="json")
        
        st.markdown("**Column Definitions:**")
        cols = bible.get("column_definitions", [])
        st.code(_safe_json(cols if cols else "No column defs", 2000), language="json")
        
        st.markdown("**Gates/Thresholds:**")
        gates = bible.get("gates", {})
        st.code(_safe_json(gates if gates else "No gates", 800), language="json")
        
        st.markdown("**Recent Notes:**")
        notes = bible.get("research_notes", [])
        st.code(_safe_json(notes[:10] if notes else "No notes yet", 1500), language="json")
    
    _append("assistant", _format_bible(bible))
    
    # Exploration
    with st.status("🔍 Phase 1: Exploration...", expanded=True) as status:
        progress = st.empty()
        exploration = _run_exploration(pl_column, progress)
        st.session_state.exploration_results = exploration
        status.update(label="🔍 Exploration complete", state="complete")
    
    with st.expander("Exploration Results", expanded=False):
        st.code(_safe_json(exploration, 3000), language="json")
    
    # Sweeps
    with st.status("🔬 Phase 2: Segment Analysis...", expanded=True) as status:
        progress = st.empty()
        sweeps = _run_sweeps(pl_column, bible, progress)
        st.session_state.sweep_results = sweeps
        promising = len(sweeps.get("bracket_sweep", {}).get("promising", [])) + len(sweeps.get("subgroup_scan", {}).get("promising", []))
        status.update(label=f"🔬 Sweeps complete ({promising} promising)", state="complete")
    
    st.markdown(f"**Found {promising} promising segments**")
    with st.expander("Sweep Results", expanded=False):
        st.code(_safe_json(sweeps, 3000), language="json")
    
    st.session_state.agent_findings.append({"phase": "exploration", "promising": promising})
    
    # Hypothesis loop
    st.markdown("---\n### 🧪 Phase 3: Hypothesis Testing")
    failures = []
    
    for i in range(1, MAX_ITERATIONS + 1):
        st.session_state.agent_iteration = i
        st.markdown(f"#### Iteration {i}/{MAX_ITERATIONS}")
        
        # Near-miss refinement
        if st.session_state.near_misses and i > 3 and i % 2 == 0:
            st.markdown("**🔧 Refining near-miss...**")
            hypothesis = _refine_near_miss(bible, st.session_state.near_misses[-1], exploration)
        else:
            hypothesis = _form_hypothesis(bible, exploration, sweeps, failures, st.session_state.near_misses, st.session_state.tested_filter_hashes)
        
        if hypothesis.get("duplicate") or not hypothesis.get("filters") or hypothesis.get("parse_error"):
            st.warning("⚠️ Invalid/duplicate hypothesis")
            failures.append({"iteration": i, "error": "Invalid"})
            continue
        
        st.markdown(f"**Hypothesis:** {hypothesis.get('hypothesis', 'N/A')}")
        st.markdown(f"**Why:** {hypothesis.get('market_inefficiency', 'N/A')}")
        st.markdown(f"**Filters:** `{hypothesis.get('filters')}`")
        
        result = _test_hypothesis(hypothesis, pl_column, bible)
        if result.get("skipped"):
            continue
        
        test = result.get("test", {})
        gates = result.get("gates_passed", result.get("gates", {}))
        gates_passed = gates if isinstance(gates, bool) else gates.get("passed", False)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Test ROI", f"{test.get('roi', 0):.4f}")
        col2.metric("Rows", test.get('rows', 0))
        col3.metric("Gates", "✅" if gates_passed else "❌")
        
        with st.expander("Full Result"):
            st.code(_safe_json(result, 2500), language="json")
        
        st.session_state.agent_findings.append({"iteration": i, "hypothesis": hypothesis, "result": result})
        
        analysis = _analyze_result(bible, hypothesis, result, i)
        st.markdown(f"**Analysis:** {analysis.get('analysis', 'N/A')[:200]}")
        st.markdown(f"**Decision:** {analysis.get('decision')}")
        
        _log(f"Iter {i}: {analysis.get('decision')} ROI={test.get('roi', 0):.4f}")
        _append("assistant", f"Iteration {i}: {analysis.get('decision')}")
        
        decision = (analysis.get("decision") or "").lower()
        
        if decision == "success":
            st.markdown("# 🎉 Strategy Found!")
            
            # Run advanced validation
            st.markdown("### 🔬 Running Advanced Validation...")
            adv_results = _run_advanced_validation(hypothesis.get("filters", []), pl_column, bible)
            
            with st.expander("Advanced Validation Results"):
                st.code(_safe_json(adv_results, 3000), language="json")
            
            conclusion = _format_conclusion(bible, st.session_state.agent_findings, True)
            st.markdown(conclusion)
            _append("assistant", conclusion)
            _run_tool("append_research_note", {"note": json.dumps({"type": "SUCCESS", "pl_column": pl_column, "filters": hypothesis.get("filters"), "test_roi": test.get("roi")}), "tags": f"success,{pl_column.replace(' ', '_')}"})
            st.session_state.agent_phase = "complete"
            st.balloons()
            return
        elif decision == "conclude_no_edge":
            break
        elif analysis.get("is_near_miss") or result.get("near_miss"):
            st.info("📌 Near-miss - will refine")
            st.session_state.near_misses.append({"iteration": i, "filters": hypothesis.get("filters"), "result": result, "gate_failures": result.get("gate_failures", [])})
        
        failures.append({"iteration": i, "hypothesis": hypothesis, "reason": analysis.get("learning", "")})
        st.session_state.past_failures = failures
        st.markdown("---")
    
    st.markdown("# 📋 No Edge Found")
    conclusion = _format_conclusion(bible, st.session_state.agent_findings, False)
    st.markdown(conclusion)
    _append("assistant", conclusion)
    _run_tool("append_research_note", {"note": json.dumps({"type": "NO_EDGE", "pl_column": pl_column, "near_misses": len(st.session_state.near_misses)}), "tags": f"no_edge,{pl_column.replace(' ', '_')}"})
    st.session_state.agent_phase = "complete"

# ============================================================
# UI
# ============================================================

st.title("⚽ Football Research Agent v4")
st.caption("v4: MARKET exploration | Sweeps | Near-miss refinement | Rolling DD | New: forward_walk, monte_carlo, correlation_check")

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
        st.session_state.tested_filter_hashes = set()
        st.session_state.bible = None
        st.session_state.run_requested = True
        st.rerun()
    
    st.divider()
    if st.session_state.agent_phase == "running":
        st.warning(f"🔄 Iteration {st.session_state.agent_iteration}")
    elif st.session_state.agent_phase == "complete":
        st.success("✅ Done")
        st.metric("Near-misses", len(st.session_state.near_misses))
    
    st.divider()
    if st.button("🗑️ Clear"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    
    if st.session_state.log:
        with st.expander("📋 Log"):
            for line in st.session_state.log[-25:]:
                st.text(line)
    st.caption("v4.0")

if st.session_state.run_requested:
    st.session_state.run_requested = False
    run_agent()
elif st.session_state.agent_phase == "idle":
    st.info("👆 Click **Start Research**")
    with st.expander("🆕 What's New in v4"):
        st.markdown("""
**Enhanced Analysis:**
- MARKET in exploration
- Bracket sweep + subgroup scan BEFORE hypothesizing
- Near-miss refinement (positive ROI but failed gates)
- Rolling 2-month drawdown (not cumulative)
- Monthly stability >40%
- Duplicate prevention
- Statistical significance

**NEW Advanced Validation Tools:**
- **combination_scan**: Test multi-filter synergies
- **forward_walk**: Walk-forward out-of-sample testing
- **monte_carlo_sim**: Bootstrap confidence intervals
- **correlation_check**: Detect data leakage
""")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
