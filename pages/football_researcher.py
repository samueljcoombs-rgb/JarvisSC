"""
Football Researcher v2 - Autonomous Agent Edition (Fixed)

Key fixes:
- Proper session state management
- Progressive display with st.empty() containers
- Non-blocking job polling
- Reliable button handling
"""

from __future__ import annotations

import os
import json
import uuid
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI
from modules import football_tools as tools

# ============================================================
# Page config MUST be first
# ============================================================
st.set_page_config(page_title="Football Research Agent v2", page_icon="⚽", layout="wide")

# ============================================================
# Configuration
# ============================================================

@st.cache_resource
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()
    return OpenAI(api_key=api_key)

def _get_model() -> str:
    return os.getenv("PREFERRED_OPENAI_MODEL") or st.secrets.get("PREFERRED_OPENAI_MODEL", "gpt-4o")

MAX_ITERATIONS = 8
MAX_MESSAGES = 150
JOB_POLL_INTERVAL = 3  # seconds
JOB_TIMEOUT = 180  # seconds

# ============================================================
# Session state initialization
# ============================================================

def _init_state():
    defaults = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "bible": None,
        "agent_phase": "idle",  # idle, exploring, hypothesizing, testing, analyzing, complete
        "agent_iteration": 0,
        "agent_findings": [],
        "current_job_id": None,
        "target_pl_column": "BO 2.5 PL",
        "exploration_results": {},
        "past_failures": [],
        "current_hypothesis": None,
        "run_requested": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ============================================================
# Message helpers
# ============================================================

def _append(role: str, content: str):
    st.session_state.messages.append({
        "role": role, 
        "content": content, 
        "ts": datetime.utcnow().isoformat()
    })
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

def _display_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ============================================================
# Tool runner
# ============================================================

def _run_tool(name: str, args: Optional[Dict] = None) -> Any:
    args = args or {}
    aliases = {
        "bracket_sweep": "start_bracket_sweep",
        "subgroup_scan": "start_subgroup_scan",
        "hyperopt_pl_lab": "start_hyperopt_pl_lab",
        "query_data": "start_query_data",
        "test_filter": "start_test_filter",
        "regime_check": "start_regime_check",
    }
    resolved = aliases.get(name, name)
    fn = getattr(tools, resolved, None)
    if not callable(fn):
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _poll_job(job_id: str, status_container, timeout: int = JOB_TIMEOUT) -> Dict:
    """Poll job with visual progress updates."""
    start = time.time()
    dots = 0
    
    while time.time() - start < timeout:
        job = _run_tool("get_job", {"job_id": job_id})
        status = (job.get("status") or "").lower()
        
        dots = (dots + 1) % 4
        status_container.text(f"⏳ Job {job_id[:8]}... {status} {'.' * dots}")
        
        if status == "done":
            result_path = job.get("result_path")
            if result_path:
                result = _run_tool("download_result", {"result_path": result_path})
                return result.get("result", job)
            return job
        elif status == "error":
            return {"error": job.get("error_message", "Job failed"), "job": job}
        
        time.sleep(JOB_POLL_INTERVAL)
    
    return {"error": "Job timeout", "job_id": job_id}

# ============================================================
# Bible helpers
# ============================================================

def _load_bible() -> Dict:
    if st.session_state.bible:
        return st.session_state.bible
    bible = _run_tool("get_research_context", {"limit_notes": 30})
    st.session_state.bible = bible
    return bible

def _format_bible(bible: Dict) -> str:
    overview = bible.get("dataset_overview") or {}
    gates = bible.get("gates") or {}
    derived = bible.get("derived") or {}
    
    return f"""## 📖 Bible Loaded

**Goal:** {overview.get('primary_goal', 'Find profitable betting strategies')}
**Output:** {overview.get('strategy_output_format', 'Explicit filter criteria')}

**Gates:**
- min_train_rows: {gates.get('min_train_rows', 300)}
- min_val_rows: {gates.get('min_val_rows', 60)}
- min_test_rows: {gates.get('min_test_rows', 60)}
- max_train_val_gap_roi: {gates.get('max_train_val_gap_roi', 0.4)}
- max_test_drawdown: {gates.get('max_test_drawdown', -50)}

**Outcome Columns (NEVER use as features):** {', '.join(derived.get('outcome_columns', []))}
"""

# ============================================================
# Agent LLM
# ============================================================

SYSTEM_PROMPT = """You are an autonomous football betting research agent.

## Mission
Find EXPLICIT filter criteria for profitable bets. Output like:
"MARKET='BO 2.5', LEAGUE IN ['EPL'], ACTUAL ODDS BETWEEN 1.7-2.3"

## Rules (THE LAW)
1. NEVER use PL columns as features - they're outcomes!
2. Split by TIME: train older, test newer
3. Check stability across months
4. Simple rules > complex rules
5. Think WHY something works

## Available Columns (use EXACTLY these names)
- MODE: Type of prediction model used
- LEAGUE: Football league name
- MARKET: Bet type (e.g., 'BO 2.5', 'BTTS', 'SHG')
- ACTUAL ODDS: Decimal odds offered
- % DRIFT: Percentage odds movement
- DRIFT IN / OUT: Direction of drift ('IN' or 'OUT')
- XG DIFF, M-XG-DIFF, etc.: Expected goals differentials

## Process
1. EXPLORE data baseline by key dimensions
2. HYPOTHESIZE with specific filters using EXACT column names
3. TEST quickly with test_filter
4. ITERATE or CONCLUDE

When forming filters, use these operators:
- "=" for exact match
- "in" for list membership: {"col": "LEAGUE", "op": "in", "values": ["EPL", "La Liga"]}
- "between" for ranges: {"col": "ACTUAL ODDS", "op": "between", "min": 1.5, "max": 2.5}
- ">=", "<=", ">", "<" for comparisons

Respond with JSON when asked for decisions."""

def _agent_decide(context: str, question: str) -> str:
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\n---\n{question}"}
            ],
            max_tokens=2000,
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Error: {e}]"

def _form_hypothesis(bible: Dict, exploration: Dict, failures: List) -> Dict:
    # Include actual column values from exploration
    context = f"""
## Bible Gates
{json.dumps(bible.get('gates', {}), indent=2)}

## Exploration Results (USE THESE EXACT VALUES)
{json.dumps(exploration, default=str, indent=2)[:3000]}

## Past Failures (DON'T REPEAT THESE)
{json.dumps(failures[-5:], default=str, indent=2) if failures else 'None yet'}

## Important
- Use EXACT column names: MODE, LEAGUE, MARKET, "ACTUAL ODDS", "% DRIFT", "DRIFT IN / OUT"
- Use EXACT values you see in exploration results
- The target is BO 2.5 PL - only rows where MARKET relates to over 2.5 goals
"""
    
    question = """Form a SPECIFIC hypothesis to test. Use exact column names and values from exploration.

Respond ONLY with JSON (no markdown):
{
    "hypothesis": "Clear statement of what you're testing",
    "reasoning": "Why this might be profitable based on the data",
    "filters": [
        {"col": "EXACT_COL_NAME", "op": "OPERATOR", "value": "OR_values_OR_min/max"}
    ],
    "confidence": "low/medium/high"
}

Example filters:
- {"col": "MODE", "op": "=", "value": "XG"}
- {"col": "LEAGUE", "op": "in", "values": ["EPL", "La Liga"]}
- {"col": "ACTUAL ODDS", "op": "between", "min": 1.8, "max": 2.5}
- {"col": "DRIFT IN / OUT", "op": "=", "value": "IN"}
"""
    
    resp = _agent_decide(context, question)
    try:
        # Try to extract JSON
        resp_clean = resp.strip()
        if resp_clean.startswith("```"):
            resp_clean = re.sub(r'^```\w*\n?', '', resp_clean)
            resp_clean = re.sub(r'\n?```$', '', resp_clean)
        
        match = re.search(r'\{[\s\S]*\}', resp_clean)
        if match:
            return json.loads(match.group())
    except Exception as e:
        pass
    
    return {"hypothesis": resp, "filters": [], "confidence": "low", "parse_error": True}

def _analyze_result(bible: Dict, hypothesis: Dict, result: Dict, iteration: int) -> Dict:
    gates = bible.get('gates', {})
    
    context = f"""
## Enforcement Gates
{json.dumps(gates, indent=2)}

## Hypothesis Tested
{json.dumps(hypothesis, default=str, indent=2)}

## Test Result
{json.dumps(result, default=str, indent=2)[:4000]}

## Progress
Iteration {iteration} of {MAX_ITERATIONS}
"""
    
    question = """Analyze this result and decide next step.

Check:
1. Did we get enough rows in train/val/test? (gates: min_train={}, min_val={}, min_test={})
2. Is ROI positive in all splits?
3. Is train-val gap acceptable? (max gap: {})
4. Is test drawdown acceptable? (max: {})

Respond ONLY with JSON:
{{
    "analysis": "What the numbers tell us",
    "metrics": {{"train_roi": X, "val_roi": Y, "test_roi": Z, "train_rows": N, "val_rows": N, "test_rows": N}},
    "passed_gates": true/false,
    "gate_failures": ["list of failed gates if any"],
    "decision": "refine|new_hypothesis|success|conclude_no_edge",
    "learning": "What to remember for next iteration",
    "refinement_suggestion": "If refining, what to change"
}}
""".format(
        gates.get('min_train_rows', 300),
        gates.get('min_val_rows', 60),
        gates.get('min_test_rows', 60),
        gates.get('max_train_val_gap_roi', 0.4),
        gates.get('max_test_drawdown', -50)
    )
    
    resp = _agent_decide(context, question)
    try:
        resp_clean = resp.strip()
        if resp_clean.startswith("```"):
            resp_clean = re.sub(r'^```\w*\n?', '', resp_clean)
            resp_clean = re.sub(r'\n?```$', '', resp_clean)
        
        match = re.search(r'\{[\s\S]*\}', resp_clean)
        if match:
            return json.loads(match.group())
    except:
        pass
    
    return {"analysis": resp, "decision": "new_hypothesis", "passed_gates": False}

def _format_conclusion(bible: Dict, findings: List, success: bool) -> str:
    context = f"""
## All Research Findings
{json.dumps(findings, default=str, indent=2)[:6000]}

## Success: {success}
"""
    
    if success:
        question = """Format the winning strategy clearly:

═══════════════════════════════════════════════════════════════
STRATEGY DISCOVERED
═══════════════════════════════════════════════════════════════

EXPLICIT CRITERIA:
- [Each filter condition]

PERFORMANCE (from test split):
- ROI: X%
- Sample size: N bets
- Max drawdown: X points
- Winning rate: X%

STABILITY:
- Train ROI: X% (N samples)
- Val ROI: X% (N samples)  
- Test ROI: X% (N samples)

RECOMMENDATION:
[How to use this strategy]

═══════════════════════════════════════════════════════════════
"""
    else:
        question = """Summarize why no profitable strategy was found:

1. What approaches were tried
2. Why each failed (specific gate failures)
3. Key learnings
4. Recommendations for future research
5. Whether the market appears efficient

Be specific about numbers and failures.
"""
    
    return _agent_decide(context, question)

# ============================================================
# Exploration phase
# ============================================================

def _run_exploration(pl_column: str, progress_container) -> Dict:
    """Run exploration queries with progress updates."""
    results = {}
    
    queries = [
        ("by_mode", {"query_type": "aggregate", "group_by": ["MODE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_drift", {"query_type": "aggregate", "group_by": ["DRIFT IN / OUT"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_league", {"query_type": "aggregate", "group_by": ["LEAGUE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"], "limit": 20}),
        ("baseline", {"query_type": "describe", "pl_column": pl_column}),
    ]
    
    for name, params in queries:
        progress_container.text(f"🔍 Exploring {name}...")
        
        job = _run_tool("query_data", params)
        if job.get("job_id"):
            result = _poll_job(job["job_id"], progress_container, timeout=120)
            results[name] = result
        else:
            results[name] = {"error": job.get("error", "Failed to submit")}
    
    progress_container.text("✅ Exploration complete")
    return results

# ============================================================
# Test hypothesis
# ============================================================

def _test_hypothesis(hypothesis: Dict, pl_column: str, bible: Dict, status_container) -> Dict:
    """Test a hypothesis with progress updates."""
    filters = hypothesis.get("filters", [])
    
    if not filters:
        return {"error": "No filters in hypothesis"}
    
    status_container.text("📤 Submitting test_filter job...")
    
    job = _run_tool("test_filter", {
        "filters": filters,
        "pl_column": pl_column,
        "enforcement": bible.get("gates", {}),
    })
    
    if not job.get("job_id"):
        return {"error": f"Failed to submit: {job.get('error', 'Unknown')}"}
    
    status_container.text(f"⏳ Testing hypothesis (job {job['job_id'][:8]})...")
    result = _poll_job(job["job_id"], status_container, timeout=JOB_TIMEOUT)
    
    return result

# ============================================================
# Main agent runner
# ============================================================

def run_agent():
    """Main agent execution with visual progress."""
    pl_column = st.session_state.target_pl_column
    
    # Create containers for progressive display
    main_container = st.container()
    
    with main_container:
        st.markdown(f"""
# 🤖 Autonomous Research Session

**Target:** {pl_column}  
**Max Iterations:** {MAX_ITERATIONS}
""")
        
        # Phase 1: Load Bible
        with st.status("📖 Loading Bible...", expanded=True) as status:
            bible = _load_bible()
            st.markdown(_format_bible(bible))
            status.update(label="📖 Bible loaded", state="complete")
        
        _append("assistant", f"# 🤖 Research Session: {pl_column}\n\n" + _format_bible(bible))
        
        # Phase 2: Exploration
        with st.status("🔍 Exploring data...", expanded=True) as status:
            progress = st.empty()
            exploration = _run_exploration(pl_column, progress)
            st.session_state.exploration_results = exploration
            
            # Show results
            st.markdown("### Exploration Results")
            
            for key, data in exploration.items():
                with st.expander(f"📊 {key}", expanded=False):
                    st.json(data)
            
            status.update(label="🔍 Exploration complete", state="complete")
        
        _append("assistant", f"**Exploration Results:**\n```json\n{json.dumps(exploration, indent=2, default=str)[:2000]}\n```")
        st.session_state.agent_findings.append({"phase": "exploration", "results": exploration})
        
        # Phase 3: Iteration loop
        st.markdown("---")
        st.markdown("## 🧪 Hypothesis Testing")
        
        failures = st.session_state.past_failures
        
        for i in range(1, MAX_ITERATIONS + 1):
            st.session_state.agent_iteration = i
            
            with st.status(f"Iteration {i}/{MAX_ITERATIONS}", expanded=True) as iter_status:
                # Form hypothesis
                st.markdown("**🧠 Forming hypothesis...**")
                hypothesis = _form_hypothesis(bible, exploration, failures)
                st.session_state.current_hypothesis = hypothesis
                
                st.markdown(f"""
**Hypothesis:** {hypothesis.get('hypothesis', 'N/A')}

**Reasoning:** {hypothesis.get('reasoning', 'N/A')}

**Filters:** 
```json
{json.dumps(hypothesis.get('filters', []), indent=2)}
```

**Confidence:** {hypothesis.get('confidence', 'N/A')}
""")
                
                if not hypothesis.get("filters") or hypothesis.get("parse_error"):
                    st.warning("⚠️ Could not form valid filters, trying again...")
                    failures.append({"iteration": i, "error": "Invalid filters", "raw": hypothesis})
                    iter_status.update(label=f"Iteration {i} - No valid filters", state="error")
                    continue
                
                # Test hypothesis
                st.markdown("**🧪 Testing...**")
                test_progress = st.empty()
                result = _test_hypothesis(hypothesis, pl_column, bible, test_progress)
                
                st.markdown("**Result:**")
                with st.expander("Full result", expanded=False):
                    st.json(result)
                
                # Save finding
                st.session_state.agent_findings.append({
                    "iteration": i,
                    "hypothesis": hypothesis,
                    "result": result,
                })
                
                # Analyze
                st.markdown("**📊 Analyzing...**")
                analysis = _analyze_result(bible, hypothesis, result, i)
                
                st.markdown(f"""
**Analysis:** {analysis.get('analysis', 'N/A')}

**Gates Passed:** {analysis.get('passed_gates', 'Unknown')}

**Decision:** {analysis.get('decision', 'Unknown')}

**Learning:** {analysis.get('learning', 'N/A')}
""")
                
                _append("assistant", f"""
### Iteration {i}
**Hypothesis:** {hypothesis.get('hypothesis')}
**Filters:** `{hypothesis.get('filters')}`
**Result:** Gates passed: {analysis.get('passed_gates')}
**Decision:** {analysis.get('decision')}
""")
                
                decision = analysis.get("decision", "").lower()
                
                if decision == "success":
                    iter_status.update(label=f"Iteration {i} - SUCCESS! 🎉", state="complete")
                    
                    st.markdown("# 🎉 Strategy Found!")
                    conclusion = _format_conclusion(bible, st.session_state.agent_findings, True)
                    st.markdown(conclusion)
                    _append("assistant", f"# 🎉 Strategy Found!\n\n{conclusion}")
                    
                    # Log to Bible
                    _run_tool("append_research_note", {
                        "note": json.dumps({
                            "type": "success",
                            "pl_column": pl_column,
                            "iterations": i,
                            "filters": hypothesis.get("filters"),
                        }),
                        "tags": f"success,{pl_column.replace(' ', '_')}"
                    })
                    
                    st.session_state.agent_phase = "complete"
                    st.balloons()
                    return
                
                elif decision == "conclude_no_edge":
                    iter_status.update(label=f"Iteration {i} - Concluding", state="complete")
                    break
                
                else:
                    iter_status.update(label=f"Iteration {i} - {decision}", state="complete")
                    failures.append({
                        "iteration": i,
                        "hypothesis": hypothesis,
                        "reason": analysis.get("learning", ""),
                        "gate_failures": analysis.get("gate_failures", []),
                    })
                    st.session_state.past_failures = failures
        
        # No edge found
        st.markdown("---")
        st.markdown("# 📋 Research Complete")
        conclusion = _format_conclusion(bible, st.session_state.agent_findings, False)
        st.markdown(conclusion)
        _append("assistant", f"# 📋 No Edge Found\n\n{conclusion}")
        
        _run_tool("append_research_note", {
            "note": json.dumps({
                "type": "no_edge",
                "pl_column": pl_column,
                "iterations": st.session_state.agent_iteration,
            }),
            "tags": f"no_edge,{pl_column.replace(' ', '_')}"
        })
        
        st.session_state.agent_phase = "complete"

# ============================================================
# Chat mode
# ============================================================

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_research_context", "description": "Load the Bible", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "start_pl_lab", "description": "Start ML pipeline", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}}, "required": ["pl_column"]}}},
    {"type": "function", "function": {"name": "start_test_filter", "description": "Test filter combination", "parameters": {"type": "object", "properties": {"filters": {"type": "array"}, "pl_column": {"type": "string"}}, "required": ["filters", "pl_column"]}}},
    {"type": "function", "function": {"name": "start_query_data", "description": "Explore data", "parameters": {"type": "object", "properties": {"query_type": {"type": "string"}, "group_by": {"type": "array"}, "metrics": {"type": "array"}}}}},
    {"type": "function", "function": {"name": "start_regime_check", "description": "Check stability", "parameters": {"type": "object", "properties": {"filters": {"type": "array"}, "pl_column": {"type": "string"}}, "required": ["filters", "pl_column"]}}},
    {"type": "function", "function": {"name": "start_bracket_sweep", "description": "Find profitable ranges", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}}, "required": ["pl_column"]}}},
    {"type": "function", "function": {"name": "start_subgroup_scan", "description": "Find profitable groups", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}}, "required": ["pl_column"]}}},
    {"type": "function", "function": {"name": "get_job", "description": "Check job status", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
    {"type": "function", "function": {"name": "wait_for_job", "description": "Wait for job", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
]

def _chat_response(user_input: str):
    _append("user", user_input)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})
    
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=2000,
        )
        
        msg = resp.choices[0].message
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                
                _append("assistant", f"🔧 Calling `{fn_name}`...")
                result = _run_tool(fn_name, fn_args)
                _append("assistant", f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```")
        
        if msg.content:
            _append("assistant", msg.content)
    
    except Exception as e:
        _append("assistant", f"Error: {e}")

# ============================================================
# UI Layout
# ============================================================

st.title("⚽ Football Research Agent v2")

# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")
    
    mode = st.radio("Mode", ["🤖 Autonomous Agent", "💬 Chat"], index=0)
    
    st.divider()
    
    if mode == "🤖 Autonomous Agent":
        pl_col = st.selectbox("Target Market", [
            "BO 2.5 PL", "BTTS PL", "SHG PL", "SHG 2+ PL", 
            "LU1.5 PL", "LFGHU0.5 PL", "BO1.5 FHG PL"
        ])
        st.session_state.target_pl_column = pl_col
        
        # Use a form to prevent accidental re-runs
        with st.form("start_form"):
            start_clicked = st.form_submit_button("🚀 Start Research", type="primary")
            
            if start_clicked:
                # Reset state for new run
                st.session_state.agent_phase = "running"
                st.session_state.agent_iteration = 0
                st.session_state.agent_findings = []
                st.session_state.past_failures = []
                st.session_state.exploration_results = {}
                st.session_state.current_hypothesis = None
                st.session_state.bible = None
                st.session_state.run_requested = True
        
        # Status display
        phase = st.session_state.agent_phase
        if phase == "running":
            st.warning(f"🔄 Running: Iteration {st.session_state.agent_iteration}")
        elif phase == "complete":
            st.success("✅ Complete")
        
        if st.session_state.agent_findings:
            st.info(f"📊 {len(st.session_state.agent_findings)} findings")
    
    st.divider()
    
    if st.button("🗑️ Clear All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    st.caption("v2.1 - Fixed Edition")

# Main content area
if mode == "🤖 Autonomous Agent":
    if st.session_state.run_requested:
        st.session_state.run_requested = False
        run_agent()
    elif st.session_state.agent_phase == "idle":
        st.info("👆 Select a market and click **Start Research** to begin.")
        
        # Show previous messages if any
        if st.session_state.messages:
            st.markdown("### Previous Session")
            _display_messages()
    elif st.session_state.agent_phase == "complete":
        st.success("Research complete! See results above or start a new session.")
        _display_messages()
    else:
        _display_messages()

else:  # Chat mode
    _display_messages()
    
    if prompt := st.chat_input("Ask me anything..."):
        _chat_response(prompt)
        st.rerun()

# Debug panel
with st.expander("🔍 Debug Info", expanded=False):
    st.json({
        "session_id": st.session_state.session_id,
        "phase": st.session_state.agent_phase,
        "iteration": st.session_state.agent_iteration,
        "messages": len(st.session_state.messages),
        "findings": len(st.session_state.agent_findings),
        "failures": len(st.session_state.past_failures),
        "bible_loaded": st.session_state.bible is not None,
        "run_requested": st.session_state.run_requested,
    })
