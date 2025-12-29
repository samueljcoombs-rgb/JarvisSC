"""
Football Researcher v2.2 - Autonomous Agent Edition (Robust)

Fixes:
- Safe JSON serialization
- Better error handling
- Progressive display
"""

from __future__ import annotations

import os
import json
import uuid
import re
import time
import traceback
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
# Safe JSON helper
# ============================================================

def _safe_json(obj: Any, max_len: int = 5000) -> str:
    """Safely convert object to JSON string."""
    try:
        result = json.dumps(obj, indent=2, default=str)
        if len(result) > max_len:
            result = result[:max_len] + "\n... (truncated)"
        return result
    except Exception as e:
        return f"{{\"error\": \"Could not serialize: {e}\"}}"

def _safe_display(container, data: Any, label: str = "Data"):
    """Safely display data in Streamlit."""
    try:
        json_str = _safe_json(data)
        container.code(json_str, language="json")
    except Exception as e:
        container.error(f"Display error: {e}")
        container.text(str(data)[:1000])

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
JOB_POLL_INTERVAL = 3
JOB_TIMEOUT = 180

# ============================================================
# Session state
# ============================================================

def _init_state():
    defaults = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "bible": None,
        "agent_phase": "idle",
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
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

def _poll_job(job_id: str, status_container, timeout: int = JOB_TIMEOUT) -> Dict:
    """Poll job with visual progress."""
    start = time.time()
    dots = 0
    
    while time.time() - start < timeout:
        try:
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
        except Exception as e:
            status_container.text(f"⚠️ Poll error: {e}")
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
    
    outcome_cols = derived.get('outcome_columns', [])
    if isinstance(outcome_cols, list):
        outcome_str = ', '.join(str(c) for c in outcome_cols)
    else:
        outcome_str = str(outcome_cols)
    
    return f"""## 📖 Bible Loaded

**Goal:** {overview.get('primary_goal', 'Find profitable betting strategies')}
**Output:** {overview.get('strategy_output_format', 'Explicit filter criteria')}

**Gates:**
- min_train_rows: {gates.get('min_train_rows', 300)}
- min_val_rows: {gates.get('min_val_rows', 60)}
- min_test_rows: {gates.get('min_test_rows', 60)}
- max_train_val_gap_roi: {gates.get('max_train_val_gap_roi', 0.4)}
- max_test_drawdown: {gates.get('max_test_drawdown', -50)}

**Outcome Columns (NEVER use as features):** {outcome_str}
"""

# ============================================================
# Agent LLM
# ============================================================

SYSTEM_PROMPT = """You are an autonomous football betting research agent.

## Mission
Find EXPLICIT filter criteria for profitable bets. Output like:
"MODE='XG', LEAGUE IN ['EPL'], ACTUAL ODDS BETWEEN 1.7-2.3"

## Rules (THE LAW)
1. NEVER use PL columns as features - they're outcomes!
2. Split by TIME: train older, test newer
3. Check stability across months
4. Simple rules > complex rules
5. Think WHY something works

## Available Columns (use EXACTLY these names)
- MODE: Type of prediction model (e.g., 'XG', 'ODDS', 'HYBRID')
- LEAGUE: Football league name
- MARKET: Bet type (e.g., 'BO 2.5', 'BTTS', 'SHG')
- ACTUAL ODDS: Decimal odds offered
- % DRIFT: Percentage odds movement
- DRIFT IN / OUT: Direction of drift ('IN' or 'OUT')

## Filter Format
Use these operators:
- "=" for exact match: {"col": "MODE", "op": "=", "value": "XG"}
- "in" for list: {"col": "LEAGUE", "op": "in", "values": ["EPL", "La Liga"]}
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
        return f"[LLM Error: {e}]"

def _parse_json_response(resp: str) -> Optional[Dict]:
    """Try to parse JSON from LLM response."""
    try:
        resp_clean = resp.strip()
        # Remove markdown code blocks
        if "```" in resp_clean:
            resp_clean = re.sub(r'```json\s*', '', resp_clean)
            resp_clean = re.sub(r'```\s*', '', resp_clean)
        
        # Find JSON object
        match = re.search(r'\{[\s\S]*\}', resp_clean)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None

def _form_hypothesis(bible: Dict, exploration: Dict, failures: List) -> Dict:
    context = f"""
## Bible Gates
{_safe_json(bible.get('gates', {}), 500)}

## Exploration Results (USE THESE EXACT VALUES)
{_safe_json(exploration, 3000)}

## Past Failures (DON'T REPEAT)
{_safe_json(failures[-5:], 1500) if failures else 'None yet'}
"""
    
    question = """Form a SPECIFIC hypothesis. Use exact column names from exploration.

Respond with JSON only:
{
    "hypothesis": "What you're testing",
    "reasoning": "Why it might work",
    "filters": [
        {"col": "COLUMN_NAME", "op": "OPERATOR", "value": "VALUE"}
    ],
    "confidence": "low/medium/high"
}

Examples:
- {"col": "MODE", "op": "=", "value": "XG"}
- {"col": "LEAGUE", "op": "in", "values": ["EPL", "La Liga"]}
- {"col": "ACTUAL ODDS", "op": "between", "min": 1.8, "max": 2.5}
"""
    
    resp = _agent_decide(context, question)
    parsed = _parse_json_response(resp)
    
    if parsed and parsed.get("filters"):
        return parsed
    
    return {"hypothesis": resp[:500], "filters": [], "confidence": "low", "parse_error": True}

def _analyze_result(bible: Dict, hypothesis: Dict, result: Dict, iteration: int) -> Dict:
    gates = bible.get('gates', {})
    
    context = f"""
## Gates
{_safe_json(gates, 500)}

## Hypothesis
{_safe_json(hypothesis, 1000)}

## Result
{_safe_json(result, 3000)}

## Progress: {iteration}/{MAX_ITERATIONS}
"""
    
    question = f"""Analyze and decide next step.

Gates to check:
- min_train_rows: {gates.get('min_train_rows', 300)}
- min_val_rows: {gates.get('min_val_rows', 60)}
- min_test_rows: {gates.get('min_test_rows', 60)}
- max_train_val_gap_roi: {gates.get('max_train_val_gap_roi', 0.4)}
- max_test_drawdown: {gates.get('max_test_drawdown', -50)}

Respond with JSON:
{{
    "analysis": "What the numbers show",
    "passed_gates": true/false,
    "decision": "refine|new_hypothesis|success|conclude_no_edge",
    "learning": "Key takeaway"
}}"""
    
    resp = _agent_decide(context, question)
    parsed = _parse_json_response(resp)
    
    if parsed:
        return parsed
    
    return {"analysis": resp[:500], "decision": "new_hypothesis", "passed_gates": False}

def _format_conclusion(bible: Dict, findings: List, success: bool) -> str:
    context = f"Findings:\n{_safe_json(findings, 5000)}"
    
    if success:
        question = """Format the winning strategy:

═══════════════════════════════════════
STRATEGY FOUND
═══════════════════════════════════════
CRITERIA: [filters]
PERFORMANCE: [ROI, samples, drawdown]
STABILITY: [train/val/test consistency]
═══════════════════════════════════════"""
    else:
        question = """Summarize why no edge found:
1. What was tried
2. Why each failed
3. Learnings
4. Recommendations"""
    
    return _agent_decide(context, question)

# ============================================================
# Exploration
# ============================================================

def _run_exploration(pl_column: str, progress_container) -> Dict:
    results = {}
    
    queries = [
        ("by_mode", {"query_type": "aggregate", "group_by": ["MODE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_drift", {"query_type": "aggregate", "group_by": ["DRIFT IN / OUT"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_league", {"query_type": "aggregate", "group_by": ["LEAGUE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"], "limit": 15}),
    ]
    
    for name, params in queries:
        try:
            progress_container.text(f"🔍 Exploring {name}...")
            
            job = _run_tool("query_data", params)
            if job.get("job_id"):
                result = _poll_job(job["job_id"], progress_container, timeout=120)
                results[name] = result
            else:
                results[name] = {"error": job.get("error", "Failed to submit")}
        except Exception as e:
            results[name] = {"error": str(e)}
    
    progress_container.text("✅ Exploration complete")
    return results

# ============================================================
# Test hypothesis
# ============================================================

def _test_hypothesis(hypothesis: Dict, pl_column: str, bible: Dict, status_container) -> Dict:
    filters = hypothesis.get("filters", [])
    
    if not filters:
        return {"error": "No filters in hypothesis"}
    
    try:
        status_container.text("📤 Submitting test_filter job...")
        
        job = _run_tool("test_filter", {
            "filters": filters,
            "pl_column": pl_column,
            "enforcement": bible.get("gates", {}),
        })
        
        if not job.get("job_id"):
            return {"error": f"Submit failed: {job.get('error', 'Unknown')}"}
        
        result = _poll_job(job["job_id"], status_container, timeout=JOB_TIMEOUT)
        return result
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

# ============================================================
# Main agent
# ============================================================

def run_agent():
    pl_column = st.session_state.target_pl_column
    
    st.markdown(f"""
# 🤖 Autonomous Research Session

**Target:** {pl_column} | **Max Iterations:** {MAX_ITERATIONS}
""")
    
    try:
        # Phase 1: Bible
        with st.status("📖 Loading Bible...", expanded=True) as status:
            bible = _load_bible()
            st.markdown(_format_bible(bible))
            status.update(label="📖 Bible loaded", state="complete")
        
        _append("assistant", f"# Research: {pl_column}\n\n" + _format_bible(bible))
        
        # Phase 2: Exploration
        with st.status("🔍 Exploring data...", expanded=True) as status:
            progress = st.empty()
            exploration = _run_exploration(pl_column, progress)
            st.session_state.exploration_results = exploration
            
            st.markdown("### Exploration Results")
            for key, data in exploration.items():
                with st.expander(f"📊 {key}", expanded=False):
                    st.code(_safe_json(data, 2000), language="json")
            
            status.update(label="🔍 Exploration complete", state="complete")
        
        _append("assistant", f"**Exploration:**\n```json\n{_safe_json(exploration, 1500)}\n```")
        st.session_state.agent_findings.append({"phase": "exploration", "results": exploration})
        
        # Phase 3: Iterations
        st.markdown("---\n## 🧪 Hypothesis Testing")
        
        failures = st.session_state.past_failures
        
        for i in range(1, MAX_ITERATIONS + 1):
            st.session_state.agent_iteration = i
            
            with st.status(f"Iteration {i}/{MAX_ITERATIONS}", expanded=True) as iter_status:
                # Form hypothesis
                st.markdown("**🧠 Forming hypothesis...**")
                hypothesis = _form_hypothesis(bible, exploration, failures)
                
                st.markdown(f"""
**Hypothesis:** {hypothesis.get('hypothesis', 'N/A')}

**Filters:**
```json
{_safe_json(hypothesis.get('filters', []), 500)}
```

**Confidence:** {hypothesis.get('confidence', 'N/A')}
""")
                
                if not hypothesis.get("filters") or hypothesis.get("parse_error"):
                    st.warning("⚠️ Invalid filters, retrying...")
                    failures.append({"iteration": i, "error": "Invalid filters"})
                    iter_status.update(label=f"Iteration {i} - Invalid", state="error")
                    continue
                
                # Test
                st.markdown("**🧪 Testing...**")
                test_progress = st.empty()
                result = _test_hypothesis(hypothesis, pl_column, bible, test_progress)
                
                with st.expander("📋 Full Result", expanded=False):
                    st.code(_safe_json(result, 3000), language="json")
                
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
""")
                
                _append("assistant", f"### Iteration {i}\n**Hypothesis:** {hypothesis.get('hypothesis')}\n**Decision:** {analysis.get('decision')}")
                
                decision = (analysis.get("decision") or "").lower()
                
                if decision == "success":
                    iter_status.update(label=f"Iteration {i} - SUCCESS! 🎉", state="complete")
                    st.markdown("# 🎉 Strategy Found!")
                    conclusion = _format_conclusion(bible, st.session_state.agent_findings, True)
                    st.markdown(conclusion)
                    _append("assistant", f"# 🎉 Success!\n\n{conclusion}")
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
                    })
                    st.session_state.past_failures = failures
        
        # No edge
        st.markdown("---\n# 📋 Research Complete")
        conclusion = _format_conclusion(bible, st.session_state.agent_findings, False)
        st.markdown(conclusion)
        _append("assistant", f"# No Edge Found\n\n{conclusion}")
        st.session_state.agent_phase = "complete"
        
    except Exception as e:
        st.error(f"Agent error: {e}")
        st.code(traceback.format_exc())
        st.session_state.agent_phase = "idle"

# ============================================================
# Chat mode
# ============================================================

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_research_context", "description": "Load the Bible", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "start_pl_lab", "description": "Start ML pipeline", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}}, "required": ["pl_column"]}}},
    {"type": "function", "function": {"name": "start_test_filter", "description": "Test filters", "parameters": {"type": "object", "properties": {"filters": {"type": "array"}, "pl_column": {"type": "string"}}, "required": ["filters", "pl_column"]}}},
    {"type": "function", "function": {"name": "start_query_data", "description": "Explore data", "parameters": {"type": "object", "properties": {"query_type": {"type": "string"}, "group_by": {"type": "array"}, "metrics": {"type": "array"}}}}},
    {"type": "function", "function": {"name": "start_bracket_sweep", "description": "Find ranges", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}}, "required": ["pl_column"]}}},
    {"type": "function", "function": {"name": "start_subgroup_scan", "description": "Find groups", "parameters": {"type": "object", "properties": {"pl_column": {"type": "string"}}, "required": ["pl_column"]}}},
    {"type": "function", "function": {"name": "get_job", "description": "Check job", "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}}},
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
                _append("assistant", f"🔧 `{fn_name}`...")
                result = _run_tool(fn_name, fn_args)
                _append("assistant", f"```json\n{_safe_json(result, 2000)}\n```")
        
        if msg.content:
            _append("assistant", msg.content)
    
    except Exception as e:
        _append("assistant", f"Error: {e}")

# ============================================================
# UI
# ============================================================

st.title("⚽ Football Research Agent v2")

with st.sidebar:
    st.header("🎛️ Controls")
    
    mode = st.radio("Mode", ["🤖 Autonomous", "💬 Chat"], index=0)
    
    st.divider()
    
    if mode == "🤖 Autonomous":
        pl_col = st.selectbox("Target Market", [
            "BO 2.5 PL", "BTTS PL", "SHG PL", "SHG 2+ PL", 
            "LU1.5 PL", "LFGHU0.5 PL", "BO1.5 FHG PL"
        ])
        st.session_state.target_pl_column = pl_col
        
        with st.form("start_form"):
            if st.form_submit_button("🚀 Start Research", type="primary"):
                st.session_state.agent_phase = "running"
                st.session_state.agent_iteration = 0
                st.session_state.agent_findings = []
                st.session_state.past_failures = []
                st.session_state.exploration_results = {}
                st.session_state.bible = None
                st.session_state.run_requested = True
        
        phase = st.session_state.agent_phase
        if phase == "running":
            st.warning(f"🔄 Iteration {st.session_state.agent_iteration}")
        elif phase == "complete":
            st.success("✅ Complete")
    
    st.divider()
    
    if st.button("🗑️ Clear"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.caption("v2.2")

# Main
if mode == "🤖 Autonomous":
    if st.session_state.run_requested:
        st.session_state.run_requested = False
        run_agent()
    elif st.session_state.agent_phase == "idle":
        st.info("👆 Select market and click **Start Research**")
        if st.session_state.messages:
            _display_messages()
    else:
        _display_messages()
else:
    _display_messages()
    if prompt := st.chat_input("Ask..."):
        _chat_response(prompt)
        st.rerun()

with st.expander("🔍 Debug", expanded=False):
    st.code(_safe_json({
        "phase": st.session_state.agent_phase,
        "iteration": st.session_state.agent_iteration,
        "findings": len(st.session_state.agent_findings),
        "failures": len(st.session_state.past_failures),
    }), language="json")
