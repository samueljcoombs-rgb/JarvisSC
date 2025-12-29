"""
Football Researcher v2 - Autonomous Agent Edition

A truly autonomous research agent that:
1. Reads and internalizes the Bible before starting
2. Forms and tests hypotheses
3. Learns from failures
4. Outputs explicit strategy criteria or explains why no edge exists
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

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

from openai import OpenAI
from modules import football_tools as tools

# ============================================================
# Configuration
# ============================================================

def _init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()
    return OpenAI(api_key=api_key)

client = _init_client()
MODEL = os.getenv("PREFERRED_OPENAI_MODEL") or st.secrets.get("PREFERRED_OPENAI_MODEL", "gpt-4o")

MAX_ITERATIONS = 8
MAX_MESSAGES = 150

# ============================================================
# Session state
# ============================================================

def _init_state():
    defaults = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "bible": None,
        "agent_active": False,
        "agent_iteration": 0,
        "agent_findings": [],
        "current_job_id": None,
        "target_pl_column": "BO 2.5 PL",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ============================================================
# Helpers
# ============================================================

def _append(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

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

## Process
1. EXPLORE data baseline
2. HYPOTHESIZE with specific filters
3. TEST quickly
4. ITERATE or CONCLUDE

Respond with JSON when asked for decisions."""

def _agent_decide(context: str, question: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
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
    context = f"""
Bible Gates: {bible.get('gates', {})}
Exploration: {json.dumps(exploration, default=str)[:2000]}
Past Failures: {json.dumps(failures[-3:], default=str) if failures else 'None'}
"""
    question = """Form a SPECIFIC hypothesis. Respond with JSON:
{
    "hypothesis": "What you're testing",
    "reasoning": "Why it might work", 
    "filters": [{"col": "COL", "op": "OP", "value": VAL}],
    "confidence": "low/medium/high"
}"""
    
    resp = _agent_decide(context, question)
    try:
        match = re.search(r'\{[\s\S]*\}', resp)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {"hypothesis": resp, "filters": [], "confidence": "low"}

def _analyze_result(bible: Dict, hypothesis: Dict, result: Dict, iteration: int) -> Dict:
    context = f"""
Gates: {bible.get('gates', {})}
Hypothesis: {json.dumps(hypothesis, default=str)}
Result: {json.dumps(result, default=str)[:3000]}
Iteration: {iteration}/{MAX_ITERATIONS}
"""
    question = """Analyze and decide next step. JSON:
{
    "analysis": "What result shows",
    "passed_gates": true/false,
    "decision": "refine|new_hypothesis|success|conclude_no_edge",
    "learning": "What to remember"
}"""
    
    resp = _agent_decide(context, question)
    try:
        match = re.search(r'\{[\s\S]*\}', resp)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {"analysis": resp, "decision": "new_hypothesis"}

def _format_conclusion(bible: Dict, findings: List, success: bool) -> str:
    context = f"Findings: {json.dumps(findings, default=str)[:4000]}"
    
    if success:
        question = """Format the winning strategy:
═══════════════════════════════════════
STRATEGY DISCOVERED: [Name]
═══════════════════════════════════════
EXPLICIT CRITERIA:
[Each filter]

PERFORMANCE:
[ROI, samples, drawdown]

STABILITY:
[Monthly consistency]
═══════════════════════════════════════"""
    else:
        question = """Explain why no edge found:
- What was tried
- Why each failed  
- Learnings
- Recommendations"""
    
    return _agent_decide(context, question)

# ============================================================
# Agent execution
# ============================================================

def _explore(pl_column: str) -> Dict:
    results = {}
    
    # By MODE
    job = _run_tool("query_data", {
        "query_type": "aggregate",
        "group_by": ["MODE"],
        "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"],
    })
    if job.get("job_id"):
        res = _run_tool("wait_for_job", {"job_id": job["job_id"], "timeout_s": 120})
        results["by_mode"] = res.get("result", {}).get("result", {})
    
    # By DRIFT
    job = _run_tool("query_data", {
        "query_type": "aggregate", 
        "group_by": ["DRIFT IN / OUT"],
        "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"],
    })
    if job.get("job_id"):
        res = _run_tool("wait_for_job", {"job_id": job["job_id"], "timeout_s": 120})
        results["by_drift"] = res.get("result", {}).get("result", {})
    
    # By LEAGUE
    job = _run_tool("query_data", {
        "query_type": "aggregate",
        "group_by": ["LEAGUE"],
        "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"],
        "limit": 15,
    })
    if job.get("job_id"):
        res = _run_tool("wait_for_job", {"job_id": job["job_id"], "timeout_s": 120})
        results["by_league"] = res.get("result", {}).get("result", {})
    
    return results

def _test_hypothesis(hypothesis: Dict, pl_column: str, bible: Dict) -> Dict:
    filters = hypothesis.get("filters", [])
    if not filters:
        return {"error": "No filters"}
    
    job = _run_tool("test_filter", {
        "filters": filters,
        "pl_column": pl_column,
        "enforcement": bible.get("gates", {}),
    })
    
    if not job.get("job_id"):
        return {"error": job.get("error", "Job failed")}
    
    result = _run_tool("wait_for_job", {"job_id": job["job_id"], "timeout_s": 300})
    return result.get("result", result)

def run_agent_session(pl_column: str):
    """Main agent session."""
    st.session_state.agent_active = True
    st.session_state.agent_iteration = 0
    st.session_state.agent_findings = []
    st.session_state.target_pl_column = pl_column
    
    _append("assistant", f"""
# 🤖 Autonomous Research Session

**Target:** {pl_column}
**Max Iterations:** {MAX_ITERATIONS}

Starting research...
""")
    
    # Load Bible
    bible = _load_bible()
    _append("assistant", _format_bible(bible))
    
    # Explore
    _append("assistant", "🔍 **Phase 1: Exploring data...**")
    exploration = _explore(pl_column)
    _append("assistant", f"**Exploration Results:**\n```json\n{json.dumps(exploration, indent=2, default=str)[:1500]}\n```")
    st.session_state.agent_findings.append({"phase": "exploration", "results": exploration})
    
    # Iteration loop
    failures = []
    
    for i in range(1, MAX_ITERATIONS + 1):
        st.session_state.agent_iteration = i
        _append("assistant", f"\n---\n### 🧪 Iteration {i}/{MAX_ITERATIONS}")
        
        # Form hypothesis
        hypothesis = _form_hypothesis(bible, exploration, failures)
        _append("assistant", f"""**Hypothesis:** {hypothesis.get('hypothesis', 'N/A')}
**Filters:** `{hypothesis.get('filters', [])}`
**Confidence:** {hypothesis.get('confidence', 'N/A')}""")
        
        if not hypothesis.get("filters"):
            failures.append({"iteration": i, "error": "No filters"})
            continue
        
        # Test
        _append("assistant", "Testing...")
        result = _test_hypothesis(hypothesis, pl_column, bible)
        
        st.session_state.agent_findings.append({
            "iteration": i,
            "hypothesis": hypothesis,
            "result": result,
        })
        
        # Analyze
        analysis = _analyze_result(bible, hypothesis, result, i)
        _append("assistant", f"""**Analysis:** {analysis.get('analysis', 'N/A')}
**Gates Passed:** {analysis.get('passed_gates', 'Unknown')}
**Decision:** {analysis.get('decision', 'Unknown')}""")
        
        decision = analysis.get("decision", "")
        
        if decision == "success":
            _append("assistant", "# 🎉 Strategy Found!")
            conclusion = _format_conclusion(bible, st.session_state.agent_findings, True)
            _append("assistant", conclusion)
            
            _run_tool("append_research_note", {
                "note": json.dumps({"type": "success", "pl_column": pl_column, "iterations": i}),
                "tags": f"success,{pl_column.replace(' ', '_')}"
            })
            
            st.session_state.agent_active = False
            return
        
        elif decision == "conclude_no_edge":
            break
        
        failures.append({
            "iteration": i,
            "hypothesis": hypothesis,
            "reason": analysis.get("learning", ""),
        })
    
    # No edge found
    _append("assistant", "# 📋 Research Complete - No Edge Found")
    conclusion = _format_conclusion(bible, st.session_state.agent_findings, False)
    _append("assistant", conclusion)
    
    _run_tool("append_research_note", {
        "note": json.dumps({"type": "no_edge", "pl_column": pl_column, "iterations": st.session_state.agent_iteration}),
        "tags": f"no_edge,{pl_column.replace(' ', '_')}"
    })
    
    st.session_state.agent_active = False

# ============================================================
# Manual chat mode
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
    {"type": "function", "function": {"name": "append_research_note", "description": "Save note", "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}}},
]

def _chat_response(user_input: str):
    """Handle chat mode with function calling."""
    _append("user", user_input)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=2000,
        )
        
        msg = resp.choices[0].message
        
        # Handle tool calls
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
# UI
# ============================================================

st.set_page_config(page_title="Football Research Agent v2", page_icon="⚽", layout="wide")
st.title("⚽ Football Research Agent v2")

# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")
    
    mode = st.radio("Mode", ["💬 Chat", "🤖 Autonomous Agent"], index=1)
    
    if mode == "🤖 Autonomous Agent":
        st.divider()
        pl_col = st.selectbox("Target Market", [
            "BO 2.5 PL", "BTTS PL", "SHG PL", "SHG 2+ PL", 
            "LU1.5 PL", "LFGHU0.5 PL", "BO1.5 FHG PL"
        ])
        
        if st.button("🚀 Start Research", type="primary", disabled=st.session_state.agent_active):
            run_agent_session(pl_col)
            st.rerun()
        
        if st.session_state.agent_active:
            st.warning(f"🔄 Agent active: Iteration {st.session_state.agent_iteration}")
        
        if st.session_state.agent_findings:
            st.success(f"📊 {len(st.session_state.agent_findings)} findings")
    
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.agent_findings = []
        st.session_state.bible = None
        st.rerun()
    
    st.divider()
    st.caption("v2 - Autonomous Agent Edition")

# Main area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if mode == "💬 Chat":
    if prompt := st.chat_input("Ask me anything..."):
        _chat_response(prompt)
        st.rerun()
else:
    st.info("👆 Select a market and click **Start Research** to begin autonomous analysis.")

# Debug panel
with st.expander("🔍 Debug Info"):
    st.json({
        "session_id": st.session_state.session_id,
        "messages": len(st.session_state.messages),
        "agent_active": st.session_state.agent_active,
        "agent_iteration": st.session_state.agent_iteration,
        "findings": len(st.session_state.agent_findings),
        "bible_loaded": st.session_state.bible is not None,
    })
