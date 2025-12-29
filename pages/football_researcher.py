"""
Football Researcher v2.3 - Simplified & Robust

Key changes:
- No nested st.status/expander (causes issues)
- Simpler display logic
- Better error handling
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

# Page config first
st.set_page_config(page_title="Football Research Agent", page_icon="⚽", layout="wide")

# ============================================================
# Helpers
# ============================================================

def _safe_json(obj: Any, max_len: int = 5000) -> str:
    """Safely convert to JSON string."""
    try:
        result = json.dumps(obj, indent=2, default=str)
        if len(result) > max_len:
            result = result[:max_len] + "\n... (truncated)"
        return result
    except Exception as e:
        return f'{{"error": "Serialize failed: {e}"}}'

# ============================================================
# Config
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
        "target_pl_column": "BO 2.5 PL",
        "exploration_results": {},
        "past_failures": [],
        "run_requested": False,
        "log": [],  # Simple log for progress
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def _log(msg: str):
    """Add to progress log."""
    st.session_state.log.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")

def _append(role: str, content: str):
    st.session_state.messages.append({
        "role": role, 
        "content": content, 
        "ts": datetime.utcnow().isoformat()
    })
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

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

def _wait_for_job(job_id: str, timeout: int = JOB_TIMEOUT) -> Dict:
    """Wait for job to complete."""
    start = time.time()
    
    while time.time() - start < timeout:
        job = _run_tool("get_job", {"job_id": job_id})
        status = (job.get("status") or "").lower()
        
        _log(f"Job {job_id[:8]}: {status}")
        
        if status == "done":
            result_path = job.get("result_path")
            if result_path:
                result = _run_tool("download_result", {"result_path": result_path})
                return result.get("result", job)
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
    bible = _run_tool("get_research_context", {"limit_notes": 30})
    st.session_state.bible = bible
    _log("Bible loaded")
    return bible

def _format_bible(bible: Dict) -> str:
    overview = bible.get("dataset_overview") or {}
    gates = bible.get("gates") or {}
    derived = bible.get("derived") or {}
    
    outcome_cols = derived.get('outcome_columns', [])
    outcome_str = ', '.join(str(c) for c in outcome_cols) if isinstance(outcome_cols, list) else str(outcome_cols)
    
    return f"""## 📖 Bible Loaded

**Goal:** {overview.get('primary_goal', 'Find profitable strategies')}
**Output:** {overview.get('strategy_output_format', 'Explicit filters')}

**Gates:** min_train={gates.get('min_train_rows', 300)}, min_val={gates.get('min_val_rows', 60)}, min_test={gates.get('min_test_rows', 60)}, max_gap={gates.get('max_train_val_gap_roi', 0.4)}, max_dd={gates.get('max_test_drawdown', -50)}

**Outcome Columns (NEVER features):** {outcome_str}
"""

# ============================================================
# LLM
# ============================================================

SYSTEM_PROMPT = """You are a football betting research agent.

## Mission
Find EXPLICIT filter criteria for profitable bets.

## Rules
1. NEVER use PL columns as features
2. Split by TIME: train older, test newer
3. Simple > complex
4. Use EXACT column names from exploration

## Filter Format
- {"col": "MODE", "op": "=", "value": "XG"}
- {"col": "LEAGUE", "op": "in", "values": ["EPL"]}
- {"col": "ACTUAL ODDS", "op": "between", "min": 1.8, "max": 2.5}

Respond with JSON when asked."""

def _llm(context: str, question: str) -> str:
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

def _parse_json(resp: str) -> Optional[Dict]:
    try:
        resp = resp.strip()
        if "```" in resp:
            resp = re.sub(r'```json\s*', '', resp)
            resp = re.sub(r'```\s*', '', resp)
        match = re.search(r'\{[\s\S]*\}', resp)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None

def _form_hypothesis(bible: Dict, exploration: Dict, failures: List) -> Dict:
    context = f"""
Gates: {_safe_json(bible.get('gates', {}), 300)}
Exploration: {_safe_json(exploration, 2500)}
Past Failures: {_safe_json(failures[-3:], 1000) if failures else 'None'}
"""
    question = """Form hypothesis. JSON only:
{"hypothesis": "...", "reasoning": "...", "filters": [...], "confidence": "low/medium/high"}"""
    
    resp = _llm(context, question)
    parsed = _parse_json(resp)
    return parsed if parsed and parsed.get("filters") else {"hypothesis": resp[:300], "filters": [], "parse_error": True}

def _analyze_result(bible: Dict, hypothesis: Dict, result: Dict, iteration: int) -> Dict:
    context = f"""
Gates: {_safe_json(bible.get('gates', {}), 300)}
Hypothesis: {_safe_json(hypothesis, 500)}
Result: {_safe_json(result, 2500)}
Iteration: {iteration}/{MAX_ITERATIONS}
"""
    question = """Analyze. JSON only:
{"analysis": "...", "passed_gates": true/false, "decision": "refine|new_hypothesis|success|conclude_no_edge", "learning": "..."}"""
    
    resp = _llm(context, question)
    parsed = _parse_json(resp)
    return parsed if parsed else {"analysis": resp[:300], "decision": "new_hypothesis", "passed_gates": False}

def _conclusion(bible: Dict, findings: List, success: bool) -> str:
    context = f"Findings: {_safe_json(findings, 4000)}"
    if success:
        question = "Format winning strategy with explicit criteria, performance, and stability."
    else:
        question = "Explain why no edge found, what was tried, and recommendations."
    return _llm(context, question)

# ============================================================
# Exploration
# ============================================================

def _explore(pl_column: str) -> Dict:
    results = {}
    
    queries = [
        ("by_mode", {"query_type": "aggregate", "group_by": ["MODE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_drift", {"query_type": "aggregate", "group_by": ["DRIFT IN / OUT"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"]}),
        ("by_league", {"query_type": "aggregate", "group_by": ["LEAGUE"], "metrics": ["count", f"sum:{pl_column}", f"mean:{pl_column}"], "limit": 15}),
    ]
    
    for name, params in queries:
        _log(f"Exploring {name}...")
        job = _run_tool("query_data", params)
        if job.get("job_id"):
            results[name] = _wait_for_job(job["job_id"], timeout=120)
        else:
            results[name] = {"error": job.get("error", "Failed")}
    
    return results

# ============================================================
# Test hypothesis
# ============================================================

def _test_hypothesis(hypothesis: Dict, pl_column: str, bible: Dict) -> Dict:
    filters = hypothesis.get("filters", [])
    if not filters:
        return {"error": "No filters"}
    
    _log(f"Testing: {filters}")
    job = _run_tool("test_filter", {
        "filters": filters,
        "pl_column": pl_column,
        "enforcement": bible.get("gates", {}),
    })
    
    if not job.get("job_id"):
        return {"error": job.get("error", "Submit failed")}
    
    return _wait_for_job(job["job_id"], timeout=JOB_TIMEOUT)

# ============================================================
# Main agent
# ============================================================

def run_agent():
    pl_column = st.session_state.target_pl_column
    st.session_state.log = []  # Reset log
    
    output = st.container()
    
    with output:
        st.markdown(f"# 🤖 Research: {pl_column}")
        
        # Bible
        st.markdown("### 📖 Loading Bible...")
        bible = _load_bible()
        st.markdown(_format_bible(bible))
        _append("assistant", _format_bible(bible))
        
        # Exploration
        st.markdown("### 🔍 Exploring data...")
        exploration = _explore(pl_column)
        st.session_state.exploration_results = exploration
        
        st.markdown("**Exploration Results:**")
        st.code(_safe_json(exploration, 3000), language="json")
        _append("assistant", f"Exploration:\n```json\n{_safe_json(exploration, 1500)}\n```")
        st.session_state.agent_findings.append({"phase": "exploration", "results": exploration})
        
        # Iterations
        st.markdown("---")
        st.markdown("### 🧪 Hypothesis Testing")
        
        failures = []
        
        for i in range(1, MAX_ITERATIONS + 1):
            st.session_state.agent_iteration = i
            st.markdown(f"#### Iteration {i}/{MAX_ITERATIONS}")
            
            # Form hypothesis
            hypothesis = _form_hypothesis(bible, exploration, failures)
            
            st.markdown(f"**Hypothesis:** {hypothesis.get('hypothesis', 'N/A')}")
            st.markdown(f"**Filters:** `{hypothesis.get('filters', [])}`")
            
            if not hypothesis.get("filters") or hypothesis.get("parse_error"):
                st.warning("⚠️ Invalid filters")
                failures.append({"iteration": i, "error": "Invalid"})
                continue
            
            # Test
            st.markdown("**Testing...**")
            result = _test_hypothesis(hypothesis, pl_column, bible)
            
            st.code(_safe_json(result, 2000), language="json")
            
            st.session_state.agent_findings.append({
                "iteration": i,
                "hypothesis": hypothesis,
                "result": result,
            })
            
            # Analyze
            analysis = _analyze_result(bible, hypothesis, result, i)
            
            st.markdown(f"**Analysis:** {analysis.get('analysis', 'N/A')}")
            st.markdown(f"**Gates Passed:** {analysis.get('passed_gates', '?')} | **Decision:** {analysis.get('decision', '?')}")
            
            _append("assistant", f"Iteration {i}: {analysis.get('decision')}")
            
            decision = (analysis.get("decision") or "").lower()
            
            if decision == "success":
                st.markdown("# 🎉 Strategy Found!")
                conclusion = _conclusion(bible, st.session_state.agent_findings, True)
                st.markdown(conclusion)
                _append("assistant", conclusion)
                st.session_state.agent_phase = "complete"
                st.balloons()
                return
            
            elif decision == "conclude_no_edge":
                break
            
            failures.append({
                "iteration": i,
                "hypothesis": hypothesis,
                "reason": analysis.get("learning", ""),
            })
            st.session_state.past_failures = failures
            
            st.markdown("---")
        
        # No edge
        st.markdown("# 📋 No Edge Found")
        conclusion = _conclusion(bible, st.session_state.agent_findings, False)
        st.markdown(conclusion)
        _append("assistant", conclusion)
        st.session_state.agent_phase = "complete"

# ============================================================
# Chat mode
# ============================================================

def _chat(user_input: str):
    _append("user", user_input)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})
    
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=messages,
            max_tokens=2000,
        )
        _append("assistant", resp.choices[0].message.content)
    except Exception as e:
        _append("assistant", f"Error: {e}")

# ============================================================
# UI
# ============================================================

st.title("⚽ Football Research Agent")

with st.sidebar:
    st.header("🎛️ Controls")
    
    mode = st.radio("Mode", ["🤖 Autonomous", "💬 Chat"], index=0)
    
    st.divider()
    
    if mode == "🤖 Autonomous":
        pl_col = st.selectbox("Market", [
            "BO 2.5 PL", "BTTS PL", "SHG PL", "SHG 2+ PL", 
            "LU1.5 PL", "LFGHU0.5 PL", "BO1.5 FHG PL"
        ])
        st.session_state.target_pl_column = pl_col
        
        if st.button("🚀 Start Research", type="primary"):
            st.session_state.agent_phase = "running"
            st.session_state.agent_iteration = 0
            st.session_state.agent_findings = []
            st.session_state.past_failures = []
            st.session_state.exploration_results = {}
            st.session_state.bible = None
            st.session_state.run_requested = True
            st.rerun()
        
        if st.session_state.agent_phase == "running":
            st.warning(f"🔄 Running...")
        elif st.session_state.agent_phase == "complete":
            st.success("✅ Done")
    
    st.divider()
    
    if st.button("🗑️ Clear"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    
    # Progress log
    if st.session_state.log:
        with st.expander("📋 Log"):
            for line in st.session_state.log[-20:]:
                st.text(line)
    
    st.caption("v2.3")

# Main content
if mode == "🤖 Autonomous":
    if st.session_state.run_requested:
        st.session_state.run_requested = False
        run_agent()
    elif st.session_state.agent_phase == "idle":
        st.info("👆 Click **Start Research**")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask..."):
        _chat(prompt)
        st.rerun()
