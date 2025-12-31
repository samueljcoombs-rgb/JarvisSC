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
# OpenAI Client (GPT-4o)
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
    """Get the OpenAI model to use - defaults to gpt-4o, can override with env var."""
    return os.getenv("OPENAI_MODEL") or os.getenv("PREFERRED_OPENAI_MODEL") or "gpt-4o"

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
    """Load research context from Google Sheets (Bible) directly."""
    try:
        if "bible" not in st.session_state:
            st.session_state.bible = None
        if st.session_state.bible:
            return st.session_state.bible
        
        _log("Loading Bible from Google Sheets...")
        
        # Import and call the Google Sheets function DIRECTLY (not via job queue)
        try:
            from football_tools import get_research_context as get_bible_from_sheets
            bible = get_bible_from_sheets(limit_notes=20)
            
            if bible.get("ok"):
                _log(f"Bible loaded: {len(bible.get('research_rules', []))} rules, {len(bible.get('column_definitions', []))} column defs")
                st.session_state.bible = bible
                return bible
            else:
                _log(f"Bible load returned not ok: {bible}")
        except ImportError as e:
            _log(f"Could not import football_tools: {e}")
        except Exception as e:
            _log(f"Error calling get_research_context from sheets: {e}")
        
        # Fallback: Try via job queue (local_compute)
        _log("Falling back to job queue for Bible...")
        bible = _run_tool("get_research_context", {"pl_column": st.session_state.target_pl_column})
        
        # If still no Bible, create minimal version
        if bible.get("error") or not bible.get("research_rules"):
            _log("Creating minimal Bible fallback")
            bible = {
                "dataset_overview": {"primary_goal": "Find profitable betting strategies"},
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
    rules = bible.get("research_rules") or []
    
    outcome_cols = derived.get('outcome_columns', OUTCOME_COLUMNS)
    outcome_str = ', '.join(str(c) for c in outcome_cols) if isinstance(outcome_cols, list) else str(outcome_cols)
    
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
{rules_text}"""

# ============================================================
# LLM Functions (GPT-4o)
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

## CRITICAL RULES
1. NEVER use PL columns as features (data leakage!)
2. Split by TIME: train older, test newer (never random!)
3. Explain WHY a filter exploits market inefficiency
4. Simple > complex - prefer fewer filters
5. Sample size matters - need 300+ train rows, 60+ test rows
6. p-value < 0.10 for statistical significance
7. Forward walk must show >60% positive periods

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
    """Call LLM for analysis with proper error handling."""
    client = _get_client()
    if not client:
        _log("ERROR: OpenAI client not available - check OPENAI_API_KEY")
        return '{"error": "OpenAI client not available - check OPENAI_API_KEY"}'
    
    try:
        model = _get_model()
        _log(f"Calling LLM: model={model}, context_len={len(context)}, question_len={len(question)}")
        
        # Build request params
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\n---\n{question}"}
            ],
            "temperature": 0.7,
        }
        
        # Different models use different token params
        if model.startswith("o1"):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        _log(f"LLM response received: {len(content) if content else 0} chars")
        return content or ""
        
    except Exception as e:
        error_msg = str(e)
        _log(f"LLM ERROR: {error_msg}")
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
st.caption("Deep Analysis Edition + Local Compute + 38 Tools + Persistence + GPT-4o")

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
    
    st.caption("v6.0 - GPT-4o + Local Compute")

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
2. **DEEP ANALYSIS**: GPT-4o analyzes after EVERY phase
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
14. **GPT-4o**: Latest model for analysis
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
