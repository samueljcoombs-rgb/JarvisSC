# memory.py
"""
Memory module for Jarvis - Now with Google Sheets integration.
Maintains backwards compatibility with local JSON storage while
also persisting to Google Sheets when available.
"""
import json
import os
import time
from pathlib import Path
from datetime import datetime

FILE = "memory.json"

# Try to import sheets_memory for Google Sheets persistence
try:
    from modules import sheets_memory as sm
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

def _load():
    """Load memories from local JSON file."""
    if not os.path.exists(FILE):
        return []
    try:
        with open(FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def _save(d):
    """Save memories to local JSON file (atomic write)."""
    path = Path(FILE)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def add_fact(text, kind="note", context=""):
    """
    Add a memory fact.
    Saves to both local JSON and Google Sheets if available.
    
    Args:
        text: The memory content
        kind: Type of memory (fact, preference, note, etc.)
        context: Optional context about where this memory came from
    """
    # Save to local JSON
    d = _load()
    d.append({
        "ts": int(time.time()),
        "kind": kind,
        "text": text,
        "context": context
    })
    _save(d)
    
    # Also save to Google Sheets if available
    if SHEETS_AVAILABLE:
        try:
            sm.add_memory(text, kind, context)
        except Exception as e:
            pass  # Silently fail, local backup exists

def add_preference(text, context=""):
    """Add a user preference to memory."""
    add_fact(text, kind="preference", context=context)

def add_note(text, context=""):
    """Add a note to memory."""
    add_fact(text, kind="note", context=context)

def recent_summary(max_chars=1200):
    """
    Get a summary of recent memories.
    Tries Google Sheets first, falls back to local JSON.
    
    Args:
        max_chars: Maximum characters to return
    
    Returns:
        String summary of recent memories
    """
    # Try Google Sheets first
    if SHEETS_AVAILABLE:
        try:
            summary = sm.get_memory_summary()
            if summary and summary.strip():
                return summary[:max_chars]
        except Exception:
            pass
    
    # Fall back to local JSON
    d = _load()
    if not d:
        return "No memories stored yet."
    
    # Format recent memories
    lines = []
    for x in d[-30:]:
        kind = x.get('kind', 'note')
        text = x.get('text', '')
        lines.append(f"- [{kind}] {text}")
    
    s = "\n".join(lines)
    return s[:max_chars]

def search_memories(query, limit=10):
    """
    Search memories for a specific query.
    
    Args:
        query: Search term
        limit: Maximum results to return
    
    Returns:
        List of matching memory dicts
    """
    # Try Google Sheets first
    if SHEETS_AVAILABLE:
        try:
            results = sm.search_memories(query, limit)
            if results:
                return results
        except Exception:
            pass
    
    # Fall back to local search
    d = _load()
    query_lower = query.lower()
    matches = [
        m for m in d
        if query_lower in m.get('text', '').lower()
    ]
    return matches[-limit:]

def get_all_memories():
    """Get all memories (from Sheets if available, else local)."""
    if SHEETS_AVAILABLE:
        try:
            return sm.get_memories(limit=1000)
        except Exception:
            pass
    return _load()

def clear_local():
    """Clear local memory file (for testing/reset)."""
    _save([])

def sync_to_sheets():
    """
    Sync local memories to Google Sheets.
    Useful for initial migration.
    """
    if not SHEETS_AVAILABLE:
        return False
    
    local = _load()
    for m in local:
        try:
            sm.add_memory(
                m.get('kind', 'note'),
                m.get('text', ''),
                m.get('context', 'synced from local')
            )
        except Exception:
            continue
    return True

def get_memory_stats():
    """Get statistics about stored memories."""
    stats = {
        "local_count": len(_load()),
        "sheets_available": SHEETS_AVAILABLE,
        "sheets_count": 0
    }
    
    if SHEETS_AVAILABLE:
        try:
            sheets_memories = sm.get_memories(limit=10000)
            stats["sheets_count"] = len(sheets_memories)
        except Exception:
            pass
    
    return stats
