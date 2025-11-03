# memory.py
import json
import os
import time
from pathlib import Path

MEMORY_FILE = Path("long_term_memory.json")


def _load():
    """Load all stored memories from disk."""
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(memories):
    """Persist memories to disk safely."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save memory: {e}")


def clean_text(text: str) -> str:
    """Remove whitespace and normalize casing."""
    return text.strip().replace("\n", " ")


def add_fact(fact: str, kind: str = "user"):
    """Add a new memory, avoiding duplicates."""
    fact = clean_text(fact)
    if not fact:
        return
    memories = _load()

    for m in memories:
        if m["text"].lower() == fact.lower():
            m["ts"] = int(time.time())  # refresh timestamp
            _save(memories)
            return

    memories.append(
        {
            "text": fact,
            "kind": kind,
            "ts": int(time.time()),
        }
    )
    _save(memories)


def recent_summary(limit: int = 15) -> str:
    """Return a summary string of the latest few memories."""
    mems = sorted(_load(), key=lambda m: m["ts"], reverse=True)[:limit]
    if not mems:
        return ""
    lines = [f"- ({m['kind']}) {m['text']}" for m in mems]
    return "\n".join(lines)


def delete_fact(index: int):
    """Remove a memory by its list index."""
    mems = _load()
    if 0 <= index < len(mems):
        mems.pop(index)
        _save(mems)


def clear_all():
    """Delete all memories."""
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
