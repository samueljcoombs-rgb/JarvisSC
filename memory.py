# memory.py — simple JSON memory for your agent
import json, os, time
from typing import List, Dict

DEFAULT_PATH = "memory.json"

def _path():
    return os.environ.get("MEMORY_FILE", DEFAULT_PATH)

def _load() -> List[Dict]:
    p = _path()
    if not os.path.exists(p): return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save(items: List[Dict]):
    with open(_path(), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def add_fact(text: str, kind: str = "note", tags: List[str] = None):
    items = _load()
    items.append({
        "ts": int(time.time()),
        "kind": kind,
        "text": text,
        "tags": tags or []
    })
    _save(items)

def all_facts() -> List[Dict]:
    return _load()

def recent_summary(max_chars: int = 800) -> str:
    """Return a compact string summary for system prompt context."""
    items = _load()[-50:]  # last 50 notes/facts
    lines = []
    for it in items:
        lines.append(f"- [{it.get('kind','note')}] {it.get('text','')}")
    s = "\n".join(lines)
    return s[:max_chars]