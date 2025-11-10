import json, os, time
FILE = "memory.json"

def _load():
    if not os.path.exists(FILE):
        return []
    try:
        with open(FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def _save(d):
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

def add_fact(text, kind="note"):
    d = _load()
    d.append({"ts": int(time.time()), "kind": kind, "text": text})
    _save(d)

def recent_summary(max_chars=1200):
    d = _load()
    s = "\n".join(f"- {x['kind']}: {x['text']}" for x in d[-30:])
    return s[:max_chars]
