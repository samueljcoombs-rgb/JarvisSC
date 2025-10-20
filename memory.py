import json, os, time
FILE = "memory.json"

def _load():
    if not os.path.exists(FILE): return []
    try:
        with open(FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def _save(data):
    with open(FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def add_fact(text, kind="note"):
    data = _load()
    data.append({"ts": int(time.time()), "kind": kind, "text": text})
    _save(data)

def recent_summary(max_chars=800):
    data = _load()
    s = "\n".join([f"- {d['kind']}: {d['text']}" for d in data[-30:]])
    return s[:max_chars]