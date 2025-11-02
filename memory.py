import json, os, time, requests

# ---------------- CONFIG ----------------
GITHUB_API_URL = "https://api.github.com"
FILE = "memory.json"

# Load credentials safely
import streamlit as st
GITHUB_PAT = st.secrets.get("GITHUB_PAT")
GITHUB_USERNAME = st.secrets.get("GITHUB_USERNAME")
GITHUB_REPO = st.secrets.get("GITHUB_REPO")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_PAT}",
    "Accept": "application/vnd.github+json",
}

# ---------------- INTERNAL HELPERS ----------------
def _github_file_url():
    return f"{GITHUB_API_URL}/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{FILE}"

def _load_local():
    if not os.path.exists(FILE):
        return []
    try:
        with open(FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def _save_local(data):
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_from_github():
    try:
        r = requests.get(_github_file_url(), headers=HEADERS)
        if r.status_code == 200:
            content = r.json()["content"]
            import base64
            decoded = base64.b64decode(content).decode("utf-8")
            return json.loads(decoded)
        else:
            print("⚠️ Could not load memory from GitHub:", r.status_code, r.text)
    except Exception as e:
        print("⚠️ Error loading from GitHub:", e)
    return _load_local()

def _save_to_github(data):
    try:
        # First get SHA of current file if it exists
        r = requests.get(_github_file_url(), headers=HEADERS)
        sha = r.json().get("sha") if r.status_code == 200 else None

        encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        import base64
        content_b64 = base64.b64encode(encoded).decode("utf-8")

        payload = {
            "message": f"Update memory.json ({time.strftime('%Y-%m-%d %H:%M:%S')})",
            "content": content_b64,
            "branch": "main",
        }
        if sha:
            payload["sha"] = sha

        r = requests.put(_github_file_url(), headers=HEADERS, json=payload)
        if r.status_code in (200, 201):
            print("✅ Memory updated on GitHub.")
        else:
            print("⚠️ GitHub save failed:", r.status_code, r.text)
    except Exception as e:
        print("⚠️ Exception saving to GitHub:", e)
        _save_local(data)

# ---------------- MAIN FUNCTIONS ----------------
def _load():
    data = _load_from_github()
    if not data:
        data = [{"ts": int(time.time()), "kind": "system", "text": "Memory initialized."}]
        _save_to_github(data)
    return data

def _save(data):
    _save_local(data)
    _save_to_github(data)

def add_fact(text, kind="note"):
    data = _load()
    data.append({"ts": int(time.time()), "kind": kind, "text": text})
    _save(data)

def recent_summary(max_chars=800):
    data = _load()
    s = "\n".join([f"- {d['kind']}: {d['text']}" for d in data[-30:]])
    return s[:max_chars]
