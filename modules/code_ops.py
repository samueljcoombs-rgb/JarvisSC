from __future__ import annotations

import os
import base64
import json
import py_compile
from typing import Any, Dict, List

import requests
import streamlit as st


# Restrict where code updates can be applied/committed from this UI/tooling
_ALLOWED = [
    "pages/football_researcher.py",
    "modules/football_tools.py",
    "modules/code_ops.py",
]


def allowed_paths() -> List[str]:
    return list(_ALLOWED)


def _require_allowed(path: str) -> None:
    path = (path or "").strip()
    if path not in _ALLOWED:
        raise ValueError(f"Path not allowed: {path}. Allowed: {', '.join(_ALLOWED)}")


def run_py_checks(paths: List[str]) -> Dict[str, Any]:
    """
    Syntax-only checks (py_compile) for specified files.
    This works in Streamlit Cloud and catches the most common breakages.
    """
    try:
        if not paths:
            return {"ok": False, "error": "No paths provided."}

        checked = []
        for p in paths:
            _require_allowed(p)
            py_compile.compile(p, doraise=True)
            checked.append(p)

        return {"ok": True, "checked": checked}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _github_cfg() -> Dict[str, str]:
    token = st.secrets.get("GITHUB_TOKEN") if hasattr(st, "secrets") else None
    token = token or os.getenv("GITHUB_TOKEN")

    repo = st.secrets.get("GITHUB_REPO") if hasattr(st, "secrets") else None
    repo = (repo or os.getenv("GITHUB_REPO") or "").strip()  # owner/repo

    branch = st.secrets.get("GITHUB_BRANCH") if hasattr(st, "secrets") else None
    branch = (branch or os.getenv("GITHUB_BRANCH") or "main").strip()

    if not token or not repo:
        raise RuntimeError("Missing GITHUB_TOKEN and/or GITHUB_REPO (owner/repo) in secrets/env.")

    return {"token": token, "repo": repo, "branch": branch}


def _gh_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def _get_existing_sha(repo: str, path: str, branch: str, token: str) -> str:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(token), params={"ref": branch}, timeout=60)
    if r.status_code == 200:
        return (r.json() or {}).get("sha") or ""
    if r.status_code == 404:
        return ""
    raise RuntimeError(f"GitHub GET contents failed ({r.status_code}): {r.text}")


def _put_contents(repo: str, path: str, branch: str, token: str, message: str, content: str) -> Dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    sha = _get_existing_sha(repo, path, branch, token)

    b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    payload: Dict[str, Any] = {
        "message": message,
        "content": b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=_gh_headers(token), data=json.dumps(payload), timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT contents failed ({r.status_code}): {r.text}")

    return r.json() if r.text else {"ok": True}


def github_commit_files(message: str, files: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Commits full file contents via GitHub Contents API.
    Commits each file as a separate commit (simpler + reliable).
    """
    try:
        cfg = _github_cfg()
        repo, branch, token = cfg["repo"], cfg["branch"], cfg["token"]

        if not message or not message.strip():
            return {"ok": False, "error": "Commit message is required."}
        if not files:
            return {"ok": False, "error": "No files provided."}

        results = []
        for f in files:
            path = (f.get("path") or "").strip()
            content = f.get("content") or ""
            _require_allowed(path)

            res = _put_contents(repo, path, branch, token, message, content)
            results.append({"path": path, "result": res})

        return {"ok": True, "repo": repo, "branch": branch, "committed": results}
    except Exception as e:
        return {"ok": False, "error": str(e)}
