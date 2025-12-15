# worker/modal_app.py
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import modal

app = modal.App("football-research-worker")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas==2.2.2",
        "numpy==2.0.2",
        "requests==2.32.3",
        "supabase==2.6.0",
    )
)

SUPABASE_SECRET = modal.Secret.from_name("football-supabase")


# -----------------------------
# Helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()


# -----------------------------
# Heavy compute + Supabase logic
# (runs inside Modal container)
# -----------------------------
def _load_csv_from_supabase_storage(sb, bucket: str, path: str):
    import pandas as pd
    import numpy as np
    from io import BytesIO

    # Download bytes from Supabase Storage
    # supabase-py returns a response-like object; .read() is supported by storage3
    res = sb.storage.from_(bucket).download(path)
    if res is None:
        raise RuntimeError(f"Storage download returned None for {bucket}/{path}")

    if isinstance(res, (bytes, bytearray)):
        raw = bytes(res)
    else:
        # fallback: try to treat it as a file-like object
        raw = res

    bio = BytesIO(raw)
    df = pd.read_csv(bio, low_memory=False)
    return df


def _to_dt(df):
    import pandas as pd

    if "DATE" in df.columns and "TIME" in df.columns:
        dt = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["TIME"].astype(str), errors="coerce"
        )
        if dt.notna().any():
            return dt
    if "DATE" in df.columns:
        dt = pd.to_datetime(df["DATE"], errors="coerce")
        if dt.notna().any():
            return dt
    return pd.to_datetime(pd.Series([None] * len(df)), errors="coerce")


def _time_split(df, ratio: float = 0.7):
    import numpy as np

    ratio = float(ratio or 0.7)
    ratio = max(0.5, min(ratio, 0.95))
    tmp = df.copy()
    tmp["_dt"] = _to_dt(tmp)
    tmp = tmp.sort_values("_dt", na_position="last")
    cut = int(np.floor(len(tmp) * ratio))
    train = tmp.iloc[:cut].drop(columns=["_dt"])
    test = tmp.iloc[cut:].drop(columns=["_dt"])
    return train, test


def _roi(df, pl_col: str, side: str, odds_col: Optional[str]) -> Dict[str, Any]:
    import pandas as pd

    d = df[df[pl_col].notna()].copy()
    d[pl_col] = pd.to_numeric(d[pl_col], errors="coerce")
    d = d[d[pl_col].notna()]
    n = int(len(d))
    total_pl = float(d[pl_col].sum()) if n else 0.0

    if n == 0:
        return {"bets": 0, "total_pl": 0.0, "roi": 0.0, "avg_pl": 0.0, "denom": 0.0}

    side = (side or "back").lower().strip()
    if side == "lay":
        if not odds_col or odds_col not in d.columns:
            return {"error": f"Lay ROI needs odds_col. Missing/invalid: {odds_col}"}
        odds = pd.to_numeric(d[odds_col], errors="coerce").fillna(0.0)
        liability = (odds - 1.0).clip(lower=0.0)
        denom = float(liability.sum())
        roi = (total_pl / denom) if denom > 0 else 0.0
        return {
            "bets": n,
            "total_pl": total_pl,
            "roi": roi,
            "avg_pl": total_pl / n,
            "denom": denom,
            "mode": "lay_liability",
        }

    denom = float(n)
    return {
        "bets": n,
        "total_pl": total_pl,
        "roi": total_pl / denom,
        "avg_pl": total_pl / denom,
        "denom": denom,
        "mode": "back_flat_1pt",
    }


def _max_drawdown(points: List[float]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in points:
        cum += float(p)
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return float(max_dd)


def _longest_losing_streak(points: List[float]) -> Dict[str, Any]:
    max_bets = 0
    cur_bets = 0
    worst_pl = 0.0
    cur_pl = 0.0
    for p in points:
        if p < 0:
            cur_bets += 1
            cur_pl += float(p)
            max_bets = max(max_bets, cur_bets)
            worst_pl = min(worst_pl, cur_pl)
        else:
            cur_bets = 0
            cur_pl = 0.0
    return {"bets": int(max_bets), "pl": float(worst_pl)}


def _game_level(df, pl_col: str) -> Dict[str, Any]:
    import pandas as pd

    if "ID" not in df.columns:
        return {"error": "Missing ID column"}
    d = df[df[pl_col].notna()].copy()
    d[pl_col] = pd.to_numeric(d[pl_col], errors="coerce")
    d = d[d[pl_col].notna()]
    if len(d) == 0:
        return {"games": 0, "max_dd": 0.0, "losing_streak": {"bets": 0, "pl": 0.0}}

    d["_dt"] = _to_dt(d)
    g = (
        d.groupby("ID", as_index=False)
        .agg(game_pl=(pl_col, "sum"), date=("_dt", "min"))
        .sort_values("date", na_position="last")
    )
    pts = g["game_pl"].astype(float).tolist()
    return {
        "games": int(len(g)),
        "max_dd": _max_drawdown(pts),
        "losing_streak": _longest_losing_streak(pts),
    }


def _mapping() -> Dict[str, Tuple[str, str]]:
    return {
        "SHG PL": ("lay", "HT CS Price"),
        "SHG 2+ PL": ("lay", "HT 2 Ahead Odds"),
        "LU1.5 PL": ("lay", "U1.5 Odds"),
        "LFGHU0.5 PL": ("lay", "FHGU0.5Odds"),
        "BO 2.5 PL": ("back", "O2.5 Odds"),
        "BO1.5 FHG PL": ("back", "FHGO1.5 Odds"),
        "BTTS PL": ("back", "BTTS Y Odds"),
    }


def _pick_market(df, split_ratio: float = 0.7) -> Dict[str, Any]:
    mp = _mapping()
    candidates = []
    for pl_col, (side, odds_col) in mp.items():
        if pl_col not in df.columns:
            continue
        filtered = df[df[pl_col].notna()].copy()
        train, test = _time_split(filtered, split_ratio)
        test_stats = _roi(test, pl_col, side, odds_col)
        candidates.append(
            (pl_col, side, odds_col, float(test_stats.get("roi", -999)), int(test_stats.get("bets", 0)))
        )

    candidates.sort(key=lambda x: (x[3], x[4]), reverse=True)
    if not candidates:
        return {"error": "No PL columns found to choose from."}

    pl_col, side, odds_col, test_roi, test_bets = candidates[0]
    return {
        "pl_column": pl_col,
        "side": side,
        "odds_col": odds_col,
        "test_roi": float(test_roi),
        "test_bets": int(test_bets),
    }


def _rule_search(df, pl_col: str, side: str, odds_col: str, split_ratio: float = 0.7) -> Dict[str, Any]:
    import pandas as pd
    import numpy as np

    numeric_cols = [
        "DIFF",
        "% DRIFT",
        "ACTUAL ODDS",
        odds_col,
        "H XG VS A XG S",
        "H XG VS A XG 6",
        "Points Diff",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    if pl_col not in df.columns:
        return {"error": f"Missing {pl_col}"}
    if odds_col not in df.columns:
        return {"error": f"Missing odds column {odds_col}"}

    d0 = df[df[pl_col].notna()].copy()
    d0[pl_col] = pd.to_numeric(d0[pl_col], errors="coerce")
    d0 = d0[d0[pl_col].notna()].copy()

    for c in available:
        d0[c] = pd.to_numeric(d0[c], errors="coerce")

    train, test = _time_split(d0, split_ratio)

    def qgrid(s: pd.Series):
        s2 = s.dropna()
        if len(s2) < 500:
            return []
        qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        vals = [float(s2.quantile(q)) for q in qs]
        return [(vals[i], vals[i + 1]) for i in range(len(vals) - 1)]

    grids = {c: qgrid(train[c]) for c in available}
    cols = [c for c in available if grids.get(c)]
    if len(cols) < 2:
        return {"error": "Not enough numeric columns with stable quantile grids to search."}

    best = []
    max_rules = 250
    tried = 0

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            for (a1, b1) in grids[c1]:
                for (a2, b2) in grids[c2]:
                    tried += 1
                    if tried > max_rules:
                        break

                    tr = train[train[c1].between(a1, b1) & train[c2].between(a2, b2)]
                    te = test[test[c1].between(a1, b1) & test[c2].between(a2, b2)]

                    if len(tr) < 300 or len(te) < 120:
                        continue

                    tr_roi = _roi(tr, pl_col, side, odds_col)
                    te_roi = _roi(te, pl_col, side, odds_col)

                    gap = float(tr_roi["roi"]) - float(te_roi["roi"])
                    te_games = _game_level(te, pl_col)
                    score = float(te_roi["roi"]) - 0.25 * max(gap, 0.0)

                    best.append(
                        {
                            "score": score,
                            "rule": [
                                {"col": c1, "min": a1, "max": b1},
                                {"col": c2, "min": a2, "max": b2},
                            ],
                            "train": tr_roi,
                            "test": te_roi,
                            "test_game_level": te_games,
                            "gap_train_minus_test": gap,
                            "samples": {"train_rows": int(len(tr)), "test_rows": int(len(te))},
                        }
                    )
                if tried > max_rules:
                    break
            if tried > max_rules:
                break
        if tried > max_rules:
            break

    best.sort(key=lambda x: x["score"], reverse=True)
    return {
        "pl_column": pl_col,
        "side": side,
        "odds_col": odds_col,
        "searched_rules": int(tried),
        "top_rules": best[:10],
    }


def _compute_task_strategy_search(sb, params: Dict[str, Any]) -> Dict[str, Any]:
    split_ratio = float(params.get("time_split_ratio", 0.7))

    storage_bucket = params.get("storage_bucket")
    storage_path = params.get("storage_path")
    if not storage_bucket or not storage_path:
        return {"error": "strategy_search requires params.storage_bucket and params.storage_path"}

    df = _load_csv_from_supabase_storage(sb, storage_bucket, storage_path)

    chosen = params.get("pl_column")
    mp = _mapping()

    if not chosen:
        picked = _pick_market(df, split_ratio)
        if "error" in picked:
            return picked
        chosen = picked["pl_column"]
        side = picked["side"]
        odds_col = picked["odds_col"]
        pick_meta = picked
    else:
        side, odds_col = mp.get(chosen, ("back", None))
        if not odds_col:
            return {"error": f"No mapping for {chosen}. Provide params.odds_col and params.side."}
        pick_meta = {"pl_column": chosen, "side": side, "odds_col": odds_col}

    out = _rule_search(df, chosen, side, odds_col, split_ratio)
    return {"picked": pick_meta, "search": out}


def _process_one(sb) -> Dict[str, Any]:
    job_rows = (
        sb.table("jobs")
        .select("*")
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
        .data
    )
    if not job_rows:
        return {"status": "idle", "message": "No queued jobs."}

    job = job_rows[0]
    job_id = job["job_id"]

    sb.table("jobs").update({"status": "running", "updated_at": _now_iso()}).eq("job_id", job_id).execute()

    try:
        task_type = job.get("task_type")
        params = job.get("params") or {}

        if task_type == "ping":
            result = {
                "job_id": job_id,
                "task_type": task_type,
                "params": params,
                "computed_at": _now_iso(),
                "message": "Compute placeholder OK. Replace with real strategy search.",
            }
        elif task_type == "strategy_search":
            result = {
                "job_id": job_id,
                "task_type": task_type,
                "params": params,
                "computed_at": _now_iso(),
                "result": _compute_task_strategy_search(sb, params),
            }
        else:
            result = {
                "job_id": job_id,
                "task_type": task_type,
                "params": params,
                "computed_at": _now_iso(),
                "error": f"Unknown task_type: {task_type}",
            }

        bucket = params.get("_results_bucket") or "football-results"
        path = f"results/{job_id}.json"
        payload = json.dumps(result, indent=2).encode("utf-8")

        sb.storage.from_(bucket).upload(
            path=path,
            file=payload,
            file_options={"content-type": "application/json", "upsert": "true"},
        )

        sb.table("jobs").update(
            {"status": "done", "result_path": path, "updated_at": _now_iso(), "error": None}
        ).eq("job_id", job_id).execute()

        return {"status": "done", "job_id": job_id, "result_path": path}

    except Exception as e:
        sb.table("jobs").update({"status": "error", "error": str(e), "updated_at": _now_iso()}).eq("job_id", job_id).execute()
        return {"status": "error", "job_id": job_id, "error": str(e)}


# -----------------------------
# Modal functions
# -----------------------------
@app.function(image=image, secrets=[SUPABASE_SECRET], timeout=60 * 60)
def run_batch(max_jobs: int = 10) -> Dict[str, Any]:
    import os
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    sb = create_client(url, key)

    max_jobs = max(1, min(int(max_jobs or 10), 50))

    results = []
    for _ in range(max_jobs):
        out = _process_one(sb)
        results.append(out)
        if out.get("status") == "idle":
            break

    return {"status": "ok", "processed": len(results), "results": results, "ts": _now_iso()}


@app.function(
    image=image,
    secrets=[SUPABASE_SECRET],
    timeout=60 * 60,
    schedule=modal.Period(minutes=1),
)
def poll_and_run():
    # IMPORTANT: this is a Modal function. Call run_batch via .remote()
    return run_batch.remote(max_jobs=10)


@app.local_entrypoint()
def main():
    print(run_batch.remote(max_jobs=5))

