# worker/modal_app.py
import json
import time
from datetime import datetime

import modal
from supabase import create_client


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


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _process_one(sb) -> dict:
    """
    Claim + process one queued job.
    Returns a status dict:
      - {"status":"idle"} if none found
      - {"status":"done", ...} / {"status":"error", ...}
    """
    # 1) Fetch one queued job
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

    # 2) Mark running (best-effort "claim")
    sb.table("jobs").update(
        {"status": "running", "updated_at": _now_iso()}
    ).eq("job_id", job_id).execute()

    try:
        task_type = job.get("task_type")
        params = job.get("params") or {}

        # ---- PLACEHOLDER HEAVY COMPUTE ----
        # Replace this block with your real strategy search / evaluation.
        time.sleep(1.0)

        result = {
            "job_id": job_id,
            "task_type": task_type,
            "params": params,
            "computed_at": _now_iso(),
            "message": "Compute placeholder OK. Replace with real strategy search.",
        }

        # 3) Upload results JSON to Supabase Storage
        bucket = params.get("_results_bucket") or "football-results"
        path = f"results/{job_id}.json"
        payload = json.dumps(result, indent=2).encode("utf-8")

        sb.storage.from_(bucket).upload(
            path=path,
            file=payload,
            file_options={"content-type": "application/json", "upsert": "true"},
        )

        # 4) Mark done
        sb.table("jobs").update(
            {"status": "done", "result_path": path, "updated_at": _now_iso(), "error": None}
        ).eq("job_id", job_id).execute()

        return {"status": "done", "job_id": job_id, "result_path": path, "bucket": bucket}

    except Exception as e:
        sb.table("jobs").update(
            {"status": "error", "error": str(e), "updated_at": _now_iso()}
        ).eq("job_id", job_id).execute()
        return {"status": "error", "job_id": job_id, "error": str(e)}


@app.function(image=image, secrets=[SUPABASE_SECRET], timeout=60 * 60)  # 1 hour
def run_batch(max_jobs: int = 10) -> dict:
    """
    Process up to max_jobs in one container invocation (prevents backlog buildup).
    """
    import os

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    sb = create_client(url, key)

    max_jobs = max(1, min(int(max_jobs or 10), 100))

    results = []
    for _ in range(max_jobs):
        out = _process_one(sb)
        results.append(out)
        if out.get("status") == "idle":
            break

    done = sum(1 for r in results if r.get("status") == "done")
    err = sum(1 for r in results if r.get("status") == "error")
    return {
        "status": "ok",
        "processed": len(results),
        "done": done,
        "errors": err,
        "results": results,
        "ts": _now_iso(),
    }


@app.function(
    image=image,
    secrets=[SUPABASE_SECRET],
    timeout=60 * 60,
    schedule=modal.Period(minutes=1),
)
def poll_and_run():
    """
    Runs every minute on Modal automatically.
    Processes a small batch each tick to keep up with load.
    """
    return run_batch(max_jobs=10)


@app.local_entrypoint()
def main():
    # Handy: run a batch once from your Mac
    print(run_batch.remote(max_jobs=5))
