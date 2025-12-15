# worker/modal_app.py
import json
import time
from datetime import datetime
from io import BytesIO

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


def _now_iso():
    return datetime.utcnow().isoformat()


@app.function(image=image, secrets=[SUPABASE_SECRET], timeout=60 * 60)  # 1 hour
def run_one_job():
    import os

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    sb = create_client(url, key)

    # 1) Fetch one queued job
    job = (
        sb.table("jobs")
        .select("*")
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
        .data
    )
    if not job:
        return {"status": "idle", "message": "No queued jobs."}

    job = job[0]
    job_id = job["job_id"]

    # 2) Mark running
    sb.table("jobs").update({"status": "running", "updated_at": _now_iso()}).eq("job_id", job_id).execute()

    try:
        task_type = job["task_type"]
        params = job.get("params") or {}

        # ---- PLACEHOLDER HEAVY COMPUTE ----
        # Replace this with your actual search / evaluation.
        # For now we just return the params as a sanity test.
        time.sleep(2)
        result = {
            "job_id": job_id,
            "task_type": task_type,
            "params": params,
            "computed_at": _now_iso(),
            "message": "Compute placeholder OK. Replace with real strategy search.",
        }

        # 3) Upload results JSON to Supabase Storage
        bucket = "football-results"
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

        return {"status": "done", "job_id": job_id, "result_path": path}

    except Exception as e:
        sb.table("jobs").update(
            {"status": "error", "error": str(e), "updated_at": _now_iso()}
        ).eq("job_id", job_id).execute()
        return {"status": "error", "job_id": job_id, "error": str(e)}


@app.function(image=image, secrets=[SUPABASE_SECRET], timeout=60 * 60, schedule=modal.Period(minutes=1))
def poll_and_run():
    # Runs every minute and processes one job per tick.
    return run_one_job.remote()


@app.local_entrypoint()
def main():
    # Run one job locally-triggered (handy for testing)
    print(run_one_job.remote())

