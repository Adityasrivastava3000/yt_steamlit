"""
Job lifecycle manager backed by Redis.

Manages job metadata creation, status updates, progress tracking, task
dispatching via Celery, and job cancellation/cleanup. All job state is
persisted in Redis hashes under keys `job:{job_id}`.
"""

import os
import json
import uuid
import shutil
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

import redis

from services.queue_manager import celery_app, REDIS_URL

logger = logging.getLogger("services.job_manager")

# Redis client for job metadata storage (separate from Celery's internal usage)
_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Lazy-initialize and return the Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# Key helpers
def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _jobs_index_key() -> str:
    return "jobs:index"


class JobManager:
    """
    High-level manager for job lifecycle operations.

    All job metadata is stored as Redis hashes. A Redis set `jobs:index`
    tracks all known job IDs for listing purposes.
    """

    def __init__(self, output_base_dir: str = "outputs"):
        self.output_base_dir = os.path.abspath(output_base_dir)
        self.r = get_redis()

    # ------------------------------------------------------------------
    # Job Creation
    # ------------------------------------------------------------------

    def create_job(self, video_path_or_url: str, params: dict) -> dict:
        """
        Create a new job record in Redis and dispatch it to Celery.

        Returns the full job detail dictionary.
        """
        job_id = str(uuid.uuid4())
        output_dir = os.path.join(self.output_base_dir, job_id)
        os.makedirs(output_dir, exist_ok=True)

        now = datetime.now().isoformat()

        job_data = {
            "job_id": job_id,
            "status": "pending",
            "video_path_or_url": video_path_or_url,
            "output_dir": output_dir,
            "created_at": now,
            "started_at": "",
            "completed_at": "",
            "error_message": "",
            "celery_task_id": "",
            # Metrics (stored as flat strings in Redis hash)
            "slides_found": "0",
            "speed": "0.0",
            "eta": "--:--:--",
            "progress": "0.0",
            "current_step": "0",
            "current_step_name": "pending",
            # Params snapshot (serialized JSON)
            "params": json.dumps(params),
        }

        # Persist to Redis
        key = _job_key(job_id)
        self.r.hset(key, mapping=job_data)
        self.r.sadd(_jobs_index_key(), job_id)

        # Dispatch Celery task
        from workers.tasks import run_extraction_task

        result = run_extraction_task.apply_async(
            args=[job_id, video_path_or_url, params],
            task_id=job_id,  # Use job_id as the Celery task_id for easy correlation
        )

        # Store the Celery task ID
        self.r.hset(key, "celery_task_id", result.id)

        logger.info(f"Job {job_id} created and dispatched to Celery (task_id={result.id})")
        return self._build_detail(job_data)

    # ------------------------------------------------------------------
    # Job Retrieval
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[dict]:
        """Retrieve full job detail from Redis."""
        key = _job_key(job_id)
        data = self.r.hgetall(key)
        if not data:
            return None
        return self._build_detail(data)

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Retrieve only the status metrics for a job."""
        key = _job_key(job_id)
        data = self.r.hgetall(key)
        if not data:
            return None
        return {
            "job_id": job_id,
            "status": data.get("status", "pending"),
            "slides_found": int(data.get("slides_found", 0)),
            "speed": float(data.get("speed", 0.0)),
            "eta": data.get("eta", "--:--:--"),
            "progress": float(data.get("progress", 0.0)),
            "current_step": int(data.get("current_step", 0)),
            "current_step_name": data.get("current_step_name", "pending"),
            "error_message": data.get("error_message"),
        }

    def list_jobs(self) -> List[dict]:
        """List all known jobs with their current details."""
        job_ids = self.r.smembers(_jobs_index_key())
        jobs = []
        for jid in job_ids:
            detail = self.get_job(jid)
            if detail:
                jobs.append(detail)
        return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)

    # ------------------------------------------------------------------
    # Progress Updates (called from within Celery tasks)
    # ------------------------------------------------------------------

    @staticmethod
    def update_progress(
        job_id: str,
        status: str = None,
        current_step: int = None,
        current_step_name: str = None,
        slides_found: int = None,
        speed: float = None,
        eta: str = None,
        progress: float = None,
        started_at: str = None,
        completed_at: str = None,
        error_message: str = None,
        ocr_duration_seconds: float = None,
        ppt_duration_seconds: float = None,
        pdf_duration_seconds: float = None,
        total_duration_seconds: float = None,
    ):
        """
        Update specific fields of a job record in Redis.

        This is a static method so Celery tasks can call it without
        instantiating the full JobManager.
        """
        r = get_redis()
        key = _job_key(job_id)

        updates = {}
        if status is not None:
            updates["status"] = status
        if current_step is not None:
            updates["current_step"] = str(current_step)
        if current_step_name is not None:
            updates["current_step_name"] = current_step_name
        if slides_found is not None:
            updates["slides_found"] = str(slides_found)
        if speed is not None:
            updates["speed"] = str(round(speed, 2))
        if eta is not None:
            updates["eta"] = eta
        if progress is not None:
            updates["progress"] = str(round(progress, 2))
        if started_at is not None:
            updates["started_at"] = started_at
        if completed_at is not None:
            updates["completed_at"] = completed_at
        if error_message is not None:
            updates["error_message"] = error_message
        if ocr_duration_seconds is not None:
            updates["ocr_duration_seconds"] = str(round(ocr_duration_seconds, 2))
        if ppt_duration_seconds is not None:
            updates["ppt_duration_seconds"] = str(round(ppt_duration_seconds, 2))
        if pdf_duration_seconds is not None:
            updates["pdf_duration_seconds"] = str(round(pdf_duration_seconds, 2))
        if total_duration_seconds is not None:
            updates["total_duration_seconds"] = str(round(total_duration_seconds, 2))

        if updates:
            r.hset(key, mapping=updates)

    # ------------------------------------------------------------------
    # Job Cancellation & Deletion
    # ------------------------------------------------------------------

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.

        Revokes the Celery task (with SIGTERM) and marks the job as cancelled.
        """
        key = _job_key(job_id)
        data = self.r.hgetall(key)
        if not data:
            return False

        celery_task_id = data.get("celery_task_id", job_id)
        status = data.get("status", "")

        if status in ("running", "pending"):
            # Revoke the task — terminate the worker process if it's running
            celery_app.control.revoke(celery_task_id, terminate=True, signal="SIGTERM")
            logger.info(f"Revoked Celery task {celery_task_id} for job {job_id}")

            self.r.hset(key, mapping={
                "status": "cancelled",
                "completed_at": datetime.now().isoformat(),
                "current_step_name": "Cancelled by user",
            })

        return True

    def delete_job(self, job_id: str) -> bool:
        """
        Cancel (if running), remove all output files from local & R2, and purge the job from Redis.
        """
        key = _job_key(job_id)
        data = self.r.hgetall(key)
        if not data:
            return False

        # Cancel first if still active
        self.cancel_job(job_id)

        # Remove local output directory (if exists)
        output_dir = data.get("output_dir", "")
        if output_dir and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                logger.info(f"Deleted local output directory: {output_dir}")
            except Exception as e:
                logger.error(f"Failed to delete local output directory {output_dir}: {e}")

        # Remove output directory from Cloudflare R2
        try:
            from services.storage_service import StorageService
            storage = StorageService()
            storage.delete_folder(f"jobs/{job_id}")
            logger.info(f"Deleted R2 output folder: jobs/{job_id}")
        except Exception as e:
            logger.error(f"Failed to delete R2 folder jobs/{job_id}: {e}")

        # Remove from Redis
        self.r.delete(key)
        self.r.srem(_jobs_index_key(), job_id)
        logger.info(f"Job {job_id} fully deleted from Redis")

        return True

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_detail(self, data: dict) -> dict:
        """Convert flat Redis hash data into structured job detail dict."""
        return {
            "job_id": data.get("job_id", ""),
            "status": data.get("status", "unknown"),
            "video_path_or_url": data.get("video_path_or_url", ""),
            "output_dir": data.get("output_dir", ""),
            "created_at": data.get("created_at", ""),
            "started_at": data.get("started_at", "") or None,
            "completed_at": data.get("completed_at", "") or None,
            "status_metrics": {
                "slides_found": int(data.get("slides_found", 0)),
                "speed": float(data.get("speed", 0.0)),
                "eta": data.get("eta", "--:--:--"),
                "progress": float(data.get("progress", 0.0)),
                "current_step": int(data.get("current_step", 0)),
                "current_step_name": data.get("current_step_name", "pending"),
            },
            "error_message": data.get("error_message", "") or None,
        }
