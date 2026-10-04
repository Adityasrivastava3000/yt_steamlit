"""
Celery application factory and queue monitoring utilities.

This module initializes the Celery app with Redis as both the message broker
and result backend. It also exposes helpers for inspecting active, scheduled,
and reserved tasks in the Celery cluster.
"""

import os
import logging
from celery import Celery

logger = logging.getLogger("services.queue_manager")

# Redis connection configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery application
celery_app = Celery(
    "lecture_extractor",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["workers.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Result expiry (keep results for 24 hours)
    result_expires=86400,

    # Worker settings
    worker_prefetch_multiplier=1,       # One task at a time per worker (heavy CV tasks)
    worker_max_tasks_per_child=5,       # Restart worker after 5 tasks to free leaked memory
    worker_hijack_root_logger=False,    # Don't override our logging config

    # Retry settings for broker connection
    broker_connection_retry_on_startup=True,

    # Task tracking
    task_track_started=True,            # Report STARTED state transitions
    task_acks_late=True,                # Acknowledge after completion (crash safety)
    task_reject_on_worker_lost=True,    # Re-queue if worker dies mid-task
)


class QueueMonitor:
    """Utility class for inspecting the state of the Celery task queue."""

    def __init__(self, app: Celery = None):
        self.app = app or celery_app

    def get_active_tasks(self) -> dict:
        """Returns currently executing tasks across all workers."""
        inspector = self.app.control.inspect()
        active = inspector.active()
        return active if active else {}

    def get_scheduled_tasks(self) -> dict:
        """Returns tasks scheduled for future execution (ETA/countdown)."""
        inspector = self.app.control.inspect()
        scheduled = inspector.scheduled()
        return scheduled if scheduled else {}

    def get_reserved_tasks(self) -> dict:
        """Returns tasks that have been claimed by a worker but not yet started."""
        inspector = self.app.control.inspect()
        reserved = inspector.reserved()
        return reserved if reserved else {}

    def get_registered_tasks(self) -> dict:
        """Returns all registered task names across workers."""
        inspector = self.app.control.inspect()
        registered = inspector.registered()
        return registered if registered else {}

    def get_worker_stats(self) -> dict:
        """Returns worker statistics (uptime, pool info, etc.)."""
        inspector = self.app.control.inspect()
        stats = inspector.stats()
        return stats if stats else {}

    def get_queue_summary(self) -> dict:
        """Returns a consolidated summary of queue health."""
        try:
            active = self.get_active_tasks()
            reserved = self.get_reserved_tasks()
            stats = self.get_worker_stats()

            total_active = sum(len(tasks) for tasks in active.values()) if active else 0
            total_reserved = sum(len(tasks) for tasks in reserved.values()) if reserved else 0
            worker_count = len(stats) if stats else 0

            return {
                "workers_online": worker_count,
                "active_tasks": total_active,
                "reserved_tasks": total_reserved,
                "worker_names": list(stats.keys()) if stats else [],
            }
        except Exception as e:
            logger.error(f"Failed to get queue summary: {e}")
            return {
                "workers_online": 0,
                "active_tasks": 0,
                "reserved_tasks": 0,
                "worker_names": [],
                "error": str(e),
            }
