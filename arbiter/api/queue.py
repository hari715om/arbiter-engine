import time
from celery import Celery
from arbiter.api.config import settings
from arbiter.logging_config import get_logger

log = get_logger(__name__)

celery_app = Celery(
    "arbiter",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)


def _fire_webhook(webhook_url: str, payload: dict) -> None:
    """POST task completion payload to the user-specified webhook URL."""
    try:
        import httpx
        resp = httpx.post(webhook_url, json=payload, timeout=5.0)
        log.info("webhook_fired", url=webhook_url, status=resp.status_code)
    except Exception as e:
        log.warning("webhook_failed", url=webhook_url, error=str(e))


@celery_app.task(name="arbiter.schedule_pending")
def schedule_pending():
    """Pull queued tasks from DB, run scheduler, persist assignments."""
    from arbiter.api.models_db import SessionLocal, TaskRecord, WorkerRecord, AssignmentRecord
    from arbiter.api.app import _log_event
    from arbiter.models.task import Task as ArbiterTask, TaskStatus
    from arbiter.models.worker import Worker as ArbiterWorker, WorkerStatus
    from arbiter.schedulers.utility_scheduler import UtilityScheduler

    db = SessionLocal()
    try:
        task_recs = db.query(TaskRecord).filter(
            TaskRecord.status.in_(["pending", "queued"])
        ).all()
        worker_recs = db.query(WorkerRecord).filter(
            WorkerRecord.status != "down"
        ).all()
        completed_ids = {
            r.id for r in db.query(TaskRecord).filter_by(status="completed").all()
        }

        if not task_recs or not worker_recs:
            return {"assignments": 0}

        tasks = []
        for r in task_recs:
            deps = [d.strip() for d in r.dependencies.split(",") if d.strip()] if r.dependencies else []
            tasks.append(ArbiterTask(
                id=r.id, compute_cost=r.compute_cost, resource_type=r.resource_type,
                deadline=r.deadline, priority=r.priority,
                failure_probability=r.failure_probability,
                estimated_duration=r.estimated_duration, status=TaskStatus.QUEUED,
                dependencies=deps, retry_count=r.retry_count, max_retries=r.max_retries,
            ))

        workers = []
        for r in worker_recs:
            resources = [x.strip() for x in r.supported_resources.split(",") if x.strip()]
            workers.append(ArbiterWorker(
                id=r.id, cpu_capacity=r.cpu_capacity, memory_capacity=r.memory_capacity,
                speed_multiplier=r.speed_multiplier, status=WorkerStatus.IDLE,
                current_load=r.current_load, supported_resources=resources,
            ))

        scheduler = UtilityScheduler()
        assignments = scheduler.schedule(tasks, workers, completed_ids)

        log.info("celery_schedule_pending", tasks=len(task_recs),
                 workers=len(worker_recs), assignments=len(assignments))

        for a in assignments:
            tr = db.query(TaskRecord).filter_by(id=a.task_id).first()
            if tr:
                tr.status = "running"
                tr.assigned_worker = a.worker_id
                tr.start_time = a.scheduled_time

            wr = db.query(WorkerRecord).filter_by(id=a.worker_id).first()
            if wr:
                t = next((t for t in tasks if t.id == a.task_id), None)
                if t:
                    wr.current_load += t.compute_cost
                    wr.status = "busy"

            db.add(AssignmentRecord(
                task_id=a.task_id, worker_id=a.worker_id,
                scheduled_time=a.scheduled_time,
            ))
            _log_event(db, "TASK_ASSIGNED", task_id=a.task_id, worker_id=a.worker_id)

        db.commit()
        return {"assignments": len(assignments)}
    finally:
        db.close()


@celery_app.task(name="arbiter.mark_completed")
def mark_task_completed(task_id: str):
    """Mark a task as completed, free the worker, and fire webhook if set."""
    from arbiter.api.models_db import SessionLocal, TaskRecord, WorkerRecord
    from arbiter.api.app import _log_event

    db = SessionLocal()
    try:
        rec = db.query(TaskRecord).filter_by(id=task_id).first()
        if not rec or rec.status != "running":
            log.warning("mark_completed_skip", task_id=task_id,
                        reason="not found or not running")
            return {"error": f"Task {task_id} not running"}

        rec.status = "completed"
        rec.completion_time = time.time()
        latency = rec.completion_time - (rec.arrival_time or rec.completion_time)

        if rec.assigned_worker:
            wr = db.query(WorkerRecord).filter_by(id=rec.assigned_worker).first()
            if wr:
                wr.current_load = max(0, wr.current_load - rec.compute_cost)
                if wr.current_load == 0:
                    wr.status = "idle"

        _log_event(db, "TASK_COMPLETED", task_id=task_id, worker_id=rec.assigned_worker)
        db.commit()

        log.info("task_completed", task_id=task_id,
                 worker_id=rec.assigned_worker, latency_s=round(latency, 2))

        # Fire webhook if configured
        if rec.webhook_url:
            _fire_webhook(rec.webhook_url, {
                "event": "task.completed",
                "task_id": task_id,
                "worker_id": rec.assigned_worker,
                "completion_time": rec.completion_time,
                "latency_seconds": round(latency, 2),
            })

        return {"status": "completed", "task_id": task_id}
    finally:
        db.close()


# periodic beat schedule
celery_app.conf.beat_schedule = {
    "schedule-pending-tasks": {
        "task": "arbiter.schedule_pending",
        "schedule": settings.schedule_interval,
    },
}
