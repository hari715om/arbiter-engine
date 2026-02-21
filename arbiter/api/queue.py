from celery import Celery
from arbiter.api.config import settings

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
    """Mark a task as completed and free the worker."""
    from arbiter.api.models_db import SessionLocal, TaskRecord, WorkerRecord
    from arbiter.api.app import _log_event
    import time

    db = SessionLocal()
    try:
        rec = db.query(TaskRecord).filter_by(id=task_id).first()
        if not rec or rec.status != "running":
            return {"error": f"Task {task_id} not running"}

        rec.status = "completed"
        rec.completion_time = time.time()

        if rec.assigned_worker:
            wr = db.query(WorkerRecord).filter_by(id=rec.assigned_worker).first()
            if wr:
                wr.current_load = max(0, wr.current_load - rec.compute_cost)
                if wr.current_load == 0:
                    wr.status = "idle"

        _log_event(db, "TASK_COMPLETED", task_id=task_id, worker_id=rec.assigned_worker)
        db.commit()
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
