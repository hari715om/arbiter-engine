import time
import uuid
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session

from arbiter.api.config import settings
from arbiter.api.models_db import init_db, get_db, TaskRecord, WorkerRecord, AssignmentRecord, EventLog
from arbiter.api.schemas import (
    TaskCreate, TaskResponse, WorkerCreate, WorkerResponse,
    MetricsSnapshot, HealthResponse, EventResponse,
)

START_TIME = time.time()
_ws_clients: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Arbiter Engine",
    description="Intelligent task scheduling API",
    version="0.7.0",
    lifespan=lifespan,
)


# -- helpers --

def _task_to_response(rec: TaskRecord) -> TaskResponse:
    deps = [d.strip() for d in rec.dependencies.split(",") if d.strip()] if rec.dependencies else []
    return TaskResponse(
        id=rec.id,
        compute_cost=rec.compute_cost,
        resource_type=rec.resource_type,
        deadline=rec.deadline,
        priority=rec.priority,
        status=rec.status,
        assigned_worker=rec.assigned_worker,
        estimated_duration=rec.estimated_duration,
        retry_count=rec.retry_count,
        dependencies=deps,
        start_time=rec.start_time,
        completion_time=rec.completion_time,
        created_at=rec.created_at,
    )


def _worker_to_response(rec: WorkerRecord) -> WorkerResponse:
    resources = [r.strip() for r in rec.supported_resources.split(",") if r.strip()]
    return WorkerResponse(
        id=rec.id,
        cpu_capacity=rec.cpu_capacity,
        memory_capacity=rec.memory_capacity,
        speed_multiplier=rec.speed_multiplier,
        status=rec.status,
        current_load=rec.current_load,
        supported_resources=resources,
        last_heartbeat=rec.last_heartbeat,
    )


def _log_event(db: Session, event_type: str, task_id: str = None, worker_id: str = None, detail: str = None):
    entry = EventLog(
        event_type=event_type,
        task_id=task_id,
        worker_id=worker_id,
        timestamp=time.time(),
        detail=detail,
    )
    db.add(entry)
    db.commit()


async def _broadcast(event: dict):
    dead = []
    msg = json.dumps(event)
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


# -- task endpoints --

@app.post("/tasks", response_model=TaskResponse, status_code=201)
def create_task(body: TaskCreate, db: Session = Depends(get_db)):
    task_id = body.id or f"task-{uuid.uuid4().hex[:8]}"
    if db.query(TaskRecord).filter_by(id=task_id).first():
        raise HTTPException(400, f"Task {task_id} already exists")

    rec = TaskRecord(
        id=task_id,
        compute_cost=body.compute_cost,
        resource_type=body.resource_type,
        deadline=body.deadline,
        priority=body.priority,
        failure_probability=body.failure_probability,
        estimated_duration=body.estimated_duration,
        max_retries=body.max_retries,
        dependencies=",".join(body.dependencies),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    _log_event(db, "TASK_CREATED", task_id=task_id)

    return _task_to_response(rec)


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, db: Session = Depends(get_db)):
    rec = db.query(TaskRecord).filter_by(id=task_id).first()
    if not rec:
        raise HTTPException(404, f"Task {task_id} not found")
    return _task_to_response(rec)


@app.get("/tasks", response_model=list[TaskResponse])
def list_tasks(
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    q = db.query(TaskRecord)
    if status:
        q = q.filter(TaskRecord.status == status)
    q = q.order_by(TaskRecord.created_at.desc()).limit(limit)
    return [_task_to_response(r) for r in q.all()]


@app.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, db: Session = Depends(get_db)):
    rec = db.query(TaskRecord).filter_by(id=task_id).first()
    if not rec:
        raise HTTPException(404, f"Task {task_id} not found")
    if rec.status == "running":
        raise HTTPException(409, "Cannot delete a running task")
    db.delete(rec)
    db.commit()
    _log_event(db, "TASK_DELETED", task_id=task_id)


# -- worker endpoints --

@app.post("/workers", response_model=WorkerResponse, status_code=201)
def register_worker(body: WorkerCreate, db: Session = Depends(get_db)):
    worker_id = body.id or f"worker-{uuid.uuid4().hex[:6]}"
    if db.query(WorkerRecord).filter_by(id=worker_id).first():
        raise HTTPException(400, f"Worker {worker_id} already exists")

    rec = WorkerRecord(
        id=worker_id,
        cpu_capacity=body.cpu_capacity,
        memory_capacity=body.memory_capacity,
        speed_multiplier=body.speed_multiplier,
        supported_resources=",".join(body.supported_resources),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    _log_event(db, "WORKER_REGISTERED", worker_id=worker_id)
    return _worker_to_response(rec)


@app.get("/workers", response_model=list[WorkerResponse])
def list_workers(db: Session = Depends(get_db)):
    recs = db.query(WorkerRecord).all()
    return [_worker_to_response(r) for r in recs]


@app.post("/workers/{worker_id}/heartbeat")
def worker_heartbeat(worker_id: str, db: Session = Depends(get_db)):
    import datetime
    rec = db.query(WorkerRecord).filter_by(id=worker_id).first()
    if not rec:
        raise HTTPException(404, f"Worker {worker_id} not found")
    rec.last_heartbeat = datetime.datetime.utcnow()
    rec.status = "idle" if rec.current_load == 0 else "busy"
    db.commit()
    return {"status": "ok"}


# -- scheduling endpoint --

@app.post("/schedule", response_model=list[dict])
def trigger_schedule(db: Session = Depends(get_db)):
    from arbiter.models.task import Task as ArbiterTask, TaskStatus
    from arbiter.models.worker import Worker as ArbiterWorker, WorkerStatus
    from arbiter.schedulers.fifo import FIFOScheduler
    from arbiter.schedulers.heuristic import HeuristicScheduler
    from arbiter.schedulers.utility_scheduler import UtilityScheduler

    schedulers = {
        "fifo": FIFOScheduler,
        "heuristic": HeuristicScheduler,
        "utility": UtilityScheduler,
    }
    sched_cls = schedulers.get(settings.scheduler_type, UtilityScheduler)
    scheduler = sched_cls()

    # load pending/queued tasks
    task_recs = db.query(TaskRecord).filter(TaskRecord.status.in_(["pending", "queued"])).all()
    worker_recs = db.query(WorkerRecord).filter(WorkerRecord.status != "down").all()
    completed_recs = db.query(TaskRecord).filter_by(status="completed").all()

    if not task_recs or not worker_recs:
        return []

    # convert to arbiter models
    tasks = []
    for r in task_recs:
        deps = [d.strip() for d in r.dependencies.split(",") if d.strip()] if r.dependencies else []
        t = ArbiterTask(
            id=r.id, compute_cost=r.compute_cost, resource_type=r.resource_type,
            deadline=r.deadline, priority=r.priority, failure_probability=r.failure_probability,
            estimated_duration=r.estimated_duration, status=TaskStatus.QUEUED,
            dependencies=deps, retry_count=r.retry_count, max_retries=r.max_retries,
        )
        tasks.append(t)

    workers = []
    for r in worker_recs:
        resources = [x.strip() for x in r.supported_resources.split(",") if x.strip()]
        w = ArbiterWorker(
            id=r.id, cpu_capacity=r.cpu_capacity, memory_capacity=r.memory_capacity,
            speed_multiplier=r.speed_multiplier, status=WorkerStatus.IDLE,
            current_load=r.current_load, supported_resources=resources,
        )
        workers.append(w)

    completed_ids = {r.id for r in completed_recs}
    assignments = scheduler.schedule(tasks, workers, completed_ids)

    results = []
    for a in assignments:
        # update DB
        task_rec = db.query(TaskRecord).filter_by(id=a.task_id).first()
        if task_rec:
            task_rec.status = "running"
            task_rec.assigned_worker = a.worker_id
            task_rec.start_time = a.scheduled_time

        worker_rec = db.query(WorkerRecord).filter_by(id=a.worker_id).first()
        if worker_rec:
            # find the task's compute cost
            t = next((t for t in tasks if t.id == a.task_id), None)
            if t:
                worker_rec.current_load += t.compute_cost
                worker_rec.status = "busy"

        db.add(AssignmentRecord(
            task_id=a.task_id,
            worker_id=a.worker_id,
            scheduled_time=a.scheduled_time,
        ))
        _log_event(db, "TASK_ASSIGNED", task_id=a.task_id, worker_id=a.worker_id)
        results.append({"task_id": a.task_id, "worker_id": a.worker_id})

    db.commit()
    return results


# -- metrics --

@app.get("/metrics", response_model=MetricsSnapshot)
def get_metrics(db: Session = Depends(get_db)):
    total = db.query(TaskRecord).count()
    completed = db.query(TaskRecord).filter_by(status="completed").count()
    failed = db.query(TaskRecord).filter_by(status="failed").count()
    pending = db.query(TaskRecord).filter(TaskRecord.status.in_(["pending", "queued"])).count()
    running = db.query(TaskRecord).filter_by(status="running").count()

    done = db.query(TaskRecord).filter_by(status="completed").all()
    latencies = []
    sla_violations = 0
    for t in done:
        if t.completion_time and t.start_time:
            latencies.append(t.completion_time - t.arrival_time)
        if t.completion_time and t.completion_time > t.deadline:
            sla_violations += 1

    avg_lat = sum(latencies) / len(latencies) if latencies else None
    sla_rate = sla_violations / len(done) if done else None

    worker_count = db.query(WorkerRecord).count()
    active = db.query(WorkerRecord).filter(WorkerRecord.status != "down").count()

    return MetricsSnapshot(
        total_tasks=total, completed=completed, failed=failed,
        pending=pending, running=running, avg_latency=avg_lat,
        sla_violation_rate=sla_rate, worker_count=worker_count,
        active_workers=active,
    )


# -- health --

@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    db_ok = True
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    redis_ok = True
    try:
        import redis as redis_lib
        r = redis_lib.from_url(settings.redis_url)
        r.ping()
    except Exception:
        redis_ok = False

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        db_connected=db_ok,
        redis_connected=redis_ok,
        scheduler_type=settings.scheduler_type,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# -- events --

@app.get("/events", response_model=list[EventResponse])
def list_events(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    recs = db.query(EventLog).order_by(EventLog.created_at.desc()).limit(limit).all()
    return [
        EventResponse(
            event_type=r.event_type, task_id=r.task_id,
            worker_id=r.worker_id, timestamp=r.timestamp, detail=r.detail,
        )
        for r in recs
    ]


# -- websocket --

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
