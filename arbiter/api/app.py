import time
import uuid
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session

from arbiter.api.config import settings
from arbiter.api.models_db import init_db, get_db, TaskRecord, WorkerRecord, AssignmentRecord, EventLog
from arbiter.api.schemas import (
    TaskCreate, TaskResponse, WorkerCreate, WorkerResponse,
    MetricsSnapshot, HealthResponse, EventResponse,
    ExplanationResponse, AlternativeAssignment,
    ChaosRequest, ChaosResponse, PolicyResponse, PolicyRule,
)
from arbiter.logging_config import configure_logging, get_logger

configure_logging(level=settings.log_level if hasattr(settings, "log_level") else "INFO")
log = get_logger(__name__)

START_TIME = time.time()
_ws_clients: list[WebSocket] = []

# ── Fix 3: Rate Limiting via slowapi ──────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False

if _HAS_SLOWAPI:
    limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
else:
    limiter = None

# ── Fix 2: Async Heartbeat Monitor ───────────────────────────────────────────
from arbiter.execution.heartbeat import AsyncHeartbeatMonitor
_heartbeat_monitor: Optional[AsyncHeartbeatMonitor] = None


async def _handle_worker_failure(worker_id: str):
    """Called by AsyncHeartbeatMonitor when a worker misses its heartbeat."""
    from arbiter.api.models_db import SessionLocal
    db = SessionLocal()
    try:
        wr = db.query(WorkerRecord).filter_by(id=worker_id).first()
        if wr and wr.status != "down":
            wr.status = "down"
            running = db.query(TaskRecord).filter_by(
                assigned_worker=worker_id, status="running"
            ).all()
            for tr in running:
                tr.status = "queued"
                tr.assigned_worker = None
            _log_event(db, "WORKER_FAILED", worker_id=worker_id,
                       detail=f"heartbeat timeout, {len(running)} tasks preempted")
            db.commit()
            log.warning("worker_failed_heartbeat", worker_id=worker_id,
                        tasks_preempted=len(running))
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _heartbeat_monitor
    log.info("arbiter_startup", scheduler=settings.scheduler_type,
             db_url=settings.database_url.split("@")[-1])  # mask credentials
    init_db()

    # Start async heartbeat monitor
    _heartbeat_monitor = AsyncHeartbeatMonitor(
        timeout=30.0, on_failure=_handle_worker_failure
    )
    await _heartbeat_monitor.start(interval=10.0)

    yield

    # Graceful shutdown
    if _heartbeat_monitor:
        await _heartbeat_monitor.stop()
    log.info("arbiter_shutdown")


app = FastAPI(
    title="Arbiter Engine",
    description="Intelligent task scheduling API — multi-objective, observable, production-ready.",
    version="0.8.0",
    lifespan=lifespan,
)

# Wire rate limiter if available
if _HAS_SLOWAPI:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: allow the React dashboard (port 5173) and anything in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request-ID middleware ──────────────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
        request_id=req_id,
    )
    response.headers["X-Request-ID"] = req_id
    return response


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_tenant(request: Request) -> str:
    """Extract tenant ID from X-Tenant-ID header (default='default')."""
    return request.headers.get("X-Tenant-ID", "default")


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
        tenant_id=rec.tenant_id,
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
        tenant_id=rec.tenant_id,
    )


def _log_event(db: Session, event_type: str, task_id: str = None,
               worker_id: str = None, detail: str = None):
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


# ── task endpoints ─────────────────────────────────────────────────────────────

@app.post("/tasks", response_model=TaskResponse, status_code=201)
def create_task(request: Request, body: TaskCreate, db: Session = Depends(get_db)):
    tenant = _get_tenant(request)
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
        webhook_url=body.webhook_url,
        tenant_id=tenant,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    log.info("task_created", task_id=task_id, priority=body.priority,
             deadline=body.deadline, tenant=tenant, has_webhook=bool(body.webhook_url))
    _log_event(db, "TASK_CREATED", task_id=task_id)
    return _task_to_response(rec)

# Apply rate limit decorator if slowapi is available
if _HAS_SLOWAPI:
    create_task = limiter.limit("100/minute")(create_task)


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, db: Session = Depends(get_db)):
    rec = db.query(TaskRecord).filter_by(id=task_id).first()
    if not rec:
        raise HTTPException(404, f"Task {task_id} not found")
    return _task_to_response(rec)


@app.get("/tasks", response_model=list[TaskResponse])
def list_tasks(
    request: Request,
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    tenant = _get_tenant(request)
    q = db.query(TaskRecord).filter_by(tenant_id=tenant)
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
    log.info("task_deleted", task_id=task_id)
    _log_event(db, "TASK_DELETED", task_id=task_id)


# ── explain endpoint ───────────────────────────────────────────────────────────

@app.get("/tasks/{task_id}/explain", response_model=ExplanationResponse)
def explain_task_assignment(task_id: str, db: Session = Depends(get_db)):
    """
    Explain why a task was assigned to its worker (or rank top alternatives).

    Uses the UtilityScheduler's objective decomposition to return per-objective
    scores and a human-readable reasoning string. If the task is unassigned,
    returns the top-3 hypothetical assignments.
    """
    from arbiter.models.task import Task as ArbiterTask, TaskStatus
    from arbiter.models.worker import Worker as ArbiterWorker, WorkerStatus
    from arbiter.schedulers.utility_scheduler import UtilityScheduler

    rec = db.query(TaskRecord).filter_by(id=task_id).first()
    if not rec:
        raise HTTPException(404, f"Task {task_id} not found")

    worker_recs = db.query(WorkerRecord).filter(WorkerRecord.status != "down").all()
    if not worker_recs:
        raise HTTPException(422, "No workers available to compute explanation")

    # Rebuild domain model for the task
    deps = [d.strip() for d in rec.dependencies.split(",") if d.strip()] if rec.dependencies else []
    task = ArbiterTask(
        id=rec.id, compute_cost=rec.compute_cost, resource_type=rec.resource_type,
        deadline=rec.deadline, priority=rec.priority,
        failure_probability=rec.failure_probability,
        estimated_duration=rec.estimated_duration,
        status=TaskStatus.QUEUED,
        dependencies=deps, retry_count=rec.retry_count, max_retries=rec.max_retries,
    )

    workers = []
    for wr in worker_recs:
        resources = [x.strip() for x in wr.supported_resources.split(",") if x.strip()]
        workers.append(ArbiterWorker(
            id=wr.id, cpu_capacity=wr.cpu_capacity, memory_capacity=wr.memory_capacity,
            speed_multiplier=wr.speed_multiplier, status=WorkerStatus.IDLE,
            current_load=wr.current_load, supported_resources=resources,
        ))

    # Score all workers via UtilityScheduler
    scheduler = UtilityScheduler()
    scored: list[tuple[ArbiterWorker, float, dict[str, float]]] = []

    # Build context for the utility function
    context = {
        "current_time": 0.0,
        "workers": workers,
        "round_capacity": {w.id: w.cpu_capacity - w.current_load for w in workers},
        "worker_reliability": {},
        "queue_depth": 1,
    }

    for worker in workers:
        # Check capacity
        if worker.current_load + task.compute_cost > worker.cpu_capacity:
            continue
        if task.resource_type not in worker.supported_resources:
            continue

        # Compute per-objective breakdown using the proper API
        breakdown = scheduler.utility_fn.evaluate_breakdown(task, worker, context)
        total = breakdown.pop("total", 0.0)
        scored.append((worker, round(total, 4), {k: round(v, 4) for k, v in breakdown.items()}))

    if not scored:
        raise HTTPException(422, "No eligible workers — all at capacity or resource mismatch")

    # Sort descending by total score
    scored.sort(key=lambda x: x[1], reverse=True)

    best_worker, best_score, best_breakdown = scored[0]
    assigned_worker_id = rec.assigned_worker or best_worker.id

    # Build alternatives (workers that were not chosen)
    alternatives = [
        AlternativeAssignment(
            worker_id=w.id,
            score=s,
            breakdown=b,
        )
        for w, s, b in scored[1:4]  # top-3 alternatives
    ]

    # Human-readable reasoning
    top_factor = max(best_breakdown, key=best_breakdown.get) if best_breakdown else "N/A"
    reasoning = (
        f"Worker {best_worker.id} scored highest ({best_score:.3f}) "
        f"driven primarily by {top_factor} ({best_breakdown.get(top_factor, 0):.3f}). "
        f"Current load: {best_worker.current_load:.2f}/{best_worker.cpu_capacity:.2f} CPU."
    )

    log.info("explain_computed", task_id=task_id, worker_id=assigned_worker_id,
             top_score=best_score, top_factor=top_factor)

    return ExplanationResponse(
        task_id=task_id,
        worker_id=assigned_worker_id,
        scheduler_name="UtilityScheduler",
        total_score=best_score,
        factors=best_breakdown,
        reasoning=reasoning,
        alternatives=alternatives,
    )


# ── worker endpoints ───────────────────────────────────────────────────────────

@app.post("/workers", response_model=WorkerResponse, status_code=201)
def register_worker(request: Request, body: WorkerCreate, db: Session = Depends(get_db)):
    tenant = _get_tenant(request)
    worker_id = body.id or f"worker-{uuid.uuid4().hex[:6]}"
    if db.query(WorkerRecord).filter_by(id=worker_id).first():
        raise HTTPException(400, f"Worker {worker_id} already exists")

    rec = WorkerRecord(
        id=worker_id,
        cpu_capacity=body.cpu_capacity,
        memory_capacity=body.memory_capacity,
        speed_multiplier=body.speed_multiplier,
        supported_resources=",".join(body.supported_resources),
        tenant_id=tenant,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    log.info("worker_registered", worker_id=worker_id, cpu=body.cpu_capacity,
             memory=body.memory_capacity, tenant=tenant)
    _log_event(db, "WORKER_REGISTERED", worker_id=worker_id)
    return _worker_to_response(rec)


@app.get("/workers", response_model=list[WorkerResponse])
def list_workers(request: Request, db: Session = Depends(get_db)):
    tenant = _get_tenant(request)
    recs = db.query(WorkerRecord).filter_by(tenant_id=tenant).all()
    return [_worker_to_response(r) for r in recs]


@app.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str, db: Session = Depends(get_db)):
    import datetime
    rec = db.query(WorkerRecord).filter_by(id=worker_id).first()
    if not rec:
        raise HTTPException(404, f"Worker {worker_id} not found")
    rec.last_heartbeat = datetime.datetime.utcnow()
    rec.status = "idle" if rec.current_load == 0 else "busy"
    db.commit()

    # Record in async heartbeat monitor
    if _heartbeat_monitor:
        await _heartbeat_monitor.record(worker_id)

    return {"status": "ok"}


# ── policy endpoints ───────────────────────────────────────────────────────────

@app.get("/policy", response_model=PolicyResponse)
def get_policy():
    """Return the currently active scheduling policy."""
    from arbiter.api.policy import get_policy_engine
    eng = get_policy_engine()
    d = eng.to_dict()
    return PolicyResponse(
        default_scheduler=d["default_scheduler"],
        rules=[PolicyRule(condition=r["if"], use=r["use"]) for r in d["rules"]],
        utility_weights=d["utility_weights"],
    )


@app.put("/policy", response_model=PolicyResponse)
def update_policy(body: PolicyResponse):
    """
    Hot-reload the scheduling policy without restarting the API.

    Accepts the same schema as GET /policy. The policy takes effect
    immediately on the next Celery scheduling cycle.
    """
    from arbiter.api.policy import get_policy_engine
    eng = get_policy_engine()
    data = {
        "default_scheduler": body.default_scheduler,
        "rules": [{"if": r.condition, "use": r.use} for r in body.rules],
        "utility_weights": body.utility_weights,
    }
    try:
        eng.load_from_dict(data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    log.info("policy_updated", default=body.default_scheduler, rules=len(body.rules))
    return get_policy()


# ── chaos endpoint ─────────────────────────────────────────────────────────────

@app.post("/chaos", response_model=ChaosResponse)
def inject_chaos(body: ChaosRequest, db: Session = Depends(get_db)):
    """
    Inject controlled chaos into the running system (Chaos Monkey style).

    Modes:
    - kill_worker:      Mark worker(s) as DOWN, preempt their tasks back to QUEUED.
                        If `target` is given, kills that worker. Otherwise kills
                        `intensity` fraction of active workers.
    - delay_tasks:      Artificially force `intensity` fraction of PENDING tasks
                        back to QUEUED (they will be re-scheduled).
    - fail_rate_spike:  Mark `intensity` fraction of RUNNING tasks as FAILED,
                        freeing the workers — simulates sudden execution failures.
    """
    import random
    affected: list[str] = []

    if body.mode == "kill_worker":
        if body.target:
            # Kill specific worker
            wr = db.query(WorkerRecord).filter_by(id=body.target).first()
            if not wr:
                raise HTTPException(404, f"Worker {body.target!r} not found")
            victims = [wr]
        else:
            # Kill intensity fraction of active workers
            active = db.query(WorkerRecord).filter(WorkerRecord.status != "down").all()
            k = max(1, int(len(active) * body.intensity))
            victims = random.sample(active, min(k, len(active)))

        for wr in victims:
            wr.status = "down"
            # Preempt running tasks back to queued
            running = db.query(TaskRecord).filter_by(
                assigned_worker=wr.id, status="running"
            ).all()
            for tr in running:
                tr.status = "queued"
                tr.assigned_worker = None
            _log_event(db, "CHAOS_WORKER_KILLED", worker_id=wr.id,
                       detail=f"chaos kill_worker intensity={body.intensity}")
            affected.append(wr.id)
        db.commit()
        msg = f"Killed {len(affected)} worker(s): {', '.join(affected)}"

    elif body.mode == "delay_tasks":
        pending = db.query(TaskRecord).filter_by(status="pending").all()
        k = max(1, int(len(pending) * body.intensity))
        victims = random.sample(pending, min(k, len(pending)))
        for tr in victims:
            tr.status = "queued"   # force reschedule cycle
            affected.append(tr.id)
        db.commit()
        msg = f"Delayed {len(affected)} task(s) back to QUEUED"

    elif body.mode == "fail_rate_spike":
        running = db.query(TaskRecord).filter_by(status="running").all()
        k = max(1, int(len(running) * body.intensity))
        victims = random.sample(running, min(k, len(running)))
        for tr in victims:
            tr.status = "failed"
            # Free the worker load
            if tr.assigned_worker:
                wr = db.query(WorkerRecord).filter_by(id=tr.assigned_worker).first()
                if wr:
                    wr.current_load = max(0.0, wr.current_load - tr.compute_cost)
                    if wr.current_load == 0:
                        wr.status = "idle"
            _log_event(db, "CHAOS_TASK_FAILED", task_id=tr.id,
                       detail=f"chaos fail_rate_spike intensity={body.intensity}")
            affected.append(tr.id)
        db.commit()
        msg = f"Force-failed {len(affected)} running task(s)"

    else:
        raise HTTPException(422, f"Unknown chaos mode: {body.mode!r}. "
                                 f"Use: kill_worker, delay_tasks, fail_rate_spike")

    log.warning("chaos_injected", mode=body.mode, intensity=body.intensity,
                affected=len(affected), target=body.target)
    return ChaosResponse(mode=body.mode, affected=affected, message=msg)


# ── scheduling endpoint ────────────────────────────────────────────────────────

@app.post("/schedule", response_model=list[dict])
def trigger_schedule(request: Request, db: Session = Depends(get_db)):
    from arbiter.models.task import Task as ArbiterTask, TaskStatus
    from arbiter.models.worker import Worker as ArbiterWorker, WorkerStatus
    from arbiter.schedulers.fifo import FIFOScheduler
    from arbiter.schedulers.heuristic import HeuristicScheduler
    from arbiter.schedulers.utility_scheduler import UtilityScheduler
    from arbiter.api.policy import get_policy_engine

    # Use policy engine to select scheduler (supports hot-reload)
    queue_depth = db.query(TaskRecord).filter_by(status="queued").count()
    failed = db.query(TaskRecord).filter_by(status="failed").count()
    total_done = db.query(TaskRecord).filter(TaskRecord.status.in_(["completed", "failed"])).count()
    failure_rate = failed / total_done if total_done > 0 else 0.0
    policy_sched = get_policy_engine().select_scheduler(
        queue_depth=queue_depth, failure_rate=failure_rate
    )

    schedulers = {
        "fifo": FIFOScheduler,
        "heuristic": HeuristicScheduler,
        "utility": UtilityScheduler,
    }
    sched_cls = schedulers.get(policy_sched, UtilityScheduler)
    scheduler = sched_cls()

    task_recs = db.query(TaskRecord).filter(TaskRecord.status.in_(["pending", "queued"])).all()
    worker_recs = db.query(WorkerRecord).filter(WorkerRecord.status != "down").all()
    completed_recs = db.query(TaskRecord).filter_by(status="completed").all()

    if not task_recs or not worker_recs:
        return []

    tasks = []
    for r in task_recs:
        deps = [d.strip() for d in r.dependencies.split(",") if d.strip()] if r.dependencies else []
        tasks.append(ArbiterTask(
            id=r.id, compute_cost=r.compute_cost, resource_type=r.resource_type,
            deadline=r.deadline, priority=r.priority, failure_probability=r.failure_probability,
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

    completed_ids = {r.id for r in completed_recs}
    assignments = scheduler.schedule(tasks, workers, completed_ids)

    log.info("schedule_run", scheduler=policy_sched,
             total_tasks=len(task_recs), total_workers=len(worker_recs),
             assignments=len(assignments))

    results = []
    for a in assignments:
        task_rec = db.query(TaskRecord).filter_by(id=a.task_id).first()
        if task_rec:
            task_rec.status = "running"
            task_rec.assigned_worker = a.worker_id
            task_rec.start_time = a.scheduled_time

        worker_rec = db.query(WorkerRecord).filter_by(id=a.worker_id).first()
        if worker_rec:
            t = next((t for t in tasks if t.id == a.task_id), None)
            if t:
                worker_rec.current_load += t.compute_cost
                worker_rec.status = "busy"

        db.add(AssignmentRecord(
            task_id=a.task_id, worker_id=a.worker_id,
            scheduled_time=a.scheduled_time,
        ))
        _log_event(db, "TASK_ASSIGNED", task_id=a.task_id, worker_id=a.worker_id)
        results.append({"task_id": a.task_id, "worker_id": a.worker_id})

    db.commit()
    return results

# Apply rate limit to schedule trigger if slowapi is available
if _HAS_SLOWAPI:
    trigger_schedule = limiter.limit("10/minute")(trigger_schedule)


# ── metrics ────────────────────────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsSnapshot)
def get_metrics(db: Session = Depends(get_db)):
    total = db.query(TaskRecord).count()
    completed = db.query(TaskRecord).filter_by(status="completed").count()
    failed = db.query(TaskRecord).filter_by(status="failed").count()
    pending = db.query(TaskRecord).filter(TaskRecord.status.in_(["pending", "queued"])).count()
    running = db.query(TaskRecord).filter_by(status="running").count()
    queue_depth = db.query(TaskRecord).filter_by(status="queued").count()

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
        pending=pending, running=running, queue_depth=queue_depth,
        avg_latency=avg_lat, sla_violation_rate=sla_rate,
        worker_count=worker_count, active_workers=active,
    )


# ── health ─────────────────────────────────────────────────────────────────────

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
        status="healthy" if (db_ok and redis_ok) else "degraded",
        db_connected=db_ok,
        redis_connected=redis_ok,
        scheduler_type=settings.scheduler_type,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


# ── events ─────────────────────────────────────────────────────────────────────

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


# ── prometheus ─────────────────────────────────────────────────────────────────

@app.get("/metrics/prometheus")
def prometheus_metrics():
    from arbiter.metrics.prometheus import get_metrics_output
    body, content_type = get_metrics_output()
    return Response(content=body, media_type=content_type)


# ── websocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    log.info("ws_client_connected", total_clients=len(_ws_clients))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        log.info("ws_client_disconnected", total_clients=len(_ws_clients))
