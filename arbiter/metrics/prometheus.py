try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


if HAS_PROMETHEUS:
    TASKS_TOTAL = Counter(
        "arbiter_tasks_total",
        "Total tasks by status",
        ["status"],
    )
    TASK_LATENCY = Histogram(
        "arbiter_task_latency_seconds",
        "Task latency from arrival to completion",
        buckets=[1, 5, 10, 30, 60, 120, 300, 600],
    )
    WORKER_UTILIZATION = Gauge(
        "arbiter_worker_utilization",
        "Current worker load ratio",
        ["worker_id"],
    )
    SLA_VIOLATIONS = Counter(
        "arbiter_sla_violations_total",
        "Total SLA deadline violations",
    )
    SCHEDULER_DECISIONS = Counter(
        "arbiter_scheduler_decisions_total",
        "Total scheduling decisions",
        ["scheduler"],
    )
    ACTIVE_TASKS = Gauge(
        "arbiter_active_tasks",
        "Currently running tasks",
    )
    QUEUE_DEPTH = Gauge(
        "arbiter_queue_depth",
        "Tasks waiting in queue",
    )
    SCHEDULER_INFO = Info(
        "arbiter_scheduler",
        "Active scheduler metadata",
    )


def record_task_created():
    if HAS_PROMETHEUS:
        TASKS_TOTAL.labels(status="created").inc()
        QUEUE_DEPTH.inc()


def record_task_started(scheduler_name: str):
    if HAS_PROMETHEUS:
        TASKS_TOTAL.labels(status="started").inc()
        SCHEDULER_DECISIONS.labels(scheduler=scheduler_name).inc()
        ACTIVE_TASKS.inc()
        QUEUE_DEPTH.dec()


def record_task_completed(latency: float):
    if HAS_PROMETHEUS:
        TASKS_TOTAL.labels(status="completed").inc()
        TASK_LATENCY.observe(latency)
        ACTIVE_TASKS.dec()


def record_task_failed():
    if HAS_PROMETHEUS:
        TASKS_TOTAL.labels(status="failed").inc()
        ACTIVE_TASKS.dec()


def record_sla_violation():
    if HAS_PROMETHEUS:
        SLA_VIOLATIONS.inc()


def update_worker_utilization(worker_id: str, load_ratio: float):
    if HAS_PROMETHEUS:
        WORKER_UTILIZATION.labels(worker_id=worker_id).set(load_ratio)


def set_scheduler_info(name: str, version: str = "0.7.0"):
    if HAS_PROMETHEUS:
        SCHEDULER_INFO.info({"name": name, "version": version})


def get_metrics_output() -> tuple[bytes, str]:
    if HAS_PROMETHEUS:
        return generate_latest(), CONTENT_TYPE_LATEST
    return b"# prometheus_client not installed\n", "text/plain"
