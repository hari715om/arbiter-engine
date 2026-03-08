import csv
import os
import random
from arbiter.models.task import Task
from arbiter.models.worker import Worker
from arbiter.logging_config import get_logger

log = get_logger(__name__)


def load_synthetic_trace(
    num_tasks: int = 500,
    num_workers: int = 20,
    seed: int = 42,
    burstiness: float = 0.3,
) -> tuple[list[Task], list[Worker]]:
    """Generate a realistic trace with bursty arrivals and heterogeneous tasks."""
    rng = random.Random(seed)
    tasks = []
    current_time = 0.0

    for i in range(num_tasks):
        # bursty arrival pattern
        if rng.random() < burstiness:
            current_time += rng.uniform(0, 0.5)  # burst: many tasks close together
        else:
            current_time += rng.expovariate(1.0 / 10.0)  # normal: Poisson

        priority = rng.choices([1,2,3,4,5,6,7,8,9,10], weights=[1,2,3,5,8,8,5,3,2,1])[0]
        cost = rng.uniform(0.5, 8.0)
        duration = rng.uniform(2.0, cost * 15)
        deadline = current_time + duration * rng.uniform(1.2, 5.0)
        fail_prob = rng.choices([0.0, 0.05, 0.1, 0.2, 0.3], weights=[5, 3, 2, 1, 1])[0]
        resource = rng.choice(["cpu", "cpu", "cpu", "gpu", "memory"])

        # some tasks depend on recent predecessors
        deps = []
        if i > 3 and rng.random() < 0.2:
            dep_count = rng.randint(1, min(3, i))
            dep_indices = rng.sample(range(max(0, i-10), i), dep_count)
            deps = [f"trace-{j}" for j in dep_indices]

        tasks.append(Task(
            id=f"trace-{i}",
            compute_cost=round(cost, 2),
            resource_type=resource,
            deadline=round(deadline, 1),
            priority=priority,
            failure_probability=fail_prob,
            estimated_duration=round(duration, 2),
            arrival_time=round(current_time, 2),
            dependencies=deps,
        ))

    workers = []
    for i in range(num_workers):
        cap = rng.choice([4.0, 8.0, 8.0, 16.0, 16.0, 32.0])
        speed = rng.uniform(0.7, 1.5)
        resources = ["cpu"]
        if rng.random() < 0.3:
            resources.append("gpu")
        resources.append("memory")

        workers.append(Worker(
            id=f"trace-worker-{i}",
            cpu_capacity=cap,
            memory_capacity=rng.choice([16.0, 32.0, 64.0]),
            speed_multiplier=round(speed, 2),
            supported_resources=resources,
        ))

    return tasks, workers


def load_csv_trace(filepath: str) -> tuple[list[Task], list[Worker]]:
    """Load tasks from a CSV file (for real cluster trace data)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trace file not found: {filepath}")

    tasks = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tasks.append(Task(
                id=row.get("task_id", f"csv-{i}"),
                compute_cost=float(row.get("cpu", 1.0)),
                resource_type=row.get("resource_type", "cpu"),
                deadline=float(row.get("deadline", 1000.0)),
                priority=int(row.get("priority", 5)),
                failure_probability=float(row.get("fail_prob", 0.0)),
                estimated_duration=float(row.get("duration", 10.0)),
                arrival_time=float(row.get("arrival_time", 0.0)),
            ))

    # generate default workers if not in trace
    workers = [
        Worker(id=f"csv-worker-{i}", cpu_capacity=8.0)
        for i in range(max(5, len(tasks) // 20))
    ]
    return tasks, workers


# ── Real Cluster Trace Loaders ─────────────────────────────────────────────────
#
# See arbiter/traces/README_TRACES.md for download instructions.
# These loaders parse the publicly released schemas and map fields to Arbiter's
# Task and Worker domain models.

def load_google_borg_trace(
    filepath: str,
    max_tasks: int = 5000,
    n_workers: int = 50,
    seed: int = 42,
) -> tuple[list[Task], list[Worker]]:
    """
    Load tasks from the Google Borg cluster trace (2019 release).

    Expected CSV schema (task_events table):
        time            — timestamp in microseconds (int)
        missing_info    — bitmask of missing fields (int)
        job_id          — unique job identifier (int)
        task_index      — task index within job (int)
        machine_id      — assigned machine (int or empty)
        event_type      — 0=SUBMIT, 1=SCHEDULE, 2=EVICT, 3=FAIL, 4=FINISH, 5=KILL, 6=LOST, 7=UPDATE
        user            — username string
        scheduling_class — 0-3 (higher = more latency-sensitive)
        priority        — 0-11 (higher = more important)
        cpu_request     — normalised CPU [0.0–1.0] relative to max machine
        memory_request  — normalised memory [0.0–1.0]
        disk_request    — normalised disk [0.0–1.0]
        different_machine — bool
        event_name      — string (same as event_type)

    Download:
        https://research.google/tools/datasets/google-cluster-workload-traces-2019/
        File: task_events/part-00000-of-00500.csv.gz (one of 500 shards)

    Args:
        filepath: Path to uncompressed task_events CSV file.
        max_tasks: Maximum number of SUBMIT events to load.
        n_workers: Number of workers to auto-generate.
        seed: RNG seed for worker generation.

    Returns:
        (tasks, workers) tuple ready for SimulationEngine.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Borg trace not found: {filepath}\n"
            f"See arbiter/traces/README_TRACES.md for download instructions."
        )

    # Borg CSVs have no header row — positional columns
    # Index: 0=time, 1=missing_info, 2=job_id, 3=task_index, 4=machine_id,
    #        5=event_type, 6=user, 7=scheduling_class, 8=priority,
    #        9=cpu_request, 10=memory_request, 11=disk_request, 12=different_machine
    SUBMIT_EVENT = "0"

    tasks: list[Task] = []
    arrival_scale = 1_000_000  # microseconds → seconds

    skipped = 0
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(tasks) >= max_tasks:
                break
            try:
                if len(row) < 13:
                    skipped += 1
                    continue
                event_type = row[5].strip()
                if event_type != SUBMIT_EVENT:
                    continue

                time_us = int(row[0]) if row[0].strip() else 0
                job_id = row[2].strip() or "0"
                task_idx = row[3].strip() or "0"
                sched_class = int(row[7]) if row[7].strip() else 0
                priority_raw = int(row[8]) if row[8].strip() else 5
                cpu_req = float(row[9]) if row[9].strip() else 0.5
                mem_req = float(row[10]) if row[10].strip() else 0.5

                # Map Borg priority (0–11) to Arbiter priority (1–10)
                priority = max(1, min(10, int(priority_raw * 10 / 11) + 1))

                # scheduling_class 3 = latency-sensitive → tight deadlines
                arrival = time_us / arrival_scale
                duration_est = max(1.0, cpu_req * 60)      # heuristic
                deadline_slack = 1.5 if sched_class >= 3 else 4.0
                deadline = arrival + duration_est * deadline_slack

                # CPU normalised to [0,1]; scale to Arbiter compute units [0.5, 16]
                compute_cost = max(0.5, min(16.0, cpu_req * 16))

                tasks.append(Task(
                    id=f"borg-{job_id}-{task_idx}",
                    compute_cost=round(compute_cost, 2),
                    resource_type="gpu" if sched_class == 3 else "cpu",
                    deadline=round(deadline, 1),
                    priority=priority,
                    failure_probability=0.05 + (0.1 * (priority_raw < 4)),
                    estimated_duration=round(duration_est, 2),
                    arrival_time=round(arrival, 2),
                ))
            except (ValueError, IndexError) as e:
                skipped += 1

    log.info("borg_trace_loaded", tasks=len(tasks), skipped=skipped, file=filepath)

    workers = _generate_workers_for_trace(tasks, n=n_workers, seed=seed,
                                          label="borg-worker")
    return tasks, workers


def load_alibaba_2018_trace(
    filepath: str,
    max_tasks: int = 5000,
    n_workers: int = 50,
    seed: int = 42,
) -> tuple[list[Task], list[Worker]]:
    """
    Load tasks from the Alibaba Cluster Trace 2018 (batch workload).

    Expected CSV schema (batch_task.csv):
        task_name       — unique task name (string)
        instance_num    — number of instances (int)
        job_name        — job identifier (string)
        task_type       — task type (string)
        status          — Terminated|Waiting|Running|Failed|Cancelled
        start_time      — seconds since trace start (float)
        end_time        — seconds since trace start (float)
        plan_cpu        — planned CPU request (float, in units of 100 = 1 core)
        plan_mem        — planned memory request (float, normalised [0,1])

    Download:
        https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
        File: batch_task.csv (from the tarball)

    Args:
        filepath: Path to batch_task.csv.
        max_tasks: Maximum number of tasks to load.
        n_workers: Number of workers to auto-generate.
        seed: RNG seed for worker generation.

    Returns:
        (tasks, workers) tuple.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Alibaba trace not found: {filepath}\n"
            f"See arbiter/traces/README_TRACES.md for download instructions."
        )

    tasks: list[Task] = []
    skipped = 0

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(tasks) >= max_tasks:
                break
            try:
                status = row.get("status", "").strip()
                # Only load submitted/terminated tasks (skip already-failed ones)
                if status not in ("Terminated", "Waiting", "Running", ""):
                    continue

                start = float(row.get("start_time") or 0.0)
                end = float(row.get("end_time") or 0.0)
                cpu = float(row.get("plan_cpu") or 100.0) / 100.0   # cores
                mem = float(row.get("plan_mem") or 0.1)

                if end <= start:
                    end = start + max(1.0, cpu * 30)

                duration = end - start
                deadline = start + duration * 2.5   # Alibaba jobs usually have loose SLAs
                compute_cost = max(0.5, min(16.0, cpu))

                tasks.append(Task(
                    id=row.get("task_name", f"ali-{i}"),
                    compute_cost=round(compute_cost, 2),
                    resource_type="cpu",
                    deadline=round(deadline, 1),
                    priority=5,          # Alibaba trace doesn't publish priorities
                    failure_probability=0.08 if status == "Failed" else 0.03,
                    estimated_duration=round(duration, 2),
                    arrival_time=round(start, 2),
                ))
            except (ValueError, KeyError, TypeError) as e:
                skipped += 1

    log.info("alibaba_trace_loaded", tasks=len(tasks), skipped=skipped, file=filepath)

    workers = _generate_workers_for_trace(tasks, n=n_workers, seed=seed,
                                          label="ali-worker")
    return tasks, workers


def _generate_workers_for_trace(
    tasks: list[Task],
    n: int,
    seed: int,
    label: str = "worker",
) -> list[Worker]:
    """
    Auto-generate a realistic heterogeneous worker fleet sized for the trace.

    Capacity is set so the fleet can handle the trace's peak CPU demand.
    """
    rng = random.Random(seed)
    if tasks:
        avg_cost = sum(t.compute_cost for t in tasks) / len(tasks)
        # Each worker should be able to handle ~3 average tasks
        target_cap = max(8.0, avg_cost * 3)
    else:
        target_cap = 8.0

    workers: list[Worker] = []
    caps = [4.0, 8.0, 8.0, 16.0, 16.0, 32.0, 64.0]
    for i in range(n):
        cap = min(target_cap * rng.uniform(0.5, 2.0), 64.0)
        cap = min(caps, key=lambda c: abs(c - cap))   # snap to standard size
        resources = ["cpu", "memory"]
        if rng.random() < 0.25:
            resources.append("gpu")
        workers.append(Worker(
            id=f"{label}-{i}",
            cpu_capacity=cap,
            memory_capacity=cap * rng.uniform(1.5, 4.0),
            speed_multiplier=round(rng.uniform(0.7, 1.5), 2),
            supported_resources=resources,
        ))
    return workers

