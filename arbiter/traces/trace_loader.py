import csv
import os
import logging
import random
from arbiter.models.task import Task
from arbiter.models.worker import Worker

logger = logging.getLogger(__name__)


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
