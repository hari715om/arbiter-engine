"""Scenario generator — creates reproducible task/worker scenarios for simulation."""

import random
from arbiter.models.task import Task
from arbiter.models.worker import Worker


class ScenarioGenerator:
    """Generates deterministic task/worker scenarios using a seeded RNG."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._task_counter = 0
        self._worker_counter = 0

    def generate_tasks(
        self,
        num_tasks: int = 50,
        deadline_tightness: float = 2.0,
        dependency_density: float = 0.2,
        max_arrival_spread: float = 50.0,
        resource_types: list[str] | None = None,
    ) -> list[Task]:
        """Generate tasks with random attributes. Dependencies only reference earlier tasks (DAG)."""
        if resource_types is None:
            resource_types = ["cpu", "gpu"]

        tasks: list[Task] = []

        for i in range(num_tasks):
            task_id = f"task-{self._task_counter:04d}"
            self._task_counter += 1

            compute_cost = round(self.rng.uniform(5.0, 50.0), 1)
            estimated_duration = round(self.rng.uniform(5.0, 30.0), 1)
            priority = self.rng.randint(1, 10)
            arrival_time = round(self.rng.uniform(0.0, max_arrival_spread), 1)
            resource_type = self.rng.choice(resource_types)
            failure_probability = round(self.rng.uniform(0.0, 0.2), 3)

            slack = self.rng.uniform(1.0, deadline_tightness)
            deadline = round(arrival_time + estimated_duration * slack, 1)

            # Dependencies: only on earlier tasks (maintains DAG property)
            dependencies: list[str] = []
            if tasks and dependency_density > 0:
                max_deps = min(3, len(tasks))
                for earlier_task in self.rng.sample(tasks, min(len(tasks), max_deps * 3)):
                    if self.rng.random() < dependency_density:
                        dependencies.append(earlier_task.id)
                        if len(dependencies) >= max_deps:
                            break

            tasks.append(Task(
                id=task_id,
                compute_cost=compute_cost,
                deadline=deadline,
                priority=priority,
                dependencies=dependencies,
                resource_type=resource_type,
                failure_probability=failure_probability,
                estimated_duration=estimated_duration,
                arrival_time=arrival_time,
            ))

        tasks.sort(key=lambda t: t.arrival_time)
        return tasks

    def generate_workers(
        self,
        num_workers: int = 5,
        resource_types: list[str] | None = None,
    ) -> list[Worker]:
        """Generate heterogeneous workers with varied capacity, speed, and resource support."""
        if resource_types is None:
            resource_types = ["cpu", "gpu"]

        workers: list[Worker] = []

        for i in range(num_workers):
            worker_id = f"worker-{self._worker_counter:03d}"
            self._worker_counter += 1

            cpu_capacity = round(self.rng.uniform(50.0, 150.0), 0)
            memory_capacity = round(self.rng.uniform(256.0, 1024.0), 0)
            speed_multiplier = round(self.rng.uniform(0.5, 2.0), 2)
            failure_rate = round(self.rng.uniform(0.0, 0.05), 3)

            if self.rng.random() < 0.4:
                supported = resource_types.copy()
            else:
                supported = ["cpu"]

            workers.append(Worker(
                id=worker_id,
                cpu_capacity=cpu_capacity,
                memory_capacity=memory_capacity,
                speed_multiplier=speed_multiplier,
                failure_rate=failure_rate,
                supported_resources=supported,
            ))

        return workers
