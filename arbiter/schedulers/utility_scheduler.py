"""Utility-Based Scheduler — globally optimal task-worker assignment.

Instead of scoring tasks and workers independently (like Heuristic) or
augmenting scores with ML predictions (like MLScheduler), the Utility
scheduler evaluates every valid (task, worker) pair with a unified
reward function and picks the assignment set that maximizes total utility.

This is the most principled approach to scheduling — every decision is
measured against the same global objective. It's how Google Borg and
Kubernetes' kube-scheduler work internally (scoring + filtering).
"""

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus
from arbiter.schedulers.base import BaseScheduler, Assignment
from arbiter.schedulers.utility import (
    UtilityFunction,
    ObjectiveWeight,
    LatencyObjective,
    ThroughputObjective,
    FairnessObjective,
    CostObjective,
    RiskObjective,
)


class UtilityScheduler(BaseScheduler):
    """Assigns tasks to maximize total utility across all assignments.

    Algorithm (greedy):
    1. Generate all valid (task, worker) candidates
    2. Score each with the utility function
    3. Pick the highest-utility pair, mark both as used
    4. Update context (capacity, load), repeat
    5. Stop when no more valid pairs remain

    Why greedy instead of optimal? With N tasks and M workers, the
    optimal solution is NP-hard (generalized assignment problem).
    Greedy gives O(N·M·log(N·M)) and produces near-optimal results
    in practice — this is what production systems use.
    """

    def __init__(
        self,
        utility_fn: UtilityFunction | None = None,
        current_time: float = 0.0,
        worker_reliability: dict[str, float] | None = None,
    ):
        self.utility_fn = utility_fn or UtilityFunction.default()
        self.current_time = current_time
        self.worker_reliability = worker_reliability or {}

    def schedule(
        self,
        tasks: list[Task],
        workers: list[Worker],
        completed_task_ids: set[str],
    ) -> list[Assignment]:
        assignments: list[Assignment] = []

        # Filter to schedulable tasks
        schedulable = [
            t for t in tasks
            if t.status == TaskStatus.QUEUED
            and self._dependencies_met(t, completed_task_ids)
        ]

        if not schedulable:
            return assignments

        # Track remaining capacity in this scheduling round
        round_capacity: dict[str, float] = {
            w.id: w.available_capacity for w in workers
        }

        # Available workers (not DOWN)
        available_workers = [w for w in workers if w.status != WorkerStatus.DOWN]

        # Build context for utility evaluation
        context = {
            "current_time": self.current_time,
            "workers": workers,
            "round_capacity": round_capacity,
            "worker_reliability": self.worker_reliability,
            "queue_depth": len(schedulable),
        }

        # Greedy assignment loop
        assigned_tasks: set[str] = set()

        while schedulable and available_workers:
            best_utility = -1.0
            best_task = None
            best_worker = None

            for task in schedulable:
                if task.id in assigned_tasks:
                    continue

                for worker in available_workers:
                    if not self._can_assign(task, worker, round_capacity):
                        continue

                    utility = self.utility_fn.evaluate(task, worker, context)

                    if utility > best_utility:
                        best_utility = utility
                        best_task = task
                        best_worker = worker

            if best_task is None or best_worker is None:
                break  # No more valid assignments

            # Commit assignment
            assignments.append(Assignment(
                task_id=best_task.id,
                worker_id=best_worker.id,
                scheduled_time=self.current_time,
            ))

            assigned_tasks.add(best_task.id)
            round_capacity[best_worker.id] -= best_task.compute_cost

            # Update context for next iteration
            context["round_capacity"] = round_capacity

            # Remove task from schedulable
            schedulable = [t for t in schedulable if t.id not in assigned_tasks]

        return assignments

    def _dependencies_met(self, task: Task, completed_task_ids: set[str]) -> bool:
        """All dependencies must be completed before scheduling."""
        return all(dep_id in completed_task_ids for dep_id in task.dependencies)

    def _can_assign(
        self, task: Task, worker: Worker, round_capacity: dict[str, float],
    ) -> bool:
        """Worker must be operational, support resource type, and have capacity."""
        return (
            worker.status != WorkerStatus.DOWN
            and task.resource_type in worker.supported_resources
            and round_capacity.get(worker.id, 0) >= task.compute_cost
        )
