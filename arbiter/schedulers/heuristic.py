"""Heuristic Scheduler — multi-factor scoring for smarter task assignment."""

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker
from arbiter.schedulers.base import BaseScheduler, Assignment


class HeuristicScheduler(BaseScheduler):
    """Scores tasks by priority/urgency/unlock-potential, picks best-fit workers."""

    def __init__(
        self,
        w_priority: float = 0.4,
        w_deadline: float = 0.3,
        w_unlock: float = 0.3,
        current_time: float = 0.0,
    ):
        self.w_priority = w_priority
        self.w_deadline = w_deadline
        self.w_unlock = w_unlock
        self.current_time = current_time

    def schedule(
        self,
        tasks: list[Task],
        workers: list[Worker],
        completed_task_ids: set[str],
    ) -> list[Assignment]:
        assignments: list[Assignment] = []

        # Build dependency reverse-map: task_id → list of tasks that depend on it
        dependents_map: dict[str, int] = {}
        for task in tasks:
            for dep_id in task.dependencies:
                dependents_map[dep_id] = dependents_map.get(dep_id, 0) + 1

        # Filter to schedulable tasks (QUEUED + dependencies met)
        schedulable = [
            t for t in tasks
            if t.status == TaskStatus.QUEUED
            and self._dependencies_met(t, completed_task_ids)
        ]

        if not schedulable:
            return assignments

        # Score and sort tasks (highest score first)
        scored_tasks = [
            (self._score_task(t, dependents_map), t)
            for t in schedulable
        ]
        scored_tasks.sort(key=lambda x: x[0], reverse=True)

        # Track remaining capacity within this scheduling round
        round_capacity: dict[str, float] = {
            w.id: w.available_capacity for w in workers
        }

        for _score, task in scored_tasks:
            # Find compatible workers and score them
            best_worker = None
            best_worker_score = -1.0

            for worker in workers:
                if not self._can_assign(task, worker, round_capacity):
                    continue
                w_score = self._score_worker(task, worker, round_capacity)
                if w_score > best_worker_score:
                    best_worker_score = w_score
                    best_worker = worker

            if best_worker is not None:
                assignments.append(
                    Assignment(
                        task_id=task.id,
                        worker_id=best_worker.id,
                        scheduled_time=task.arrival_time,
                    )
                )
                round_capacity[best_worker.id] -= task.compute_cost

        return assignments

    def _score_task(self, task: Task, dependents_map: dict[str, int]) -> float:
        """Score a task: higher = should be scheduled sooner."""
        # Priority: normalized to [0, 1]
        priority_score = task.priority / 10.0

        # Deadline urgency: tighter deadline = higher score
        time_remaining = max(task.deadline - self.current_time, 0.01)
        urgency_score = min(1.0, task.estimated_duration / time_remaining)

        # Unlock potential: how many tasks are waiting on this one
        unlock_count = dependents_map.get(task.id, 0)
        unlock_score = min(1.0, unlock_count / 3.0)

        return (
            self.w_priority * priority_score
            + self.w_deadline * urgency_score
            + self.w_unlock * unlock_score
        )

    def _score_worker(
        self, task: Task, worker: Worker, round_capacity: dict[str, float]
    ) -> float:
        """Score a worker for a task: higher = better fit."""
        # Speed: faster workers are better
        speed_score = min(1.0, worker.speed_multiplier / 2.0)

        # Capacity fit: prefer workers where task fills capacity tightly
        remaining = round_capacity.get(worker.id, 0)
        if remaining <= 0:
            return -1.0
        fit_score = 1.0 - abs(remaining - task.compute_cost) / remaining

        # Resource specialization: prefer specialists over generalists
        specialization_score = 1.0 / len(worker.supported_resources)

        return 0.5 * speed_score + 0.3 * fit_score + 0.2 * specialization_score

    def _dependencies_met(
        self, task: Task, completed_task_ids: set[str]
    ) -> bool:
        """All dependencies must be completed."""
        return all(dep_id in completed_task_ids for dep_id in task.dependencies)

    def _can_assign(
        self, task: Task, worker: Worker, round_capacity: dict[str, float]
    ) -> bool:
        """Worker must be operational, support resource type, and have capacity."""
        return (
            worker.status != worker.status.DOWN
            and task.resource_type in worker.supported_resources
            and round_capacity.get(worker.id, 0) >= task.compute_cost
        )
