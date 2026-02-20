"""FIFO Scheduler — baseline first-come-first-served scheduler."""

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker
from arbiter.schedulers.base import BaseScheduler, Assignment


class FIFOScheduler(BaseScheduler):
    """Assigns tasks strictly in arrival order to the first compatible worker."""

    def schedule(
        self,
        tasks: list[Task],
        workers: list[Worker],
        completed_task_ids: set[str],
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)

        # Track remaining capacity within this scheduling round
        round_capacity: dict[str, float] = {
            w.id: w.available_capacity for w in workers
        }

        for task in sorted_tasks:
            if task.status != TaskStatus.QUEUED:
                continue
            if not self._dependencies_met(task, completed_task_ids):
                continue

            for worker in workers:
                if self._can_assign(task, worker, round_capacity):
                    assignments.append(
                        Assignment(
                            task_id=task.id,
                            worker_id=worker.id,
                            scheduled_time=task.arrival_time,
                        )
                    )
                    round_capacity[worker.id] -= task.compute_cost
                    break

        return assignments

    def _dependencies_met(
        self, task: Task, completed_task_ids: set[str]
    ) -> bool:
        """All dependencies must be in the completed set."""
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
