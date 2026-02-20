"""
Tests for the FIFO Scheduler.

These tests verify:
    1. Tasks are assigned in arrival order (FIFO)
    2. Capacity constraints are respected
    3. Resource type compatibility is enforced
    4. Dependencies block assignment until satisfied
    5. Unassignable tasks remain in the queue
"""

import pytest

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus
from arbiter.schedulers.fifo import FIFOScheduler


class TestFIFOScheduler:
    """Tests for the FIFO scheduler."""

    def setup_method(self):
        """Create a fresh scheduler for each test."""
        self.scheduler = FIFOScheduler()

    def _make_task(self, id: str, arrival: float = 0.0, cost: float = 10.0,
                   resource: str = "cpu", deps: list[str] | None = None,
                   status: TaskStatus = TaskStatus.QUEUED) -> Task:
        """Helper to create a task with sensible defaults."""
        return Task(
            id=id,
            compute_cost=cost,
            deadline=100.0,
            priority=5,
            dependencies=deps or [],
            resource_type=resource,
            failure_probability=0.0,
            estimated_duration=10.0,
            arrival_time=arrival,
            status=status,
        )

    def _make_worker(self, id: str, capacity: float = 100.0,
                     resources: list[str] | None = None) -> Worker:
        """Helper to create a worker with sensible defaults."""
        return Worker(
            id=id,
            cpu_capacity=capacity,
            memory_capacity=512.0,
            supported_resources=resources or ["cpu"],
        )

    def test_fifo_order(self):
        """Tasks should be assigned in arrival order."""
        tasks = [
            self._make_task("task-c", arrival=3.0),
            self._make_task("task-a", arrival=1.0),
            self._make_task("task-b", arrival=2.0),
        ]
        workers = [self._make_worker("w1")]

        assignments = self.scheduler.schedule(tasks, workers, set())

        # Should be in arrival order: a, b, c
        assert len(assignments) == 3
        assert assignments[0].task_id == "task-a"
        assert assignments[1].task_id == "task-b"
        assert assignments[2].task_id == "task-c"

    def test_capacity_constraint(self):
        """Workers at capacity should not receive more tasks."""
        tasks = [
            self._make_task("task-1", cost=60.0),
            self._make_task("task-2", cost=60.0),
        ]
        # Worker can handle 100, so task-1 (60) fits but task-2 (60) doesn't
        workers = [self._make_worker("w1", capacity=100.0)]

        assignments = self.scheduler.schedule(tasks, workers, set())

        assert len(assignments) == 1
        assert assignments[0].task_id == "task-1"

    def test_resource_type_compatibility(self):
        """Tasks should only go to workers supporting their resource type."""
        tasks = [
            self._make_task("cpu-task", resource="cpu"),
            self._make_task("gpu-task", resource="gpu"),
        ]
        workers = [
            self._make_worker("cpu-worker", resources=["cpu"]),
            self._make_worker("gpu-worker", resources=["gpu"]),
        ]

        assignments = self.scheduler.schedule(tasks, workers, set())

        assert len(assignments) == 2
        cpu_assignment = next(a for a in assignments if a.task_id == "cpu-task")
        gpu_assignment = next(a for a in assignments if a.task_id == "gpu-task")
        assert cpu_assignment.worker_id == "cpu-worker"
        assert gpu_assignment.worker_id == "gpu-worker"

    def test_dependencies_block_scheduling(self):
        """Tasks with unmet dependencies should NOT be assigned."""
        tasks = [
            self._make_task("task-1"),
            self._make_task("task-2", deps=["task-1"]),  # depends on task-1
        ]
        workers = [self._make_worker("w1")]

        # task-1 is NOT completed yet
        assignments = self.scheduler.schedule(tasks, workers, completed_task_ids=set())

        # Only task-1 should be assigned (task-2 blocked by dependency)
        assert len(assignments) == 1
        assert assignments[0].task_id == "task-1"

    def test_dependencies_satisfied(self):
        """Tasks with ALL dependencies completed should be assigned."""
        tasks = [
            self._make_task("task-2", deps=["task-1"]),
        ]
        workers = [self._make_worker("w1")]

        # task-1 IS completed
        assignments = self.scheduler.schedule(
            tasks, workers, completed_task_ids={"task-1"}
        )

        assert len(assignments) == 1
        assert assignments[0].task_id == "task-2"

    def test_skip_non_queued_tasks(self):
        """Only QUEUED tasks should be scheduled."""
        tasks = [
            self._make_task("pending-task", status=TaskStatus.PENDING),
            self._make_task("running-task", status=TaskStatus.RUNNING),
            self._make_task("queued-task", status=TaskStatus.QUEUED),
        ]
        workers = [self._make_worker("w1")]

        assignments = self.scheduler.schedule(tasks, workers, set())

        assert len(assignments) == 1
        assert assignments[0].task_id == "queued-task"

    def test_no_tasks_returns_empty(self):
        """Empty task list should return no assignments."""
        workers = [self._make_worker("w1")]
        assignments = self.scheduler.schedule([], workers, set())
        assert assignments == []

    def test_no_workers_returns_empty(self):
        """No workers available should return no assignments."""
        tasks = [self._make_task("task-1")]
        assignments = self.scheduler.schedule(tasks, [], set())
        assert assignments == []

    def test_down_worker_skipped(self):
        """DOWN workers should not receive tasks."""
        tasks = [self._make_task("task-1")]
        worker = self._make_worker("w1")
        worker.status = WorkerStatus.DOWN

        assignments = self.scheduler.schedule(tasks, [worker], set())
        assert assignments == []

    def test_multiple_workers_load_distribution(self):
        """Tasks should spread across workers when capacity is limited."""
        tasks = [
            self._make_task("task-1", cost=60.0, arrival=0.0),
            self._make_task("task-2", cost=60.0, arrival=1.0),
        ]
        workers = [
            self._make_worker("w1", capacity=100.0),
            self._make_worker("w2", capacity=100.0),
        ]

        assignments = self.scheduler.schedule(tasks, workers, set())

        assert len(assignments) == 2
        # task-1 goes to w1 (first available), task-2 to w2 (w1 doesn't have room)
        assert assignments[0].worker_id == "w1"
        assert assignments[1].worker_id == "w2"
