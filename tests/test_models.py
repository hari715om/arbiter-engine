"""
Tests for Task and Worker data models.

These tests verify:
    1. Task creation with valid/invalid fields
    2. Worker capacity management (assign/release)
    3. Status transitions
    4. Computed properties (latency, SLA violation, available capacity)
    5. Pydantic validation (rejects bad data)
"""

import pytest
from pydantic import ValidationError

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus


# ══════════════════════════════════════════════════════════════════════
# TASK MODEL TESTS
# ══════════════════════════════════════════════════════════════════════

class TestTask:
    """Tests for the Task model."""

    def test_create_valid_task(self):
        """A task with all required fields should be created successfully."""
        task = Task(
            id="task-001",
            compute_cost=25.0,
            deadline=100.0,
            priority=7,
            estimated_duration=15.0,
            arrival_time=0.0,
        )
        assert task.id == "task-001"
        assert task.compute_cost == 25.0
        assert task.status == TaskStatus.PENDING
        assert task.assigned_worker is None
        assert task.dependencies == []
        assert task.resource_type == "cpu"  # default

    def test_task_defaults(self):
        """Default values should be set correctly."""
        task = Task(
            id="task-002",
            compute_cost=10.0,
            deadline=50.0,
            priority=5,
            estimated_duration=10.0,
        )
        assert task.arrival_time == 0.0
        assert task.start_time is None
        assert task.completion_time is None
        assert task.failure_probability == 0.0  # default: no failure
        assert task.resource_type == "cpu"

    def test_task_with_dependencies(self):
        """Tasks can have a list of dependency IDs."""
        task = Task(
            id="task-003",
            compute_cost=10.0,
            deadline=50.0,
            priority=5,
            dependencies=["task-001", "task-002"],
            estimated_duration=10.0,
        )
        assert len(task.dependencies) == 2
        assert "task-001" in task.dependencies

    def test_invalid_priority_too_high(self):
        """Priority > 10 should be rejected."""
        with pytest.raises(ValidationError):
            Task(
                id="bad-task",
                compute_cost=10.0,
                deadline=50.0,
                priority=11,  # Invalid: max is 10
                estimated_duration=10.0,
            )

    def test_invalid_priority_too_low(self):
        """Priority < 1 should be rejected."""
        with pytest.raises(ValidationError):
            Task(
                id="bad-task",
                compute_cost=10.0,
                deadline=50.0,
                priority=0,  # Invalid: min is 1
                estimated_duration=10.0,
            )

    def test_invalid_negative_compute_cost(self):
        """Compute cost must be > 0."""
        with pytest.raises(ValidationError):
            Task(
                id="bad-task",
                compute_cost=-5.0,  # Invalid
                deadline=50.0,
                priority=5,
                estimated_duration=10.0,
            )

    def test_invalid_failure_probability(self):
        """Failure probability must be between 0 and 1."""
        with pytest.raises(ValidationError):
            Task(
                id="bad-task",
                compute_cost=10.0,
                deadline=50.0,
                priority=5,
                failure_probability=1.5,  # Invalid: max is 1.0
                estimated_duration=10.0,
            )

    def test_latency_property(self):
        """Latency should be completion_time - arrival_time."""
        task = Task(
            id="task-004",
            compute_cost=10.0,
            deadline=50.0,
            priority=5,
            estimated_duration=10.0,
            arrival_time=5.0,
        )
        # Before completion
        assert task.latency is None

        # After completion
        task.completion_time = 20.0
        assert task.latency == 15.0  # 20.0 - 5.0

    def test_sla_violated_property(self):
        """SLA should be violated when completion > deadline."""
        task = Task(
            id="task-005",
            compute_cost=10.0,
            deadline=30.0,
            priority=5,
            estimated_duration=10.0,
        )
        # Not completed yet
        assert task.is_sla_violated is False

        # Completed ON TIME
        task.completion_time = 25.0
        assert task.is_sla_violated is False

        # Completed LATE
        task.completion_time = 35.0
        assert task.is_sla_violated is True

    def test_is_ready_property(self):
        """Task is ready only when status is QUEUED."""
        task = Task(
            id="task-006",
            compute_cost=10.0,
            deadline=50.0,
            priority=5,
            estimated_duration=10.0,
        )
        assert task.is_ready is False  # PENDING

        task.status = TaskStatus.QUEUED
        assert task.is_ready is True

        task.status = TaskStatus.RUNNING
        assert task.is_ready is False

    def test_status_transitions(self):
        """Task status can transition through the full lifecycle."""
        task = Task(
            id="task-007",
            compute_cost=10.0,
            deadline=50.0,
            priority=5,
            estimated_duration=10.0,
        )
        assert task.status == TaskStatus.PENDING
        task.status = TaskStatus.QUEUED
        assert task.status == TaskStatus.QUEUED
        task.status = TaskStatus.RUNNING
        assert task.status == TaskStatus.RUNNING
        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED


# ══════════════════════════════════════════════════════════════════════
# WORKER MODEL TESTS
# ══════════════════════════════════════════════════════════════════════

class TestWorker:
    """Tests for the Worker model."""

    def test_create_valid_worker(self):
        """A worker with required fields should be created successfully."""
        worker = Worker(
            id="worker-001",
            cpu_capacity=100.0,
            memory_capacity=512.0,
        )
        assert worker.id == "worker-001"
        assert worker.cpu_capacity == 100.0
        assert worker.current_load == 0.0
        assert worker.status == WorkerStatus.IDLE
        assert worker.active_tasks == []

    def test_available_capacity(self):
        """Available capacity = cpu_capacity - current_load."""
        worker = Worker(
            id="worker-002",
            cpu_capacity=100.0,
            memory_capacity=512.0,
            current_load=30.0,
        )
        assert worker.available_capacity == 70.0

    def test_is_available(self):
        """Worker is available when not DOWN and has spare capacity."""
        worker = Worker(
            id="worker-003",
            cpu_capacity=100.0,
            memory_capacity=512.0,
        )
        assert worker.is_available is True

        worker.status = WorkerStatus.DOWN
        assert worker.is_available is False

    def test_can_handle_compatible_task(self):
        """Worker should accept a task it can handle."""
        worker = Worker(
            id="worker-004",
            cpu_capacity=100.0,
            memory_capacity=512.0,
            supported_resources=["cpu", "gpu"],
        )
        assert worker.can_handle("cpu", 50.0) is True
        assert worker.can_handle("gpu", 50.0) is True

    def test_cannot_handle_incompatible_resource(self):
        """Worker should reject tasks needing unsupported resources."""
        worker = Worker(
            id="worker-005",
            cpu_capacity=100.0,
            memory_capacity=512.0,
            supported_resources=["cpu"],  # CPU only
        )
        assert worker.can_handle("gpu", 50.0) is False

    def test_cannot_handle_over_capacity(self):
        """Worker should reject tasks exceeding remaining capacity."""
        worker = Worker(
            id="worker-006",
            cpu_capacity=100.0,
            memory_capacity=512.0,
            current_load=80.0,
        )
        assert worker.can_handle("cpu", 30.0) is False  # 80 + 30 > 100
        assert worker.can_handle("cpu", 20.0) is True   # 80 + 20 = 100

    def test_assign_task(self):
        """Assigning a task should increase load and add to active_tasks."""
        worker = Worker(
            id="worker-007",
            cpu_capacity=100.0,
            memory_capacity=512.0,
        )
        worker.assign_task("task-001", 30.0)

        assert worker.current_load == 30.0
        assert worker.status == WorkerStatus.BUSY
        assert "task-001" in worker.active_tasks

    def test_release_task(self):
        """Releasing a task should decrease load and update status."""
        worker = Worker(
            id="worker-008",
            cpu_capacity=100.0,
            memory_capacity=512.0,
        )
        worker.assign_task("task-001", 30.0)
        worker.assign_task("task-002", 20.0)
        assert worker.current_load == 50.0

        worker.release_task("task-001", 30.0)
        assert worker.current_load == 20.0
        assert worker.status == WorkerStatus.BUSY  # Still has task-002
        assert "task-001" not in worker.active_tasks

        worker.release_task("task-002", 20.0)
        assert worker.current_load == 0.0
        assert worker.status == WorkerStatus.IDLE  # No more tasks

    def test_cannot_handle_when_down(self):
        """DOWN workers should reject all tasks."""
        worker = Worker(
            id="worker-009",
            cpu_capacity=100.0,
            memory_capacity=512.0,
        )
        worker.status = WorkerStatus.DOWN
        assert worker.can_handle("cpu", 10.0) is False
