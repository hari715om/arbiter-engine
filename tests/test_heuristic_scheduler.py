"""Tests for the Heuristic Scheduler."""

import pytest

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker
from arbiter.schedulers.heuristic import HeuristicScheduler
from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.simulator.engine import SimulationEngine
from arbiter.simulator.generator import ScenarioGenerator


class TestHeuristicScheduler:
    """Tests for the multi-factor heuristic scheduler."""

    def _make_task(self, id: str, arrival: float = 0.0, cost: float = 10.0,
                   duration: float = 10.0, resource: str = "cpu",
                   priority: int = 5, deadline: float = 200.0,
                   deps: list[str] | None = None,
                   failure_prob: float = 0.0) -> Task:
        return Task(
            id=id, compute_cost=cost, deadline=deadline,
            priority=priority, dependencies=deps or [],
            resource_type=resource, failure_probability=failure_prob,
            estimated_duration=duration, arrival_time=arrival,
            status=TaskStatus.QUEUED,
        )

    def _make_worker(self, id: str, capacity: float = 100.0,
                     speed: float = 1.0,
                     resources: list[str] | None = None) -> Worker:
        return Worker(
            id=id, cpu_capacity=capacity, memory_capacity=512.0,
            speed_multiplier=speed,
            supported_resources=resources or ["cpu"],
        )

    def test_high_priority_first(self):
        """High-priority tasks should be scheduled before low-priority ones."""
        low = self._make_task("low", priority=1, cost=50.0)
        high = self._make_task("high", priority=10, cost=50.0)
        worker = self._make_worker("w1", capacity=50.0)

        scheduler = HeuristicScheduler()
        assignments = scheduler.schedule([low, high], [worker], set())

        assert len(assignments) == 1
        assert assignments[0].task_id == "high"

    def test_urgent_deadline_first(self):
        """Tasks with tighter deadlines should be preferred when priority is equal."""
        relaxed = self._make_task("relaxed", priority=5, deadline=1000.0, cost=50.0)
        urgent = self._make_task("urgent", priority=5, deadline=11.0, cost=50.0)
        worker = self._make_worker("w1", capacity=50.0)

        scheduler = HeuristicScheduler(w_priority=0.0, w_deadline=1.0, w_unlock=0.0)
        assignments = scheduler.schedule([relaxed, urgent], [worker], set())

        assert len(assignments) == 1
        assert assignments[0].task_id == "urgent"

    def test_dependency_unlock_preferred(self):
        """Tasks that unlock more dependents should be preferred."""
        # blocker unlocks 3 tasks, loner unlocks 0
        blocker = self._make_task("blocker", priority=5, cost=50.0)
        loner = self._make_task("loner", priority=5, cost=50.0)
        dep1 = self._make_task("d1", deps=["blocker"])
        dep2 = self._make_task("d2", deps=["blocker"])
        dep3 = self._make_task("d3", deps=["blocker"])
        worker = self._make_worker("w1", capacity=50.0)

        scheduler = HeuristicScheduler(w_priority=0.0, w_deadline=0.0, w_unlock=1.0)
        all_tasks = [blocker, loner, dep1, dep2, dep3]
        assignments = scheduler.schedule(all_tasks, [worker], set())

        assert len(assignments) == 1
        assert assignments[0].task_id == "blocker"

    def test_respects_capacity_constraint(self):
        """Should not assign tasks exceeding worker capacity."""
        big_task = self._make_task("big", cost=200.0)
        worker = self._make_worker("w1", capacity=100.0)

        scheduler = HeuristicScheduler()
        assignments = scheduler.schedule([big_task], [worker], set())

        assert len(assignments) == 0

    def test_respects_resource_constraint(self):
        """Should not assign GPU tasks to CPU-only workers."""
        gpu_task = self._make_task("gpu-task", resource="gpu")
        cpu_worker = self._make_worker("w1", resources=["cpu"])

        scheduler = HeuristicScheduler()
        assignments = scheduler.schedule([gpu_task], [cpu_worker], set())

        assert len(assignments) == 0

    def test_respects_dependency_constraint(self):
        """Should not schedule tasks whose dependencies aren't completed."""
        task = self._make_task("blocked", deps=["nonexistent"])
        worker = self._make_worker("w1")

        scheduler = HeuristicScheduler()
        assignments = scheduler.schedule([task], [worker], set())

        assert len(assignments) == 0

    def test_prefers_faster_worker(self):
        """Should prefer faster workers for task assignment."""
        task = self._make_task("t1", cost=10.0)
        slow = self._make_worker("slow", speed=0.5, capacity=100.0)
        fast = self._make_worker("fast", speed=2.0, capacity=100.0)

        scheduler = HeuristicScheduler()
        assignments = scheduler.schedule([task], [slow, fast], set())

        assert len(assignments) == 1
        assert assignments[0].worker_id == "fast"

    def test_multiple_assignments(self):
        """Should assign multiple tasks when capacity allows."""
        t1 = self._make_task("t1", cost=20.0, priority=8)
        t2 = self._make_task("t2", cost=20.0, priority=3)
        w1 = self._make_worker("w1", capacity=50.0)
        w2 = self._make_worker("w2", capacity=50.0)

        scheduler = HeuristicScheduler()
        assignments = scheduler.schedule([t1, t2], [w1, w2], set())

        assert len(assignments) == 2
        task_ids = {a.task_id for a in assignments}
        assert task_ids == {"t1", "t2"}

    def test_configurable_weights(self):
        """Changing weights should change scheduling priority."""
        low_prio_urgent = self._make_task("urgent", priority=1, deadline=11.0, cost=50.0)
        high_prio_relaxed = self._make_task("relaxed", priority=10, deadline=1000.0, cost=50.0)
        worker = self._make_worker("w1", capacity=50.0)

        # Priority-heavy: should pick high priority
        prio_scheduler = HeuristicScheduler(w_priority=1.0, w_deadline=0.0, w_unlock=0.0)
        assignments = prio_scheduler.schedule(
            [low_prio_urgent, high_prio_relaxed], [worker], set()
        )
        assert assignments[0].task_id == "relaxed"

        # Deadline-heavy: should pick urgent
        deadline_scheduler = HeuristicScheduler(w_priority=0.0, w_deadline=1.0, w_unlock=0.0)
        assignments = deadline_scheduler.schedule(
            [low_prio_urgent, high_prio_relaxed], [worker], set()
        )
        assert assignments[0].task_id == "urgent"

    def test_integration_with_engine(self):
        """Heuristic scheduler should work with SimulationEngine and produce metrics."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=20, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=HeuristicScheduler(), seed=42,
        )
        metrics = engine.run()
        report = metrics.report

        assert report.total_tasks == 20
        assert report.tasks_completed + report.tasks_failed + report.tasks_pending == 20
        assert report.tasks_completed > 0
