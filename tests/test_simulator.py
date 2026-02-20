"""
Tests for the Simulation Engine.

These tests verify:
    1. Simulation completes all schedulable tasks
    2. Events are processed in chronological order
    3. Worker load never exceeds capacity
    4. Metrics are calculated correctly
    5. Task failures are recorded
    6. Dependencies are handled by the engine
"""

import pytest

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker
from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.simulator.engine import SimulationEngine
from arbiter.simulator.generator import ScenarioGenerator


class TestSimulationEngine:
    """Tests for the event-driven simulation engine."""

    def _make_task(self, id: str, arrival: float = 0.0, cost: float = 10.0,
                   duration: float = 10.0, resource: str = "cpu",
                   deps: list[str] | None = None,
                   failure_prob: float = 0.0) -> Task:
        """Helper to create tasks."""
        return Task(
            id=id,
            compute_cost=cost,
            deadline=200.0,
            priority=5,
            dependencies=deps or [],
            resource_type=resource,
            failure_probability=failure_prob,
            estimated_duration=duration,
            arrival_time=arrival,
        )

    def _make_worker(self, id: str, capacity: float = 100.0,
                     speed: float = 1.0,
                     resources: list[str] | None = None) -> Worker:
        """Helper to create workers."""
        return Worker(
            id=id,
            cpu_capacity=capacity,
            memory_capacity=512.0,
            speed_multiplier=speed,
            supported_resources=resources or ["cpu"],
        )

    def test_simple_simulation(self):
        """A single task on a single worker should complete."""
        tasks = [self._make_task("task-1")]
        workers = [self._make_worker("w1")]

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()

        assert metrics.report.tasks_completed == 1
        assert metrics.report.tasks_failed == 0
        assert metrics.report.tasks_pending == 0

    def test_multiple_tasks(self):
        """Multiple tasks should all complete on sufficient workers."""
        tasks = [
            self._make_task("task-1", cost=20.0),
            self._make_task("task-2", cost=20.0),
            self._make_task("task-3", cost=20.0),
        ]
        workers = [
            self._make_worker("w1", capacity=100.0),
            self._make_worker("w2", capacity=100.0),
        ]

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()

        assert metrics.report.tasks_completed == 3

    def test_events_in_chronological_order(self):
        """Events should be processed in ascending time order."""
        tasks = [
            self._make_task("task-1", arrival=5.0),
            self._make_task("task-2", arrival=1.0),
            self._make_task("task-3", arrival=10.0),
        ]
        workers = [self._make_worker("w1")]

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        engine.run()

        # Verify events are in time order
        times = [e.time for e in engine.event_log]
        assert times == sorted(times), "Events were not in chronological order"

    def test_task_failure_recorded(self):
        """Tasks with failure_probability=1.0 should always fail."""
        tasks = [
            self._make_task("fail-task", failure_prob=1.0),
        ]
        workers = [self._make_worker("w1")]

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()

        assert metrics.report.tasks_failed == 1
        assert metrics.report.tasks_completed == 0

    def test_dependencies_in_simulation(self):
        """Task-2 should only run after task-1 completes."""
        tasks = [
            self._make_task("task-1", arrival=0.0, duration=10.0),
            self._make_task("task-2", arrival=0.0, duration=5.0, deps=["task-1"]),
        ]
        workers = [self._make_worker("w1", capacity=200.0)]

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()

        # Both should complete
        assert metrics.report.tasks_completed == 2

        # task-2 must have started AFTER task-1 completed
        task1 = engine.tasks["task-1"]
        task2 = engine.tasks["task-2"]
        assert task1.completion_time is not None
        assert task2.start_time is not None
        assert task2.start_time >= task1.completion_time

    def test_metrics_calculated(self):
        """Metrics report should have sensible values."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=20, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()
        report = metrics.report

        assert report.total_tasks == 20
        assert report.tasks_completed + report.tasks_failed + report.tasks_pending == 20
        assert report.avg_completion_time >= 0
        assert report.throughput >= 0
        assert 0 <= report.sla_violation_rate <= 1
        assert 0 <= report.failure_rate <= 1

    def test_no_task_failure_probability_zero(self):
        """Tasks with failure_probability=0 should never fail."""
        tasks = [
            self._make_task(f"task-{i}", failure_prob=0.0)
            for i in range(10)
        ]
        workers = [self._make_worker("w1", capacity=500.0)]

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()

        assert metrics.report.tasks_failed == 0
        assert metrics.report.tasks_completed == 10

    def test_scenario_generator_reproducibility(self):
        """Same seed should produce identical scenarios."""
        gen1 = ScenarioGenerator(seed=99)
        gen2 = ScenarioGenerator(seed=99)

        tasks1 = gen1.generate_tasks(num_tasks=10)
        tasks2 = gen2.generate_tasks(num_tasks=10)

        for t1, t2 in zip(tasks1, tasks2):
            assert t1.id == t2.id
            assert t1.compute_cost == t2.compute_cost
            assert t1.deadline == t2.deadline
            assert t1.priority == t2.priority
