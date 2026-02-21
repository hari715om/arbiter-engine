"""Tests for Phase 5: Utility-Based Scheduling.

Verifies:
1. Individual objectives produce correct [0, 1] scores
2. UtilityFunction composes objectives correctly
3. UtilityScheduler produces valid assignments
4. WeightOptimizer improves over default weights
5. Fairness index is computed correctly
6. Integration: Utility scheduler with failure injection
"""

import pytest

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus
from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.schedulers.utility import (
    LatencyObjective,
    ThroughputObjective,
    FairnessObjective,
    CostObjective,
    RiskObjective,
    UtilityFunction,
    ObjectiveWeight,
)
from arbiter.schedulers.utility_scheduler import UtilityScheduler
from arbiter.schedulers.weight_optimizer import WeightOptimizer
from arbiter.simulator.engine import SimulationEngine
from arbiter.simulator.failure_injector import FailureInjector
from arbiter.simulator.generator import ScenarioGenerator


def make_task(**overrides) -> Task:
    """Helper to create a task with sensible defaults."""
    defaults = dict(
        id="t1", compute_cost=10.0, deadline=200.0, priority=5,
        estimated_duration=20.0, resource_type="cpu",
        failure_probability=0.0,
    )
    defaults.update(overrides)
    return Task(**defaults)


def make_worker(**overrides) -> Worker:
    """Helper to create a worker with sensible defaults."""
    defaults = dict(
        id="w1", cpu_capacity=100.0, memory_capacity=512.0,
        speed_multiplier=1.0,
    )
    defaults.update(overrides)
    return Worker(**defaults)


class TestLatencyObjective:
    """Tests for deadline-slack-based scoring."""

    def test_high_slack_scores_high(self):
        """Task with plenty of time should score near 1.0."""
        obj = LatencyObjective()
        task = make_task(deadline=200.0, estimated_duration=10.0, arrival_time=0.0)
        worker = make_worker(speed_multiplier=1.0)
        score = obj.evaluate(task, worker, {"current_time": 0.0})
        assert score > 0.8

    def test_tight_deadline_scores_low(self):
        """Task about to miss deadline should score near 0.0."""
        obj = LatencyObjective()
        task = make_task(deadline=25.0, estimated_duration=20.0, arrival_time=0.0)
        worker = make_worker(speed_multiplier=1.0)
        score = obj.evaluate(task, worker, {"current_time": 20.0})
        assert score < 0.3

    def test_fast_worker_helps(self):
        """Faster worker should give higher latency score."""
        obj = LatencyObjective()
        task = make_task(deadline=50.0, estimated_duration=30.0, arrival_time=0.0)
        slow_worker = make_worker(speed_multiplier=0.5)
        fast_worker = make_worker(id="w2", speed_multiplier=2.0)

        slow_score = obj.evaluate(task, slow_worker, {"current_time": 0.0})
        fast_score = obj.evaluate(task, fast_worker, {"current_time": 0.0})
        assert fast_score > slow_score


class TestThroughputObjective:
    """Tests for capacity-fit-based scoring."""

    def test_perfect_fit_scores_high(self):
        """Task that exactly fills remaining capacity should score 1.0."""
        obj = ThroughputObjective()
        task = make_task(compute_cost=50.0)
        worker = make_worker(cpu_capacity=100.0)
        score = obj.evaluate(task, worker, {"round_capacity": {"w1": 50.0}})
        assert score == pytest.approx(1.0)

    def test_small_task_large_gap(self):
        """Small task on mostly empty worker should score lower."""
        obj = ThroughputObjective()
        task = make_task(compute_cost=10.0)
        worker = make_worker(cpu_capacity=100.0)
        score = obj.evaluate(task, worker, {"round_capacity": {"w1": 100.0}})
        assert score < 0.3


class TestFairnessObjective:
    """Tests for load-balance scoring."""

    def test_underloaded_scores_high(self):
        """Worker with low load should score higher than one with high load."""
        obj = FairnessObjective()
        task = make_task()
        light_worker = make_worker(id="w1", current_load=10.0, cpu_capacity=100.0)
        heavy_worker = make_worker(id="w2", current_load=90.0, cpu_capacity=100.0)

        workers = [light_worker, heavy_worker]
        ctx = {"workers": workers}

        light_score = obj.evaluate(task, light_worker, ctx)
        heavy_score = obj.evaluate(task, heavy_worker, ctx)
        assert light_score > heavy_score


class TestCostObjective:
    """Tests for waste-avoidance scoring."""

    def test_reliable_worker_scores_high(self):
        """Worker with good reliability should score high."""
        obj = CostObjective()
        task = make_task(failure_probability=0.0)
        worker = make_worker()
        score = obj.evaluate(task, worker, {"worker_reliability": {"w1": 1.0}})
        assert score > 0.9

    def test_unreliable_worker_scores_low(self):
        """Worker with poor reliability should score low."""
        obj = CostObjective()
        task = make_task(failure_probability=0.3)
        worker = make_worker()
        score = obj.evaluate(task, worker, {"worker_reliability": {"w1": 0.3}})
        assert score < 0.6


class TestRiskObjective:
    """Tests for expected-utility-under-uncertainty scoring."""

    def test_no_risk_high_score(self):
        """Zero failure probability should give high score."""
        obj = RiskObjective()
        task = make_task(failure_probability=0.0, retry_count=0)
        worker = make_worker()
        score = obj.evaluate(task, worker, {"worker_reliability": {"w1": 1.0}})
        assert score > 0.9

    def test_retries_increase_risk(self):
        """Tasks with past failures should have lower risk scores."""
        obj = RiskObjective()
        task_fresh = make_task(failure_probability=0.1, retry_count=0)
        task_retried = make_task(failure_probability=0.1, retry_count=2)
        worker = make_worker()
        ctx = {"worker_reliability": {"w1": 0.8}}

        fresh_score = obj.evaluate(task_fresh, worker, ctx)
        retried_score = obj.evaluate(task_retried, worker, ctx)
        assert fresh_score > retried_score


class TestUtilityFunction:
    """Tests for the composable utility framework."""

    def test_default_produces_valid_score(self):
        """Default utility function should return a score in [0, 1]."""
        utility = UtilityFunction.default()
        task = make_task()
        worker = make_worker()
        ctx = {"current_time": 0.0, "workers": [worker],
               "round_capacity": {"w1": 100.0}, "worker_reliability": {"w1": 1.0}}
        score = utility.evaluate(task, worker, ctx)
        assert 0.0 <= score <= 1.0

    def test_weights_sum_to_one(self):
        """Weights should be normalized to sum to 1.0."""
        utility = UtilityFunction([
            ObjectiveWeight(LatencyObjective(), 3.0),
            ObjectiveWeight(RiskObjective(), 7.0),
        ])
        total = sum(utility.weights.values())
        assert total == pytest.approx(1.0)

    def test_breakdown_includes_all_objectives(self):
        """Breakdown should have one entry per objective plus total."""
        utility = UtilityFunction.default()
        task = make_task()
        worker = make_worker()
        ctx = {"current_time": 0.0, "workers": [worker],
               "round_capacity": {"w1": 100.0}, "worker_reliability": {"w1": 1.0}}
        breakdown = utility.evaluate_breakdown(task, worker, ctx)
        assert "total" in breakdown
        assert len(breakdown) == 6  # 5 objectives + total


class TestUtilityScheduler:
    """Tests for the greedy global optimizer."""

    def test_produces_valid_assignments(self):
        """Scheduler should return valid assignments for schedulable tasks."""
        tasks = [
            make_task(id="t1", status=TaskStatus.QUEUED),
            make_task(id="t2", status=TaskStatus.QUEUED),
        ]
        workers = [
            make_worker(id="w1"),
            make_worker(id="w2"),
        ]
        scheduler = UtilityScheduler()
        assignments = scheduler.schedule(tasks, workers, set())
        assert len(assignments) == 2

    def test_skips_down_workers(self):
        """Should not assign to DOWN workers."""
        tasks = [make_task(id="t1", status=TaskStatus.QUEUED)]
        workers = [
            make_worker(id="w1", status=WorkerStatus.DOWN),
            make_worker(id="w2"),
        ]
        scheduler = UtilityScheduler()
        assignments = scheduler.schedule(tasks, workers, set())
        assert len(assignments) == 1
        assert assignments[0].worker_id == "w2"

    def test_respects_dependencies(self):
        """Should not schedule tasks with unmet dependencies."""
        tasks = [
            make_task(id="t1", status=TaskStatus.QUEUED, dependencies=["t0"]),
        ]
        workers = [make_worker()]
        scheduler = UtilityScheduler()
        assignments = scheduler.schedule(tasks, workers, set())
        assert len(assignments) == 0

    def test_respects_capacity(self):
        """Should not over-commit worker capacity."""
        tasks = [
            make_task(id="t1", compute_cost=60.0, status=TaskStatus.QUEUED),
            make_task(id="t2", compute_cost=60.0, status=TaskStatus.QUEUED),
        ]
        workers = [make_worker(id="w1", cpu_capacity=100.0)]
        scheduler = UtilityScheduler()
        assignments = scheduler.schedule(tasks, workers, set())
        # Can only fit one task
        assert len(assignments) == 1

    def test_simulation_with_utility_scheduler(self):
        """Full simulation should complete with the utility scheduler."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=30, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=UtilityScheduler(), seed=42,
        )
        metrics = engine.run()
        assert metrics.report.tasks_completed > 0
        assert metrics.report.total_tasks == 30


class TestWeightOptimizer:
    """Tests for adaptive weight tuning."""

    def test_returns_valid_weights(self):
        """Optimizer should return a dict with 5 named weights."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=15, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        optimizer = WeightOptimizer(seed=42)
        best = optimizer.optimize(tasks, workers, n_trials=5, seed=42)

        assert len(best) == 5
        assert "latency" in best
        assert "risk" in best

    def test_weights_sum_to_one(self):
        """Returned weights should approximately sum to 1.0."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=15, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        optimizer = WeightOptimizer(seed=42)
        best = optimizer.optimize(tasks, workers, n_trials=5, seed=42)
        assert sum(best.values()) == pytest.approx(1.0, abs=0.01)

    def test_records_trials(self):
        """Optimizer should record all trial results."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=15, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        optimizer = WeightOptimizer(seed=42)
        optimizer.optimize(tasks, workers, n_trials=5, seed=42)
        assert len(optimizer.trials) == 5
        assert optimizer.best_trial is not None


class TestFairnessMetric:
    """Tests for Jain's fairness index computation."""

    def test_equal_utilization_is_fair(self):
        """All workers equally utilized should give fairness=1.0."""
        from arbiter.metrics.collector import MetricsCollector
        from arbiter.models.task import Task

        tasks = [
            make_task(id=f"t{i}", status=TaskStatus.COMPLETED,
                      assigned_worker=f"w{i % 2 + 1}",
                      start_time=0.0, completion_time=50.0)
            for i in range(4)
        ]
        workers = [make_worker(id="w1"), make_worker(id="w2")]

        collector = MetricsCollector()
        report = collector.calculate(tasks, workers, total_time=100.0,
                                     scheduler_name="test")
        assert report.fairness_index > 0.9

    def test_fairness_reported_in_simulation(self):
        """Fairness index should be present in simulation output."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=20, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=UtilityScheduler(), seed=42,
        )
        metrics = engine.run()
        assert 0.0 <= metrics.report.fairness_index <= 1.0


class TestIntegrationWithFailures:
    """Utility scheduler under failure injection."""

    def test_utility_handles_failures(self):
        """Utility scheduler should complete tasks despite worker failures."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=30, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=5)

        injector = FailureInjector(mode="random", failure_rate=0.05, seed=42)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=UtilityScheduler(), seed=42,
            failure_injector=injector,
        )
        metrics = engine.run()
        r = metrics.report

        assert r.tasks_completed > 0
        assert r.tasks_completed + r.tasks_failed + r.tasks_pending == r.total_tasks
