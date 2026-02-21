"""Tests for the ML integration pipeline."""

import pytest

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker
from arbiter.ml.data_generator import TrainingDataGenerator
from arbiter.ml.models import RuntimePredictor, FailureClassifier
from arbiter.schedulers.ml_scheduler import MLScheduler
from arbiter.simulator.engine import SimulationEngine
from arbiter.simulator.generator import ScenarioGenerator


class TestTrainingDataGenerator:
    """Tests for simulation-based training data generation."""

    def test_generates_samples(self):
        """Should produce training samples from simulations."""
        gen = TrainingDataGenerator(base_seed=42)
        data = gen.generate(num_simulations=3, tasks_per_sim=20, workers_per_sim=3)

        assert len(data) > 0
        sample = data[0]

        expected_keys = [
            "compute_cost", "estimated_duration", "priority",
            "failure_probability", "dependency_count", "resource_type_encoded",
            "retry_count",
            "worker_speed", "worker_capacity", "worker_load_ratio",
            "worker_active_tasks", "worker_reliability",
            "queue_depth",
            "actual_runtime", "did_fail",
        ]
        for key in expected_keys:
            assert key in sample, f"Missing key: {key}"

    def test_targets_are_valid(self):
        """Targets should have correct types and ranges."""
        gen = TrainingDataGenerator(base_seed=42)
        data = gen.generate(num_simulations=3, tasks_per_sim=20, workers_per_sim=3)

        for sample in data:
            assert sample["actual_runtime"] >= 0
            assert sample["did_fail"] in (0, 1)

    def test_deterministic(self):
        """Same seed should produce same data."""
        gen1 = TrainingDataGenerator(base_seed=42)
        gen2 = TrainingDataGenerator(base_seed=42)

        data1 = gen1.generate(num_simulations=2, tasks_per_sim=10, workers_per_sim=2)
        data2 = gen2.generate(num_simulations=2, tasks_per_sim=10, workers_per_sim=2)

        assert len(data1) == len(data2)
        for d1, d2 in zip(data1, data2):
            assert d1 == d2

    def test_retry_data_collected(self):
        """Training data should include retry information."""
        gen = TrainingDataGenerator(base_seed=42)
        data = gen.generate(num_simulations=5, tasks_per_sim=30, workers_per_sim=3)

        retry_counts = [d["retry_count"] for d in data]
        assert all(r >= 0 for r in retry_counts)

    def test_worker_reliability_collected(self):
        """Training data should include worker reliability scores."""
        gen = TrainingDataGenerator(base_seed=42)
        data = gen.generate(num_simulations=5, tasks_per_sim=30, workers_per_sim=3)

        reliabilities = [d["worker_reliability"] for d in data]
        assert all(0 <= r <= 1.0 for r in reliabilities)


class TestRuntimePredictor:
    """Tests for the runtime prediction model."""

    @pytest.fixture
    def training_data(self):
        gen = TrainingDataGenerator(base_seed=42)
        return gen.generate(num_simulations=5, tasks_per_sim=30, workers_per_sim=3)

    def test_train_and_predict(self, training_data):
        """Model should train and produce positive predictions."""
        predictor = RuntimePredictor(model_type="random_forest")
        metrics = predictor.train(training_data)

        assert predictor.is_trained
        assert metrics["rmse"] > 0
        assert "r2" in metrics

        task = Task(
            id="t1", compute_cost=20.0, deadline=100.0, priority=5,
            estimated_duration=15.0, resource_type="cpu",
        )
        worker = Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)

        prediction = predictor.predict(task, worker)
        assert prediction > 0

    def test_predict_with_context(self, training_data):
        """Model should accept context dict for richer predictions."""
        predictor = RuntimePredictor()
        predictor.train(training_data)

        task = Task(
            id="t1", compute_cost=20.0, deadline=100.0, priority=5,
            estimated_duration=15.0, resource_type="cpu",
        )
        worker = Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)

        ctx = {"queue_depth": 50, "worker_reliability": 0.7}
        prediction = predictor.predict(task, worker, context=ctx)
        assert prediction > 0

    def test_fallback_when_untrained(self):
        """Untrained model should return estimated_duration."""
        predictor = RuntimePredictor()
        task = Task(
            id="t1", compute_cost=10.0, deadline=50.0, priority=5,
            estimated_duration=12.0, resource_type="cpu",
        )
        worker = Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)

        assert predictor.predict(task, worker) == 12.0


class TestFailureClassifier:
    """Tests for the failure classification model."""

    @pytest.fixture
    def training_data(self):
        gen = TrainingDataGenerator(base_seed=42)
        return gen.generate(num_simulations=10, tasks_per_sim=50, workers_per_sim=3)

    def test_train_and_predict(self, training_data):
        """Classifier should train and output probabilities in [0, 1]."""
        classifier = FailureClassifier()
        metrics = classifier.train(training_data)

        assert classifier.is_trained
        assert "precision" in metrics

        task = Task(
            id="t1", compute_cost=20.0, deadline=100.0, priority=5,
            estimated_duration=15.0, resource_type="cpu",
            failure_probability=0.1,
        )
        worker = Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)

        proba = classifier.predict_proba(task, worker)
        assert 0.0 <= proba <= 1.0

    def test_fallback_when_untrained(self):
        """Untrained classifier should return task.failure_probability."""
        classifier = FailureClassifier()
        task = Task(
            id="t1", compute_cost=10.0, deadline=50.0, priority=5,
            estimated_duration=12.0, resource_type="cpu",
            failure_probability=0.15,
        )
        worker = Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)

        assert classifier.predict_proba(task, worker) == 0.15


class TestMLScheduler:
    """Tests for the ML-enhanced scheduler."""

    def test_works_without_trained_models(self):
        """MLScheduler should fall back to heuristic behavior when untrained."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=10, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=2)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=MLScheduler(), seed=42,
        )
        metrics = engine.run()
        report = metrics.report

        assert report.total_tasks == 10
        assert report.tasks_completed > 0

    def test_works_with_trained_models(self):
        """MLScheduler should work with pre-trained models."""
        data_gen = TrainingDataGenerator(base_seed=99)
        data = data_gen.generate(num_simulations=5, tasks_per_sim=20, workers_per_sim=3)

        runtime_pred = RuntimePredictor()
        runtime_pred.train(data)

        failure_clf = FailureClassifier()
        failure_clf.train(data)

        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=15, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        scheduler = MLScheduler(
            runtime_predictor=runtime_pred,
            failure_classifier=failure_clf,
        )
        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=scheduler, seed=42,
        )
        metrics = engine.run()
        report = metrics.report

        assert report.total_tasks == 15
        assert report.tasks_completed + report.tasks_failed + report.tasks_pending == 15
        assert report.tasks_completed > 0


class TestRetrySystem:
    """Tests for the task retry system in the engine."""

    def test_tasks_can_retry(self):
        """Tasks with retries remaining should re-queue on failure."""
        task = Task(
            id="t1", compute_cost=10.0, deadline=500.0, priority=5,
            estimated_duration=5.0, resource_type="cpu",
            failure_probability=0.99, max_retries=3,
        )
        worker = Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)

        from arbiter.schedulers.fifo import FIFOScheduler
        engine = SimulationEngine(
            tasks=[task], workers=[worker],
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()

        final_task = engine.tasks["t1"]
        assert final_task.retry_count > 0

    def test_retry_limit_reached(self):
        """Tasks should permanently fail after max retries."""
        task = Task(
            id="t1", compute_cost=10.0, deadline=500.0, priority=5,
            estimated_duration=5.0, resource_type="cpu",
            failure_probability=0.99, max_retries=2,
        )
        worker = Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)

        from arbiter.schedulers.fifo import FIFOScheduler
        engine = SimulationEngine(
            tasks=[task], workers=[worker],
            scheduler=FIFOScheduler(), seed=42,
        )
        engine.run()

        final_task = engine.tasks["t1"]
        assert final_task.status == TaskStatus.FAILED
        assert final_task.retry_count <= 2


class TestWorkerReliability:
    """Tests for correlated worker failure patterns."""

    def test_reliability_degrades_on_failure(self):
        """Worker reliability should decrease when tasks fail on it."""
        task = Task(
            id="t1", compute_cost=10.0, deadline=500.0, priority=5,
            estimated_duration=5.0, resource_type="cpu",
            failure_probability=0.99, max_retries=0,
        )
        worker = Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)

        from arbiter.schedulers.fifo import FIFOScheduler
        engine = SimulationEngine(
            tasks=[task], workers=[worker],
            scheduler=FIFOScheduler(), seed=42,
        )
        engine.run()

        # Worker reliability should have degraded
        assert engine.worker_reliability["w1"] < 1.0

    def test_reliability_recovers_on_success(self):
        """Worker reliability should slowly recover on successful tasks."""
        # Use multiple low-risk tasks so recovery accumulates
        tasks = [
            Task(
                id=f"t{i}", compute_cost=10.0, deadline=500.0, priority=5,
                estimated_duration=5.0, resource_type="cpu",
                failure_probability=0.0,  # low base risk
            )
            for i in range(10)
        ]
        worker = Worker(id="w1", cpu_capacity=200.0, memory_capacity=512.0)

        from arbiter.schedulers.fifo import FIFOScheduler
        engine = SimulationEngine(
            tasks=tasks, workers=[worker],
            scheduler=FIFOScheduler(), seed=42,
        )
        # Degrade reliability slightly (small enough that combined risk stays low)
        engine.worker_reliability["w1"] = 0.95
        engine.run()

        # Reliability should have improved from 0.95 toward 1.0
        final_rel = engine.worker_reliability["w1"]
        assert final_rel > 0.95, f"Expected reliability to recover above 0.95, got {final_rel}"


class TestCostEfficiency:
    """Tests for failure cost metrics."""

    def test_cost_efficiency_tracked(self):
        """Cost efficiency should be between 0 and 1."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=20, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        from arbiter.schedulers.fifo import FIFOScheduler
        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()
        report = metrics.report

        assert 0.0 <= report.cost_efficiency <= 1.0
        assert report.total_retries >= 0
        assert report.total_wasted_time >= 0
