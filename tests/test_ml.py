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

        # Check all expected feature columns
        expected_keys = [
            "compute_cost", "estimated_duration", "priority",
            "failure_probability", "dependency_count", "resource_type_encoded",
            "worker_speed", "worker_capacity", "worker_load_ratio",
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

    def test_linear_regression(self, training_data):
        """Linear model should also train successfully."""
        predictor = RuntimePredictor(model_type="linear")
        metrics = predictor.train(training_data)

        assert predictor.is_trained
        assert "rmse" in metrics

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
        # Train models
        data_gen = TrainingDataGenerator(base_seed=99)
        data = data_gen.generate(num_simulations=5, tasks_per_sim=20, workers_per_sim=3)

        runtime_pred = RuntimePredictor()
        runtime_pred.train(data)

        failure_clf = FailureClassifier()
        failure_clf.train(data)

        # Run simulation with trained models
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
