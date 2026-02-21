"""ML-Enhanced Scheduler — heuristic scheduler augmented with ML predictions."""

from typing import Optional

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker
from arbiter.schedulers.heuristic import HeuristicScheduler
from arbiter.schedulers.base import Assignment
from arbiter.ml.models import RuntimePredictor, FailureClassifier


class MLScheduler(HeuristicScheduler):
    """Extends HeuristicScheduler with ML-predicted runtime and failure risk."""

    def __init__(
        self,
        runtime_predictor: Optional[RuntimePredictor] = None,
        failure_classifier: Optional[FailureClassifier] = None,
        w_priority: float = 0.3,
        w_deadline: float = 0.3,
        w_unlock: float = 0.2,
        w_risk: float = 0.2,
        current_time: float = 0.0,
    ):
        super().__init__(
            w_priority=w_priority,
            w_deadline=w_deadline,
            w_unlock=w_unlock,
            current_time=current_time,
        )
        self.runtime_predictor = runtime_predictor or RuntimePredictor()
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.w_risk = w_risk

    def _score_task(self, task: Task, dependents_map: dict[str, int]) -> float:
        """Score task using ML-predicted runtime for urgency calculation."""
        priority_score = task.priority / 10.0

        # Use ML-predicted runtime instead of estimated_duration for urgency
        predicted_runtime = self.runtime_predictor.predict(task, self._dummy_worker())
        time_remaining = max(task.deadline - self.current_time, 0.01)
        urgency_score = min(1.0, predicted_runtime / time_remaining)

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
        """Score worker factoring in ML-predicted failure risk."""
        base_score = super()._score_worker(task, worker, round_capacity)
        if base_score < 0:
            return base_score

        # Penalize risky task-worker pairs
        predicted_failure = self.failure_classifier.predict_proba(task, worker)
        risk_penalty = 1.0 - predicted_failure  # 0.0 = always fails, 1.0 = never fails

        return (1.0 - self.w_risk) * base_score + self.w_risk * risk_penalty

    def _dummy_worker(self) -> Worker:
        """Dummy worker for task-only predictions (average worker stats)."""
        return Worker(
            id="__dummy__", cpu_capacity=100.0, memory_capacity=512.0,
            speed_multiplier=1.0,
        )

    @classmethod
    def from_model_dir(cls, model_dir: str, **kwargs) -> "MLScheduler":
        """Load pre-trained models from a directory."""
        from pathlib import Path
        model_path = Path(model_dir)

        runtime_pred = RuntimePredictor()
        failure_clf = FailureClassifier()

        runtime_file = model_path / "runtime_predictor.joblib"
        failure_file = model_path / "failure_classifier.joblib"

        if runtime_file.exists():
            runtime_pred.load(str(runtime_file))
        if failure_file.exists():
            failure_clf.load(str(failure_file))

        return cls(
            runtime_predictor=runtime_pred,
            failure_classifier=failure_clf,
            **kwargs,
        )
