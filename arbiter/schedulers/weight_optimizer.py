"""Weight Optimizer — finds optimal utility weights through simulation.

Instead of guessing weights (e.g., latency=0.3, risk=0.2), the optimizer
runs N trial simulations with different weight combinations and picks the
set that produces the best overall outcome.

This is a simple form of hyperparameter optimization — like grid search
for ML models, but applied to scheduling weights.

Real-world analog: A/B testing in production load balancers, or
Bayesian optimization for tuning Kubernetes scheduling parameters.
"""

import random
from dataclasses import dataclass

from arbiter.models.task import Task
from arbiter.models.worker import Worker
from arbiter.schedulers.utility import (
    UtilityFunction,
    ObjectiveWeight,
    LatencyObjective,
    ThroughputObjective,
    FairnessObjective,
    CostObjective,
    RiskObjective,
)
from arbiter.schedulers.utility_scheduler import UtilityScheduler


@dataclass
class TrialResult:
    """Result of a single weight configuration trial."""
    weights: dict[str, float]
    score: float
    tasks_completed: int
    sla_violation_rate: float
    fairness_index: float


class WeightOptimizer:
    """Finds optimal utility weights through simulated evaluation.

    Algorithm:
    1. Generate N random weight vectors (summing to 1.0)
    2. For each, run a simulation with those weights
    3. Score the outcome using a meta-objective:
       score = completed_ratio - sla_violations - (1 - fairness)
    4. Return the best-performing weights

    This is random search — simple but effective for 5 dimensions.
    For production use, you'd want Bayesian optimization (e.g., Optuna).
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.trials: list[TrialResult] = []

    def optimize(
        self,
        tasks: list[Task],
        workers: list[Worker],
        n_trials: int = 30,
        seed: int = 42,
    ) -> dict[str, float]:
        """Run trials and return best weight configuration.

        Args:
            tasks: Task list for evaluation.
            workers: Worker list for evaluation.
            n_trials: Number of random weight configs to try.
            seed: Simulation seed for reproducibility.

        Returns:
            Dict mapping objective name → optimal weight.
        """
        from arbiter.simulator.engine import SimulationEngine

        best_score = -float("inf")
        best_weights: dict[str, float] = {}

        for trial_idx in range(n_trials):
            # Generate random weights (5 objectives)
            raw = [self.rng.random() for _ in range(5)]
            total = sum(raw)
            weights = [w / total for w in raw]

            # Build utility function with these weights
            utility_fn = UtilityFunction([
                ObjectiveWeight(LatencyObjective(), weights[0]),
                ObjectiveWeight(ThroughputObjective(), weights[1]),
                ObjectiveWeight(FairnessObjective(), weights[2]),
                ObjectiveWeight(CostObjective(), weights[3]),
                ObjectiveWeight(RiskObjective(), weights[4]),
            ])

            scheduler = UtilityScheduler(utility_fn=utility_fn)

            # Run simulation
            engine = SimulationEngine(
                tasks=tasks,
                workers=workers,
                scheduler=scheduler,
                seed=seed,
            )
            metrics = engine.run()
            report = metrics.report

            # Compute meta-score
            completed_ratio = report.tasks_completed / max(report.total_tasks, 1)
            sla_rate = report.sla_violation_rate
            fairness = report.fairness_index if hasattr(report, 'fairness_index') else 0.5

            # Meta-objective: maximize completions, minimize SLA violations
            score = (
                0.4 * completed_ratio
                + 0.3 * (1.0 - sla_rate)
                + 0.2 * fairness
                + 0.1 * report.cost_efficiency
            )

            result = TrialResult(
                weights={
                    "latency": weights[0],
                    "throughput": weights[1],
                    "fairness": weights[2],
                    "cost": weights[3],
                    "risk": weights[4],
                },
                score=score,
                tasks_completed=report.tasks_completed,
                sla_violation_rate=sla_rate,
                fairness_index=fairness,
            )
            self.trials.append(result)

            if score > best_score:
                best_score = score
                best_weights = result.weights

        return best_weights

    @property
    def best_trial(self) -> TrialResult | None:
        """Return the trial with the highest score."""
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t.score)

    def make_utility_fn(self, weights: dict[str, float]) -> UtilityFunction:
        """Create a UtilityFunction from a weight dict."""
        return UtilityFunction([
            ObjectiveWeight(LatencyObjective(), weights.get("latency", 0.3)),
            ObjectiveWeight(ThroughputObjective(), weights.get("throughput", 0.2)),
            ObjectiveWeight(FairnessObjective(), weights.get("fairness", 0.15)),
            ObjectiveWeight(CostObjective(), weights.get("cost", 0.15)),
            ObjectiveWeight(RiskObjective(), weights.get("risk", 0.2)),
        ])
