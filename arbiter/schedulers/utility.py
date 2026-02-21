"""Utility Functions — composable objectives for scheduling decisions.

Instead of ad-hoc scoring formulas scattered across schedulers, utility
functions provide a single formal framework for measuring assignment quality.

Each Objective answers one question:
  "How good is assigning task T to worker W, considering factor X?"

The UtilityFunction combines multiple objectives via weighted sum:
  total_utility = w1·latency + w2·throughput + w3·fairness + w4·cost + w5·risk

This is how production systems like Google Borg and AWS Spot make decisions —
every choice is evaluated against a global reward signal.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus


class Objective(ABC):
    """Base class for individual optimization objectives.

    Each objective produces a score in [0, 1] where:
    - 1.0 = best possible outcome for this objective
    - 0.0 = worst possible outcome
    """

    @abstractmethod
    def evaluate(self, task: Task, worker: Worker, context: dict) -> float:
        """Score this assignment for one specific objective."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class LatencyObjective(Objective):
    """Reward for meeting deadlines — more slack = higher score.

    Measures how much time buffer remains between when the task will
    finish and its deadline. Tasks about to miss their deadline get
    a score near 0; tasks with plenty of slack get near 1.

    Real-world analog: SLA compliance score in fleet management.
    """

    def evaluate(self, task: Task, worker: Worker, context: dict) -> float:
        current_time = context.get("current_time", 0.0)
        # Estimated completion: now + runtime / worker_speed
        est_runtime = task.estimated_duration / max(worker.speed_multiplier, 0.1)
        est_finish = current_time + est_runtime

        slack = task.deadline - est_finish
        total_budget = task.deadline - task.arrival_time

        if total_budget <= 0:
            return 0.0
        # Normalize: 1.0 = full budget remaining, 0.0 = at/past deadline
        return max(0.0, min(1.0, slack / total_budget))


class ThroughputObjective(Objective):
    """Reward for maximizing worker utilization — tighter capacity fit = higher score.

    Prefers assignments that fill workers efficiently rather than leaving
    large capacity gaps. This maximizes cluster throughput.

    Real-world analog: bin-packing efficiency in container orchestration.
    """

    def evaluate(self, task: Task, worker: Worker, context: dict) -> float:
        capacity = worker.available_capacity
        if capacity <= 0:
            return 0.0

        round_capacity = context.get("round_capacity", {})
        remaining = round_capacity.get(worker.id, capacity)
        if remaining <= 0:
            return 0.0

        # How well does this task fill the remaining capacity?
        # Perfect fit = 1.0, large gap = lower score
        fit_ratio = task.compute_cost / remaining
        return max(0.0, min(1.0, fit_ratio))


class FairnessObjective(Objective):
    """Penalize load imbalance — prefer underloaded workers.

    Uses relative load across the fleet: workers with lower current
    load get higher scores. This distributes work evenly, preventing
    hot spots and improving fault tolerance.

    Real-world analog: weighted least-connections in load balancers.
    """

    def evaluate(self, task: Task, worker: Worker, context: dict) -> float:
        workers = context.get("workers", [])
        if not workers:
            return 0.5

        loads = []
        for w in workers:
            if w.status != WorkerStatus.DOWN:
                load_ratio = w.current_load / w.cpu_capacity if w.cpu_capacity > 0 else 0
                loads.append(load_ratio)

        if not loads:
            return 0.5

        # Worker's load ratio
        worker_load = worker.current_load / worker.cpu_capacity if worker.cpu_capacity > 0 else 0
        max_load = max(loads) if loads else 1.0

        # Score: less loaded workers get higher scores
        if max_load <= 0:
            return 1.0
        return max(0.0, 1.0 - worker_load / max(max_load, 0.01))


class CostObjective(Objective):
    """Penalize wasted compute — high failure risk × long duration = expensive waste.

    Estimates the "expected waste" of an assignment: if the task is likely
    to fail, the compute time is wasted. Prefer low-risk assignments.

    Real-world analog: spot instance cost optimization in cloud computing.
    """

    def evaluate(self, task: Task, worker: Worker, context: dict) -> float:
        worker_reliability = context.get("worker_reliability", {})
        rel = worker_reliability.get(worker.id, 1.0)

        # Expected waste = failure_prob × duration
        combined_fail = task.failure_probability + (1.0 - rel) * 0.3
        combined_fail = min(combined_fail, 0.95)

        # Score: lower failure probability = higher score
        return 1.0 - combined_fail


class RiskObjective(Objective):
    """Expected utility under failure uncertainty.

    Discounts the assignment's value by the probability of failure:
      E[utility] = (1 - p_fail) × base_reward

    Also considers retry count — tasks that have already failed multiple
    times are riskier and should be assigned to more reliable workers.

    Real-world analog: risk-adjusted return in portfolio optimization.
    """

    def evaluate(self, task: Task, worker: Worker, context: dict) -> float:
        worker_reliability = context.get("worker_reliability", {})
        rel = worker_reliability.get(worker.id, 1.0)

        # Base failure probability
        combined_fail = task.failure_probability + (1.0 - rel) * 0.3
        combined_fail = min(combined_fail, 0.95)

        # Retry penalty: each retry makes the task riskier
        retry_penalty = 1.0 - (task.retry_count * 0.15)
        retry_penalty = max(0.3, retry_penalty)

        # Expected success rate
        return (1.0 - combined_fail) * retry_penalty


@dataclass
class ObjectiveWeight:
    """Named weight for an objective."""
    objective: Objective
    weight: float


class UtilityFunction:
    """Combines multiple objectives into a single reward signal.

    Usage:
        utility = UtilityFunction([
            ObjectiveWeight(LatencyObjective(), 0.3),
            ObjectiveWeight(ThroughputObjective(), 0.2),
            ObjectiveWeight(FairnessObjective(), 0.2),
            ObjectiveWeight(CostObjective(), 0.15),
            ObjectiveWeight(RiskObjective(), 0.15),
        ])
        score = utility.evaluate(task, worker, context)
    """

    def __init__(self, objective_weights: list[ObjectiveWeight]):
        self.objective_weights = objective_weights
        # Normalize weights so they sum to 1.0
        total = sum(ow.weight for ow in objective_weights)
        if total > 0:
            for ow in self.objective_weights:
                ow.weight /= total

    def evaluate(self, task: Task, worker: Worker, context: dict) -> float:
        """Compute total utility: weighted sum of normalized objectives."""
        total = 0.0
        for ow in self.objective_weights:
            score = ow.objective.evaluate(task, worker, context)
            total += ow.weight * score
        return total

    def evaluate_breakdown(self, task: Task, worker: Worker, context: dict) -> dict[str, float]:
        """Return per-objective scores for debugging/analysis."""
        breakdown = {}
        for ow in self.objective_weights:
            name = ow.objective.name
            score = ow.objective.evaluate(task, worker, context)
            breakdown[name] = score
        breakdown["total"] = self.evaluate(task, worker, context)
        return breakdown

    @property
    def weights(self) -> dict[str, float]:
        """Return current weight configuration."""
        return {ow.objective.name: ow.weight for ow in self.objective_weights}

    @classmethod
    def default(cls) -> "UtilityFunction":
        """Create utility function with balanced default weights."""
        return cls([
            ObjectiveWeight(LatencyObjective(), 0.30),
            ObjectiveWeight(ThroughputObjective(), 0.20),
            ObjectiveWeight(FairnessObjective(), 0.15),
            ObjectiveWeight(CostObjective(), 0.15),
            ObjectiveWeight(RiskObjective(), 0.20),
        ])
