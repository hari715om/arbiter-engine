from dataclasses import dataclass, field
from arbiter.models.task import Task
from arbiter.models.worker import Worker


@dataclass
class SchedulingExplanation:
    task_id: str
    worker_id: str
    scheduler_name: str
    total_score: float
    factors: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    alternatives: list[dict] = field(default_factory=list)


def explain_utility_assignment(
    task: Task,
    chosen_worker: Worker,
    all_workers: list[Worker],
    current_time: float = 0.0,
    worker_reliability: dict[str, float] | None = None,
) -> SchedulingExplanation:
    from arbiter.schedulers.utility import (
        UtilityFunction, LatencyObjective, ThroughputObjective,
        FairnessObjective, CostObjective, RiskObjective,
    )

    objectives = [
        LatencyObjective(), ThroughputObjective(), FairnessObjective(),
        CostObjective(), RiskObjective(),
    ]
    uf = UtilityFunction(objectives=objectives)
    reliability = worker_reliability or {}

    # score every worker for comparison
    scores = {}
    for w in all_workers:
        total = 0.0
        breakdown = {}
        for obj in objectives:
            s = obj.score(task, w, current_time, reliability.get(w.id, 1.0))
            weighted = s * obj.weight
            breakdown[obj.__class__.__name__] = round(weighted, 3)
            total += weighted
        scores[w.id] = {"total": round(total, 3), "breakdown": breakdown}

    chosen_scores = scores.get(chosen_worker.id, {})
    factors = chosen_scores.get("breakdown", {})
    total = chosen_scores.get("total", 0)

    # build human-readable reasoning
    sorted_factors = sorted(factors.items(), key=lambda x: -abs(x[1]))
    top = sorted_factors[:3]
    parts = [f"{name}={val:.2f}" for name, val in top]
    reasoning = f"Assigned to {chosen_worker.id} (score={total:.2f}). Top factors: {', '.join(parts)}."

    # alternatives
    alts = []
    for wid, data in sorted(scores.items(), key=lambda x: -x[1]["total"]):
        if wid != chosen_worker.id:
            alts.append({"worker_id": wid, "score": data["total"], "breakdown": data["breakdown"]})
    alts = alts[:3]

    return SchedulingExplanation(
        task_id=task.id,
        worker_id=chosen_worker.id,
        scheduler_name="UtilityScheduler",
        total_score=total,
        factors=factors,
        reasoning=reasoning,
        alternatives=alts,
    )


def explain_heuristic_assignment(
    task: Task,
    chosen_worker: Worker,
    all_workers: list[Worker],
    all_tasks: list[Task],
    completed_ids: set[str],
) -> SchedulingExplanation:
    priority_score = task.priority / 10.0
    urgency = max(0, 1.0 - (task.deadline - task.estimated_duration) / max(task.deadline, 0.01))

    blocked_by_task = sum(
        1 for t in all_tasks
        if t.id in task.dependencies and t.id not in completed_ids
    )
    unlock_score = 0.0
    for other in all_tasks:
        if task.id in other.dependencies:
            unlock_score += 0.1

    total = priority_score * 0.4 + urgency * 0.3 + unlock_score * 0.3
    factors = {
        "priority": round(priority_score * 0.4, 3),
        "urgency": round(urgency * 0.3, 3),
        "unlock_potential": round(unlock_score * 0.3, 3),
    }

    reasoning = (
        f"Assigned to {chosen_worker.id}. "
        f"Priority={task.priority}/10, urgency={urgency:.2f}, "
        f"unlocks {int(unlock_score/0.1)} downstream tasks."
    )

    return SchedulingExplanation(
        task_id=task.id,
        worker_id=chosen_worker.id,
        scheduler_name="HeuristicScheduler",
        total_score=round(total, 3),
        factors=factors,
        reasoning=reasoning,
    )
