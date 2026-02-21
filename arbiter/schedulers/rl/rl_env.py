import numpy as np
from dataclasses import dataclass
from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus


@dataclass
class SchedulingState:
    """Observation vector for the RL agent."""
    queue_depth: int
    avg_worker_load: float
    task_priority: int
    task_cost: float
    task_deadline_pressure: float  # (deadline - now) / estimated_duration
    task_failure_prob: float
    task_dependency_count: int
    task_retry_count: int
    worker_cpu_free: list[float]
    worker_speeds: list[float]
    worker_reliability: list[float]

    def to_vector(self) -> np.ndarray:
        base = [
            self.queue_depth / 100.0,  # normalize
            self.avg_worker_load,
            self.task_priority / 10.0,
            min(self.task_cost / 10.0, 1.0),
            min(max(self.task_deadline_pressure, 0), 5.0) / 5.0,
            self.task_failure_prob,
            min(self.task_dependency_count / 5.0, 1.0),
            self.task_retry_count / 3.0,
        ]
        # pad worker features to fixed size (max 20 workers)
        max_w = 20
        cpus = (self.worker_cpu_free + [0.0] * max_w)[:max_w]
        speeds = (self.worker_speeds + [0.0] * max_w)[:max_w]
        reliab = (self.worker_reliability + [0.0] * max_w)[:max_w]
        return np.array(base + cpus + speeds + reliab, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return 8 + 20 * 3  # 68


def build_state(
    task: Task,
    workers: list[Worker],
    all_tasks: list[Task],
    current_time: float,
    worker_reliability: dict[str, float] | None = None,
) -> SchedulingState:
    queued = sum(1 for t in all_tasks if t.status == TaskStatus.QUEUED)
    loads = [w.current_load / max(w.cpu_capacity, 0.01) for w in workers]
    avg_load = sum(loads) / len(loads) if loads else 0

    dp = (task.deadline - current_time) / max(task.estimated_duration, 0.01)
    reliability = worker_reliability or {}

    return SchedulingState(
        queue_depth=queued,
        avg_worker_load=avg_load,
        task_priority=task.priority,
        task_cost=task.compute_cost,
        task_deadline_pressure=dp,
        task_failure_prob=task.failure_probability,
        task_dependency_count=len(task.dependencies),
        task_retry_count=task.retry_count,
        worker_cpu_free=[max(0, w.cpu_capacity - w.current_load) for w in workers],
        worker_speeds=[w.speed_multiplier for w in workers],
        worker_reliability=[reliability.get(w.id, 1.0) for w in workers],
    )


def compute_reward(
    task: Task,
    completed: bool,
    sla_violated: bool,
    fairness_bonus: float = 0.0,
) -> float:
    if not completed:
        return -1.0
    reward = 1.0
    if sla_violated:
        reward -= 0.5
    reward += fairness_bonus * 0.2
    if task.retry_count > 0:
        reward -= 0.1 * task.retry_count
    return reward
