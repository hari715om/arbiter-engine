import numpy as np
import logging
from arbiter.schedulers.base import BaseScheduler, Assignment
from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker

logger = logging.getLogger(__name__)


def compute_workload_fingerprint(tasks: list[Task], workers: list[Worker]) -> np.ndarray:
    queued = [t for t in tasks if t.status == TaskStatus.QUEUED]
    if not queued:
        return np.zeros(5, dtype=np.float32)

    priorities = [t.priority for t in queued]
    costs = [t.compute_cost for t in queued]
    deadlines = [t.deadline for t in queued]
    fail_probs = [t.failure_probability for t in queued]
    dep_counts = [len(t.dependencies) for t in queued]

    # 5-dim fingerprint
    burstiness = len(queued) / max(len(workers), 1)  # queue pressure
    priority_skew = np.std(priorities) / max(np.mean(priorities), 0.01)
    avg_fail_rate = np.mean(fail_probs) if fail_probs else 0
    dependency_density = np.mean(dep_counts) / max(len(queued), 1)
    deadline_tightness = np.mean([
        max(0, 1.0 - (d - t.estimated_duration) / max(d, 0.01))
        for t, d in zip(queued, deadlines)
    ])

    return np.array([
        min(burstiness / 10.0, 1.0),
        min(priority_skew, 1.0),
        avg_fail_rate,
        min(dependency_density, 1.0),
        deadline_tightness,
    ], dtype=np.float32)


class MetaScheduler(BaseScheduler):
    """Dynamically selects the best scheduler based on workload characteristics."""

    def __init__(self, schedulers: dict[str, BaseScheduler] = None):
        from arbiter.schedulers.fifo import FIFOScheduler
        from arbiter.schedulers.heuristic import HeuristicScheduler
        from arbiter.schedulers.utility_scheduler import UtilityScheduler

        self._schedulers = schedulers or {
            "fifo": FIFOScheduler(),
            "heuristic": HeuristicScheduler(),
            "utility": UtilityScheduler(),
        }
        # learned mapping: fingerprint region → best scheduler
        # initialized with sensible defaults
        self._strategy_rules = [
            # (condition_fn, scheduler_key, reason)
            (lambda fp: fp[0] < 0.2 and fp[2] < 0.1, "fifo", "low load, low failure → FIFO sufficient"),
            (lambda fp: fp[2] > 0.3, "utility", "high failure rate → utility handles risk"),
            (lambda fp: fp[4] > 0.5, "utility", "tight deadlines → utility optimizes latency"),
            (lambda fp: fp[0] > 0.5, "utility", "high queue pressure → utility balances load"),
            (lambda fp: fp[3] > 0.3, "heuristic", "complex dependencies → heuristic unlocks chains"),
        ]
        self._current_scheduler = "utility"
        self._switch_history: list[dict] = []

    def schedule(self, tasks: list[Task], workers: list[Worker],
                 completed_task_ids: set[str]) -> list[Assignment]:
        fp = compute_workload_fingerprint(tasks, workers)
        chosen = self._select_strategy(fp)

        if chosen != self._current_scheduler:
            logger.info("Meta-scheduler switching: %s → %s", self._current_scheduler, chosen)
            self._switch_history.append({
                "from": self._current_scheduler,
                "to": chosen,
                "fingerprint": fp.tolist(),
                "reason": self._last_reason,
            })
            self._current_scheduler = chosen

        return self._schedulers[self._current_scheduler].schedule(
            tasks, workers, completed_task_ids
        )

    def _select_strategy(self, fingerprint: np.ndarray) -> str:
        self._last_reason = "default"
        for condition, key, reason in self._strategy_rules:
            try:
                if condition(fingerprint):
                    self._last_reason = reason
                    return key
            except Exception:
                continue
        return "utility"

    @property
    def switch_history(self) -> list[dict]:
        return self._switch_history

    @property
    def current_strategy(self) -> str:
        return self._current_scheduler

    @property
    def name(self) -> str:
        return f"MetaScheduler({self._current_scheduler})"
