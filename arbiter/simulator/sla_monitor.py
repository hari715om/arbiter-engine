"""SLA Monitor — proactive deadline risk detection.

Models production SLA alerting systems (like Datadog monitors, PagerDuty
triggers, or Kubernetes liveness probes). Instead of waiting for deadlines
to be missed, the monitor periodically scans tasks and flags those at risk.
"""

from dataclasses import dataclass
from enum import Enum
from arbiter.models.task import Task, TaskStatus


class RiskLevel(str, Enum):
    """Severity levels for SLA risk alerts."""
    SAFE = "safe"           # On track to meet deadline
    WARNING = "warning"     # Less than 20% slack remaining
    CRITICAL = "critical"   # Will likely miss deadline


@dataclass
class SLARiskAlert:
    """A single SLA risk alert for a task."""
    task_id: str
    risk_level: RiskLevel
    deadline: float
    estimated_finish: float
    slack: float            # deadline - estimated_finish (negative = will miss)
    current_time: float


class SLAMonitor:
    """Detects tasks at risk of SLA violation before they breach.

    The monitor checks two types of tasks:
    - RUNNING: estimates completion based on elapsed time and estimated_duration
    - QUEUED: estimates delay based on queue position and average processing time
    """

    def __init__(self, warning_threshold: float = 0.2):
        """
        Args:
            warning_threshold: Fraction of remaining deadline time below which
                               a WARNING is raised. Default 0.2 = warn when
                               less than 20% of deadline budget remains.
        """
        self.warning_threshold = warning_threshold
        self.alerts_history: list[SLARiskAlert] = []

    def check(
        self,
        tasks: dict[str, Task],
        current_time: float,
    ) -> list[SLARiskAlert]:
        """Scan all active tasks and return risk alerts.

        Active = RUNNING or QUEUED (we don't check COMPLETED/FAILED/PENDING).

        For RUNNING tasks:
            estimated_finish = start_time + estimated_duration

        For QUEUED tasks:
            estimated_finish = current_time + estimated_duration
            (worst case: hasn't started yet, full duration remaining)
        """
        alerts: list[SLARiskAlert] = []

        for task in tasks.values():
            if task.status == TaskStatus.RUNNING:
                if task.start_time is not None:
                    # Estimate: how much time has passed vs expected duration
                    elapsed = current_time - task.start_time
                    remaining = max(0.0, task.estimated_duration - elapsed)
                    estimated_finish = current_time + remaining
                else:
                    estimated_finish = current_time + task.estimated_duration

            elif task.status == TaskStatus.QUEUED:
                # Worst case: still needs full duration to complete
                estimated_finish = current_time + task.estimated_duration

            else:
                continue  # Skip PENDING, COMPLETED, FAILED

            slack = task.deadline - estimated_finish
            total_budget = task.deadline - task.arrival_time

            if slack < 0:
                risk = RiskLevel.CRITICAL
            elif total_budget > 0 and (slack / total_budget) < self.warning_threshold:
                risk = RiskLevel.WARNING
            else:
                risk = RiskLevel.SAFE

            if risk != RiskLevel.SAFE:
                alert = SLARiskAlert(
                    task_id=task.id,
                    risk_level=risk,
                    deadline=task.deadline,
                    estimated_finish=estimated_finish,
                    slack=slack,
                    current_time=current_time,
                )
                alerts.append(alert)
                self.alerts_history.append(alert)

        return alerts

    @property
    def total_alerts(self) -> int:
        """Total number of alerts raised across all checks."""
        return len(self.alerts_history)

    @property
    def critical_count(self) -> int:
        """Number of CRITICAL alerts raised."""
        return sum(1 for a in self.alerts_history if a.risk_level == RiskLevel.CRITICAL)

    @property
    def warning_count(self) -> int:
        """Number of WARNING alerts raised."""
        return sum(1 for a in self.alerts_history if a.risk_level == RiskLevel.WARNING)
