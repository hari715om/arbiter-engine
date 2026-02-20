"""Event types for the discrete event simulation."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class EventType(str, Enum):
    """Types of events the simulation engine processes."""
    TASK_ARRIVAL = "task_arrival"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    WORKER_FAILURE = "worker_failure"
    WORKER_RECOVERY = "worker_recovery"
    SCHEDULE_TICK = "schedule_tick"


@dataclass(order=True)
class Event:
    """
    A single simulation event, ordered by time then sequence for heap ordering.
    Fields with compare=False are excluded from ordering (only time + sequence matter).
    """
    time: float
    sequence: int
    event_type: EventType = field(compare=False)
    task_id: Optional[str] = field(default=None, compare=False)
    worker_id: Optional[str] = field(default=None, compare=False)
    metadata: dict = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        parts = [f"Event(t={self.time:.2f}, type={self.event_type.value}"]
        if self.task_id:
            parts.append(f", task={self.task_id}")
        if self.worker_id:
            parts.append(f", worker={self.worker_id}")
        parts.append(")")
        return "".join(parts)
