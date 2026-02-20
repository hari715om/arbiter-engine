"""Base Scheduler — abstract interface for all scheduling algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from arbiter.models.task import Task
from arbiter.models.worker import Worker


@dataclass(frozen=True)
class Assignment:
    """Immutable scheduling decision: assign a task to a worker."""
    task_id: str
    worker_id: str
    scheduled_time: float


class BaseScheduler(ABC):
    """Abstract base class for all schedulers. Subclasses implement schedule()."""

    @abstractmethod
    def schedule(
        self,
        tasks: list[Task],
        workers: list[Worker],
        completed_task_ids: set[str],
    ) -> list[Assignment]:
        """Return task-to-worker assignments for the current state."""
        ...

    @property
    def name(self) -> str:
        """Human-readable scheduler name for reports."""
        return self.__class__.__name__
