"""Worker model — a compute resource that executes tasks."""

from enum import Enum
from pydantic import BaseModel, Field


class WorkerStatus(str, Enum):
    """Operational states: IDLE ↔ BUSY → DOWN → IDLE"""
    IDLE = "idle"
    BUSY = "busy"
    DOWN = "down"


class Worker(BaseModel):
    """A machine/node with limited resources that executes tasks."""

    id: str = Field(description="Unique worker identifier")
    cpu_capacity: float = Field(gt=0, description="Max compute units simultaneously")
    memory_capacity: float = Field(gt=0, description="Max memory units available")
    current_load: float = Field(default=0.0, ge=0, description="Compute units currently in use")
    speed_multiplier: float = Field(default=1.0, gt=0, description="Runtime = duration / speed_multiplier")
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Failure probability per time unit")
    supported_resources: list[str] = Field(default_factory=lambda: ["cpu"], description="Resource types this worker handles")
    status: WorkerStatus = Field(default=WorkerStatus.IDLE, description="Current operational state")
    active_tasks: list[str] = Field(default_factory=list, description="Task IDs currently running")

    @property
    def available_capacity(self) -> float:
        """Remaining compute capacity."""
        return self.cpu_capacity - self.current_load

    @property
    def is_available(self) -> bool:
        """Can accept tasks if not DOWN and has spare capacity."""
        return self.status != WorkerStatus.DOWN and self.available_capacity > 0

    def can_handle(self, resource_type: str, compute_cost: float) -> bool:
        """Check if this worker can accept a task with given requirements."""
        return (
            self.status != WorkerStatus.DOWN
            and resource_type in self.supported_resources
            and self.available_capacity >= compute_cost
        )

    def assign_task(self, task_id: str, compute_cost: float) -> None:
        """Assign a task — increases load, sets status to BUSY."""
        self.current_load += compute_cost
        self.active_tasks.append(task_id)
        self.status = WorkerStatus.BUSY

    def release_task(self, task_id: str, compute_cost: float) -> None:
        """Release a task — decreases load, sets IDLE if no active tasks."""
        self.current_load = max(0.0, self.current_load - compute_cost)
        if task_id in self.active_tasks:
            self.active_tasks.remove(task_id)
        if not self.active_tasks:
            self.status = WorkerStatus.IDLE

    def __repr__(self) -> str:
        return (
            f"Worker(id={self.id!r}, load={self.current_load}/{self.cpu_capacity}, "
            f"status={self.status.value}, tasks={len(self.active_tasks)})"
        )
