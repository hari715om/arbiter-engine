"""Task model — the fundamental unit of work in Arbiter Engine."""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class TaskStatus(str, Enum):
    """Lifecycle states: PENDING → QUEUED → RUNNING → COMPLETED | FAILED"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """A unit of computation that needs to be scheduled onto a Worker."""

    id: str = Field(description="Unique task identifier")
    compute_cost: float = Field(gt=0, description="CPU units consumed while running")
    resource_type: str = Field(default="cpu", description="Required resource: cpu, gpu, memory")
    deadline: float = Field(gt=0, description="Must complete by this simulation time")
    priority: int = Field(ge=1, le=10, description="Scheduling priority (1=low, 10=high)")
    dependencies: list[str] = Field(default_factory=list, description="Task IDs that must complete first")
    failure_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Chance of failure during execution")
    estimated_duration: float = Field(gt=0, description="Expected execution time")
    arrival_time: float = Field(default=0.0, ge=0, description="When this task enters the system")
    start_time: Optional[float] = Field(default=None, description="When execution began")
    completion_time: Optional[float] = Field(default=None, description="When execution finished")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current lifecycle state")
    assigned_worker: Optional[str] = Field(default=None, description="Worker ID if assigned")

    @property
    def is_ready(self) -> bool:
        """Task is ready to schedule if it's QUEUED."""
        return self.status == TaskStatus.QUEUED

    @property
    def latency(self) -> Optional[float]:
        """Time from arrival to completion."""
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None

    @property
    def is_sla_violated(self) -> bool:
        """True if the task completed after its deadline."""
        if self.completion_time is not None:
            return self.completion_time > self.deadline
        return False

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id!r}, priority={self.priority}, "
            f"cost={self.compute_cost}, status={self.status.value})"
        )
