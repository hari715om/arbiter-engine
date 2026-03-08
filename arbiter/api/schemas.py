from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TaskCreate(BaseModel):
    id: Optional[str] = None
    compute_cost: float = Field(gt=0)
    resource_type: str = "cpu"
    deadline: float = Field(gt=0)
    priority: int = Field(ge=1, le=10, default=5)
    failure_probability: float = Field(ge=0, le=1, default=0.0)
    estimated_duration: float = Field(gt=0)
    dependencies: list[str] = Field(default_factory=list)
    max_retries: int = Field(ge=0, default=2)
    webhook_url: Optional[str] = None   # POST to this URL when task completes


class TaskResponse(BaseModel):
    id: str
    compute_cost: float
    resource_type: str
    deadline: float
    priority: int
    status: str
    assigned_worker: Optional[str]
    estimated_duration: float
    retry_count: int
    dependencies: list[str]
    start_time: Optional[float]
    completion_time: Optional[float]
    created_at: Optional[datetime]
    tenant_id: str = "default"


class WorkerCreate(BaseModel):
    id: Optional[str] = None
    cpu_capacity: float = Field(gt=0)
    memory_capacity: float = Field(gt=0, default=16.0)
    speed_multiplier: float = Field(gt=0, default=1.0)
    supported_resources: list[str] = Field(default_factory=lambda: ["cpu", "gpu", "memory"])


class WorkerResponse(BaseModel):
    id: str
    cpu_capacity: float
    memory_capacity: float
    speed_multiplier: float
    status: str
    current_load: float
    supported_resources: list[str]
    last_heartbeat: Optional[datetime]
    tenant_id: str = "default"


class MetricsSnapshot(BaseModel):
    total_tasks: int
    completed: int
    failed: int
    pending: int
    running: int
    queue_depth: int
    avg_latency: Optional[float]
    sla_violation_rate: Optional[float]
    worker_count: int
    active_workers: int


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    redis_connected: bool
    scheduler_type: str
    uptime_seconds: float


class EventResponse(BaseModel):
    event_type: str
    task_id: Optional[str]
    worker_id: Optional[str]
    timestamp: float
    detail: Optional[str]


class AlternativeAssignment(BaseModel):
    worker_id: str
    score: float
    breakdown: dict[str, float]


class ExplanationResponse(BaseModel):

    task_id: str
    worker_id: Optional[str]
    scheduler_name: str
    total_score: float
    factors: dict[str, float]
    reasoning: str
    alternatives: list[AlternativeAssignment]



class PolicyRule(BaseModel):
    """One conditional scheduling rule."""
    condition: str = Field(description="e.g. 'queue_depth > 100'")
    use: str = Field(description="Scheduler name: fifo, heuristic, utility, ml, rl, meta")


class PolicyResponse(BaseModel):
    default_scheduler: str
    rules: list[PolicyRule]
    utility_weights: dict[str, float]



class ChaosRequest(BaseModel):

    mode: str = Field(description="kill_worker | delay_tasks | fail_rate_spike")
    target: Optional[str] = Field(None, description="Worker ID (for kill_worker)")
    intensity: float = Field(0.5, ge=0.0, le=1.0,
                             description="Fraction of workers/tasks affected (0-1)")
    duration_seconds: Optional[float] = Field(
        None, description="Auto-recover after this many seconds (not yet implemented)"
    )


class ChaosResponse(BaseModel):
    mode: str
    affected: list[str]
    message: str
