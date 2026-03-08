import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Enum as SAEnum, ForeignKey, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker, relationship
from arbiter.api.config import settings


class Base(DeclarativeBase):
    pass


class TaskRecord(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True)
    compute_cost = Column(Float, nullable=False)
    resource_type = Column(String, default="cpu")
    deadline = Column(Float, nullable=False)
    priority = Column(Integer, default=5)
    failure_probability = Column(Float, default=0.0)
    estimated_duration = Column(Float, nullable=False)
    status = Column(String, default="pending")
    assigned_worker = Column(String, ForeignKey("workers.id"), nullable=True)
    arrival_time = Column(Float, default=0.0)
    start_time = Column(Float, nullable=True)
    completion_time = Column(Float, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=2)
    dependencies = Column(Text, default="")  # comma-separated task IDs
    webhook_url = Column(Text, nullable=True)  # POST to this URL on completion
    tenant_id = Column(String, default="default", nullable=False, index=True)  # Feature 1: multi-tenancy
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class WorkerRecord(Base):
    __tablename__ = "workers"

    id = Column(String, primary_key=True)
    cpu_capacity = Column(Float, nullable=False)
    memory_capacity = Column(Float, default=16.0)
    speed_multiplier = Column(Float, default=1.0)
    status = Column(String, default="idle")
    current_load = Column(Float, default=0.0)
    supported_resources = Column(Text, default="cpu,gpu,memory")
    last_heartbeat = Column(DateTime, nullable=True)
    tenant_id = Column(String, default="default", nullable=False, index=True)  # Feature 1: multi-tenancy
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    tasks = relationship("TaskRecord", backref="worker", lazy="selectin")


class AssignmentRecord(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False)
    worker_id = Column(String, ForeignKey("workers.id"), nullable=False)
    scheduled_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class EventLog(Base):
    __tablename__ = "event_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    task_id = Column(String, nullable=True)
    worker_id = Column(String, nullable=True)
    timestamp = Column(Float, nullable=False)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
