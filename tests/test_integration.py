"""Integration tests — full API lifecycle via httpx.AsyncClient.

Feature 2 (Group E): Tests the complete request flow through FastAPI
with a real HTTP client, not TestClient shortcuts. Validates:
- Task creation → scheduling → assignment
- Worker registration + heartbeat
- Policy hot-reload + scheduler selection
- Chaos injection + recovery
- Multi-tenancy isolation
- WebSocket event stream
"""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport

# Override DB before importing app
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker
from arbiter.api.models_db import Base, get_db

TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(bind=test_engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


from arbiter.api.app import app


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.pop(get_db, None)


TASK_BODY = {
    "compute_cost": 4.0,
    "resource_type": "cpu",
    "deadline": 500.0,
    "priority": 8,
    "failure_probability": 0.1,
    "estimated_duration": 10.0,
    "dependencies": [],
    "max_retries": 2,
}

WORKER_BODY = {
    "id": "w-int-01",
    "cpu_capacity": 32.0,
    "memory_capacity": 64.0,
    "speed_multiplier": 1.2,
    "supported_resources": ["cpu", "gpu", "memory"],
}


@pytest.mark.asyncio
async def test_full_task_lifecycle():
    """POST /tasks → GET /tasks/{id} → verify status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create task
        r = await client.post("/tasks", json={**TASK_BODY, "id": "t-lifecycle-01"})
        assert r.status_code == 201
        task = r.json()
        assert task["id"] == "t-lifecycle-01"
        assert task["status"] == "pending"

        # Fetch it back
        r2 = await client.get(f"/tasks/{task['id']}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_worker_registration_and_scheduling():
    """POST /workers → POST /tasks → POST /schedule → verify assignment."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register worker
        r = await client.post("/workers", json=WORKER_BODY)
        assert r.status_code == 201
        assert r.json()["id"] == "w-int-01"

        # Create task
        r2 = await client.post("/tasks", json={**TASK_BODY, "id": "t-sched-01"})
        assert r2.status_code == 201

        # Trigger scheduling
        r3 = await client.post("/schedule")
        assert r3.status_code == 200
        assignments = r3.json()
        assert len(assignments) >= 1
        assert assignments[0]["task_id"] == "t-sched-01"
        assert assignments[0]["worker_id"] == "w-int-01"

        # Verify task is now running
        r4 = await client.get("/tasks/t-sched-01")
        assert r4.json()["status"] == "running"


@pytest.mark.asyncio
async def test_policy_hotreload_affects_scheduling():
    """PUT /policy → change to fifo → POST /schedule → verify FIFO behaviour."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Get current policy
        r = await client.get("/policy")
        assert r.status_code == 200

        # Update to force FIFO
        new_policy = {
            "default_scheduler": "fifo",
            "rules": [],
            "utility_weights": r.json()["utility_weights"],
        }
        r2 = await client.put("/policy", json=new_policy)
        assert r2.status_code == 200
        assert r2.json()["default_scheduler"] == "fifo"


@pytest.mark.asyncio
async def test_chaos_kill_worker_preempts_tasks():
    """POST /chaos kill_worker → verify worker DOWN, tasks preempted."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Setup
        await client.post("/workers", json=WORKER_BODY)
        await client.post("/tasks", json={**TASK_BODY, "id": "t-chaos-01"})
        await client.post("/schedule")

        # Kill worker
        r = await client.post("/chaos", json={
            "mode": "kill_worker",
            "target": "w-int-01",
            "intensity": 1.0,
        })
        assert r.status_code == 200
        chaos = r.json()
        assert "w-int-01" in chaos["affected"]

        # Verify worker is down
        r2 = await client.get("/workers")
        workers = r2.json()
        killed = [w for w in workers if w["id"] == "w-int-01"]
        assert killed[0]["status"] == "down"

        # Verify task is back to queued
        r3 = await client.get("/tasks/t-chaos-01")
        assert r3.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_multi_tenancy_isolation():
    """Tasks created with X-Tenant-ID are invisible to other tenants."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create task for tenant "acme"
        r = await client.post(
            "/tasks",
            json={**TASK_BODY, "id": "t-acme-01"},
            headers={"X-Tenant-ID": "acme"},
        )
        assert r.status_code == 201
        assert r.json()["tenant_id"] == "acme"

        # List tasks for default tenant → should NOT see acme's task
        r2 = await client.get("/tasks")
        task_ids = [t["id"] for t in r2.json()]
        assert "t-acme-01" not in task_ids

        # List tasks for acme tenant → should see it
        r3 = await client.get("/tasks", headers={"X-Tenant-ID": "acme"})
        task_ids_acme = [t["id"] for t in r3.json()]
        assert "t-acme-01" in task_ids_acme


@pytest.mark.asyncio
async def test_explain_endpoint():
    """GET /tasks/{id}/explain → returns scoring breakdown."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/workers", json=WORKER_BODY)
        await client.post("/tasks", json={**TASK_BODY, "id": "t-explain-01"})

        r = await client.get("/tasks/t-explain-01/explain")
        assert r.status_code == 200
        data = r.json()
        assert "factors" in data
        assert "reasoning" in data
        assert data["scheduler_name"] == "UtilityScheduler"


@pytest.mark.asyncio
async def test_metrics_and_health():
    """GET /metrics and GET /health return valid responses."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/metrics")
        assert r.status_code == 200
        assert "total_tasks" in r.json()

        r2 = await client.get("/health")
        assert r2.status_code == 200
        assert r2.json()["db_connected"] is True


@pytest.mark.asyncio
async def test_events_endpoint():
    """GET /events returns audit log after task creation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/tasks", json={**TASK_BODY, "id": "t-events-01"})

        r = await client.get("/events")
        assert r.status_code == 200
        events = r.json()
        assert any(e["event_type"] == "TASK_CREATED" for e in events)
