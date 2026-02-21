import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from arbiter.api.models_db import Base, get_db
from arbiter.api.app import app

# single shared in-memory SQLite connection across all test requests
test_engine = create_engine(
    "sqlite:///:memory:",
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


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


client = TestClient(app)


# -- task CRUD --

class TestTaskEndpoints:
    def test_create_task(self):
        resp = client.post("/tasks", json={
            "compute_cost": 2.0, "deadline": 100.0,
            "estimated_duration": 10.0, "priority": 7,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert data["priority"] == 7
        assert data["compute_cost"] == 2.0

    def test_create_task_custom_id(self):
        resp = client.post("/tasks", json={
            "id": "my-task", "compute_cost": 1.0,
            "deadline": 50.0, "estimated_duration": 5.0,
        })
        assert resp.status_code == 201
        assert resp.json()["id"] == "my-task"

    def test_duplicate_task_rejected(self):
        body = {"id": "dup", "compute_cost": 1.0, "deadline": 50.0, "estimated_duration": 5.0}
        client.post("/tasks", json=body)
        resp = client.post("/tasks", json=body)
        assert resp.status_code == 400

    def test_get_task(self):
        client.post("/tasks", json={
            "id": "t1", "compute_cost": 1.0,
            "deadline": 50.0, "estimated_duration": 5.0,
        })
        resp = client.get("/tasks/t1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "t1"

    def test_get_missing_task(self):
        resp = client.get("/tasks/nonexistent")
        assert resp.status_code == 404

    def test_list_tasks(self):
        for i in range(3):
            client.post("/tasks", json={
                "id": f"lt-{i}", "compute_cost": 1.0,
                "deadline": 50.0, "estimated_duration": 5.0,
            })
        resp = client.get("/tasks")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_list_tasks_filters_by_status(self):
        client.post("/tasks", json={
            "id": "filt-1", "compute_cost": 1.0,
            "deadline": 50.0, "estimated_duration": 5.0,
        })
        resp = client.get("/tasks?status=pending")
        assert len(resp.json()) >= 1
        assert all(t["status"] == "pending" for t in resp.json())

    def test_delete_task(self):
        client.post("/tasks", json={
            "id": "del-1", "compute_cost": 1.0,
            "deadline": 50.0, "estimated_duration": 5.0,
        })
        resp = client.delete("/tasks/del-1")
        assert resp.status_code == 204
        assert client.get("/tasks/del-1").status_code == 404

    def test_delete_missing_task(self):
        assert client.delete("/tasks/nope").status_code == 404

    def test_task_with_dependencies(self):
        resp = client.post("/tasks", json={
            "compute_cost": 1.0, "deadline": 50.0,
            "estimated_duration": 5.0, "dependencies": ["dep1", "dep2"],
        })
        data = resp.json()
        assert data["dependencies"] == ["dep1", "dep2"]


# -- worker endpoints --

class TestWorkerEndpoints:
    def test_register_worker(self):
        resp = client.post("/workers", json={"cpu_capacity": 8.0})
        assert resp.status_code == 201
        data = resp.json()
        assert data["cpu_capacity"] == 8.0
        assert data["status"] == "idle"

    def test_register_worker_custom_id(self):
        resp = client.post("/workers", json={"id": "w1", "cpu_capacity": 4.0})
        assert resp.status_code == 201
        assert resp.json()["id"] == "w1"

    def test_duplicate_worker_rejected(self):
        body = {"id": "wdup", "cpu_capacity": 4.0}
        client.post("/workers", json=body)
        resp = client.post("/workers", json=body)
        assert resp.status_code == 400

    def test_list_workers(self):
        client.post("/workers", json={"id": "lw-1", "cpu_capacity": 4.0})
        client.post("/workers", json={"id": "lw-2", "cpu_capacity": 8.0})
        resp = client.get("/workers")
        assert resp.status_code == 200
        assert len(resp.json()) >= 2

    def test_worker_heartbeat(self):
        client.post("/workers", json={"id": "hb-w", "cpu_capacity": 4.0})
        resp = client.post("/workers/hb-w/heartbeat")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_heartbeat_missing_worker(self):
        assert client.post("/workers/ghost/heartbeat").status_code == 404


# -- scheduling --

class TestScheduling:
    def test_schedule_empty(self):
        resp = client.post("/schedule")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_schedule_assigns_tasks(self):
        client.post("/workers", json={"id": "sw-1", "cpu_capacity": 10.0})
        client.post("/tasks", json={
            "id": "st-1", "compute_cost": 2.0,
            "deadline": 100.0, "estimated_duration": 5.0, "priority": 5,
        })
        resp = client.post("/schedule")
        assert resp.status_code == 200
        assignments = resp.json()
        assert len(assignments) >= 1
        assert assignments[0]["task_id"] == "st-1"
        assert assignments[0]["worker_id"] == "sw-1"

        task = client.get("/tasks/st-1").json()
        assert task["status"] == "running"
        assert task["assigned_worker"] == "sw-1"


# -- metrics --

class TestMetrics:
    def test_metrics_empty(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tasks"] == 0
        assert data["worker_count"] == 0

    def test_metrics_with_data(self):
        client.post("/tasks", json={
            "id": "m-1", "compute_cost": 1.0,
            "deadline": 50.0, "estimated_duration": 5.0,
        })
        client.post("/workers", json={"id": "m-w", "cpu_capacity": 4.0})
        resp = client.get("/metrics")
        data = resp.json()
        assert data["total_tasks"] == 1
        assert data["pending"] >= 1
        assert data["worker_count"] == 1


# -- events --

class TestEvents:
    def test_events_logged_on_create(self):
        client.post("/tasks", json={
            "id": "ev-1", "compute_cost": 1.0,
            "deadline": 50.0, "estimated_duration": 5.0,
        })
        resp = client.get("/events")
        assert resp.status_code == 200
        events = resp.json()
        assert any(e["event_type"] == "TASK_CREATED" for e in events)


# -- health --

class TestHealth:
    def test_health_returns_status(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scheduler_type"] == "utility"
        assert data["uptime_seconds"] >= 0
