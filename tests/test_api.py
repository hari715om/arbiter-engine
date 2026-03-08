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


# -- policy CRUD --

class TestPolicy:
    def test_get_policy_default(self):
        resp = client.get("/policy")
        assert resp.status_code == 200
        data = resp.json()
        assert "default_scheduler" in data
        assert "rules" in data
        assert "utility_weights" in data
        assert isinstance(data["utility_weights"], dict)

    def test_put_policy_updates(self):
        new_policy = {
            "default_scheduler": "fifo",
            "rules": [
                {"condition": "queue_depth > 50", "use": "utility"}
            ],
            "utility_weights": {
                "latency": 0.5, "throughput": 0.2,
                "fairness": 0.1, "cost": 0.1, "risk": 0.1,
            },
        }
        resp = client.put("/policy", json=new_policy)
        assert resp.status_code == 200
        data = resp.json()
        assert data["default_scheduler"] == "fifo"
        assert len(data["rules"]) == 1

    def test_put_policy_invalid_scheduler(self):
        bad = {
            "default_scheduler": "nonexistent_scheduler",
            "rules": [],
            "utility_weights": {
                "latency": 0.5, "throughput": 0.2,
                "fairness": 0.1, "cost": 0.1, "risk": 0.1,
            },
        }
        resp = client.put("/policy", json=bad)
        assert resp.status_code == 422


# -- chaos mode --

class TestChaos:
    def _make_worker(self, wid, load=0.0):
        client.post("/workers", json={
            "id": wid, "cpu_capacity": 8.0,
            "memory_capacity": 16.0,
        })
        # Manually set the load for kill + fail tests if needed
        return wid

    def _make_task(self, tid, status="pending"):
        client.post("/tasks", json={
            "id": tid, "compute_cost": 1.0,
            "deadline": 100.0, "estimated_duration": 5.0,
        })
        return tid

    def test_chaos_kill_worker_by_target(self):
        self._make_worker("chaos-w1")
        resp = client.post("/chaos", json={
            "mode": "kill_worker",
            "target": "chaos-w1",
            "intensity": 1.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "chaos-w1" in data["affected"]
        # verify worker is now down
        wr = client.get("/workers").json()
        w = next((w for w in wr if w["id"] == "chaos-w1"), None)
        assert w and w["status"] == "down"

    def test_chaos_kill_worker_missing_target(self):
        resp = client.post("/chaos", json={
            "mode": "kill_worker",
            "target": "nonexistent-worker",
            "intensity": 1.0,
        })
        assert resp.status_code == 404

    def test_chaos_delay_tasks(self):
        for i in range(3):
            self._make_task(f"chaos-t{i}")
        resp = client.post("/chaos", json={
            "mode": "delay_tasks",
            "intensity": 1.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["affected"]) >= 1

    def test_chaos_fail_rate_spike_no_running(self):
        """When no tasks are running, affected list should be empty."""
        resp = client.post("/chaos", json={
            "mode": "fail_rate_spike",
            "intensity": 1.0,
        })
        assert resp.status_code == 200
        assert resp.json()["affected"] == []

    def test_chaos_invalid_mode(self):
        resp = client.post("/chaos", json={
            "mode": "destroy_everything",
            "intensity": 1.0,
        })
        assert resp.status_code == 422

