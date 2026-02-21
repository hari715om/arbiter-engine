import time
import pytest
from arbiter.execution.executor import SimulatedExecutor, ExecutionResult
from arbiter.execution.heartbeat import HeartbeatMonitor


class TestSimulatedExecutor:
    def test_execution_returns_result(self):
        exe = SimulatedExecutor(failure_rate=0.0)
        result = exe.execute("t1", "w1", "alpine", "echo hello", timeout=1.0)
        assert isinstance(result, ExecutionResult)
        assert result.task_id == "t1"
        assert result.worker_id == "w1"
        assert result.success is True
        assert result.exit_code == 0
        assert result.duration > 0

    def test_execution_can_fail(self):
        exe = SimulatedExecutor(failure_rate=1.0)
        result = exe.execute("t2", "w1", "alpine", "echo fail")
        assert result.success is False
        assert result.exit_code == 1
        assert result.error == "Simulated failure"

    def test_cancel_returns_true(self):
        exe = SimulatedExecutor()
        assert exe.cancel("t1") is True

    def test_speed_affects_duration(self):
        slow = SimulatedExecutor(failure_rate=0.0, speed=0.5)
        fast = SimulatedExecutor(failure_rate=0.0, speed=2.0)
        r_slow = slow.execute("t1", "w1", "img", "cmd", timeout=10.0)
        r_fast = fast.execute("t2", "w1", "img", "cmd", timeout=10.0)
        assert r_slow.duration > r_fast.duration


class TestHeartbeatMonitor:
    def test_record_and_check(self):
        monitor = HeartbeatMonitor(timeout=1.0)
        monitor.record("w1")
        failed = monitor.check_all()
        assert failed == []

    def test_timeout_detection(self):
        detected = []
        monitor = HeartbeatMonitor(timeout=0.1, on_failure=lambda wid: detected.append(wid))
        monitor.record("w1")
        time.sleep(0.2)
        failed = monitor.check_all()
        assert "w1" in failed
        assert "w1" in detected

    def test_remove_worker(self):
        monitor = HeartbeatMonitor(timeout=0.1)
        monitor.record("w1")
        monitor.remove("w1")
        assert "w1" not in monitor.tracked_workers

    def test_tracked_workers(self):
        monitor = HeartbeatMonitor()
        monitor.record("w1")
        monitor.record("w2")
        assert set(monitor.tracked_workers) == {"w1", "w2"}

    def test_seconds_since(self):
        monitor = HeartbeatMonitor()
        monitor.record("w1")
        time.sleep(0.05)
        elapsed = monitor.seconds_since("w1")
        assert elapsed is not None
        assert elapsed >= 0.04

    def test_seconds_since_unknown(self):
        monitor = HeartbeatMonitor()
        assert monitor.seconds_since("ghost") is None

    def test_background_thread(self):
        detected = []
        monitor = HeartbeatMonitor(timeout=0.1, on_failure=lambda wid: detected.append(wid))
        monitor.record("w1")
        monitor.start(interval=0.05)
        time.sleep(0.3)
        monitor.stop()
        assert "w1" in detected

    def test_multiple_heartbeats_reset_timer(self):
        monitor = HeartbeatMonitor(timeout=0.2)
        monitor.record("w1")
        time.sleep(0.1)
        monitor.record("w1")  # reset
        time.sleep(0.1)
        failed = monitor.check_all()
        assert failed == []


class TestPrometheusMetrics:
    def test_get_metrics_output(self):
        from arbiter.metrics.prometheus import get_metrics_output
        body, content_type = get_metrics_output()
        assert isinstance(body, bytes)
        assert len(body) > 0

    def test_record_functions_dont_crash(self):
        from arbiter.metrics import prometheus as p
        p.record_task_created()
        p.record_task_started("utility")
        p.record_task_completed(5.0)
        p.record_task_failed()
        p.record_sla_violation()
        p.update_worker_utilization("w1", 0.5)
        p.set_scheduler_info("utility")

    def test_prometheus_api_endpoint(self):
        from fastapi.testclient import TestClient
        from arbiter.api.app import app
        client = TestClient(app)
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200
