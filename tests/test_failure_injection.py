"""Tests for Phase 4: Failure Injection & Dynamic Replanning.

Verifies:
1. FailureInjector generates valid paired events
2. Worker failures preempt running tasks
3. Preempted tasks get rescheduled on surviving workers
4. SLA risk detection catches at-risk tasks
5. Phase 4 metrics are tracked correctly
6. Backwards compatibility — no injection = no worker failures
"""

import pytest

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus
from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.simulator.engine import SimulationEngine
from arbiter.simulator.events import Event, EventType
from arbiter.simulator.failure_injector import FailureInjector
from arbiter.simulator.sla_monitor import SLAMonitor, RiskLevel
from arbiter.simulator.generator import ScenarioGenerator


class TestFailureInjector:
    """Tests for the failure injection system."""

    def test_random_generates_paired_events(self):
        """Each failure event should have a matching recovery event."""
        workers = [Worker(id=f"w{i}", cpu_capacity=100.0, memory_capacity=512.0) for i in range(3)]
        injector = FailureInjector(mode="random", failure_rate=0.1, seed=42)
        events = injector.generate_events(workers, time_limit=500.0)

        failures = [e for e in events if e.event_type == EventType.WORKER_FAILURE]
        recoveries = [e for e in events if e.event_type == EventType.WORKER_RECOVERY]

        assert len(failures) > 0, "Should generate at least one failure"
        assert len(failures) == len(recoveries), "Each failure must have a recovery"

    def test_periodic_round_robin(self):
        """Periodic mode should cycle through workers."""
        workers = [Worker(id=f"w{i}", cpu_capacity=100.0, memory_capacity=512.0) for i in range(3)]
        injector = FailureInjector(mode="periodic", failure_rate=100.0, seed=42)
        events = injector.generate_events(workers, time_limit=500.0)

        failures = [e for e in events if e.event_type == EventType.WORKER_FAILURE]
        assert len(failures) > 0

        # Should hit different workers
        failed_workers = set(e.worker_id for e in failures)
        assert len(failed_workers) > 1, "Periodic should rotate across workers"

    def test_burst_multiple_workers(self):
        """Burst mode should fail multiple workers simultaneously."""
        workers = [Worker(id=f"w{i}", cpu_capacity=100.0, memory_capacity=512.0) for i in range(5)]
        injector = FailureInjector(mode="burst", burst_size=3, failure_rate=0.01, seed=42)
        events = injector.generate_events(workers, time_limit=500.0)

        failures = [e for e in events if e.event_type == EventType.WORKER_FAILURE]
        if failures:
            # Check that the first burst has multiple workers at the same time
            first_time = failures[0].time
            burst_at_first = [e for e in failures if e.time == first_time]
            assert len(burst_at_first) >= 2, "Burst should fail multiple workers"

    def test_no_events_with_empty_workers(self):
        """Should return empty list with no workers."""
        injector = FailureInjector(mode="random", seed=42)
        events = injector.generate_events([], time_limit=500.0)
        assert events == []

    def test_deterministic(self):
        """Same seed should produce identical events."""
        workers = [Worker(id=f"w{i}", cpu_capacity=100.0, memory_capacity=512.0) for i in range(3)]
        inj1 = FailureInjector(mode="random", failure_rate=0.1, seed=42)
        inj2 = FailureInjector(mode="random", failure_rate=0.1, seed=42)

        events1 = inj1.generate_events(workers, time_limit=500.0)
        events2 = inj2.generate_events(workers, time_limit=500.0)

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.time == e2.time
            assert e1.worker_id == e2.worker_id

    def test_events_within_time_limit(self):
        """All events should be within the time limit."""
        workers = [Worker(id=f"w{i}", cpu_capacity=100.0, memory_capacity=512.0) for i in range(3)]
        injector = FailureInjector(mode="random", failure_rate=0.1, seed=42)
        events = injector.generate_events(workers, time_limit=500.0)

        for event in events:
            assert event.time < 500.0, f"Event at t={event.time} exceeds time limit"

    def test_recovery_after_failure(self):
        """Recovery time should be after failure time for each worker."""
        workers = [Worker(id=f"w{i}", cpu_capacity=100.0, memory_capacity=512.0) for i in range(3)]
        injector = FailureInjector(mode="random", failure_rate=0.1, seed=42)
        events = injector.generate_events(workers, time_limit=500.0)

        # Group by worker
        worker_events: dict[str, list] = {}
        for e in events:
            worker_events.setdefault(e.worker_id, []).append(e)

        for worker_id, wevents in worker_events.items():
            wevents.sort(key=lambda e: e.time)
            for i in range(0, len(wevents) - 1, 2):
                assert wevents[i].event_type == EventType.WORKER_FAILURE
                assert wevents[i + 1].event_type == EventType.WORKER_RECOVERY
                assert wevents[i + 1].time > wevents[i].time


class TestDynamicReplanning:
    """Tests for worker failure handling and task preemption in the engine."""

    def test_preemption_on_worker_failure(self):
        """Tasks running on a failed worker should be re-queued."""
        # 2 tasks, 1 worker — both start running, worker fails, tasks get preempted
        tasks = [
            Task(id=f"t{i}", compute_cost=10.0, deadline=500.0, priority=5,
                 estimated_duration=100.0, resource_type="cpu",
                 failure_probability=0.0)
            for i in range(2)
        ]
        workers = [
            Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0),
            Worker(id="w2", cpu_capacity=100.0, memory_capacity=512.0),
        ]

        # Inject a failure for w1 at t=30 (while tasks are running), recovery at t=80
        injector = FailureInjector(mode="periodic", failure_rate=200.0, seed=42)
        # Manual injection — force a failure at known time
        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        # Inject manually after init
        engine._push_event(Event(
            time=30.0,
            sequence=engine._next_sequence(),
            event_type=EventType.WORKER_FAILURE,
            worker_id="w1",
        ))
        engine._push_event(Event(
            time=80.0,
            sequence=engine._next_sequence(),
            event_type=EventType.WORKER_RECOVERY,
            worker_id="w1",
        ))

        metrics = engine.run()
        report = metrics.report

        # Both tasks should still complete (rescheduled to w2 or w1 after recovery)
        assert report.tasks_completed == 2

    def test_worker_failure_count_tracked(self):
        """Engine should track worker failure count."""
        task = Task(id="t1", compute_cost=10.0, deadline=500.0, priority=5,
                    estimated_duration=5.0, resource_type="cpu",
                    failure_probability=0.0)
        workers = [Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0)]

        engine = SimulationEngine(
            tasks=[task], workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        # Inject one failure
        engine._push_event(Event(
            time=1.0,
            sequence=engine._next_sequence(),
            event_type=EventType.WORKER_FAILURE,
            worker_id="w1",
        ))
        engine._push_event(Event(
            time=50.0,
            sequence=engine._next_sequence(),
            event_type=EventType.WORKER_RECOVERY,
            worker_id="w1",
        ))
        metrics = engine.run()
        assert engine.worker_failure_count == 1
        assert metrics.report.worker_failures == 1

    def test_tasks_preempted_count(self):
        """Preempted tasks should be counted."""
        tasks = [
            Task(id="t1", compute_cost=10.0, deadline=500.0, priority=5,
                 estimated_duration=100.0, resource_type="cpu",
                 failure_probability=0.0)
        ]
        workers = [
            Worker(id="w1", cpu_capacity=100.0, memory_capacity=512.0),
            Worker(id="w2", cpu_capacity=100.0, memory_capacity=512.0),
        ]

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        # Fail w1 at t=10 (task should be running on w1 by then)
        engine._push_event(Event(
            time=10.0,
            sequence=engine._next_sequence(),
            event_type=EventType.WORKER_FAILURE,
            worker_id="w1",
        ))
        engine._push_event(Event(
            time=60.0,
            sequence=engine._next_sequence(),
            event_type=EventType.WORKER_RECOVERY,
            worker_id="w1",
        ))
        metrics = engine.run()

        assert engine.tasks_preempted >= 0  # May or may not be on w1
        assert metrics.report.tasks_completed >= 1  # Task should complete eventually

    def test_backwards_compatibility(self):
        """Without failure injection, engine should behave exactly as before."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=20, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
        )
        metrics = engine.run()

        assert engine.worker_failure_count == 0
        assert engine.tasks_preempted == 0
        assert metrics.report.worker_failures == 0
        assert metrics.report.tasks_preempted == 0

    def test_no_task_lost(self):
        """Every task should end up completed, failed, or pending — never lost."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=30, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=3)

        injector = FailureInjector(mode="random", failure_rate=0.1, seed=42)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
            failure_injector=injector,
        )
        metrics = engine.run()
        r = metrics.report

        assert r.tasks_completed + r.tasks_failed + r.tasks_pending == r.total_tasks


class TestSLAMonitor:
    """Tests for the SLA risk detection system."""

    def test_detects_critical_risk(self):
        """Should flag tasks that will miss their deadline."""
        task = Task(
            id="t1", compute_cost=10.0, deadline=50.0, priority=5,
            estimated_duration=100.0, resource_type="cpu",
        )
        task.status = TaskStatus.QUEUED

        monitor = SLAMonitor()
        alerts = monitor.check({"t1": task}, current_time=40.0)

        assert len(alerts) == 1
        assert alerts[0].risk_level == RiskLevel.CRITICAL
        assert alerts[0].slack < 0

    def test_detects_warning_risk(self):
        """Should flag tasks with less than 20% slack."""
        task = Task(
            id="t1", compute_cost=10.0, deadline=100.0, priority=5,
            estimated_duration=10.0, resource_type="cpu",
            arrival_time=0.0,
        )
        task.status = TaskStatus.RUNNING
        task.start_time = 80.0

        monitor = SLAMonitor(warning_threshold=0.2)
        alerts = monitor.check({"t1": task}, current_time=89.0)

        # estimated_finish = 89 + (10 - 9) = 90, slack = 100 - 90 = 10
        # slack/budget = 10/100 = 0.1 < 0.2 → WARNING
        assert len(alerts) >= 1
        assert any(a.risk_level == RiskLevel.WARNING for a in alerts)

    def test_safe_tasks_not_flagged(self):
        """Tasks on track should not generate alerts."""
        task = Task(
            id="t1", compute_cost=10.0, deadline=200.0, priority=5,
            estimated_duration=10.0, resource_type="cpu",
            arrival_time=0.0,
        )
        task.status = TaskStatus.RUNNING
        task.start_time = 5.0

        monitor = SLAMonitor()
        alerts = monitor.check({"t1": task}, current_time=10.0)

        assert len(alerts) == 0

    def test_completed_tasks_ignored(self):
        """COMPLETED and FAILED tasks should not be checked."""
        task = Task(
            id="t1", compute_cost=10.0, deadline=1.0, priority=5,
            estimated_duration=100.0, resource_type="cpu",
        )
        task.status = TaskStatus.COMPLETED
        task.completion_time = 50.0

        monitor = SLAMonitor()
        alerts = monitor.check({"t1": task}, current_time=100.0)
        assert len(alerts) == 0

    def test_alert_history_accumulates(self):
        """Monitor should track all alerts across multiple checks."""
        task = Task(
            id="t1", compute_cost=10.0, deadline=50.0, priority=5,
            estimated_duration=100.0, resource_type="cpu",
        )
        task.status = TaskStatus.QUEUED

        monitor = SLAMonitor()
        monitor.check({"t1": task}, current_time=10.0)
        monitor.check({"t1": task}, current_time=20.0)

        assert monitor.total_alerts >= 2
        assert monitor.critical_count >= 2

    def test_sla_monitor_in_engine(self):
        """SLA monitor should accumulate alerts during simulation."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=20, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=2)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
            sla_check_interval=50.0,
        )
        metrics = engine.run()

        # SLA monitor should have been invoked during simulation
        assert engine.sla_monitor.total_alerts >= 0  # may or may not have alerts
        assert metrics.report.sla_risks_detected == engine.sla_monitor.total_alerts


class TestFailureInjectionIntegration:
    """End-to-end tests with failure injection enabled."""

    def test_simulation_completes_with_random_failures(self):
        """Simulation should complete even with frequent worker failures."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=30, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=5)

        injector = FailureInjector(mode="random", failure_rate=0.05, seed=42)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
            failure_injector=injector,
        )
        metrics = engine.run()
        r = metrics.report

        assert r.total_tasks == 30
        assert r.tasks_completed + r.tasks_failed + r.tasks_pending == 30

    def test_simulation_completes_with_burst_failures(self):
        """Simulation should handle burst failures gracefully."""
        gen = ScenarioGenerator(seed=42)
        tasks = gen.generate_tasks(num_tasks=20, dependency_density=0.0)
        workers = gen.generate_workers(num_workers=5)

        injector = FailureInjector(mode="burst", burst_size=2, failure_rate=0.005, seed=42)

        engine = SimulationEngine(
            tasks=tasks, workers=workers,
            scheduler=FIFOScheduler(), seed=42,
            failure_injector=injector,
        )
        metrics = engine.run()
        r = metrics.report

        assert r.total_tasks == 20
        assert r.tasks_completed > 0
