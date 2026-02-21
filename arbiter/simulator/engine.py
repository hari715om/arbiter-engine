"""Simulation Engine — core event-driven execution loop."""

import heapq
import random
from typing import Optional

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus
from arbiter.schedulers.base import BaseScheduler, Assignment
from arbiter.simulator.events import Event, EventType
from arbiter.metrics.collector import MetricsCollector


class SimulationEngine:
    """Event-driven simulation engine for task scheduling."""

    def __init__(
        self,
        tasks: list[Task],
        workers: list[Worker],
        scheduler: BaseScheduler,
        seed: int = 42,
        time_limit: float = 10000.0,
    ):
        self.tasks: dict[str, Task] = {t.id: t.model_copy() for t in tasks}
        self.workers: dict[str, Worker] = {w.id: w.model_copy() for w in workers}
        self.scheduler = scheduler
        self.rng = random.Random(seed)
        self.time_limit = time_limit

        self._event_queue: list[Event] = []
        self._event_counter: int = 0
        self._completed_task_ids: set[str] = set()
        self._current_time: float = 0.0
        self._metrics = MetricsCollector()
        self.event_log: list[Event] = []

        # Worker reliability tracking (models hardware degradation patterns)
        # 1.0 = fully reliable, degrades on failure, recovers slowly on success
        self.worker_reliability: dict[str, float] = {
            w.id: 1.0 for w in workers
        }

    def run(self) -> MetricsCollector:
        """Run the simulation: push arrivals → process events → return metrics."""
        for task in self.tasks.values():
            self._push_event(Event(
                time=task.arrival_time,
                sequence=self._next_sequence(),
                event_type=EventType.TASK_ARRIVAL,
                task_id=task.id,
            ))

        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            if event.time > self.time_limit:
                break

            self._current_time = event.time
            self.event_log.append(event)

            match event.event_type:
                case EventType.TASK_ARRIVAL:
                    self._handle_task_arrival(event)
                case EventType.SCHEDULE_TICK:
                    self._handle_schedule_tick(event)
                case EventType.TASK_COMPLETION:
                    self._handle_task_completion(event)
                case EventType.TASK_FAILURE:
                    self._handle_task_failure(event)
                case EventType.WORKER_FAILURE:
                    self._handle_worker_failure(event)
                case EventType.WORKER_RECOVERY:
                    self._handle_worker_recovery(event)

        self._metrics.calculate(
            tasks=list(self.tasks.values()),
            workers=list(self.workers.values()),
            total_time=self._current_time,
            scheduler_name=self.scheduler.name,
        )

        return self._metrics

    # ── Event Handlers ────────────────────────────────────────────────

    def _handle_task_arrival(self, event: Event) -> None:
        """Mark task QUEUED and trigger scheduling."""
        task = self.tasks[event.task_id]
        task.status = TaskStatus.QUEUED
        self._push_event(Event(
            time=self._current_time,
            sequence=self._next_sequence(),
            event_type=EventType.SCHEDULE_TICK,
        ))

    def _handle_schedule_tick(self, event: Event) -> None:
        """Run the scheduler and execute resulting assignments."""
        queued_tasks = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.QUEUED
        ]
        if not queued_tasks:
            return

        assignments = self.scheduler.schedule(
            tasks=queued_tasks,
            workers=list(self.workers.values()),
            completed_task_ids=self._completed_task_ids,
        )

        for assignment in assignments:
            self._execute_assignment(assignment)

    def _execute_assignment(self, assignment: Assignment) -> None:
        """Start a task on a worker: update state, schedule completion or failure.

        Models real-world behaviors:
        - Resource contention: busy workers run tasks slower (like CPU throttling)
        - Correlated failures: unreliable workers have higher failure rates
        """
        task = self.tasks[assignment.task_id]
        worker = self.workers[assignment.worker_id]

        task.status = TaskStatus.RUNNING
        task.start_time = self._current_time
        task.assigned_worker = worker.id
        worker.assign_task(task.id, task.compute_cost)

        # Resource contention: more load → slower execution (real CPU/memory contention)
        load_ratio = worker.current_load / worker.cpu_capacity if worker.cpu_capacity > 0 else 0
        contention_factor = 1.0 + 0.3 * load_ratio

        jitter = self.rng.uniform(0.8, 1.2)
        actual_runtime = (task.estimated_duration / worker.speed_multiplier) * jitter * contention_factor
        completion_time = self._current_time + actual_runtime

        # Correlated failure: combine task risk with worker reliability
        worker_rel = self.worker_reliability.get(worker.id, 1.0)
        combined_failure_prob = task.failure_probability + (1.0 - worker_rel) * 0.3
        combined_failure_prob = min(combined_failure_prob, 0.95)  # cap at 95%

        will_fail = self.rng.random() < combined_failure_prob

        if will_fail:
            fail_time = self._current_time + actual_runtime * self.rng.uniform(0.3, 0.9)
            self._push_event(Event(
                time=fail_time,
                sequence=self._next_sequence(),
                event_type=EventType.TASK_FAILURE,
                task_id=task.id,
                worker_id=worker.id,
            ))
        else:
            self._push_event(Event(
                time=completion_time,
                sequence=self._next_sequence(),
                event_type=EventType.TASK_COMPLETION,
                task_id=task.id,
                worker_id=worker.id,
            ))

    def _handle_task_completion(self, event: Event) -> None:
        """Mark COMPLETED, free worker, improve worker reliability, trigger scheduling."""
        task = self.tasks[event.task_id]
        worker = self.workers[event.worker_id]

        task.status = TaskStatus.COMPLETED
        task.completion_time = self._current_time
        worker.release_task(task.id, task.compute_cost)
        self._completed_task_ids.add(task.id)

        # Worker reliability slowly recovers on successful execution
        rel = self.worker_reliability.get(worker.id, 1.0)
        self.worker_reliability[worker.id] = min(1.0, rel * 1.05)

        self._push_event(Event(
            time=self._current_time,
            sequence=self._next_sequence(),
            event_type=EventType.SCHEDULE_TICK,
        ))

    def _handle_task_failure(self, event: Event) -> None:
        """Handle task failure with retry logic (like Kubernetes pod restarts).

        If retries remain: re-queue with backoff delay.
        If exhausted: mark permanently FAILED.
        Worker reliability degrades on failure.
        """
        task = self.tasks[event.task_id]
        worker = self.workers[event.worker_id]

        worker.release_task(task.id, task.compute_cost)

        # Degrade worker reliability (hardware failure correlation)
        rel = self.worker_reliability.get(worker.id, 1.0)
        self.worker_reliability[worker.id] = max(0.1, rel * 0.85)

        task.retry_count += 1

        if task.retry_count < task.max_retries:
            # Re-queue with exponential backoff (like K8s CrashLoopBackOff)
            backoff = 2.0 * (2 ** (task.retry_count - 1))  # 2s, 4s, 8s...
            task.status = TaskStatus.QUEUED
            task.assigned_worker = None
            task.start_time = None

            self._push_event(Event(
                time=self._current_time + backoff,
                sequence=self._next_sequence(),
                event_type=EventType.SCHEDULE_TICK,
            ))
        else:
            # Permanent failure — exhausted retries
            task.status = TaskStatus.FAILED
            task.completion_time = self._current_time

            self._push_event(Event(
                time=self._current_time,
                sequence=self._next_sequence(),
                event_type=EventType.SCHEDULE_TICK,
            ))

    def _handle_worker_failure(self, event: Event) -> None:
        """Mark worker DOWN. (Placeholder — full logic in Phase 4)"""
        worker = self.workers[event.worker_id]
        worker.status = WorkerStatus.DOWN

    def _handle_worker_recovery(self, event: Event) -> None:
        """Restore worker to IDLE and trigger scheduling."""
        worker = self.workers[event.worker_id]
        worker.status = WorkerStatus.IDLE
        worker.current_load = 0.0
        worker.active_tasks = []

        self._push_event(Event(
            time=self._current_time,
            sequence=self._next_sequence(),
            event_type=EventType.SCHEDULE_TICK,
        ))

    # ── Utilities ─────────────────────────────────────────────────────

    def _push_event(self, event: Event) -> None:
        """Add an event to the priority queue."""
        heapq.heappush(self._event_queue, event)

    def _next_sequence(self) -> int:
        """Monotonically increasing sequence number for tie-breaking."""
        self._event_counter += 1
        return self._event_counter
