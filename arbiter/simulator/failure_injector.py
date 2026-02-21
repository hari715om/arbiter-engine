"""Failure Injector — generates worker failure/recovery events for simulation.

Models real-world infrastructure failures:
- Periodic: scheduled maintenance windows (like K8s node drain)
- Random: unpredictable hardware failures (like disk errors, OOM kills)
- Burst: correlated failures (like rack switch failure, AZ outage)
"""

import random
from arbiter.models.worker import Worker
from arbiter.simulator.events import Event, EventType


class FailureInjector:
    """Generates worker FAILURE/RECOVERY event pairs for the simulation.

    Usage:
        injector = FailureInjector(mode="random", failure_rate=0.02, seed=42)
        events = injector.generate_events(workers, time_limit=10000.0)
        # Feed events into engine._event_queue before simulation starts
    """

    def __init__(
        self,
        mode: str = "random",
        failure_rate: float = 0.02,
        downtime_range: tuple[float, float] = (10.0, 50.0),
        burst_size: int = 2,
        seed: int = 42,
    ):
        """
        Args:
            mode: "random" | "periodic" | "burst"
            failure_rate: For random mode — probability of failure per check interval.
                          For periodic mode — interval between scheduled failures.
            downtime_range: (min, max) time units a worker stays down.
            burst_size: For burst mode — how many workers fail simultaneously.
            seed: RNG seed for reproducibility.
        """
        self.mode = mode
        self.failure_rate = failure_rate
        self.downtime_range = downtime_range
        self.burst_size = burst_size
        self.rng = random.Random(seed)

    def generate_events(
        self,
        workers: list[Worker],
        time_limit: float,
        start_time: float = 20.0,
    ) -> list[Event]:
        """Pre-generate all failure/recovery events before simulation.

        Returns a list of Event objects to inject into the engine.
        Each failure event is paired with a recovery event (failure → downtime → recovery).

        Args:
            workers: Available workers that can fail.
            time_limit: Simulation time limit.
            start_time: Don't inject failures before this time (let simulation stabilize).
        """
        if not workers:
            return []

        if self.mode == "random":
            return self._generate_random(workers, time_limit, start_time)
        elif self.mode == "periodic":
            return self._generate_periodic(workers, time_limit, start_time)
        elif self.mode == "burst":
            return self._generate_burst(workers, time_limit, start_time)
        else:
            raise ValueError(f"Unknown failure mode: {self.mode}")

    def _generate_random(
        self, workers: list[Worker], time_limit: float, start_time: float,
    ) -> list[Event]:
        """Random failures: each worker has a chance of failing at each check interval.

        Models unpredictable hardware failures — disk errors, memory corruption,
        network partitions. Each worker is independently evaluated every 50 time units.
        """
        events: list[Event] = []
        check_interval = 50.0
        seq = 0

        t = start_time
        while t < time_limit:
            for worker in workers:
                if self.rng.random() < self.failure_rate:
                    downtime = self.rng.uniform(*self.downtime_range)

                    # Don't let recovery exceed time limit
                    recovery_time = min(t + downtime, time_limit - 1.0)
                    if recovery_time <= t:
                        continue

                    events.append(Event(
                        time=t,
                        sequence=seq,
                        event_type=EventType.WORKER_FAILURE,
                        worker_id=worker.id,
                        metadata={"injected": True, "mode": "random"},
                    ))
                    seq += 1

                    events.append(Event(
                        time=recovery_time,
                        sequence=seq,
                        event_type=EventType.WORKER_RECOVERY,
                        worker_id=worker.id,
                        metadata={"injected": True, "mode": "random"},
                    ))
                    seq += 1

            t += check_interval

        return events

    def _generate_periodic(
        self, workers: list[Worker], time_limit: float, start_time: float,
    ) -> list[Event]:
        """Periodic failures: simulate scheduled maintenance windows.

        Models planned downtime — OS patches, hardware replacement, rolling restarts.
        Workers are failed one at a time in round-robin order.
        """
        events: list[Event] = []
        interval = max(10.0, self.failure_rate)  # failure_rate = interval in periodic mode
        seq = 0
        worker_idx = 0

        t = start_time
        while t < time_limit:
            worker = workers[worker_idx % len(workers)]
            downtime = self.rng.uniform(*self.downtime_range)
            recovery_time = min(t + downtime, time_limit - 1.0)

            if recovery_time > t:
                events.append(Event(
                    time=t,
                    sequence=seq,
                    event_type=EventType.WORKER_FAILURE,
                    worker_id=worker.id,
                    metadata={"injected": True, "mode": "periodic"},
                ))
                seq += 1

                events.append(Event(
                    time=recovery_time,
                    sequence=seq,
                    event_type=EventType.WORKER_RECOVERY,
                    worker_id=worker.id,
                    metadata={"injected": True, "mode": "periodic"},
                ))
                seq += 1

            worker_idx += 1
            t += interval

        return events

    def _generate_burst(
        self, workers: list[Worker], time_limit: float, start_time: float,
    ) -> list[Event]:
        """Burst failures: multiple workers fail simultaneously.

        Models correlated infrastructure outages — rack switch failure,
        datacenter cooling issue, AZ-level outage. Multiple workers go down
        at once, stressing the scheduler's ability to handle reduced capacity.
        """
        events: list[Event] = []
        burst_interval = max(50.0, 1.0 / self.failure_rate if self.failure_rate > 0 else 200.0)
        seq = 0

        t = start_time
        while t < time_limit:
            # Pick burst_size workers (or all if fewer available)
            burst_count = min(self.burst_size, len(workers))
            failing_workers = self.rng.sample(workers, burst_count)

            downtime = self.rng.uniform(*self.downtime_range)
            recovery_time = min(t + downtime, time_limit - 1.0)

            if recovery_time > t:
                for worker in failing_workers:
                    events.append(Event(
                        time=t,
                        sequence=seq,
                        event_type=EventType.WORKER_FAILURE,
                        worker_id=worker.id,
                        metadata={"injected": True, "mode": "burst"},
                    ))
                    seq += 1

                    events.append(Event(
                        time=recovery_time,
                        sequence=seq,
                        event_type=EventType.WORKER_RECOVERY,
                        worker_id=worker.id,
                        metadata={"injected": True, "mode": "burst"},
                    ))
                    seq += 1

            t += burst_interval

        return events
