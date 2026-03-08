import time
import threading
from typing import Callable

from arbiter.logging_config import get_logger

log = get_logger(__name__)


class HeartbeatMonitor:
    """Tracks worker heartbeats and detects failures via timeout."""

    def __init__(self, timeout: float = 30.0, on_failure: Callable[[str], None] = None):
        self._timeout = timeout
        self._on_failure = on_failure or (lambda wid: None)
        self._heartbeats: dict[str, float] = {}
        self._active = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def record(self, worker_id: str):
        with self._lock:
            self._heartbeats[worker_id] = time.time()

    def check_all(self) -> list[str]:
        now = time.time()
        failed = []
        with self._lock:
            for wid, last in list(self._heartbeats.items()):
                if now - last > self._timeout:
                    failed.append(wid)
        for wid in failed:
            log.warning("worker_heartbeat_timeout", worker_id=wid,
                        seconds_elapsed=round(now - self._heartbeats.get(wid, now), 1),
                        timeout=self._timeout)
            self._on_failure(wid)
        return failed

    def remove(self, worker_id: str):
        with self._lock:
            self._heartbeats.pop(worker_id, None)

    def start(self, interval: float = 10.0):
        if self._active:
            return
        self._active = True

        def _loop():
            while self._active:
                self.check_all()
                time.sleep(interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._active = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def tracked_workers(self) -> list[str]:
        with self._lock:
            return list(self._heartbeats.keys())

    def seconds_since(self, worker_id: str) -> float | None:
        with self._lock:
            ts = self._heartbeats.get(worker_id)
        if ts is None:
            return None
        return time.time() - ts
