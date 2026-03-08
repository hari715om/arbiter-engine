"""Heartbeat monitoring — asyncio-native (Group E Fix 2).

Replaces the threading-based HeartbeatMonitor with an asyncio-native
implementation that integrates cleanly with FastAPI's event loop.
The old threading version caused potential deadlocks when mixing OS threads
with asyncio and made graceful shutdown fragile.

The monitor is started/stopped via the FastAPI lifespan context manager:
    async with lifespan(app):
        monitor = AsyncHeartbeatMonitor(...)
        await monitor.start()
        yield
        await monitor.stop()
"""

import asyncio
import time
from typing import Callable, Awaitable, Optional

from arbiter.logging_config import get_logger

log = get_logger(__name__)


class AsyncHeartbeatMonitor:
    """Asyncio-native heartbeat monitor for worker liveness detection.

    Workers send periodic heartbeats. If a worker misses the timeout window,
    the on_failure callback fires (e.g. to mark the worker DOWN and preempt tasks).
    """

    def __init__(
        self,
        timeout: float = 30.0,
        on_failure: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self._timeout = timeout
        self._on_failure = on_failure or (lambda wid: asyncio.sleep(0))
        self._heartbeats: dict[str, float] = {}
        self._lock = asyncio.Lock()  # asyncio.Lock, NOT threading.Lock
        self._task: Optional[asyncio.Task] = None

    async def record(self, worker_id: str):
        """Record a heartbeat from a worker. Called from API endpoint."""
        async with self._lock:
            self._heartbeats[worker_id] = time.time()

    async def check_all(self) -> list[str]:
        """Check all workers and return list of timed-out worker IDs."""
        now = time.time()
        failed = []
        async with self._lock:
            for wid, last in list(self._heartbeats.items()):
                if now - last > self._timeout:
                    failed.append(wid)

        for wid in failed:
            log.warning(
                "worker_heartbeat_timeout",
                worker_id=wid,
                seconds_elapsed=round(now - self._heartbeats.get(wid, now), 1),
                timeout=self._timeout,
            )
            try:
                await self._on_failure(wid)
            except Exception as e:
                log.error("heartbeat_failure_callback_error",
                          worker_id=wid, error=str(e))

        return failed

    async def remove(self, worker_id: str):
        """Stop tracking a worker."""
        async with self._lock:
            self._heartbeats.pop(worker_id, None)

    async def start(self, interval: float = 10.0):
        """Start the background heartbeat checking loop."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(interval))
        log.info("heartbeat_monitor_started", interval=interval,
                 timeout=self._timeout)

    async def stop(self):
        """Gracefully stop the background loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            log.info("heartbeat_monitor_stopped")

    async def _loop(self, interval: float):
        """Background loop — yields control to event loop via asyncio.sleep."""
        try:
            while True:
                await self.check_all()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise  # propagate for clean shutdown

    @property
    def tracked_workers(self) -> list[str]:
        """Return list of currently tracked worker IDs (sync-safe snapshot)."""
        # Safe to read without lock for display purposes
        return list(self._heartbeats.keys())

    def seconds_since(self, worker_id: str) -> Optional[float]:
        """Return seconds since last heartbeat, or None if not tracked."""
        ts = self._heartbeats.get(worker_id)
        if ts is None:
            return None
        return time.time() - ts


# ── Backward compatibility shim ────────────────────────────────────────────────
# Keep the old class name so existing tests don't break on import,
# but it now delegates to the async version for new code.

class HeartbeatMonitor:
    """Sync wrapper around AsyncHeartbeatMonitor for backward compatibility.

    Tests that use synchronous record() / check_all() will still work.
    New code should use AsyncHeartbeatMonitor directly.
    """

    def __init__(self, timeout: float = 30.0, on_failure=None):
        self._timeout = timeout
        self._sync_on_failure = on_failure or (lambda wid: None)
        self._heartbeats: dict[str, float] = {}
        self._active = False
        self._thread = None

    def record(self, worker_id: str):
        self._heartbeats[worker_id] = time.time()

    def check_all(self) -> list[str]:
        now = time.time()
        failed = []
        for wid, last in list(self._heartbeats.items()):
            if now - last > self._timeout:
                failed.append(wid)
        for wid in failed:
            try:
                self._sync_on_failure(wid)
            except Exception:
                pass
        return failed

    def remove(self, worker_id: str):
        self._heartbeats.pop(worker_id, None)

    def start(self, interval: float = 10.0):
        import threading
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
        return list(self._heartbeats.keys())

    def seconds_since(self, worker_id: str) -> Optional[float]:
        ts = self._heartbeats.get(worker_id)
        if ts is None:
            return None
        return time.time() - ts
