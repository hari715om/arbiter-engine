import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    task_id: str
    worker_id: str
    success: bool
    duration: float
    exit_code: int = 0
    output: str = ""
    error: str = ""


class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, task_id: str, worker_id: str, image: str, command: str,
                timeout: float = 300.0) -> ExecutionResult:
        ...

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        ...


class SimulatedExecutor(BaseExecutor):
    """Runs tasks as time-delayed completions (no real execution)."""

    def __init__(self, failure_rate: float = 0.1, speed: float = 1.0):
        self._failure_rate = failure_rate
        self._speed = speed
        import random
        self._rng = random.Random(42)

    def execute(self, task_id: str, worker_id: str, image: str, command: str,
                timeout: float = 300.0) -> ExecutionResult:
        duration = max(0.1, timeout * 0.1 / self._speed)
        time.sleep(min(duration, 2.0))  # cap actual wait

        failed = self._rng.random() < self._failure_rate
        return ExecutionResult(
            task_id=task_id,
            worker_id=worker_id,
            success=not failed,
            duration=duration,
            exit_code=1 if failed else 0,
            output="" if failed else f"Task {task_id} completed",
            error="Simulated failure" if failed else "",
        )

    def cancel(self, task_id: str) -> bool:
        logger.info("Cancelled simulated task %s", task_id)
        return True


class DockerExecutor(BaseExecutor):
    """Runs tasks as Docker containers."""

    def __init__(self):
        try:
            import docker
            self._client = docker.from_env()
            self._client.ping()
            self._containers: dict[str, object] = {}
        except Exception as e:
            raise RuntimeError(f"Docker not available: {e}")

    def execute(self, task_id: str, worker_id: str, image: str, command: str,
                timeout: float = 300.0) -> ExecutionResult:
        start = time.time()
        try:
            container = self._client.containers.run(
                image=image,
                command=command,
                detach=True,
                labels={"arbiter.task_id": task_id, "arbiter.worker_id": worker_id},
                mem_limit="512m",
                cpu_period=100000,
                cpu_quota=50000,  # 0.5 CPU
            )
            self._containers[task_id] = container

            result = container.wait(timeout=timeout)
            logs = container.logs(tail=200).decode("utf-8", errors="replace")
            duration = time.time() - start
            exit_code = result.get("StatusCode", -1)

            container.remove(force=True)
            del self._containers[task_id]

            return ExecutionResult(
                task_id=task_id,
                worker_id=worker_id,
                success=(exit_code == 0),
                duration=duration,
                exit_code=exit_code,
                output=logs if exit_code == 0 else "",
                error=logs if exit_code != 0 else "",
            )
        except Exception as e:
            duration = time.time() - start
            logger.error("Docker execution failed for %s: %s", task_id, e)
            if task_id in self._containers:
                try:
                    self._containers[task_id].remove(force=True)
                except Exception:
                    pass
                del self._containers[task_id]
            return ExecutionResult(
                task_id=task_id, worker_id=worker_id,
                success=False, duration=duration, exit_code=-1,
                error=str(e),
            )

    def cancel(self, task_id: str) -> bool:
        container = self._containers.get(task_id)
        if not container:
            return False
        try:
            container.kill()
            container.remove(force=True)
            del self._containers[task_id]
            return True
        except Exception as e:
            logger.error("Failed to cancel %s: %s", task_id, e)
            return False


def get_executor(mode: str = "simulated") -> BaseExecutor:
    if mode == "docker":
        return DockerExecutor()
    return SimulatedExecutor()
