"""Training data generator — runs simulations and collects task-level features + outcomes."""

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker
from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.schedulers.heuristic import HeuristicScheduler
from arbiter.simulator.generator import ScenarioGenerator
from arbiter.simulator.engine import SimulationEngine


RESOURCE_ENCODING = {"cpu": 0, "gpu": 1, "memory": 2}


class TrainingDataGenerator:
    """Runs multiple simulations and extracts feature vectors for ML training."""

    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed

    def generate(
        self,
        num_simulations: int = 50,
        tasks_per_sim: int = 100,
        workers_per_sim: int = 5,
    ) -> list[dict]:
        """Run simulations and return a list of feature dicts (one per completed/failed task)."""
        all_samples: list[dict] = []

        for sim_idx in range(num_simulations):
            seed = self.base_seed + sim_idx

            gen = ScenarioGenerator(seed=seed)
            tasks = gen.generate_tasks(num_tasks=tasks_per_sim, dependency_density=0.1)
            workers = gen.generate_workers(num_workers=workers_per_sim)

            # Alternate between schedulers for diverse training data
            scheduler = HeuristicScheduler() if sim_idx % 2 == 0 else FIFOScheduler()

            engine = SimulationEngine(
                tasks=tasks, workers=workers,
                scheduler=scheduler, seed=seed,
            )
            engine.run()

            # Build worker lookup for feature extraction
            worker_map = {w.id: w for w in workers}

            for task in engine.tasks.values():
                if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    continue
                if task.assigned_worker is None or task.start_time is None:
                    continue

                worker = worker_map.get(task.assigned_worker)
                if worker is None:
                    continue

                actual_runtime = (
                    task.completion_time - task.start_time
                    if task.completion_time is not None else 0.0
                )

                all_samples.append(self._extract_features(task, worker, actual_runtime))

        return all_samples

    def _extract_features(self, task: Task, worker: Worker, actual_runtime: float) -> dict:
        """Extract a feature dict from a completed/failed task."""
        return {
            # Task features
            "compute_cost": task.compute_cost,
            "estimated_duration": task.estimated_duration,
            "priority": task.priority,
            "failure_probability": task.failure_probability,
            "dependency_count": len(task.dependencies),
            "resource_type_encoded": RESOURCE_ENCODING.get(task.resource_type, 0),
            # Worker features
            "worker_speed": worker.speed_multiplier,
            "worker_capacity": worker.cpu_capacity,
            "worker_load_ratio": task.compute_cost / worker.cpu_capacity,
            # Targets
            "actual_runtime": actual_runtime,
            "did_fail": 1 if task.status == TaskStatus.FAILED else 0,
        }
