"""Train the RL scheduler against simulated workloads."""

import time
import argparse
import logging
import numpy as np

from arbiter.simulator.engine import SimulationEngine
from arbiter.schedulers.rl.rl_scheduler import RLScheduler
from arbiter.schedulers.rl.rl_env import build_state, compute_reward
from arbiter.traces.trace_loader import load_synthetic_trace
from arbiter.metrics.collector import MetricsCollector
from arbiter.models.task import TaskStatus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_episode(scheduler: RLScheduler, num_tasks: int, num_workers: int, seed: int):
    tasks, workers = load_synthetic_trace(num_tasks=num_tasks, num_workers=num_workers, seed=seed)
    engine = SimulationEngine(tasks=tasks, workers=workers, scheduler=scheduler)
    engine.run()

    completed = [t for t in engine.tasks.values() if t.status == TaskStatus.COMPLETED]
    failed = [t for t in engine.tasks.values() if t.status == TaskStatus.FAILED]

    total_reward = 0.0
    for t in completed:
        state_before = build_state(t, workers, tasks, t.arrival_time)
        sla_violated = t.completion_time > t.deadline if t.completion_time else False
        reward = compute_reward(t, completed=True, sla_violated=sla_violated)
        total_reward += reward

    for t in failed:
        reward = compute_reward(t, completed=False, sla_violated=False)
        total_reward += reward

    return total_reward, len(completed), len(failed)


def train(episodes: int = 50, num_tasks: int = 100, num_workers: int = 10,
          save_path: str = "models/rl_policy.json", epsilon_decay: float = 0.95):
    scheduler = RLScheduler(model_path=save_path, epsilon=1.0)

    best_reward = float("-inf")
    for ep in range(episodes):
        seed = ep * 13 + 7  # deterministic but varied seeds
        scheduler._epsilon = max(0.05, scheduler._epsilon * epsilon_decay)

        try:
            total_reward, completed, failed = run_episode(scheduler, num_tasks, num_workers, seed)
        except Exception as e:
            logger.warning("Episode %d failed: %s", ep + 1, e)
            continue

        if total_reward > best_reward:
            best_reward = total_reward
            scheduler.save()
            logger.info("Episode %d/%d  reward=%.2f  completed=%d  failed=%d  [SAVED]",
                        ep + 1, episodes, total_reward, completed, failed)
        else:
            logger.info("Episode %d/%d  reward=%.2f  completed=%d  failed=%d",
                        ep + 1, episodes, total_reward, completed, failed)

    logger.info("Training complete. Best reward: %.2f. Model saved to %s", best_reward, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL scheduler")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--save", default="models/rl_policy.json")
    args = parser.parse_args()
    train(episodes=args.episodes, num_tasks=args.tasks,
          num_workers=args.workers, save_path=args.save)
