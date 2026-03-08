"""RL Ablation Study — Feature 4 (Group E).

Compares RL scheduler variants against the Utility baseline across
5 seeds to measure the contribution of each DQN component:
  1. Full RL (baseline)
  2. RL without replay buffer
  3. RL without target network
  4. RL without priority sorting
  5. Utility Scheduler (control group)

Metrics per condition (mean ± std across seeds):
  - Completion rate
  - SLA violation rate
  - Avg task latency
  - IQM reward (interquartile mean — robust per Agarwal et al. 2021)
  - Throughput (tasks/s)

Usage:
    python scripts/rl_ablation.py [--tasks 500] [--seeds 5] [--workers 10]
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arbiter.simulator.generator import TaskGenerator, WorkerGenerator
from arbiter.simulator.engine import SimulationEngine
from arbiter.schedulers.utility_scheduler import UtilityScheduler
from arbiter.schedulers.rl.rl_scheduler import RLScheduler, QNetwork, ReplayBuffer


def iqm(values: list[float]) -> float:
    """Interquartile Mean — robust central tendency metric for RL evaluation.

    Discards the bottom and top 25% of values, averages the middle 50%.
    Recommended by Agarwal et al. (2021), 'Deep Reinforcement Learning
    at the Edge of the Statistical Precipice'.
    """
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    lo = n // 4
    hi = n - n // 4
    if lo >= hi:
        return float(np.mean(arr))
    return float(np.mean(arr[lo:hi]))


class RLNoReplay(RLScheduler):
    """RL variant with replay buffer disabled (direct online update only)."""
    name = "RL-NoReplay"

    def update(self, state, action, reward, next_state, done):
        # Skip replay buffer, do direct Bellman update
        q_current = self.q_net.forward(state)
        q_target_vals = self.target_net.forward(next_state)
        target = reward + (1 - done) * self.gamma * np.max(q_target_vals)
        error = target - q_current[action]
        # Gradient step on output layer only
        hidden = np.maximum(0, state @ self.q_net.w1 + self.q_net.b1)
        self.q_net.w2[:, action] += self.lr * error * hidden
        self.q_net.b2[action] += self.lr * error


class RLNoTargetNet(RLScheduler):
    """RL variant without target network (uses online Q-net for both)."""
    name = "RL-NoTarget"

    def update(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        for i in range(self.batch_size):
            q_current = self.q_net.forward(states[i])
            # Use online net instead of target net
            q_next = self.q_net.forward(next_states[i])
            target = rewards[i] + (1 - dones[i]) * self.gamma * np.max(q_next)
            error = target - q_current[int(actions[i])]
            hidden = np.maximum(0, states[i] @ self.q_net.w1 + self.q_net.b1)
            self.q_net.w2[:, int(actions[i])] += self.lr * error * hidden
            self.q_net.b2[int(actions[i])] += self.lr * error


class RLNoPrioritySort(RLScheduler):
    """RL variant without priority-based task sorting before assignment."""
    name = "RL-NoSort"

    def schedule(self, tasks, workers, completed_ids=None):
        from arbiter.models.task import TaskStatus
        from arbiter.schedulers.base import Assignment
        from arbiter.schedulers.rl.rl_env import build_state

        ready = [t for t in tasks if t.status in (TaskStatus.QUEUED, TaskStatus.PENDING)]
        if completed_ids:
            ready = [t for t in ready if all(d in completed_ids for d in t.dependencies)]
        # No sorting (the key ablation: remove priority ordering)

        available = {w.id: w for w in workers if w.current_load < w.cpu_capacity}
        assignments = []
        used = set()
        for task in ready:
            if not available:
                break
            avail_list = [w for wid, w in available.items() if wid not in used]
            if not avail_list:
                break
            state = build_state(task, avail_list, tasks, current_time=0.0)
            q_values = self.q_net.forward(state)
            for i, w in enumerate(avail_list):
                if w.current_load + task.compute_cost > w.cpu_capacity:
                    q_values[i] = -1e9
            action = int(np.argmax(q_values[:len(avail_list)]))
            worker = avail_list[action]
            if worker.current_load + task.compute_cost <= worker.cpu_capacity:
                assignments.append(Assignment(task_id=task.id, worker_id=worker.id))
                worker.current_load += task.compute_cost
                used.add(worker.id)
        return assignments


def run_condition(scheduler, tasks, workers, seed):
    """Run a single scheduler on a deep-copied trace and return metrics."""
    t_copy = copy.deepcopy(tasks)
    w_copy = copy.deepcopy(workers)

    engine = SimulationEngine(
        tasks=t_copy, workers=w_copy, scheduler=scheduler, seed=seed,
    )
    t0 = time.perf_counter()
    report = engine.run()
    wall = time.perf_counter() - t0

    total = len(tasks)
    completed = report.tasks_completed
    failed = report.tasks_failed
    sla_viols = getattr(report, "sla_violations", 0)

    latencies = [
        t.completion_time - t.arrival_time
        for t in engine.tasks.values()
        if t.completion_time is not None and t.arrival_time is not None
    ]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = completed / wall if wall > 0 else 0.0

    # Compute per-task rewards for IQM
    rewards = []
    for t in engine.tasks.values():
        if t.completion_time is not None:
            sla_viol = (t.completion_time > t.deadline) if t.deadline else False
            r = 1.0  # completed
            if sla_viol:
                r -= 0.5
            r -= 0.1 * t.retry_count
            rewards.append(r)
        else:
            rewards.append(-1.0)

    return {
        "completion_rate": completed / total if total else 0.0,
        "sla_violation_rate": sla_viols / max(completed, 1),
        "avg_latency": avg_lat,
        "iqm_reward": iqm(rewards),
        "throughput": throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="RL Ablation Study")
    parser.add_argument("--tasks", type=int, default=500)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    conditions = [
        ("Full RL", lambda: RLScheduler()),
        ("RL-NoReplay", lambda: RLNoReplay()),
        ("RL-NoTarget", lambda: RLNoTargetNet()),
        ("RL-NoSort", lambda: RLNoPrioritySort()),
        ("Utility", lambda: UtilityScheduler()),
    ]

    print(f"\n  RL Ablation Study — {args.tasks} tasks × {args.workers} workers × {args.seeds} seeds\n")

    all_results = {}  # condition_name → list[dict] across seeds

    for seed_i in range(args.seeds):
        seed = 42 + seed_i
        task_gen = TaskGenerator(seed=seed)
        worker_gen = WorkerGenerator(seed=seed)
        tasks = task_gen.generate(n=args.tasks, max_deadline=500.0, failure_prob_range=(0.0, 0.15))
        workers = worker_gen.generate(n=args.workers)

        for name, factory in conditions:
            try:
                sched = factory()
                result = run_condition(sched, tasks, workers, seed)
                all_results.setdefault(name, []).append(result)
                print(f"  seed={seed}  {name:15s}  completion={result['completion_rate']:.1%}")
            except Exception as e:
                print(f"  seed={seed}  {name:15s}  ERROR: {e}")

    # ── Print comparison table ────────────────────────────────────────────────
    metrics = ["completion_rate", "sla_violation_rate", "avg_latency", "iqm_reward", "throughput"]
    col_w = [15, 18, 18, 16, 16, 16]
    headers = ["Condition", "Completion %", "SLA Viol. %", "Avg Lat (s)", "IQM Reward", "Throughput"]

    sep = "─" * (sum(col_w) + len(col_w) * 3)
    print(f"\n{'Ablation Results (mean ± std across seeds)':^{len(sep)}}")
    print(sep)
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print(sep)

    for name, _ in conditions:
        runs = all_results.get(name, [])
        if not runs:
            continue
        vals = {m: [r[m] for r in runs] for m in metrics}
        row = [name.ljust(col_w[0])]
        for i, m in enumerate(metrics):
            mean = np.mean(vals[m])
            std = np.std(vals[m])
            if m in ("completion_rate", "sla_violation_rate"):
                row.append(f"{mean*100:.1f}±{std*100:.1f}%".ljust(col_w[i + 1]))
            elif m == "avg_latency":
                row.append(f"{mean:.1f}±{std:.1f}s".ljust(col_w[i + 1]))
            elif m == "iqm_reward":
                row.append(f"{mean:.3f}±{std:.3f}".ljust(col_w[i + 1]))
            else:
                row.append(f"{mean:.1f}±{std:.1f}".ljust(col_w[i + 1]))
        print("  " + "  ".join(row))

    print(sep)

    # Winner
    best = max(all_results.items(), key=lambda x: np.mean([r["iqm_reward"] for r in x[1]]))
    print(f"\n  Best by IQM Reward: {best[0]} "
          f"(IQM={np.mean([r['iqm_reward'] for r in best[1]]):.3f})\n")


if __name__ == "__main__":
    main()
