"""Comprehensive Evaluation Report — benchmarks all schedulers at scale.

Runs FIFO, Heuristic, ML, and Utility schedulers across multiple scenarios:
  - Normal (no failures)
  - Random failures
  - Burst failures
  - High-scale (1000 tasks)
  - Tight deadlines

Generates structured output for documentation.

Usage:
    python scripts/evaluation_report.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.schedulers.heuristic import HeuristicScheduler
from arbiter.schedulers.ml_scheduler import MLScheduler
from arbiter.schedulers.utility_scheduler import UtilityScheduler
from arbiter.simulator.generator import ScenarioGenerator
from arbiter.simulator.engine import SimulationEngine
from arbiter.simulator.failure_injector import FailureInjector


def load_ml_scheduler():
    """Try to load pre-trained ML scheduler, fall back to default."""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    if os.path.exists(model_dir):
        try:
            return MLScheduler.from_model_dir(model_dir)
        except Exception:
            pass
    return MLScheduler()


def run_scenario(scheduler, tasks, workers, seed, injector=None):
    """Run one simulation and return the report + wall-clock time."""
    t_copy = [t.model_copy(deep=True) for t in tasks]
    w_copy = [w.model_copy(deep=True) for w in workers]
    start = time.perf_counter()
    engine = SimulationEngine(
        tasks=t_copy, workers=w_copy,
        scheduler=scheduler, seed=seed,
        failure_injector=injector,
    )
    metrics = engine.run()
    elapsed = time.perf_counter() - start
    return metrics.report, elapsed


def format_row(name, report, elapsed):
    """Format a single result row."""
    r = report
    return {
        "scheduler": name,
        "completed": r.tasks_completed,
        "failed": r.tasks_failed,
        "pending": r.tasks_pending,
        "avg_latency": r.avg_completion_time,
        "p95_latency": r.p95_latency,
        "throughput": r.throughput,
        "sla_violation": r.sla_violation_rate,
        "failure_rate": r.failure_rate,
        "fairness": r.fairness_index,
        "utilization": r.avg_worker_utilization,
        "cost_efficiency": r.cost_efficiency,
        "worker_failures": r.worker_failures,
        "preempted": r.tasks_preempted,
        "retries": r.total_retries,
        "wall_clock_ms": elapsed * 1000,
    }


def print_table(title, rows, total_tasks):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"  {total_tasks} tasks")
    print(f"{'='*80}")
    print(f"{'Scheduler':>12s} | {'Done':>4s} {'Fail':>4s} {'Pend':>4s} | "
          f"{'AvgLat':>7s} {'P95':>7s} | {'SLA%':>6s} {'Fair':>5s} {'Util%':>5s} | "
          f"{'CostEff':>7s} {'Retries':>7s} | {'WF':>3s} {'Pre':>3s} | {'Time':>6s}")
    print("-" * 110)
    for row in rows:
        r = row
        wf = f"{r['worker_failures']:3d}" if r['worker_failures'] else "  -"
        pre = f"{r['preempted']:3d}" if r['preempted'] else "  -"
        print(f"{r['scheduler']:>12s} | {r['completed']:4d} {r['failed']:4d} {r['pending']:4d} | "
              f"{r['avg_latency']:7.1f} {r['p95_latency']:7.1f} | "
              f"{r['sla_violation']:5.1%} {r['fairness']:.3f} {r['utilization']:5.1%} | "
              f"{r['cost_efficiency']:6.1%} {r['retries']:7d} | "
              f"{wf} {pre} | {r['wall_clock_ms']:5.0f}ms")


def run_benchmark(scenario_name, tasks, workers, seed, injector=None, schedulers=None):
    """Run all schedulers on one scenario."""
    if schedulers is None:
        schedulers = {
            "FIFO": FIFOScheduler(),
            "Heuristic": HeuristicScheduler(),
            "ML": load_ml_scheduler(),
            "Utility": UtilityScheduler(),
        }

    rows = []
    for name, sched in schedulers.items():
        report, elapsed = run_scenario(sched, tasks, workers, seed, injector)
        rows.append(format_row(name, report, elapsed))

    print_table(scenario_name, rows, len(tasks))
    return rows


def main():
    seed = 42
    gen = ScenarioGenerator(seed=seed)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           ARBITER ENGINE — COMPREHENSIVE EVALUATION            ║")
    print("║           4-Way Scheduler Comparison (FIFO/Heur/ML/Util)       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Scenario 1: Normal (100 tasks, 5 workers)
    tasks = gen.generate_tasks(num_tasks=100)
    workers = gen.generate_workers(num_workers=5)
    all_results = {}
    all_results["normal"] = run_benchmark("Scenario 1: Normal (100 tasks, 5 workers)", tasks, workers, seed)

    # Scenario 2: Random failures
    inj = FailureInjector(mode="random", failure_rate=0.02, seed=seed)
    all_results["random"] = run_benchmark("Scenario 2: Random Failures (rate=0.02)", tasks, workers, seed, inj)

    # Scenario 3: Burst failures
    inj_burst = FailureInjector(mode="burst", failure_rate=0.02, seed=seed)
    all_results["burst"] = run_benchmark("Scenario 3: Burst Failures (AZ outage)", tasks, workers, seed, inj_burst)

    # Scenario 4: Tight deadlines
    tasks_tight = gen.generate_tasks(num_tasks=100, deadline_tightness=1.2)
    all_results["tight"] = run_benchmark("Scenario 4: Tight Deadlines (tightness=1.2)", tasks_tight, workers, seed)

    # Scenario 5: High scale (1000 tasks, 20 workers)
    # Skip ML at scale — per-pair prediction calls make it O(N²·M) per tick
    gen2 = ScenarioGenerator(seed=seed)
    tasks_large = gen2.generate_tasks(num_tasks=1000, dependency_density=0.1)
    workers_large = gen2.generate_workers(num_workers=20)
    fast_schedulers = {
        "FIFO": FIFOScheduler(),
        "Heuristic": HeuristicScheduler(),
        "Utility": UtilityScheduler(),
    }
    all_results["scale"] = run_benchmark(
        "Scenario 5: High Scale (1000 tasks, 20 workers)",
        tasks_large, workers_large, seed, schedulers=fast_schedulers,
    )

    # Scenario 6: High scale + failures
    inj_scale = FailureInjector(mode="random", failure_rate=0.01, seed=seed)
    fast_schedulers_2 = {
        "FIFO": FIFOScheduler(),
        "Heuristic": HeuristicScheduler(),
        "Utility": UtilityScheduler(),
    }
    all_results["scale_fail"] = run_benchmark(
        "Scenario 6: High Scale + Random Failures (1000 tasks, 20 workers)",
        tasks_large, workers_large, seed, inj_scale, schedulers=fast_schedulers_2,
    )

    # Summary analysis
    print(f"\n{'='*80}")
    print("  SUMMARY: Win/Loss Analysis (best in each metric per scenario)")
    print(f"{'='*80}")

    # Collect all scheduler names across scenarios
    all_schedulers = set()
    for rows in all_results.values():
        for r in rows:
            all_schedulers.add(r["scheduler"])
    wins = {name: 0 for name in sorted(all_schedulers)}

    for scenario_name, rows in all_results.items():
        # Best = most completed
        best_completed = max(rows, key=lambda r: r["completed"])
        wins[best_completed["scheduler"]] += 1
        # Best = lowest SLA violations (among those who completed > 0)
        valid = [r for r in rows if r["completed"] > 0]
        if valid:
            best_sla = min(valid, key=lambda r: r["sla_violation"])
            wins[best_sla["scheduler"]] += 1
        # Best = fairness
        best_fair = max(rows, key=lambda r: r["fairness"])
        wins[best_fair["scheduler"]] += 1

    for name, count in sorted(wins.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {name:>12s}: {count:2d} wins  {bar}")

    print(f"\nTotal scenarios: {len(all_results)}, metrics scored: completed, SLA, fairness")


if __name__ == "__main__":
    main()
