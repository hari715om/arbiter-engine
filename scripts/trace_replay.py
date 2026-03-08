import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arbiter.simulator.generator import TaskGenerator, WorkerGenerator
from arbiter.simulator.engine import SimulationEngine
from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.schedulers.heuristic import HeuristicScheduler
from arbiter.schedulers.utility_scheduler import UtilityScheduler
from arbiter.schedulers.meta_scheduler import MetaScheduler


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _ms(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1f}s"


def run_scheduler(scheduler, tasks, workers, seed: int) -> dict:
    import copy
    t_copy = copy.deepcopy(tasks)
    w_copy = copy.deepcopy(workers)

    engine = SimulationEngine(
        tasks=t_copy,
        workers=w_copy,
        scheduler=scheduler,
        seed=seed,
    )
    t0 = time.perf_counter()
    report = engine.run()
    wall = time.perf_counter() - t0

    latencies = [
        t.completion_time - t.arrival_time
        for t in engine.tasks.values()
        if t.completion_time is not None and t.arrival_time is not None
    ]
    p95_lat = None
    if latencies:
        latencies_sorted = sorted(latencies)
        idx = int(len(latencies_sorted) * 0.95)
        p95_lat = latencies_sorted[max(0, idx - 1)]

    total = len(tasks)
    completed = report.tasks_completed
    failed = report.tasks_failed
    sla_viols = getattr(report, "sla_violations", 0)

    return {
        "scheduler": scheduler.name,
        "completed": completed,
        "total": total,
        "completion_rate": completed / total if total else 0.0,
        "failed": failed,
        "sla_violations": sla_viols,
        "sla_violation_rate": sla_viols / max(completed, 1),
        "avg_latency": report.avg_latency,
        "p95_latency": p95_lat,
        "preempted": getattr(report, "tasks_preempted", 0),
        "fairness": getattr(report, "fairness_index", None),
        "wall_time_s": round(wall, 2),
    }


def print_table(results: list[dict]) -> None:
    """Print a rich comparison table to stdout."""
    col_widths = [20, 14, 16, 14, 14, 14, 10, 10]
    headers = [
        "Scheduler", "Completed", "Completion%", "SLA Viols%",
        "Avg Latency", "P95 Latency", "Preempted", "Wall(s)",
    ]

    sep = "─" * (sum(col_widths) + len(col_widths) * 3)
    print(f"\n{'Arbiter Engine — Trace Replay Benchmark':^{len(sep)}}")
    print(sep)
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"  {header_row}")
    print(sep)

    # Sort by completion rate descending
    results_sorted = sorted(results, key=lambda r: r["completion_rate"], reverse=True)

    for r in results_sorted:
        row_vals = [
            r["scheduler"][:col_widths[0]],
            f"{r['completed']}/{r['total']}",
            _pct(r["completion_rate"]),
            _pct(r["sla_violation_rate"]),
            _ms(r["avg_latency"]),
            _ms(r["p95_latency"]),
            str(r["preempted"]),
            str(r["wall_time_s"]),
        ]
        row = "  ".join(v.ljust(w) for v, w in zip(row_vals, col_widths))
        print(f"  {row}")

    print(sep)

    # Winner
    best = results_sorted[0]
    worst = results_sorted[-1]
    gain = (best["completion_rate"] - worst["completion_rate"]) * 100
    print(f"\n  Winner: {best['scheduler']} ({_pct(best['completion_rate'])} completion)")
    print(f"  vs worst ({worst['scheduler']}): +{gain:.1f}pp completion rate\n")


def main():
    parser = argparse.ArgumentParser(description="Arbiter trace replay benchmark")
    parser.add_argument("--tasks", type=int, default=200, help="Number of tasks in trace")
    parser.add_argument("--workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--failure-rate", type=float, default=0.15,
                        help="Worker failure probability per task")
    parser.add_argument("--burstiness", type=float, default=0.3,
                        help="Task arrival burstiness [0-1]")
    parser.add_argument("--trace", type=str, default="synthetic",
                        choices=["synthetic", "google", "alibaba"],
                        help="Trace source (google/alibaba require data file)")
    parser.add_argument("--trace-file", type=str, default=None,
                        help="Path to trace CSV file (for --trace google/alibaba)")
    parser.add_argument("--out", type=str, default=None,
                        help="Save CSV results to this path")
    args = parser.parse_args()

    print(f"\n  Loading trace: {args.trace} | tasks={args.tasks} | "
          f"workers={args.workers} | seed={args.seed}")

    # ── Load trace ─────────────────────────────────────────────────────────────
    if args.trace == "synthetic":
        task_gen = TaskGenerator(seed=args.seed)
        worker_gen = WorkerGenerator(seed=args.seed)
        tasks = task_gen.generate(
            n=args.tasks,
            max_deadline=500.0,
            failure_prob_range=(0.0, args.failure_rate),
        )
        workers = worker_gen.generate(n=args.workers)
    elif args.trace in ("google", "alibaba"):
        if not args.trace_file:
            print(f"\n  ERROR: --trace-file required for trace={args.trace}")
            print(f"  See arbiter/traces/README_TRACES.md for download instructions.")
            sys.exit(1)
        from arbiter.traces.trace_loader import (
            load_google_borg_trace, load_alibaba_2018_trace
        )
        loader = load_google_borg_trace if args.trace == "google" else load_alibaba_2018_trace
        tasks, workers = loader(args.trace_file, max_tasks=args.tasks, n_workers=args.workers)
    else:
        raise ValueError(f"Unknown trace: {args.trace}")

    print(f"  Loaded {len(tasks)} tasks, {len(workers)} workers")

    schedulers = [
        FIFOScheduler(),
        HeuristicScheduler(),
        UtilityScheduler(),
        MetaScheduler(),
    ]

    try:
        from arbiter.schedulers.ml_scheduler import MLScheduler
        schedulers.append(MLScheduler())
        print("  MLScheduler: loaded")
    except Exception as e:
        print(f"  MLScheduler: skipped ({e})")

    try:
        from arbiter.schedulers.rl_scheduler import RLScheduler
        schedulers.append(RLScheduler())
        print("  RLScheduler: loaded")
    except Exception as e:
        print(f"  RLScheduler: skipped ({e})")

    print(f"\n  Running {len(schedulers)} schedulers...")

    results = []
    for sched in schedulers:
        try:
            r = run_scheduler(sched, tasks, workers, seed=args.seed)
            results.append(r)
            print(f"    ✓ {r['scheduler']}: {r['completed']}/{r['total']} "
                  f"({_pct(r['completion_rate'])})")
        except Exception as e:
            print(f"    ✗ {sched.name}: ERROR — {e}")

    if results:
        print_table(results)

    if args.out and results:
        import csv
        out_path = Path(args.out)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
