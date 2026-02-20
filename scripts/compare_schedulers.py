"""Compare schedulers side-by-side on the same scenario.

Usage:
    python scripts/compare_schedulers.py --tasks 100 --workers 5 --seed 42
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.schedulers.heuristic import HeuristicScheduler
from arbiter.simulator.generator import ScenarioGenerator
from arbiter.simulator.engine import SimulationEngine
from arbiter.metrics.collector import MetricsReport

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def run_with_scheduler(scheduler, tasks, workers, seed):
    """Run simulation with a given scheduler and return the metrics report."""
    engine = SimulationEngine(
        tasks=tasks, workers=workers,
        scheduler=scheduler, seed=seed,
    )
    metrics = engine.run()
    return metrics.report


def print_comparison(fifo_report: MetricsReport, heuristic_report: MetricsReport):
    """Print side-by-side comparison of two scheduler runs."""
    f, h = fifo_report, heuristic_report

    def delta(new, old, lower_better=True):
        if old == 0:
            return ""
        diff = new - old
        pct = (diff / old) * 100 if old != 0 else 0
        sign = "+" if diff > 0 else ""
        color = "red" if (diff > 0 and lower_better) or (diff < 0 and not lower_better) else "green"
        return f"[{color}]{sign}{pct:.1f}%[/]" if HAS_RICH else f"{sign}{pct:.1f}%"

    rows = [
        ("Tasks Completed", f.tasks_completed, h.tasks_completed, delta(h.tasks_completed, f.tasks_completed, lower_better=False)),
        ("Tasks Failed", f.tasks_failed, h.tasks_failed, delta(h.tasks_failed, f.tasks_failed)),
        ("Tasks Pending", f.tasks_pending, h.tasks_pending, delta(h.tasks_pending, f.tasks_pending)),
        ("Avg Completion Time", f"{f.avg_completion_time:.2f}", f"{h.avg_completion_time:.2f}", delta(h.avg_completion_time, f.avg_completion_time)),
        ("P95 Latency", f"{f.p95_latency:.2f}", f"{h.p95_latency:.2f}", delta(h.p95_latency, f.p95_latency)),
        ("Throughput", f"{f.throughput:.4f}", f"{h.throughput:.4f}", delta(h.throughput, f.throughput, lower_better=False)),
        ("SLA Violation Rate", f"{f.sla_violation_rate:.1%}", f"{h.sla_violation_rate:.1%}", delta(h.sla_violation_rate, f.sla_violation_rate)),
        ("Failure Rate", f"{f.failure_rate:.1%}", f"{h.failure_rate:.1%}", delta(h.failure_rate, f.failure_rate)),
        ("Avg Utilization", f"{f.avg_worker_utilization:.1%}", f"{h.avg_worker_utilization:.1%}", delta(h.avg_worker_utilization, f.avg_worker_utilization, lower_better=False)),
    ]

    if HAS_RICH:
        table = Table(title="FIFO vs Heuristic Scheduler", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("FIFO", justify="right")
        table.add_column("Heuristic", justify="right")
        table.add_column("Change", justify="right")
        for name, fifo_val, heur_val, change in rows:
            table.add_row(name, str(fifo_val), str(heur_val), change)
        console.print(table)
    else:
        print(f"\n{'Metric':<25} {'FIFO':>12} {'Heuristic':>12} {'Change':>10}")
        print("-" * 65)
        for name, fifo_val, heur_val, change in rows:
            print(f"{name:<25} {str(fifo_val):>12} {str(heur_val):>12} {change:>10}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare FIFO vs Heuristic schedulers")
    parser.add_argument("--tasks", type=int, default=100, help="Number of tasks (default: 100)")
    parser.add_argument("--workers", type=int, default=5, help="Number of workers (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--deadline-tightness", type=float, default=2.0)
    parser.add_argument("--dependency-density", type=float, default=0.2)

    args = parser.parse_args()

    # Generate ONE scenario — same for both schedulers
    gen = ScenarioGenerator(seed=args.seed)
    tasks = gen.generate_tasks(
        num_tasks=args.tasks,
        deadline_tightness=args.deadline_tightness,
        dependency_density=args.dependency_density,
    )
    workers = gen.generate_workers(num_workers=args.workers)

    if HAS_RICH:
        console.print(f"[bold]Scenario:[/bold] {len(tasks)} tasks, {len(workers)} workers, seed={args.seed}\n")

    fifo_report = run_with_scheduler(FIFOScheduler(), tasks, workers, args.seed)
    heuristic_report = run_with_scheduler(HeuristicScheduler(), tasks, workers, args.seed)

    print_comparison(fifo_report, heuristic_report)


if __name__ == "__main__":
    main()
