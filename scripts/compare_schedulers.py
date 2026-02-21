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
from arbiter.schedulers.ml_scheduler import MLScheduler
from arbiter.schedulers.utility_scheduler import UtilityScheduler
from arbiter.simulator.generator import ScenarioGenerator
from arbiter.simulator.engine import SimulationEngine
from arbiter.simulator.failure_injector import FailureInjector
from arbiter.metrics.collector import MetricsReport

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def run_with_scheduler(scheduler, tasks, workers, seed, failure_injector=None):
    """Run simulation with a given scheduler and return the metrics report."""
    engine = SimulationEngine(
        tasks=tasks, workers=workers,
        scheduler=scheduler, seed=seed,
        failure_injector=failure_injector,
    )
    metrics = engine.run()
    return metrics.report


def get_ml_scheduler():
    """Load ML scheduler with pre-trained models if available."""
    from pathlib import Path
    model_dir = Path("models")
    if model_dir.exists():
        return MLScheduler.from_model_dir(str(model_dir))
    return MLScheduler()


def print_comparison(reports: dict[str, MetricsReport]):
    """Print side-by-side comparison of scheduler runs."""
    names = list(reports.keys())

    def fmt_delta(new, baseline, lower_better=True):
        if baseline == 0:
            return ""
        pct = ((new - baseline) / baseline) * 100
        sign = "+" if pct > 0 else ""
        color = "red" if (pct > 0 and lower_better) or (pct < 0 and not lower_better) else "green"
        return f"[{color}]{sign}{pct:.1f}%[/]" if HAS_RICH else f"{sign}{pct:.1f}%"

    metric_defs = [
        ("Tasks Completed", lambda r: r.tasks_completed, False),
        ("Tasks Failed", lambda r: r.tasks_failed, True),
        ("Tasks Pending", lambda r: r.tasks_pending, True),
        ("Avg Completion Time", lambda r: r.avg_completion_time, True),
        ("P95 Latency", lambda r: r.p95_latency, True),
        ("Throughput", lambda r: r.throughput, False),
        ("SLA Violation Rate", lambda r: r.sla_violation_rate, True),
        ("Failure Rate", lambda r: r.failure_rate, True),
        ("Total Retries", lambda r: r.total_retries, True),
        ("Wasted Time", lambda r: r.total_wasted_time, True),
        ("Cost Efficiency", lambda r: r.cost_efficiency, False),
        ("Worker Failures", lambda r: r.worker_failures, True),
        ("Tasks Preempted", lambda r: r.tasks_preempted, True),
        ("SLA Risks", lambda r: r.sla_risks_detected, True),
        ("Fairness Index", lambda r: r.fairness_index, False),
        ("Avg Utilization", lambda r: r.avg_worker_utilization, False),
    ]

    def fmt_val(val, is_rate=False):
        if isinstance(val, int):
            return str(val)
        if is_rate:
            return f"{val:.1%}"
        return f"{val:.4f}" if val < 1 else f"{val:.2f}"

    rate_metrics = {"SLA Violation Rate", "Failure Rate", "Avg Utilization", "Cost Efficiency"}

    if HAS_RICH:
        title = " vs ".join(names)
        table = Table(title=title, border_style="cyan")
        table.add_column("Metric", style="bold")
        for name in names:
            table.add_column(name, justify="right")
        if len(names) > 1:
            for name in names[1:]:
                table.add_column(f"Δ vs {names[0]}", justify="right")

        for metric_name, extract_fn, lower_better in metric_defs:
            is_rate = metric_name in rate_metrics
            row = [metric_name]
            vals = {n: extract_fn(reports[n]) for n in names}
            for n in names:
                row.append(fmt_val(vals[n], is_rate=is_rate))
            for n in names[1:]:
                row.append(fmt_delta(vals[n], vals[names[0]], lower_better))
            table.add_row(*row)

        console.print(table)
    else:
        header = f"{'Metric':<25}"
        for n in names:
            header += f" {n:>12}"
        print(f"\n{header}")
        print("-" * (25 + 13 * len(names)))
        for metric_name, extract_fn, lower_better in metric_defs:
            is_rate = metric_name in rate_metrics
            line = f"{metric_name:<25}"
            vals = {n: extract_fn(reports[n]) for n in names}
            for n in names:
                line += f" {fmt_val(vals[n], is_rate=is_rate):>12}"
            print(line)
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare FIFO vs Heuristic vs ML schedulers")
    parser.add_argument("--tasks", type=int, default=100, help="Number of tasks (default: 100)")
    parser.add_argument("--workers", type=int, default=5, help="Number of workers (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--deadline-tightness", type=float, default=2.0)
    parser.add_argument("--dependency-density", type=float, default=0.2)
    parser.add_argument("--failure-mode", type=str, default=None,
                        choices=["random", "periodic", "burst"],
                        help="Enable worker failure injection")
    parser.add_argument("--failure-rate", type=float, default=0.02,
                        help="Failure rate for injection (default: 0.02)")

    args = parser.parse_args()

    gen = ScenarioGenerator(seed=args.seed)
    tasks = gen.generate_tasks(
        num_tasks=args.tasks,
        deadline_tightness=args.deadline_tightness,
        dependency_density=args.dependency_density,
    )
    workers = gen.generate_workers(num_workers=args.workers)

    if HAS_RICH:
        mode_str = f", failure_mode={args.failure_mode}" if args.failure_mode else ""
        console.print(f"[bold]Scenario:[/bold] {len(tasks)} tasks, {len(workers)} workers, seed={args.seed}{mode_str}\n")

    # Phase 4: create failure injector if requested
    injector = None
    if args.failure_mode:
        injector = FailureInjector(
            mode=args.failure_mode,
            failure_rate=args.failure_rate,
            seed=args.seed,
        )

    reports = {
        "FIFO": run_with_scheduler(FIFOScheduler(), tasks, workers, args.seed, injector),
        "Heuristic": run_with_scheduler(HeuristicScheduler(), tasks, workers, args.seed, injector),
        "ML": run_with_scheduler(get_ml_scheduler(), tasks, workers, args.seed, injector),
        "Utility": run_with_scheduler(UtilityScheduler(), tasks, workers, args.seed, injector),
    }

    print_comparison(reports)


if __name__ == "__main__":
    main()
