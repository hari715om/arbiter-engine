"""Entry point for running Arbiter Engine simulations.

Usage:
    python scripts/run_simulation.py --tasks 50 --workers 5 --scheduler fifo
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbiter.models.task import Task
from arbiter.models.worker import Worker
from arbiter.schedulers.fifo import FIFOScheduler
from arbiter.simulator.generator import ScenarioGenerator
from arbiter.simulator.engine import SimulationEngine

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def get_scheduler(name: str):
    """Factory function to get a scheduler by name."""
    schedulers = {
        "fifo": FIFOScheduler,
    }
    if name.lower() not in schedulers:
        available = ", ".join(schedulers.keys())
        raise ValueError(f"Unknown scheduler: {name}. Available: {available}")
    return schedulers[name.lower()]()


def print_scenario_summary(tasks: list[Task], workers: list[Worker]) -> None:
    """Print a summary of the generated scenario."""
    if HAS_RICH:
        console.print("\n[bold cyan]Generated Scenario[/bold cyan]")
        console.print(f"  Tasks:   {len(tasks)}")
        console.print(f"  Workers: {len(workers)}")

        resource_counts = {}
        for t in tasks:
            resource_counts[t.resource_type] = resource_counts.get(t.resource_type, 0) + 1
        console.print(f"  Task types: {resource_counts}")

        dep_count = sum(1 for t in tasks if t.dependencies)
        console.print(f"  Tasks with dependencies: {dep_count}")

        avg_priority = sum(t.priority for t in tasks) / len(tasks) if tasks else 0
        console.print(f"  Avg priority: {avg_priority:.1f}")

        for w in workers:
            console.print(
                f"  {w.id}: capacity={w.cpu_capacity:.0f}, "
                f"speed={w.speed_multiplier:.2f}x, "
                f"resources={w.supported_resources}"
            )
        console.print()
    else:
        print(f"\n--- Generated Scenario ---")
        print(f"Tasks: {len(tasks)}, Workers: {len(workers)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Arbiter Engine — Intelligent Resource Allocation Simulator"
    )
    parser.add_argument("--tasks", type=int, default=50, help="Number of tasks (default: 50)")
    parser.add_argument("--workers", type=int, default=5, help="Number of workers (default: 5)")
    parser.add_argument("--scheduler", type=str, default="fifo", help="Scheduler: fifo (default: fifo)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--deadline-tightness", type=float, default=2.0, help="1.0=tight, 5.0=relaxed (default: 2.0)")
    parser.add_argument("--dependency-density", type=float, default=0.2, help="Dependency probability (default: 0.2)")

    args = parser.parse_args()

    if HAS_RICH:
        console.print("[bold]Arbiter Engine[/bold] — Starting simulation...\n")

    generator = ScenarioGenerator(seed=args.seed)
    tasks = generator.generate_tasks(
        num_tasks=args.tasks,
        deadline_tightness=args.deadline_tightness,
        dependency_density=args.dependency_density,
    )
    workers = generator.generate_workers(num_workers=args.workers)

    print_scenario_summary(tasks, workers)

    scheduler = get_scheduler(args.scheduler)

    engine = SimulationEngine(
        tasks=tasks,
        workers=workers,
        scheduler=scheduler,
        seed=args.seed,
    )

    metrics = engine.run()
    metrics.print_report()

    if HAS_RICH:
        console.print(
            f"\n[dim]Processed {len(engine.event_log)} events "
            f"in {metrics.report.total_simulation_time:.2f} simulation time units[/dim]"
        )


if __name__ == "__main__":
    main()
