"""Metrics Collector — measures scheduling performance."""

from dataclasses import dataclass, field
from typing import Optional

from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


@dataclass
class MetricsReport:
    """Container for all computed metrics."""
    scheduler_name: str = ""
    total_tasks: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_pending: int = 0
    avg_completion_time: float = 0.0
    p95_latency: float = 0.0
    max_latency: float = 0.0
    throughput: float = 0.0
    avg_worker_utilization: float = 0.0
    sla_violation_rate: float = 0.0
    failure_rate: float = 0.0
    total_simulation_time: float = 0.0
    per_worker_utilization: dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Computes and reports scheduling performance metrics."""

    def __init__(self):
        self.report: Optional[MetricsReport] = None

    def calculate(
        self,
        tasks: list[Task],
        workers: list[Worker],
        total_time: float,
        scheduler_name: str,
    ) -> MetricsReport:
        """Compute all metrics from final task/worker states."""
        report = MetricsReport(
            scheduler_name=scheduler_name,
            total_tasks=len(tasks),
            total_simulation_time=total_time,
        )

        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in tasks if t.status == TaskStatus.FAILED]
        pending_tasks = [t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.QUEUED)]

        report.tasks_completed = len(completed_tasks)
        report.tasks_failed = len(failed_tasks)
        report.tasks_pending = len(pending_tasks)

        # Latency = completion_time - arrival_time
        latencies = [t.latency for t in completed_tasks if t.latency is not None]
        if latencies:
            latencies.sort()
            report.avg_completion_time = sum(latencies) / len(latencies)
            report.max_latency = max(latencies)
            p95_index = int(len(latencies) * 0.95)
            report.p95_latency = latencies[min(p95_index, len(latencies) - 1)]

        if total_time > 0:
            report.throughput = report.tasks_completed / total_time

        if completed_tasks:
            sla_violations = sum(1 for t in completed_tasks if t.is_sla_violated)
            report.sla_violation_rate = sla_violations / len(completed_tasks)

        finished_tasks = len(completed_tasks) + len(failed_tasks)
        if finished_tasks > 0:
            report.failure_rate = len(failed_tasks) / finished_tasks

        # Per-worker utilization: busy_time / total_time
        if total_time > 0 and workers:
            for worker in workers:
                worker_tasks = [
                    t for t in completed_tasks
                    if t.assigned_worker == worker.id
                ]
                if worker_tasks:
                    busy_time = sum(
                        (t.completion_time - t.start_time)
                        for t in worker_tasks
                        if t.start_time is not None and t.completion_time is not None
                    )
                    utilization = min(1.0, busy_time / total_time)
                else:
                    utilization = 0.0
                report.per_worker_utilization[worker.id] = utilization

            report.avg_worker_utilization = (
                sum(report.per_worker_utilization.values()) / len(workers)
            )

        self.report = report
        return report

    def print_report(self) -> None:
        """Print formatted metrics report (rich if available, plain otherwise)."""
        if self.report is None:
            print("No metrics calculated yet. Run calculate() first.")
            return

        r = self.report
        if HAS_RICH:
            self._print_rich_report(r)
        else:
            self._print_plain_report(r)

    def _print_rich_report(self, r: MetricsReport) -> None:
        console = Console()

        console.print(Panel(
            f"[bold cyan]Arbiter Engine — Simulation Report[/bold cyan]\n"
            f"Scheduler: [bold yellow]{r.scheduler_name}[/bold yellow]",
            border_style="cyan",
        ))

        task_table = Table(title="Task Summary", border_style="blue")
        task_table.add_column("Metric", style="bold")
        task_table.add_column("Value", justify="right")
        task_table.add_row("Total Tasks", str(r.total_tasks))
        task_table.add_row("Completed", f"[green]{r.tasks_completed}[/green]")
        task_table.add_row("Failed", f"[red]{r.tasks_failed}[/red]")
        task_table.add_row("Pending", f"[yellow]{r.tasks_pending}[/yellow]")
        console.print(task_table)

        perf_table = Table(title="Performance Metrics", border_style="green")
        perf_table.add_column("Metric", style="bold")
        perf_table.add_column("Value", justify="right")
        perf_table.add_row("Avg Completion Time", f"{r.avg_completion_time:.2f}")
        perf_table.add_row("P95 Latency", f"{r.p95_latency:.2f}")
        perf_table.add_row("Max Latency", f"{r.max_latency:.2f}")
        perf_table.add_row("Throughput (tasks/time)", f"{r.throughput:.4f}")
        perf_table.add_row(
            "SLA Violation Rate",
            f"[{'red' if r.sla_violation_rate > 0.1 else 'green'}]"
            f"{r.sla_violation_rate:.1%}[/]"
        )
        perf_table.add_row(
            "Failure Rate",
            f"[{'red' if r.failure_rate > 0.1 else 'green'}]{r.failure_rate:.1%}[/]"
        )
        perf_table.add_row("Simulation Time", f"{r.total_simulation_time:.2f}")
        console.print(perf_table)

        if r.per_worker_utilization:
            worker_table = Table(title="Worker Utilization", border_style="magenta")
            worker_table.add_column("Worker", style="bold")
            worker_table.add_column("Utilization", justify="right")
            for worker_id, util in sorted(r.per_worker_utilization.items()):
                bar_len = int(util * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                worker_table.add_row(worker_id, f"{bar} {util:.1%}")
            worker_table.add_row(
                "[bold]Average[/bold]",
                f"[bold]{r.avg_worker_utilization:.1%}[/bold]",
            )
            console.print(worker_table)

    def _print_plain_report(self, r: MetricsReport) -> None:
        print(f"\n{'='*50}")
        print(f"  Arbiter Engine — Simulation Report")
        print(f"  Scheduler: {r.scheduler_name}")
        print(f"{'='*50}")
        print(f"  Total Tasks:        {r.total_tasks}")
        print(f"  Completed:          {r.tasks_completed}")
        print(f"  Failed:             {r.tasks_failed}")
        print(f"  Pending:            {r.tasks_pending}")
        print(f"{'─'*50}")
        print(f"  Avg Completion Time: {r.avg_completion_time:.2f}")
        print(f"  P95 Latency:         {r.p95_latency:.2f}")
        print(f"  Throughput:          {r.throughput:.4f}")
        print(f"  SLA Violation Rate:  {r.sla_violation_rate:.1%}")
        print(f"  Failure Rate:        {r.failure_rate:.1%}")
        print(f"  Avg Utilization:     {r.avg_worker_utilization:.1%}")
        print(f"  Simulation Time:     {r.total_simulation_time:.2f}")
        print(f"{'='*50}\n")
