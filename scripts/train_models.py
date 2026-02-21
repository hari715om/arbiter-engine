"""End-to-end ML training pipeline.

Usage:
    python scripts/train_models.py --simulations 50 --tasks 100 --workers 5
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from arbiter.ml.data_generator import TrainingDataGenerator
from arbiter.ml.models import RuntimePredictor, FailureClassifier

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def main():
    parser = argparse.ArgumentParser(description="Train ML models for Arbiter Engine")
    parser.add_argument("--simulations", type=int, default=50, help="Number of simulation runs (default: 50)")
    parser.add_argument("--tasks", type=int, default=100, help="Tasks per simulation (default: 100)")
    parser.add_argument("--workers", type=int, default=5, help="Workers per simulation (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--model-dir", type=str, default="models", help="Output directory for models (default: models)")

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)

    # Step 1: Generate training data
    if HAS_RICH:
        console.print("[bold cyan]Step 1:[/bold cyan] Generating training data...")
    else:
        print("Step 1: Generating training data...")

    generator = TrainingDataGenerator(base_seed=args.seed)
    data = generator.generate(
        num_simulations=args.simulations,
        tasks_per_sim=args.tasks,
        workers_per_sim=args.workers,
    )

    total_samples = len(data)
    failures = sum(1 for d in data if d["did_fail"] == 1)

    if HAS_RICH:
        console.print(f"  Collected [bold]{total_samples}[/bold] samples ({failures} failures, {total_samples - failures} completions)\n")
    else:
        print(f"  Collected {total_samples} samples ({failures} failures, {total_samples - failures} completions)\n")

    if total_samples < 10:
        print("ERROR: Not enough training data. Increase --simulations or --tasks.")
        sys.exit(1)

    # Step 2: Train runtime predictor
    if HAS_RICH:
        console.print("[bold cyan]Step 2:[/bold cyan] Training runtime predictor (Random Forest)...")
    else:
        print("Step 2: Training runtime predictor (Random Forest)...")

    runtime_pred = RuntimePredictor(model_type="random_forest")
    runtime_metrics = runtime_pred.train(data)

    if HAS_RICH:
        table = Table(title="Runtime Predictor Metrics", border_style="green")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("RMSE", f"{runtime_metrics['rmse']:.3f}")
        table.add_row("R²", f"{runtime_metrics['r2']:.3f}")
        table.add_row("Train Samples", str(runtime_metrics['train_samples']))
        table.add_row("Test Samples", str(runtime_metrics['test_samples']))
        console.print(table)
    else:
        print(f"  RMSE: {runtime_metrics['rmse']:.3f}")
        print(f"  R²:   {runtime_metrics['r2']:.3f}")

    # Step 3: Train failure classifier
    if HAS_RICH:
        console.print("\n[bold cyan]Step 3:[/bold cyan] Training failure classifier (Random Forest)...")
    else:
        print("\nStep 3: Training failure classifier (Random Forest)...")

    failure_clf = FailureClassifier()
    failure_metrics = failure_clf.train(data)

    if HAS_RICH:
        table = Table(title="Failure Classifier Metrics", border_style="magenta")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Precision", f"{failure_metrics['precision']:.3f}")
        table.add_row("Recall", f"{failure_metrics['recall']:.3f}")
        table.add_row("F1 Score", f"{failure_metrics['f1']:.3f}")
        table.add_row("Failure Rate (train)", f"{failure_metrics['failure_rate_train']:.1%}")
        console.print(table)
    else:
        print(f"  Precision: {failure_metrics['precision']:.3f}")
        print(f"  Recall:    {failure_metrics['recall']:.3f}")
        print(f"  F1:        {failure_metrics['f1']:.3f}")

    # Step 4: Save models
    runtime_path = str(model_dir / "runtime_predictor.joblib")
    failure_path = str(model_dir / "failure_classifier.joblib")

    runtime_pred.save(runtime_path)
    failure_clf.save(failure_path)

    if HAS_RICH:
        console.print(f"\n[bold green]Models saved to:[/bold green]")
        console.print(f"  {runtime_path}")
        console.print(f"  {failure_path}")
    else:
        print(f"\nModels saved to: {runtime_path}, {failure_path}")


if __name__ == "__main__":
    main()
