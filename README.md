# 🧠 Arbiter Engine

**An AI-driven intelligent scheduling and resource allocation engine.**

Arbiter Engine is a decision engine that answers: *Which task should run next? On which worker? In what order? With what expected cost?*

It integrates **heuristic search**, **ML-based runtime prediction**, **utility-based multi-objective optimization**, and **failure-resilient replanning** to optimize distributed task allocation under resource constraints.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Arbiter Engine                       │
├───────────────┬──────────────────┬───────────────────────┤
│    Models     │    Schedulers    │      Simulator         │
│  Task         │  FIFO            │  Event Engine          │
│  Worker       │  Heuristic       │  Task Generator        │
│               │  ML-Enhanced     │  Failure Injection     │
│               │  Utility-Based   │  SLA Monitoring        │
├───────────────┴──────────────────┴───────────────────────┤
│              ML Predictor Layer                            │
│  Runtime Prediction │ Failure Classification               │
├──────────────────────────────────────────────────────────┤
│              Utility Optimization Layer                    │
│  Latency │ Throughput │ Fairness │ Cost │ Risk             │
├──────────────────────────────────────────────────────────┤
│              Metrics & Observability                       │
│  Jain's Fairness │ SLA Compliance │ Cost Efficiency        │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Run tests (109 tests)
python -m pytest tests/ -v

# Run a simulation
python scripts/run_simulation.py --tasks 50 --workers 5 --scheduler fifo

# Compare all 4 schedulers
python scripts/compare_schedulers.py --tasks 100 --workers 5

# Compare with failure injection
python scripts/compare_schedulers.py --tasks 100 --workers 5 --failure-mode random

# Run comprehensive 6-scenario evaluation
python scripts/evaluation_report.py

# Train ML models
python scripts/train_models.py --simulations 50 --tasks 100
```

## Project Structure

```
arbiter-engine/
├── arbiter/
│   ├── models/         # Task & Worker data models
│   ├── schedulers/     # Scheduling algorithms (FIFO, Heuristic, ML, Utility)
│   ├── ml/             # ML pipeline (training data, models)
│   ├── simulator/      # Event-driven engine, failure injection, SLA monitoring
│   └── metrics/        # Performance metrics & fairness index
├── tests/              # 109 unit & integration tests
├── scripts/            # CLI entry points & benchmarking
├── docs/               # Architecture deep-dive
└── pyproject.toml      # Project configuration
```

## Schedulers

| Scheduler | Approach | Best For |
|-----------|----------|----------|
| **FIFO** | First-come, first-served | Baseline, predictable ordering |
| **Heuristic** | Multi-factor scoring (priority, urgency, unlock potential) | Low latency, SLA compliance |
| **ML-Enhanced** | Heuristic + ML-predicted runtime & failure risk | Data-driven environments |
| **Utility** | 5-objective weighted optimization (latency, throughput, fairness, cost, risk) | Failure-resilient, high completion rate |

## Benchmark Highlights

100 tasks, 5 workers — Utility scheduler dominance under failures:

| Scenario | FIFO | Heuristic | Utility |
|----------|------|-----------|---------|
| Normal | 66 done | 57 done | **69 done** |
| Random Failures | 67 done | 54 done | **95 done** |
| Burst Failures | 45 done | 50 done | **66 done** |

Under random failures, Utility completes **42% more tasks** than FIFO and **76% more** than Heuristic by systematically accounting for risk and capacity fit.

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ | Project setup |
| 1 | ✅ | Core simulation (Task/Worker models, FIFO, Event engine) |
| 2 | ✅ | Heuristic scheduler (multi-factor scoring) |
| 3 | ✅ | ML integration (runtime prediction, failure classification) |
| 3.5 | ✅ | Realistic simulation (retries, contention, correlated failures) |
| 4 | ✅ | Failure injection & dynamic replanning |
| 5 | ✅ | Utility-based multi-objective optimization |
| 6 | ✅ | Observability & benchmarking |

See [docs/architecture.md](docs/architecture.md) for a comprehensive deep-dive.

## License

MIT
