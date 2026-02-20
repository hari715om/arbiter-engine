# 🧠 Arbiter Engine

**An AI-driven intelligent scheduling and resource allocation engine.**

Arbiter Engine is a decision engine that answers: *Which task should run next? On which worker? In what order? With what expected cost?*

It integrates **heuristic search (A\*)**, **constraint satisfaction**, and **ML-based runtime prediction** to optimize distributed task allocation under resource constraints.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Arbiter Engine                  │
├─────────────┬──────────────┬────────────────────┤
│   Models    │  Schedulers  │    Simulator        │
│  Task       │  FIFO        │  Event Engine       │
│  Worker     │  Heuristic   │  Task Generator     │
│             │  A* Search   │  Failure Injection   │
├─────────────┴──────────────┴────────────────────┤
│              ML Predictor Layer                   │
│  Runtime Prediction │ Failure Classification      │
├───────────────────────────────────────────────────┤
│              Metrics & Observability              │
└───────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run a simulation
python scripts/run_simulation.py --tasks 50 --workers 5 --scheduler fifo
```

## Project Structure

```
arbiter-engine/
├── arbiter/
│   ├── models/         # Task & Worker data models
│   ├── schedulers/     # Scheduling algorithms (FIFO, Heuristic, A*)
│   ├── simulator/      # Event-driven simulation engine
│   └── metrics/        # Performance metrics collection
├── tests/              # Unit & integration tests
├── scripts/            # CLI entry points
└── pyproject.toml      # Project configuration
```

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ | Project setup |
| 1 | 🔄 | Core simulation (Task/Worker models, FIFO, Simulator) |
| 2 | ⬜ | Heuristic scheduler (A* search, constraint satisfaction) |
| 3 | ⬜ | ML integration (runtime prediction, failure classification) |
| 4 | ⬜ | Failure & dynamic replanning |
| 5 | ⬜ | Advanced features (multi-objective optimization) |
| 6 | ⬜ | Observability & benchmarking |

## License

MIT
