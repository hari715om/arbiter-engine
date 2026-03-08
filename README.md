# Arbiter Engine

An intelligent, production-ready distributed task scheduling system built from the ground up in Python.

## What it is

Arbiter Engine is a **real scheduling system** — not a simulation toy. It exposes a REST API, persists state in PostgreSQL, runs background workers via Celery, and ships with five scheduling algorithms including a custom DQN reinforcement learning scheduler.

## Architecture

```
                   ┌───────────────────────────────────────┐
                   │           FastAPI REST API            │
                   │  /tasks  /workers  /schedule /metrics │
                   │  /health  /events  /ws/events         │
                   └────────────┬─────────────┬────────────┘
                                │             │
              ┌─────────────────▼──┐   ┌──────▼──────────────┐
              │  PostgreSQL (state)│   │  Redis (task queue) │
              └─────────────────┬──┘   └──────┬──────────────┘
                                │             │
                    ┌───────────▼─────────────▼───────────┐
                    │         Celery Worker Pool          │
                    │  schedule_pending (beat: 5s)        │
                    │  mark_task_completed                │
                    └───────────────┬────────────────────-┘
                                    │
                    ┌──────────────-▼────────────────────────┐
                    │          Scheduler Engine              │
                    │  FIFO │ Heuristic │ Utility │ RL │ Meta│
                    └────────────────────────────────────────┘
```

## Schedulers

| Scheduler | Technique | Best For |
|-----------|-----------|----------|
| FIFO | First-in, first-out | Baseline / fairness |
| Heuristic | Priority + deadline + dependency unlock | DAG workloads |
| ML | Random Forest latency prediction | Pattern-heavy workloads |
| Utility | Composable multi-objective optimization | General purpose |
| **RL** | Custom DQN (numpy, no PyTorch) | Adaptive learning |
| **Meta** | Workload fingerprinting + strategy selection | Mixed workloads |

## Quick Start

**Local (requires PostgreSQL + Redis running):**
```bash
git clone https://github.com/you/arbiter-engine
cd arbiter-engine
python -m venv venv && venv\Scripts\activate
pip install -e ".[dev]"
arbiter-serve        # starts FastAPI on :8000
```

**Docker (full stack):**
```bash
docker compose up
```
Starts: PostgreSQL + Redis + API + Celery Worker + Celery Beat

**API Docs:** http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tasks` | Submit a task |
| GET | `/tasks/{id}` | Get task status |
| GET | `/tasks?status=pending` | List tasks with filter |
| DELETE | `/tasks/{id}` | Delete a task |
| POST | `/workers` | Register a worker |
| GET | `/workers` | List workers |
| POST | `/workers/{id}/heartbeat` | Worker health ping |
| POST | `/schedule` | Trigger scheduling cycle |
| GET | `/metrics` | JSON metrics snapshot |
| GET | `/metrics/prometheus` | Prometheus scrape endpoint |
| GET | `/health` | DB + Redis health check |
| GET | `/events` | Audit event log |
| WS | `/ws/events` | Real-time event stream |

## Observability

- **Prometheus:** scrape `/metrics/prometheus` — 8 custom collectors (task counters, latency histogram, SLA violations, queue depth, worker utilization)
- **Event Log:** every scheduling decision written to `event_log` table
- **WebSocket:** `/ws/events` for live dashboard integration
- **Heartbeat Monitor:** background thread detects worker failures via configurable timeout

## Novelty Features

### Custom DQN Scheduler
No PyTorch/TensorFlow dependency. Pure numpy 2-layer Q-network with replay buffer (10K capacity), epsilon-greedy exploration, and target network sync every 100 steps. Learns from a 68-dimensional observation vector encoding task urgency, worker capacity, and reliability.

### Adaptive Meta-Scheduler
Computes a 5-dimensional **workload fingerprint** (burstiness, priority skew, failure rate, dependency density, deadline tightness) and dynamically selects the best scheduling strategy. Logs all strategy switches with reasons.

### Explainable Scheduling
Every assignment is explainable: per-objective score contributions, human-readable reasoning string, and ranked alternative assignments.

### Synthetic Trace Generator
Generates realistic bursty workloads with Poisson + burst arrival patterns, priority distributions, dependency chains, and heterogeneous workers. Also loads real CSV cluster traces.

## Execution Backends

| Mode | Description |
|------|-------------|
| `simulated` | Fast time-delayed execution (default) |
| `docker` | Real containers with resource limits (0.5 CPU, 512MB) |

## Testing

```bash
pytest tests/ -q          # 173 tests
pytest tests/ --cov=arbiter --cov-report=term
```

**Test breakdown:**
- 109 core scheduler/simulation tests
- 22 API endpoint tests (SQLite in-memory, no PostgreSQL required)
- 15 execution + heartbeat + prometheus tests  
- 27 RL + meta-scheduler + explainer + trace tests

## Configuration

All settings via environment variables prefixed `ARBITER_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARBITER_DATABASE_URL` | `postgresql+psycopg://...` | PostgreSQL connection |
| `ARBITER_REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `ARBITER_SCHEDULER_TYPE` | `utility` | Active scheduler |
| `ARBITER_EXECUTION_MODE` | `simulated` | `simulated` or `docker` |
| `ARBITER_SCHEDULE_INTERVAL` | `5.0` | Celery beat interval (seconds) |

## RL Training

```bash
python scripts/train_rl_scheduler.py --episodes 100 --tasks 200 --workers 10
# model saved to models/rl_policy.json, auto-loaded on next start
```


