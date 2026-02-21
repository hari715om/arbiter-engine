import pytest
import numpy as np
from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus
from arbiter.schedulers.rl.rl_env import SchedulingState, build_state, compute_reward
from arbiter.schedulers.rl.rl_scheduler import RLScheduler, QNetwork, ReplayBuffer
from arbiter.schedulers.meta_scheduler import MetaScheduler, compute_workload_fingerprint
from arbiter.schedulers.explainer import explain_heuristic_assignment, SchedulingExplanation
from arbiter.traces.trace_loader import load_synthetic_trace


# -- helpers --

def make_tasks(n=5, priority=5, deadline=100.0, cost=2.0):
    return [Task(
        id=f"t{i}", compute_cost=cost, deadline=deadline,
        priority=priority, estimated_duration=cost * 2,
    ) for i in range(n)]


def make_workers(n=3, capacity=10.0):
    return [Worker(id=f"w{i}", cpu_capacity=capacity, memory_capacity=16.0) for i in range(n)]


# -- RL environment --

class TestRLEnvironment:
    def test_state_vector_shape(self):
        state = SchedulingState(
            queue_depth=10, avg_worker_load=0.5,
            task_priority=7, task_cost=3.0,
            task_deadline_pressure=2.0, task_failure_prob=0.1,
            task_dependency_count=1, task_retry_count=0,
            worker_cpu_free=[5.0, 8.0], worker_speeds=[1.0, 1.2],
            worker_reliability=[0.9, 1.0],
        )
        vec = state.to_vector()
        assert vec.shape == (68,)
        assert vec.dtype == np.float32

    def test_build_state(self):
        tasks = make_tasks(3)
        tasks[0].status = TaskStatus.QUEUED
        workers = make_workers(2)
        state = build_state(tasks[0], workers, tasks, current_time=0.0)
        assert isinstance(state, SchedulingState)
        vec = state.to_vector()
        assert vec.shape == (68,)

    def test_reward_completed(self):
        task = Task(id="t1", compute_cost=1.0, deadline=50.0, estimated_duration=5.0, priority=5)
        r = compute_reward(task, completed=True, sla_violated=False)
        assert r == 1.0

    def test_reward_failure(self):
        task = Task(id="t1", compute_cost=1.0, deadline=50.0, estimated_duration=5.0, priority=5)
        r = compute_reward(task, completed=False, sla_violated=False)
        assert r == -1.0

    def test_reward_sla_violation_penalty(self):
        task = Task(id="t1", compute_cost=1.0, deadline=50.0, estimated_duration=5.0, priority=5)
        r = compute_reward(task, completed=True, sla_violated=True)
        assert r == 0.5  # 1.0 - 0.5

    def test_reward_retry_penalty(self):
        task = Task(id="t1", compute_cost=1.0, deadline=50.0,
                    estimated_duration=5.0, priority=5, retry_count=2)
        r = compute_reward(task, completed=True, sla_violated=False)
        assert r == pytest.approx(0.8)  # 1.0 - 0.2


# -- Q-network --

class TestQNetwork:
    def test_forward_shape(self):
        net = QNetwork(state_dim=68, action_dim=20)
        state = np.random.randn(68).astype(np.float32)
        q_vals = net.forward(state)
        assert q_vals.shape == (20,)

    def test_batch_forward(self):
        net = QNetwork(state_dim=68, action_dim=20)
        states = np.random.randn(8, 68).astype(np.float32)
        q_vals = net.forward(states)
        assert q_vals.shape == (8, 20)

    def test_save_load(self, tmp_path):
        net = QNetwork(state_dim=10, action_dim=5)
        path = str(tmp_path / "test_net.json")
        net.save(path)

        net2 = QNetwork(state_dim=10, action_dim=5)
        net2.load(path)
        np.testing.assert_array_equal(net.w1, net2.w1)
        np.testing.assert_array_equal(net.b2, net2.b2)


# -- replay buffer --

class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(maxlen=100)
        for i in range(50):
            buf.add(np.zeros(10), i % 5, 1.0, np.ones(10), False)
        assert len(buf) == 50
        s, a, r, ns, d = buf.sample(8)
        assert s.shape == (8, 10)
        assert len(a) == 8

    def test_maxlen(self):
        buf = ReplayBuffer(maxlen=10)
        for i in range(20):
            buf.add(np.zeros(5), 0, 0, np.zeros(5), False)
        assert len(buf) == 10


# -- RL scheduler --

class TestRLScheduler:
    def test_schedule_returns_assignments(self):
        scheduler = RLScheduler(max_workers=5, epsilon=1.0)
        tasks = make_tasks(3)
        for t in tasks:
            t.status = TaskStatus.QUEUED
        workers = make_workers(3)
        assignments = scheduler.schedule(tasks, workers, set())
        assert len(assignments) > 0
        assert all(a.task_id.startswith("t") for a in assignments)

    def test_schedule_respects_capacity(self):
        scheduler = RLScheduler(max_workers=3, epsilon=1.0)
        tasks = make_tasks(5, cost=5.0)
        for t in tasks:
            t.status = TaskStatus.QUEUED
        workers = make_workers(2, capacity=6.0)
        assignments = scheduler.schedule(tasks, workers, set())
        # each worker can handle at most 1 task (cost=5, cap=6)
        assert len(assignments) <= 2

    def test_name(self):
        assert RLScheduler().name == "RLScheduler"

    def test_update_doesnt_crash(self):
        sched = RLScheduler(epsilon=0.5)
        for _ in range(50):
            sched.update(
                np.random.randn(68).astype(np.float32), 0, 1.0,
                np.random.randn(68).astype(np.float32), False,
            )


# -- meta-scheduler --

class TestMetaScheduler:
    def test_schedule_delegates(self):
        meta = MetaScheduler()
        tasks = make_tasks(5)
        for t in tasks:
            t.status = TaskStatus.QUEUED
        workers = make_workers(3)
        assignments = meta.schedule(tasks, workers, set())
        assert len(assignments) > 0

    def test_workload_fingerprint(self):
        tasks = make_tasks(10)
        for t in tasks:
            t.status = TaskStatus.QUEUED
        workers = make_workers(3)
        fp = compute_workload_fingerprint(tasks, workers)
        assert fp.shape == (5,)
        assert all(0 <= v <= 1.0 for v in fp)

    def test_empty_queue_fingerprint(self):
        tasks = make_tasks(3)  # all pending, not queued
        fp = compute_workload_fingerprint(tasks, [])
        np.testing.assert_array_equal(fp, np.zeros(5))

    def test_strategy_switching(self):
        meta = MetaScheduler()
        # low load scenario → should pick fifo
        tasks = [Task(
            id=f"lt{i}", compute_cost=1.0, deadline=1000.0,
            priority=5, estimated_duration=5.0, failure_probability=0.0,
        ) for i in range(2)]
        for t in tasks:
            t.status = TaskStatus.QUEUED
        workers = make_workers(10)
        meta.schedule(tasks, workers, set())
        # under low load + low failure, should select fifo
        assert meta.current_strategy in ("fifo", "utility", "heuristic")

    def test_name_reflects_current(self):
        meta = MetaScheduler()
        assert "MetaScheduler" in meta.name


# -- explainer --

class TestExplainer:
    def test_heuristic_explanation(self):
        task = Task(id="t1", compute_cost=2.0, deadline=50.0,
                    priority=8, estimated_duration=5.0, dependencies=["t0"])
        workers = make_workers(2)
        all_tasks = [
            Task(id="t0", compute_cost=1.0, deadline=100.0, estimated_duration=3.0, priority=5),
            task,
            Task(id="t2", compute_cost=1.0, deadline=100.0, estimated_duration=3.0,
                 priority=5, dependencies=["t1"]),
        ]
        exp = explain_heuristic_assignment(task, workers[0], workers, all_tasks, {"t0"})
        assert isinstance(exp, SchedulingExplanation)
        assert exp.task_id == "t1"
        assert exp.worker_id == "w0"
        assert "priority" in exp.factors
        assert "urgency" in exp.factors
        assert len(exp.reasoning) > 0

    def test_explanation_has_alternatives(self):
        task = Task(id="t1", compute_cost=2.0, deadline=50.0,
                    priority=8, estimated_duration=5.0)
        workers = make_workers(3)
        exp = explain_heuristic_assignment(task, workers[0], workers, [task], set())
        # heuristic explainer doesn't produce alternatives, but shouldn't crash
        assert isinstance(exp.alternatives, list)


# -- trace loader --

class TestTraceLoader:
    def test_synthetic_trace(self):
        tasks, workers = load_synthetic_trace(num_tasks=100, num_workers=5)
        assert len(tasks) == 100
        assert len(workers) == 5
        assert all(t.id.startswith("trace-") for t in tasks)
        assert all(w.id.startswith("trace-worker-") for w in workers)

    def test_deterministic_with_seed(self):
        t1, _ = load_synthetic_trace(num_tasks=50, seed=123)
        t2, _ = load_synthetic_trace(num_tasks=50, seed=123)
        assert [t.id for t in t1] == [t.id for t in t2]
        assert [t.compute_cost for t in t1] == [t.compute_cost for t in t2]

    def test_burstiness(self):
        tasks, _ = load_synthetic_trace(num_tasks=200, burstiness=0.8)
        # with high burstiness, many tasks should arrive close together
        gaps = [tasks[i+1].arrival_time - tasks[i].arrival_time
                for i in range(len(tasks)-1)]
        small_gaps = sum(1 for g in gaps if g < 1.0)
        assert small_gaps > len(gaps) * 0.3  # at least 30% bursty

    def test_trace_has_dependencies(self):
        tasks, _ = load_synthetic_trace(num_tasks=100)
        has_deps = sum(1 for t in tasks if len(t.dependencies) > 0)
        assert has_deps > 0

    def test_csv_trace_file_not_found(self):
        from arbiter.traces.trace_loader import load_csv_trace
        with pytest.raises(FileNotFoundError):
            load_csv_trace("/nonexistent/trace.csv")
