import numpy as np
import json
import os
import logging
from collections import deque
from arbiter.schedulers.base import BaseScheduler, Assignment
from arbiter.schedulers.rl.rl_env import build_state, SchedulingState
from arbiter.models.task import Task, TaskStatus
from arbiter.models.worker import Worker, WorkerStatus

logger = logging.getLogger(__name__)

# lightweight Q-network using numpy (no torch/tf dependency)

class QNetwork:
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        rng = np.random.default_rng(42)
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.w1 = rng.normal(0, scale1, (state_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.normal(0, scale2, (hidden, action_dim)).astype(np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)

    def forward(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(0, state @ self.w1 + self.b1)  # ReLU
        return h @ self.w2 + self.b2

    def save(self, path: str):
        data = {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.w1 = np.array(data["w1"], dtype=np.float32)
        self.b1 = np.array(data["b1"], dtype=np.float32)
        self.w2 = np.array(data["w2"], dtype=np.float32)
        self.b2 = np.array(data["b2"], dtype=np.float32)


class ReplayBuffer:
    def __init__(self, maxlen: int = 10000):
        self._buf = deque(maxlen=maxlen)

    def add(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self._buf), min(batch_size, len(self._buf)), replace=False)
        batch = [self._buf[i] for i in indices]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self._buf)


class RLScheduler(BaseScheduler):
    def __init__(self, max_workers: int = 20, model_path: str = "models/rl_policy.json",
                 epsilon: float = 0.1, learning_rate: float = 0.001, gamma: float = 0.99):
        self._max_workers = max_workers
        self._model_path = model_path
        self._epsilon = epsilon
        self._lr = learning_rate
        self._gamma = gamma
        self._state_dim = 68
        self._q_net = QNetwork(self._state_dim, max_workers)
        self._target_net = QNetwork(self._state_dim, max_workers)
        self._buffer = ReplayBuffer()
        self._steps = 0
        self._trained = False
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self._model_path):
            try:
                self._q_net.load(self._model_path)
                self._target_net.load(self._model_path)
                self._trained = True
                logger.info("Loaded RL policy from %s", self._model_path)
            except Exception as e:
                logger.warning("Failed to load RL policy: %s", e)

    def schedule(self, tasks: list[Task], workers: list[Worker],
                 completed_task_ids: set[str]) -> list[Assignment]:
        available = [w for w in workers if w.status != WorkerStatus.DOWN
                     and w.current_load < w.cpu_capacity]
        ready = [t for t in tasks if t.status == TaskStatus.QUEUED
                 and t.id not in completed_task_ids
                 and all(d in completed_task_ids for d in t.dependencies)]

        if not available or not ready:
            return []

        assignments = []
        used_workers = set()
        num_workers = min(len(available), self._max_workers)

        for task in sorted(ready, key=lambda t: (-t.priority, t.deadline)):
            state = build_state(task, available, tasks, 0.0)
            state_vec = state.to_vector()

            # epsilon-greedy
            if not self._trained or np.random.random() < self._epsilon:
                action = np.random.randint(0, num_workers)
            else:
                q_vals = self._q_net.forward(state_vec)
                # mask unavailable workers
                for i in range(self._max_workers):
                    if i >= num_workers or available[i].id in used_workers:
                        q_vals[i] = -1e9
                action = int(np.argmax(q_vals))

            if action >= num_workers:
                action = 0

            worker = available[action]
            if worker.id in used_workers:
                # fallback: pick first free worker
                for i, w in enumerate(available):
                    if w.id not in used_workers:
                        worker = w
                        break
                else:
                    continue

            if worker.current_load + task.compute_cost > worker.cpu_capacity:
                continue

            used_workers.add(worker.id)
            assignments.append(Assignment(
                task_id=task.id, worker_id=worker.id, scheduled_time=0.0,
            ))

        return assignments

    def update(self, state_vec, action, reward, next_state_vec, done):
        self._buffer.add(state_vec, action, reward, next_state_vec, done)
        if len(self._buffer) < 32:
            return

        states, actions, rewards, next_states, dones = self._buffer.sample(32)

        # compute targets
        next_q = self._target_net.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self._gamma * max_next_q * (1 - dones)

        # current Q values
        current_q = self._q_net.forward(states)
        for i in range(len(actions)):
            error = targets[i] - current_q[i, actions[i]]
            # gradient step on output layer for the chosen action
            h = np.maximum(0, states[i] @ self._q_net.w1 + self._q_net.b1)
            self._q_net.w2[:, actions[i]] += self._lr * error * h
            self._q_net.b2[actions[i]] += self._lr * error

        self._steps += 1
        if self._steps % 100 == 0:
            self._target_net.w1 = self._q_net.w1.copy()
            self._target_net.b1 = self._q_net.b1.copy()
            self._target_net.w2 = self._q_net.w2.copy()
            self._target_net.b2 = self._q_net.b2.copy()

    def save(self):
        self._q_net.save(self._model_path)
        self._trained = True

    @property
    def name(self) -> str:
        return "RLScheduler"
