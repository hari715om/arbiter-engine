from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from arbiter.logging_config import get_logger

log = get_logger(__name__)

_VALID_SCHEDULERS = {"fifo", "heuristic", "utility", "ml", "rl", "meta"}


@dataclass
class PolicyRule:
    condition: str              # e.g. "queue_depth > 100"
    scheduler: str

    def evaluate(self, ctx: dict) -> bool:
        m = re.match(
            r'^(\w+)\s*(>|<|>=|<=|==|!=)\s*([0-9.]+)$',
            self.condition.strip(),
        )
        if not m:
            return False
        var, op, val_str = m.groups()
        actual = ctx.get(var)
        if actual is None:
            return False
        try:
            threshold = float(val_str)
            actual_f = float(actual)
        except (ValueError, TypeError):
            return False
        return eval(f"{actual_f} {op} {threshold}", {"__builtins__": {}})  # noqa: S307


@dataclass
class Policy:
    default_scheduler: str = "utility"
    rules: list[PolicyRule] = field(default_factory=list)
    utility_weights: dict[str, float] = field(default_factory=lambda: {
        "latency": 0.40,
        "throughput": 0.25,
        "fairness": 0.15,
        "cost": 0.10,
        "risk": 0.10,
    })


def _parse_policy(data: dict) -> Policy:
    """Parse a YAML dict into a Policy object."""
    default = data.get("default_scheduler", "utility")
    if default not in _VALID_SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {default}")

    raw_rules = data.get("rules", [])
    rules: list[PolicyRule] = []
    for r in raw_rules:
        cond = r.get("if", "")
        use = r.get("use", "")
        if use not in _VALID_SCHEDULERS:
            raise ValueError(f"Unknown scheduler in rule: {use}")
        rules.append(PolicyRule(condition=cond, scheduler=use))

    weights = data.get("utility_weights", {})
    default_weights = {
        "latency": 0.40, "throughput": 0.25, "fairness": 0.15,
        "cost": 0.10, "risk": 0.10,
    }
    merged = {**default_weights, **{k: float(v) for k, v in weights.items()}}

    return Policy(default_scheduler=default, rules=rules, utility_weights=merged)


class PolicyEngine:


    def __init__(self, path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._policy = Policy()
        if path and Path(path).exists():
            self.load_from_file(path)

    def load_from_file(self, path: str) -> Policy:
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required for policy files. Run: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return self._apply(data, source=f"file:{path}")

    def load_from_dict(self, data: dict) -> Policy:
        return self._apply(data, source="api")

    def _apply(self, data: dict, source: str) -> Policy:
        policy = _parse_policy(data)
        with self._lock:
            self._policy = policy
        log.info("policy_loaded", source=source,
                 default=policy.default_scheduler,
                 rules=len(policy.rules),
                 weights=policy.utility_weights)
        return policy

    def select_scheduler(self, **context) -> str:

        with self._lock:
            policy = self._policy
        for rule in policy.rules:
            if rule.evaluate(context):
                log.debug("policy_rule_matched",
                          condition=rule.condition, scheduler=rule.scheduler)
                return rule.scheduler
        return policy.default_scheduler

    @property
    def policy(self) -> Policy:
        with self._lock:
            return self._policy

    def to_dict(self) -> dict:
        with self._lock:
            p = self._policy
        return {
            "default_scheduler": p.default_scheduler,
            "rules": [{"if": r.condition, "use": r.scheduler} for r in p.rules],
            "utility_weights": p.utility_weights,
        }


# Global singleton shared by the API and Celery beat
_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    global _engine
    if _engine is None:
        _engine = PolicyEngine(path="scheduler_policy.yml")
    return _engine
