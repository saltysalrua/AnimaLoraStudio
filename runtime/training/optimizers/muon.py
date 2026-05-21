"""Muon optimizer build wrapper（ADR 0003 PR-C）。"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="muon",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        momentum=float(getattr(args, "muon_momentum", 0.95)),
        nesterov=bool(getattr(args, "muon_nesterov", True)),
        ns_steps=int(getattr(args, "muon_ns_steps", 5)),
    )
