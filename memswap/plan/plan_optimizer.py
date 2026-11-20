"""Plan: Plan Optimizer. Computes the plan sigma that maximises expected accuracy under the token budget B.
"""
from __future__ import annotations

from itertools import product

from memswap.knowledge.knowledge_base import MECHANISMS, TOLERANCE

TOL = TOLERANCE


class _R:                      # the name the copied code uses for the mechanism list (R.MECHS)
    MECHS = MECHANISMS


R = _R


def exact_plan(kb, mix, B):
    total = sum(mix.get(t, 0) for t in kb) or 1
    w = {t: mix.get(t, 0) / total for t in kb}
    types = sorted(kb)
    best = None
    for asg in product(R.MECHS, repeat=len(types)):
        tok = sum(w[t] * kb[t]["tok"][m] for t, m in zip(types, asg))
        if tok > B * (1 + TOL):
            continue
        acc = sum(w[t] * kb[t]["acc"][m] for t, m in zip(types, asg))
        if best is None or (acc, -tok) > (best[0], -best[1]):
            best = (acc, tok, asg)
    if best is None:   # nothing feasible: fall back to the cheapest assignment
        asg = tuple(min(R.MECHS, key=lambda m: kb[t]["tok"][m]) for t in types)
        best = (0, 0, asg)
    a = dict(zip(types, best[2]))
    start = {t: kb[t]["start"] for t in kb}
    moves = [{"type": t, "from": start[t], "to": a[t]} for t in types if a[t] != start[t]]
    return a, moves


class PlanOptimizer:
    def __init__(self, kb):
        self.kb = kb

    def planner_input(self, fold):
        """the mechanism profiles of a fold in the planner's format."""
        return {t: {"acc": r["accuracy"], "tok": r["mean_tokens"], "start": r["chosen"], "n": r["n"]}
                for t, r in self.kb.estimates.batch[fold].items()}

    def plan(self, fold):
        """returns (assignment, moves) for one fold under the current budget and type shares."""
        return exact_plan(self.planner_input(fold), self.kb.shares(fold), self.kb.goals.budget)
