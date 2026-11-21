"""Plan: Mechanism Selector. Applies the selection rule to a type's mechanism profiles and records sigma_0(t) or m*_t.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from memswap.knowledge.knowledge_base import MECHANISMS

MECHS = MECHANISMS
Z_SPEC = 1.645                      # z of the one-sided 95% Wilson bound
Z_SM = float(norm.ppf(0.95))
EPS = 1e-12


def wilson_lb_formula(p, n, z):
    """One-sided lower Wilson bound; 0 when n == 0."""
    if n == 0:
        return 0.0
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return float((centre - half) / denom)


def wilson_lb_statsmodels(k, n):
    """One-sided 95% lower bound = lower end of the two-sided 90% Wilson interval."""
    if n == 0:
        return 0.0
    lo, _ = proportion_confint(int(k), int(n), alpha=0.10, method="wilson")
    return float(lo)


Z_DISAGREEMENTS = {"count": 0, "total_rule_calls": 0, "examples": []}


def select(k, n, tok, margin, context=None):
    """Selection rule (Eq. 2 of the paper).

    k[m]   : number of correct answers of mechanism m on the profiled questions
    n      : number of profiled questions (same for every mechanism)
    tok[m] : mean memory tokens of mechanism m on the profiled questions
    margin : accuracy margin as a fraction (1 point = 0.01)
    Returns dict with best, chosen (statsmodels bound), chosen_z1645, lower bounds.
    """
    acc = {m: (k[m] / n if n else 0.0) for m in MECHS}
    # best: highest accuracy, ties -> lower mean tokens, then fixed order
    best = sorted(MECHS, key=lambda m: (-acc[m], tok[m], MECHS.index(m)))[0]
    lb_sm = {m: wilson_lb_statsmodels(k[m], n) for m in MECHS}
    lb_sp = {m: wilson_lb_formula(acc[m], n, Z_SPEC) for m in MECHS}

    def pick(lb):
        cands = [m for m in MECHS if tok[m] < tok[best] and lb[m] >= acc[best] - margin - EPS]
        if not cands:
            return best
        return sorted(cands, key=lambda m: (tok[m], MECHS.index(m)))[0]

    chosen = pick(lb_sm)
    chosen_sp = pick(lb_sp)
    Z_DISAGREEMENTS["total_rule_calls"] += 1
    if chosen != chosen_sp:
        Z_DISAGREEMENTS["count"] += 1
        if len(Z_DISAGREEMENTS["examples"]) < 20:
            Z_DISAGREEMENTS["examples"].append(
                {"context": context, "n": int(n), "acc": acc, "mean_tokens": tok,
                 "chosen_statsmodels": chosen, "chosen_z1645": chosen_sp})
    return {"best": best, "chosen": chosen, "chosen_z1645": chosen_sp, "n": int(n),
            "accuracy": acc, "mean_tokens": tok, "wilson_lb": lb_sm}


class MechanismSelector:
    """sigma_0(t) after batch profiling; m*_t (or an early cheaper choice) during runtime profiling."""

    def __init__(self, kb):
        self.kb = kb

    def decide_batch(self, fold, profiled, default_profile, context=""):
        """profiled: type -> (k, n, tok) from the MechanismEvaluator's batch; default_profile: the same over all first-batch
        questions as one type (m0). Records the rule's output as the mechanism profiles and writes sigma_0 to the table."""
        default = select(*default_profile, self.kb.goals.margin, context=context + ":default")
        self.kb.estimates.batch.setdefault(fold, {})
        self.kb.estimates.batch_default[fold] = default
        self.kb.routing.table[fold] = {}
        self.kb.routing.default[fold] = default["chosen"]
        for t in sorted(profiled):
            r = select(*profiled[t], self.kb.goals.margin, context=context + f":fb:{t}")
            self.kb.estimates.batch[fold][t] = r
            self.kb.routing.table[fold][t] = r["chosen"]
        return default

    def decide_runtime(self, qtype, context=""):
        """after a runtime-profiled question. Returns the rule's
        output when the type is decided, else None."""
        p = self.kb.estimates.runtime[qtype]
        r = select(p["k"], p["n"], {m: p["tok"][m] / p["n"] for m in MECHS}, self.kb.goals.margin,
                   context=context + f":runtime:{qtype}:n{p['n']}")
        if r["chosen"] != r["best"] or p["n"] >= self.kb.goals.cap:
            return r
        return None
