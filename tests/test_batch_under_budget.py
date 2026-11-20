"""Algorithm 1, line 8: after the Mechanism Selector writes sigma_0 for a labelled batch, the Plan Optimizer re-plans
if a budget is set. Also checks the type shares modes of the Knowledge Base.

  python -m unittest discover -s tests -v        (from the package folder; no model calls, no data files)
"""
import os
import sys
import unittest
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memswap.knowledge.knowledge_base import AdaptationGoals, MECHANISMS   # noqa: E402
from memswap.loop import MemSwap                                              # noqa: E402

SL, PR, IS = MECHANISMS


def toy(n_a=40, n_b=40):
    """two question types. Type a: single lookup is the most accurate (90%) and costs 6,000 tokens; planned read is
    cheap (2,000) and 70% accurate. Type b: iterative search is the most accurate (100%) at 5,000 tokens."""
    rows, types = {}, {}
    for i in range(n_a):
        rows[f"a{i}"] = ({SL: int(i % 10 != 0), PR: int(i % 10 < 7), IS: int(i % 10 < 8)}, {SL: 6000.0, PR: 2000.0, IS: 7000.0})
        types[f"a{i}"] = "a"
    for i in range(n_b):
        rows[f"b{i}"] = ({SL: int(i % 10 < 2), PR: int(i % 10 < 8), IS: 1}, {SL: 6000.0, PR: 2000.0, IS: 5000.0})
        types[f"b{i}"] = "b"
    S = pd.DataFrame({q: r[0] for q, r in rows.items()}).T[MECHANISMS]
    TOK = pd.DataFrame({q: r[1] for q, r in rows.items()}).T[MECHANISMS]
    return S, TOK, types


class Labels:
    def __init__(self, types):
        self.types = types

    def classify(self, qid):
        return self.types[qid]


def make(budget=None, shares="first_batch_fold", mix=None):
    S, TOK, types = toy()
    labels = Labels(types)
    ms = MemSwap(AdaptationGoals(margin=0.01, cap=0, budget=budget), labels,
                 lambda m, qid: (S.at[qid, m], TOK.at[qid, m]), S, TOK, mix=mix, shares=shares)
    return ms, labels, list(S.index)


class BatchUnderBudget(unittest.TestCase):
    def test_no_budget_keeps_sigma0(self):
        ms, labels, qids = make()
        ms.first_batch("f", qids, labels.classify)
        self.assertEqual(ms.kb.routing.table["f"], {"a": SL, "b": IS})   # sigma_0 = the most accurate mechanism
        self.assertEqual(ms.replans, [])

    def test_budget_set_replans_after_batch(self):
        # sigma_0 costs 0.5 * 6000 + 0.5 * 5000 = 5500 tokens per question; the budget allows 3600
        ms, labels, qids = make(budget=3600.0, shares="first_batch")
        ms.first_batch("f", qids, labels.classify)
        table = ms.kb.routing.table["f"]
        cost = 0.5 * {SL: 6000, PR: 2000, IS: 7000}[table["a"]] + 0.5 * {SL: 6000, PR: 2000, IS: 5000}[table["b"]]
        self.assertLessEqual(cost, 3600.0 * 1.005)
        self.assertEqual(table, {"a": PR, "b": IS})                      # the most accurate plan that fits
        self.assertEqual(ms.replans, [{"at": None, "why": "labelled batch", "B": 3600.0}])

    def test_same_plan_as_a_budget_set_after_the_batch(self):
        after, labels, qids = make(shares="first_batch")
        after.first_batch("f", qids, labels.classify)
        after.set_budget(3600.0)
        before, labels, qids = make(budget=3600.0, shares="first_batch")
        before.first_batch("f", qids, labels.classify)
        self.assertEqual(before.kb.routing.table["f"], after.kb.routing.table["f"])

    def test_budget_that_sigma0_fits_changes_nothing(self):
        ms, labels, qids = make(budget=6000.0, shares="first_batch")
        ms.first_batch("f", qids, labels.classify)
        self.assertEqual(ms.kb.routing.table["f"], {"a": SL, "b": IS})


class Shares(unittest.TestCase):
    def test_default_is_first_batch_fold(self):
        S, TOK, types = toy()
        labels = Labels(types)
        ms = MemSwap(AdaptationGoals(margin=0.01, cap=0), labels, lambda m, qid: (S.at[qid, m], TOK.at[qid, m]), S, TOK)
        self.assertEqual(ms.kb.shares_mode, "first_batch_fold")

    def test_modes(self):
        S, TOK, types = toy()
        batch = [f"a{i}" for i in range(30)] + [f"b{i}" for i in range(10)]          # first batch: 30 a, 10 b
        for mode, expected in (("given", Counter({"a": 1, "b": 99 + 2})),             # the caller's counts + served
                               ("first_batch", Counter({"a": 31, "b": 12})),          # first batch + all served
                               ("first_batch_fold", Counter({"a": 31, "b": 11}))):    # first batch + served from the fold
            labels = Labels(types)
            ms = MemSwap(AdaptationGoals(margin=0.01, cap=0), labels, lambda m, qid: (S.at[qid, m], TOK.at[qid, m]),
                         S, TOK, mix={"b": 99} if mode == "given" else None, shares=mode)
            ms.first_batch("f", batch, labels.classify)
            ms.first_batch("g", batch, labels.classify)
            ms.handle("a30", "f", 1); ms.handle("b10", "f", 2); ms.handle("b11", "g", 3)
            self.assertEqual(ms.kb.shares("f"), expected, mode)


if __name__ == "__main__":
    unittest.main()
