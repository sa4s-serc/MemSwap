"""Unit tests of the components of the managing system and of the Router. No data files, no model calls."""
import pandas as pd
import pytest

from memswap.analyze.cost_analyzer import CostAnalyzer
from memswap.analyze.workload_analyzer import WorkloadAnalyzer
from memswap.execute.plan_enactor import PlanEnactor
from memswap.knowledge.knowledge_base import AdaptationGoals, KnowledgeBase, MECHANISMS, WINDOW
from memswap.loop import MemSwap
from memswap.managed.memory_manager.router import Router
from memswap.monitor.usage_tracker import UsageTracker
from memswap.plan.mechanism_selector import select, wilson_lb_formula, wilson_lb_statsmodels
from memswap.plan.plan_optimizer import exact_plan

SL, PR, IS = MECHANISMS


def test_wilson_bound_is_cautious_on_few_questions():
    # a measured 80% is worth less on 30 questions than on 300
    assert round(100 * wilson_lb_statsmodels(24, 30), 1) == 65.7
    assert round(100 * wilson_lb_statsmodels(240, 300), 1) == 75.9
    assert wilson_lb_statsmodels(0, 0) == 0.0
    assert wilson_lb_formula(0.8, 30, 1.645) == pytest.approx(wilson_lb_statsmodels(24, 30), abs=1e-4)


def test_selection_rule_keeps_the_best_when_nothing_cheaper_is_sufficient():
    r = select({SL: 85, PR: 80, IS: 84}, 100, {SL: 6000.0, PR: 2400.0, IS: 6700.0}, 0.01)
    assert r["best"] == SL and r["chosen"] == SL


def test_selection_rule_takes_the_cheapest_sufficient_mechanism():
    # planned read is cheaper and, on 1,000 questions, surely within ten points of the best
    r = select({SL: 850, PR: 800, IS: 840}, 1000, {SL: 6000.0, PR: 2400.0, IS: 6700.0}, 0.10)
    assert r["best"] == SL and r["chosen"] == PR


def test_ties_go_to_the_cheaper_mechanism():
    r = select({SL: 50, PR: 50, IS: 50}, 100, {SL: 6000.0, PR: 2400.0, IS: 6700.0}, 0.01)
    assert r["best"] == PR


def _estimates():
    return {"a": {"acc": {SL: 0.9, PR: 0.7, IS: 0.8}, "tok": {SL: 6000.0, PR: 2000.0, IS: 7000.0}, "start": SL, "n": 40},
            "b": {"acc": {SL: 0.2, PR: 0.8, IS: 1.0}, "tok": {SL: 6000.0, PR: 2000.0, IS: 5000.0}, "start": IS, "n": 40}}


def test_plan_optimizer_picks_the_most_accurate_plan_that_fits():
    plan, moves = exact_plan(_estimates(), {"a": 1, "b": 1}, 3600.0)
    assert plan == {"a": PR, "b": IS}
    assert moves == [{"type": "a", "from": SL, "to": PR}]


def test_plan_optimizer_without_a_feasible_plan_takes_the_cheapest_mechanisms():
    plan, _ = exact_plan(_estimates(), {"a": 1, "b": 1}, 100.0)
    assert plan == {"a": PR, "b": PR}


def test_plan_optimizer_follows_the_shares():
    # when almost every question has type b, type a can keep its expensive mechanism
    plan, _ = exact_plan(_estimates(), {"a": 1, "b": 99}, 5100.0)
    assert plan == {"a": SL, "b": IS}


def test_usage_tracker_closes_a_window_every_50_questions():
    u = UsageTracker()
    means = [u.add(100.0, 150.0) for _ in range(WINDOW)]
    assert means[:-1] == [None] * (WINDOW - 1) and means[-1] == 100.0
    assert u.windows == [{"end": WINDOW, "mean": 100.0, "B": 150.0, "kept": True}]
    assert UsageTracker().add(100.0, None) is None          # no budget, no window check


def test_cost_analyzer_flags_two_consecutive_windows_over_budget():
    c = CostAnalyzer()
    assert c.violation(120.0, 100.0) is False
    assert c.violation(90.0, 100.0) is False                # the run is broken
    assert c.violation(120.0, 100.0) is False
    assert c.violation(130.0, 100.0) is True
    assert c.budget_changed(80.0, 100.0) is True and c.budget_changed(80.0, 80.0) is False


def test_router_uses_the_default_for_an_unknown_question_type_and_the_analyzer_flags_it():
    kb = KnowledgeBase(AdaptationGoals(margin=0.01))
    kb.routing.default["f"] = IS
    PlanEnactor(kb).update("f", {"a": SL})
    router = Router(kb, lambda m, qid: (1, 10.0))
    assert router.route("f", "a") == SL and router.route("f", "new") == IS
    w = WorkloadAnalyzer(kb)
    assert w.is_known("f", "a") and not w.is_known("f", "new")


def test_the_managing_system_changes_only_later_questions():
    # the request path answers with the routing table as it stands; a re-plan shows on the next question
    ids = [f"q{i}" for i in range(200)]
    S = pd.DataFrame({SL: 1, PR: 0, IS: 1}, index=ids)
    TOK = pd.DataFrame({SL: 6000.0, PR: 2000.0, IS: 7000.0}, index=ids)

    class Labels:
        def classify(self, qid):
            return "a"

    ms = MemSwap(AdaptationGoals(margin=0.01, cap=0), Labels(), lambda m, q: (S.at[q, m], TOK.at[q, m]), S, TOK)
    ms.first_batch("f", ids[:100], Labels().classify)
    assert ms.handle(ids[100], "f", 1)["mechanism"] == SL
    before = ms.plan_enactor.updates
    ms.set_budget(2500.0, at=2)                             # the architect lowers the budget
    assert ms.plan_enactor.updates == before + 1         # Execute wrote the new plan once
    assert ms.handle(ids[101], "f", 2)["mechanism"] == PR
