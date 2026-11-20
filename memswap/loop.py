"""MemSwap's MAPE-K loop for one question, in the order of Algorithm 1.

  Execute (request path)  Question Classifier: t = classify(q); Router: m = R(t); the mechanism reads the store and answers
  Monitor                 Usage Tracker adds the question's memory tokens; the type shares w_t are updated
  Analyze                 Workload Analyzer: is t known?  Cost Analyzer: two windows over budget?
  Plan                    for a new type with cap N > 0: MechanismEvaluator trial-runs every mechanism, Mechanism Selector may
                          decide; on a budget flag: Plan Optimizer computes a new plan
  Execute                 writes the decision or the plan into the routing table, for the next question

The Router answers with the routing table as it stands, so the rest of the loop changes only later questions.
A budget change set by the architect is a separate event (set_budget): Analyze flags it, Plan re-plans, Execute writes.

This file only calls the components, in the order above.
"""
from __future__ import annotations

from collections import Counter

from memswap.analyze.cost_analyzer import CostAnalyzer
from memswap.analyze.workload_analyzer import WorkloadAnalyzer
from memswap.execute.plan_enactor import PlanEnactor
from memswap.managed.memory_manager.router import Router
from memswap.knowledge.knowledge_base import KnowledgeBase
from memswap.monitor.mechanism_evaluator import MechanismEvaluator
from memswap.monitor.usage_tracker import UsageTracker
from memswap.plan.plan_optimizer import PlanOptimizer
from memswap.plan.mechanism_selector import MechanismSelector


class MemSwap:
    def __init__(self, settings, classifier, read, S_sel, TOK, mix=None, shares="first_batch_fold"):
        self.kb = KnowledgeBase(settings, mix=Counter(mix or {}), shares_mode=shares)
        self.question_classifier = classifier
        self.router = Router(self.kb, read)
        self.usage_tracker = UsageTracker()
        self.mechanism_evaluator = MechanismEvaluator(self.kb, S_sel, TOK)
        self.workload_analyzer = WorkloadAnalyzer(self.kb)
        self.cost_analyzer = CostAnalyzer()
        self.mechanism_selector = MechanismSelector(self.kb)
        self.plan_optimizer = PlanOptimizer(self.kb)
        self.plan_enactor = PlanEnactor(self.kb)
        self.replans = []

    # ---- before deployment: batch profiling of a first batch ----
    def first_batch(self, fold, qids, type_of, context=""):
        """a labelled batch from the architect (Algorithm 1, lines 7 and 8): the MechanismEvaluator measures every mechanism,
        the Mechanism Selector writes sigma_0 into the routing table, and, if a budget is set, the Plan Optimizer
        re-plans the fold."""
        profiled, as_one_type = self.mechanism_evaluator.batch_profile(qids, type_of)
        self.kb.first_batch_mix[fold] = Counter(type_of(qid) for qid in qids)
        default = self.mechanism_selector.decide_batch(fold, profiled, as_one_type, context)
        if self.kb.goals.budget is not None:
            assignment, _ = self.plan_optimizer.plan(fold)      # Plan
            self.plan_enactor.update(fold, assignment)       # Execute
            self.replans.append({"at": None, "why": "labelled batch", "B": self.kb.goals.budget})
        return default

    # ---- architect event: a new budget ----
    def set_budget(self, budget, at=None):
        """Analyze flags the change; Plan re-plans every fold; Execute writes the plans. Returns the first fold's moves.
        `at` (1-based question position) is recorded as a re-plan; None for the budget in force from the start."""
        changed = self.cost_analyzer.budget_changed(budget, self.kb.goals.budget)
        self.kb.goals.budget = budget
        if not changed:
            return None
        moves = self._plan_and_write()
        if at is not None:
            self.replans.append({"at": at, "why": "budget change", "B": budget})
        return moves

    def _plan_and_write(self):
        first_moves = None
        plans = {g: self.plan_optimizer.plan(g) for g in sorted(self.kb.routing.table)}   # Plan
        for g, (a, mv) in plans.items():                                                  # Execute
            self.plan_enactor.update(g, a)
            if first_moves is None:
                first_moves = mv
        return first_moves

    # ---- one question ----
    def handle(self, qid, fold, position, context=""):
        # Execute, request path: Question Classifier, then Router (reads the routing table as it stands)
        t = self.question_classifier.classify(qid)
        known = self.workload_analyzer.is_known(fold, t)          # Analyze reads the same table state
        m, outcome, tokens = self.router.answer(fold, t, qid)
        # Monitor: Usage Tracker and type shares
        self.kb.mix[t] += 1
        self.kb.served_by_fold.setdefault(fold, Counter())[t] += 1
        wmean = self.usage_tracker.add(tokens, self.kb.goals.budget)
        # Analyze: Cost Analyzer
        flag = wmean is not None and self.cost_analyzer.violation(wmean, self.kb.goals.budget)
        # Plan: runtime profiling and the Mechanism Selector for a new type; the Plan Optimizer on a flag
        charged, decision = tokens, None
        profiled = (not known) and self.kb.goals.cap > 0
        if profiled:
            charged = self.mechanism_evaluator.runtime_profile(t, qid)
            decision = self.mechanism_selector.decide_runtime(t, context)
        # Execute: write into the routing table
        if decision is not None:
            self.plan_enactor.update_one(fold, t, decision["chosen"])
        if flag:
            self._plan_and_write()
            self.replans.append({"at": position, "why": "two windows over budget", "B": self.kb.goals.budget})
        return {"type": t, "known": known, "mechanism": m, "outcome": outcome, "tokens": tokens,
                "charged_tokens": charged, "profiled": profiled, "decision": decision,
                "n_profiled": self.kb.estimates.runtime[t]["n"] if profiled else None}
