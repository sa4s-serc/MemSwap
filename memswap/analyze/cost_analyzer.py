"""Analyze: Cost Analyzer. Flags a budget change when the architect sets a new budget, and a budget violation when
the window mean exceeds the budget in two consecutive windows.
"""
from __future__ import annotations


class CostAnalyzer:
    def __init__(self):
        self.over_run = 0

    def budget_changed(self, new_budget, budget):
        """a new budget is a budget change; it resets the count of windows over budget."""
        if new_budget != budget:
            self.over_run = 0
            return True
        return False

    def violation(self, wmean, budget):
        """two consecutive windows over budget are a budget violation."""
        self.over_run = self.over_run + 1 if wmean > budget else 0
        if self.over_run >= 2:
            self.over_run = 0
            return True
        return False
