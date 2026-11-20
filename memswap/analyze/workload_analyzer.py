"""Analyze: Workload Analyzer. A type is known once the Plan phase has recorded a mechanism for it.
"""
from __future__ import annotations


class WorkloadAnalyzer:
    def __init__(self, kb):
        self.kb = kb

    def is_known(self, fold, qtype):
        return qtype in self.kb.routing.table[fold]
