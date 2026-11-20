"""Execute: Plan Enactor. Applies a new plan to the managed system through the `update` interface, the effector
of the managing system. It is the only component that writes the routing table R that the Router reads.

Execute runs only when Plan changes R, so the managing system is never on the request path.
"""
from __future__ import annotations


class PlanEnactor:
    def __init__(self, kb):
        self.kb = kb
        self.updates = 0              # how many times R was written

    def update(self, fold, assignment):
        """send a whole plan (question type -> mechanism) for one fold."""
        self.kb.routing.table[fold] = assignment
        self.updates += 1

    def update_one(self, fold, qtype, mechanism):
        """send the mechanism of one question type."""
        self.kb.routing.write(fold, qtype, mechanism)
        self.updates += 1
