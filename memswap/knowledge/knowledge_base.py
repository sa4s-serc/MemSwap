"""Knowledge: the repositories that the four phases share (paper, Fig. 1).

  MechanismProfiles   per question type: questions profiled, correct answers and mean memory tokens of each mechanism
  RoutingTable       the mechanism each known type uses now, and the default mechanism m0 for every other type
  AdaptationGoals  accuracy margin, profiling cap and token budget, set by the architect

Evaluation note: the leave-one-conversation-out protocol gives every held-out conversation its own first batch, so
the evaluation keeps one set of estimates and one routing table per first batch ("fold"). A deployed system has a
single fold, so there the shares are the first batch plus every question served.

Shares modes (KnowledgeBase.shares_mode); "first_batch_fold" is the default:
  "first_batch_fold"  a fold's shares start from the question types of its first labelled batch and are updated with
                      the questions served from that fold (the Usage Tracker of the paper; the paper's tables)
  "first_batch"       as "first_batch_fold", but updated with every question served, from any fold
  "given"             the shares the caller passes as `mix`, plus every question served (for example the type counts of
                      the whole dataset, which include questions not yet served)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# mechanism names and their fixed order
MECHANISMS = ["single_lookup", "planned_read", "iterative_search"]
WINDOW = 50                 # questions per window
TOLERANCE = 0.005           # budget tolerance (0.5%)


@dataclass
class AdaptationGoals:
    margin: float                   # accuracy margin as a fraction (1 point = 0.01)
    cap: int = 0                    # profiling cap N
    budget: float | None = None     # token budget B; None = no budget


@dataclass
class MechanismProfiles:
    """batch: the selection rule's record per type of a first batch;
    runtime: running sums of runtime profiling per type."""
    batch: dict = field(default_factory=dict)      # fold -> type -> rule_on record
    batch_default: dict = field(default_factory=dict)   # fold -> rule_on record over the whole first batch (m0)
    runtime: dict = field(default_factory=dict)    # type -> {"k": {m: int}, "tok": {m: float}, "n": int}


@dataclass
class RoutingTable:
    """R(t) = sigma(t) for a known type, m0 for every other type (paper, Execute)."""
    table: dict = field(default_factory=dict)      # fold -> type -> mechanism
    default: dict = field(default_factory=dict)    # fold -> m0

    def lookup(self, fold, qtype):
        return self.table[fold].get(qtype, self.default[fold])

    def write(self, fold, qtype, mechanism):
        self.table[fold][qtype] = mechanism


@dataclass
class KnowledgeBase:
    goals: AdaptationGoals
    estimates: MechanismProfiles = field(default_factory=MechanismProfiles)
    routing: RoutingTable = field(default_factory=RoutingTable)
    mix: Counter = field(default_factory=Counter)  # type shares w_t: questions per type
    shares_mode: str = "first_batch_fold"
    first_batch_mix: dict = field(default_factory=dict)   # fold -> Counter of the question types of its labelled batches
    served_by_fold: dict = field(default_factory=dict)    # fold -> Counter of the question types served from that fold

    def shares(self, fold):
        """the question type counts that the Plan Optimizer weights fold `fold` with (see the module header)."""
        if self.shares_mode == "given":
            return self.mix
        if self.shares_mode == "first_batch":
            return self.first_batch_mix.get(fold, Counter()) + self.mix
        if self.shares_mode == "first_batch_fold":
            return self.first_batch_mix.get(fold, Counter()) + self.served_by_fold.get(fold, Counter())
        raise ValueError(self.shares_mode)
