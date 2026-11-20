"""Monitor: MechanismEvaluator. Runs every mechanism on profiled questions and has the Judge score the answers; records the
outcomes as mechanism profiles.

In this package the trial runs and the Judge's verdicts are the logged outcomes (data/answers.csv): `outcomes` gives,
for a question, each mechanism's verdict and memory tokens. A live system passes the mechanisms of
memswap/managed/memory_manager/mechanisms/ and a judge instead.

batch_profile counts correct answers and mean tokens per mechanism; runtime_profile does the same for one question.
"""
from __future__ import annotations

from memswap.knowledge.knowledge_base import MECHANISMS

MECHS = MECHANISMS


def profile_questions(qids, S, TOK):
    """counts per mechanism the correct answers and the mean memory tokens of the profiled questions."""
    n = len(qids)
    if n == 0:
        k = {m: 0 for m in MECHS}
        tok = {m: 0.0 for m in MECHS}
    else:
        sub = S.loc[qids, MECHS]
        k = {m: int(sub[m].sum()) for m in MECHS}
        tok = {m: float(TOK.loc[qids, m].mean()) for m in MECHS}
    return k, n, tok


class MechanismEvaluator:
    def __init__(self, kb, S_sel, TOK):
        self.kb = kb
        self.S_sel, self.TOK = S_sel, TOK     # judged outcomes of every mechanism (selection verdicts) and tokens

    def batch_profile(self, qids, type_of):
        """batch profiling of a first batch: (k, n, tok) per type, and over all questions as one type (for m0)."""
        by_type = {}
        for qid in qids:
            by_type.setdefault(type_of(qid), []).append(qid)
        return ({t: profile_questions(tq, self.S_sel, self.TOK) for t, tq in by_type.items()},
                profile_questions(list(qids), self.S_sel, self.TOK))

    def runtime_profile(self, qtype, qid):
        """runtime profiling of one question; returns the memory tokens charged (all three mechanisms)."""
        charged = sum(self.TOK.at[qid, m] for m in MECHS)
        p = self.kb.estimates.runtime.setdefault(qtype, {"k": {m: 0 for m in MECHS}, "tok": {m: 0.0 for m in MECHS}, "n": 0})
        p["n"] += 1
        for m in MECHS:
            p["k"][m] += int(self.S_sel.at[qid, m])
            p["tok"][m] += float(self.TOK.at[qid, m])
        return charged
