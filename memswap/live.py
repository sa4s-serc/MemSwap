"""Connecting MemSwap to a live agent.

The replay in this package reads logged outcomes. A live agent needs only two functions of its own:

  mechanisms[m](question) -> (answer, memory_tokens)   one per reading mechanism m in MECHANISMS
  judge(question, answer)  -> 1 or 0                    whether the answer is correct; used only on labelled questions

`profile_batch` runs every mechanism on the questions of a labelled batch and has the judge score each answer. This is
the trial run of the Mechanism Evaluator, done live. `LiveMemSwap` then builds the MAPE-K loop on these outcomes and
answers each new question with the mechanism that the Routing Table gives its question type. A question is any
object with an `id`, a `text` and, for labelled questions, a `gold` answer and a `qtype`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from memswap.knowledge.knowledge_base import MECHANISMS, AdaptationGoals
from memswap.loop import MemSwap


@dataclass
class Question:
    id: str
    text: str
    qtype: str | None = None          # the question type; for a labelled question, its label
    gold: str | None = None           # the gold answer; only labelled questions have one
    extra: dict = field(default_factory=dict)


def profile_batch(batch, mechanisms, judge):
    """runs every mechanism on every labelled question and judges each answer.
    Returns the judged outcomes S, the memory tokens TOK (question id x mechanism) and the answers."""
    S = pd.DataFrame(index=[q.id for q in batch], columns=MECHANISMS, dtype=float)
    TOK = pd.DataFrame(index=[q.id for q in batch], columns=MECHANISMS, dtype=float)
    answers = {}
    for q in batch:
        for m in MECHANISMS:
            answer, tokens = mechanisms[m](q)
            S.at[q.id, m] = float(judge(q, answer))
            TOK.at[q.id, m] = float(tokens)
            answers[(q.id, m)] = answer
    return S, TOK, answers


class _Types:
    """the question type of each question seen so far (the Question Classifier's output)."""
    def __init__(self, classify):
        self.classify_text, self.of = classify, {}

    def classify(self, qid):
        return self.of[qid]


class LiveMemSwap:
    """MemSwap around live mechanisms. `classify(question) -> question type` is the Question Classifier."""

    def __init__(self, mechanisms, judge, classify, goals: AdaptationGoals, fold="live"):
        self.mechanisms, self.judge, self.fold = mechanisms, judge, fold
        self.types = _Types(classify)
        self.goals = goals
        self.questions, self.ms, self.position = {}, None, 0

    def _read(self, m, qid):
        return self.mechanisms[m](self.questions[qid])        # Router -> reading mechanism -> (answer, tokens)

    def labelled_batch(self, batch):
        """profiles a labelled batch and writes the Routing Table (Algorithm 1, lines 9 to 15)."""
        S, TOK, answers = profile_batch(batch, self.mechanisms, self.judge)
        for q in batch:
            self.questions[q.id] = q
            self.types.of[q.id] = q.qtype
        self.ms = MemSwap(self.goals, self.types, self._read, S, TOK)
        self.ms.first_batch(self.fold, [q.id for q in batch], lambda qid: self.types.of[qid])
        return S, TOK, answers

    def ask(self, question):
        """answers one question on the request path; the loop runs after it (Algorithm 1, lines 1 to 8)."""
        assert self.ms is not None, "profile a labelled batch first"
        self.questions[question.id] = question
        self.types.of[question.id] = self.types.classify_text(question)
        self.position += 1
        return self.ms.handle(question.id, self.fold, self.position)

    def set_budget(self, budget):
        return self.ms.set_budget(budget, at=self.position)

    @property
    def routing_table(self):
        return dict(self.ms.kb.routing.table[self.fold]), self.ms.kb.routing.default[self.fold]
