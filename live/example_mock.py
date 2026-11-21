"""The live wiring of MemSwap with mock mechanisms and a mock judge. No model, no network, no API key.

  python live/example_mock.py

The three mock mechanisms answer from a tiny memory and report a fixed number of memory tokens, as the real ones
report the input tokens of their model calls. They differ by question type, like the real mechanisms in Table I.
Single lookup finds dated facts but misses what the assistant said; planned read is cheap but misses order questions;
iterative search answers everything at the highest cost. The run shows the steps of a live deployment: a labelled
batch is profiled, the Routing Table is written, questions are routed by their type, a question type that was not
profiled goes to the default mechanism m0, and a lower token budget makes the Plan Optimizer re-plan.
The same code path is used by live/smoke_test.py with the real mechanisms, and by tests/test_live_wiring.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memswap.knowledge.knowledge_base import AdaptationGoals   # noqa: E402
from memswap.live import LiveMemSwap, Question                  # noqa: E402

MEMORY = {"order": "2026-03-02", "promise": "a refund within 5 days", "city": "Lisbon"}
COST = {"single_lookup": 3000, "planned_read": 1500, "iterative_search": 7000}
ANSWERS = {                      # what each mock mechanism can answer, by question type
    "single_lookup": {"temporal", "fact"},
    "planned_read": {"assistant", "fact"},
    "iterative_search": {"temporal", "assistant", "fact"},
}


def mechanism(name):
    def read(q):
        key = q.extra["key"]
        found = name == "iterative_search" or q.qtype in ANSWERS[name]     # iterative search answers every type
        answer = MEMORY[key] if found else "not mentioned"
        return answer, COST[name]
    return read


MECHANISMS = {m: mechanism(m) for m in COST}


def judge(q, answer):
    return int(answer == q.gold)


def classify(q):                 # the mock Question Classifier: the type is attached to the question
    return q.qtype


def labelled_batch():
    out = []
    for i in range(6):
        out.append(Question(f"b{i}t", "When did I place my last order?", "temporal", MEMORY["order"], {"key": "order"}))
        out.append(Question(f"b{i}a", "What did you promise me?", "assistant", MEMORY["promise"], {"key": "promise"}))
        out.append(Question(f"b{i}f", "Which city do I live in?", "fact", MEMORY["city"], {"key": "city"}))
    return out


def run(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    live = LiveMemSwap(MECHANISMS, judge, classify, AdaptationGoals(margin=0.01))
    S, TOK, _ = live.labelled_batch(labelled_batch())
    table, default = live.routing_table
    say("profiled accuracy per mechanism and question type:")
    types = {"temporal": "t", "assistant": "a", "fact": "f"}
    for t, s in types.items():
        ids = [i for i in S.index if i.endswith(s)]
        say(f"  {t:9s}", {m: round(float(S.loc[ids, m].mean()), 2) for m in S.columns})
    say("routing table:", table, "| default m0:", default)

    served = []
    for i, (qtype, key, text) in enumerate([("temporal", "order", "When was my last order?"),
                                            ("assistant", "promise", "What was I promised?"),
                                            ("fact", "city", "Where do I live?"),
                                            ("returns", "order", "How do I return my order?")]):
        rec = live.ask(Question(f"s{i}", text, qtype, extra={"key": key}))
        served.append(rec)
        say(f"served {text!r:32s} type={qtype:9s} -> {rec['mechanism']:16s} answer={rec['outcome']!r} tokens={rec['tokens']}")

    moves = live.set_budget(1800)
    table_low, _ = live.routing_table
    say("after a budget of 1,800 tokens per question: moves", moves, "| routing table:", table_low)
    return {"table": table, "default": default, "served": served, "table_low_budget": table_low, "moves": moves}


if __name__ == "__main__":
    run()
