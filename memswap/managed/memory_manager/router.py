"""Memory manager: Router. The entry of MemSwap: reads the store with R(classify(q)) and returns the answer to the agent.

`read` is the mechanism call. In this package's replay it returns the logged outcome of that mechanism on the
question (its verdict and memory tokens); a live system passes the functions of memswap/managed/memory_manager/mechanisms/.
"""
from __future__ import annotations


class Router:
    def __init__(self, kb, read):
        self.kb = kb
        self.read = read              # read(mechanism, qid) -> (outcome, memory tokens)

    def route(self, fold, qtype):
        return self.kb.routing.lookup(fold, qtype)

    def answer(self, fold, qtype, qid):
        m = self.route(fold, qtype)
        outcome, tokens = self.read(m, qid)
        return m, outcome, tokens
