"""Monitor: Usage Tracker. Tracks the memory tokens of served questions in non-overlapping windows of 50.
"""
from __future__ import annotations

import numpy as np

from memswap.knowledge.knowledge_base import WINDOW


class UsageTracker:
    def __init__(self, window=WINDOW):
        self.window = window
        self.tokens = []          # memory tokens of every served question, in order
        self.windows = []         # completed windows: {"end", "mean", "B", "kept"}

    def add(self, tokens, budget):
        """record one served question; returns the window mean when this question completes a window, else None."""
        self.tokens.append(tokens)
        i = len(self.tokens) - 1
        if budget is not None and (i + 1) % self.window == 0:   # without a budget, windows are not checked
            t = np.asarray(self.tokens[i + 1 - self.window:i + 1], dtype=float)
            wmean = float(t.mean())
            self.windows.append({"end": i + 1, "mean": wmean, "B": budget, "kept": wmean <= budget + 1e-9})
            return wmean
        return None
