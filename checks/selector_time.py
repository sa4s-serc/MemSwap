"""Decision time of the Mechanism Selector. Offline, no model calls.

One decision = one call of memswap.plan.mechanism_selector.select for one question type of one fold's first batch
(leave-one-conversation-out, primary judge, margin one point). Each decision is timed 200 times with perf_counter_ns.
Every decision is checked against results/replay.json. Prints the entry for efficiency.json as JSON.
Run from the package root: python checks/selector_time.py"""
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import evaluation as E  # noqa: E402
from memswap.plan.mechanism_selector import MECHS, select  # noqa: E402

REPS = 200


def summ(v):
    v = np.asarray(v, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}


def main():
    a, s, q, SJ, _, TOK = E.load()
    S = SJ["j1"]
    rj = json.load(open(os.path.join(ROOT, "results", "replay.json")))["results.json"]["memswitch_within_dataset_leave_one_conversation_out"]
    out = {"note": "checks/selector_time.py: one call of select() per fold and question type of the first batch, margin 1 point, "
                   "primary judge, %d repetitions each; excludes profiling (the mechanisms and the judge)" % REPS,
           "python": platform.python_version(), "reps": REPS}
    for ds in ("locomo", "longmemeval_s_star", "longmemeval_r"):
        dq = q[q.dataset == ds]
        want = rj[ds]["margin_pp=1"]["judge1"]["selections_per_held_out_conversation"]
        times, n_dec = [], 0
        for conv in sorted(dq.conversation_id.unique()):
            fb = dq[dq.conversation_id != conv]
            for t in sorted(fb.question_type.unique()):
                ids = fb.index[fb.question_type == t]
                k = {m: int(S.loc[ids, m].sum()) for m in MECHS}
                tok = {m: float(TOK.loc[ids, m].mean()) for m in MECHS}
                n = len(ids)
                r = select(k, n, tok, 0.01)
                if t in want[conv]:
                    assert r["chosen"] == want[conv][t]["chosen"] and n == want[conv][t]["n_profiled"], (ds, conv, t)
                n_dec += 1
                for _ in range(REPS):
                    t0 = time.perf_counter_ns(); select(k, n, tok, 0.01); times.append((time.perf_counter_ns() - t0) / 1e6)
        out[ds] = {"decisions": n_dec, "ms": summ(times)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
