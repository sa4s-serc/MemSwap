"""MemSwap against Letta with and without Letta's empty answers. Offline replay, no model calls.

MemSwap: leave-one-conversation-out, margin 1 point, cap 0, benchmark labels, as in Table II. Under the second judge,
MemSwap is profiled with that judge's verdicts. An empty answer is an answer whose text is empty after stripping; the
texts are in data/answer_texts.jsonl. Writes results/letta_empty_answers.json.
Run from the package root: python checks/letta_empty_answers.py"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import evaluation as E  # noqa: E402
from memswap.managed.memory_manager.question_classifier import LabelTypes  # noqa: E402
from memswap.knowledge.knowledge_base import AdaptationGoals  # noqa: E402
from memswap.loop import MemSwap  # noqa: E402

FULL = os.path.join(os.path.dirname(HERE), "data", "answer_texts.jsonl")


def main():
    a, s, q, SJ, _, TOK = E.load()
    labels = LabelTypes(q)
    full = [json.loads(l) for l in open(FULL)]
    empty = {r["question_id"] for r in full if r["system"] == "letta" and not r["answer"].strip()}
    out = {"source_of_answer_texts": "data/answer_texts.jsonl", "settings": "margin 1 point, cap 0, benchmark labels, leave-one-conversation-out", "datasets": {}}
    for ds in ("locomo", "longmemeval_s_star"):
        dq = q[q.dataset == ds]
        ids = dq.index.tolist()
        keep = np.array([i not in empty for i in ids])
        entry = {"n": len(ids), "n_empty": int((~keep).sum()), "empty_question_ids": sorted(i for i in ids if i in empty)}
        for jname, jk in (("judge1", "j1"), ("judge2", "j2")):
            S = SJ[jk]
            v = {}
            ms = MemSwap(AdaptationGoals(margin=0.01, cap=0), labels, lambda m, qid: (S.at[qid, m], TOK.at[qid, m]), S, TOK)
            for conv in sorted(dq.conversation_id.unique()):
                ms.first_batch(conv, dq.index[dq.conversation_id != conv].tolist(), labels.classify)
                for i, qid in enumerate(dq.index[dq.conversation_id == conv]):
                    v[qid] = ms.handle(qid, conv, i + 1)["outcome"]
            mv = np.array([v[i] for i in ids], dtype=float)
            lv = S.loc[ids, "letta"].to_numpy(dtype=float)
            je = {}
            for name, k in (("all_questions", np.ones(len(ids), dtype=bool)), ("without_empty_answers", keep)):
                r = E.mcnemar(mv[k], lv[k])
                je[name] = {"n": int(k.sum()), "memswap_accuracy": float(mv[k].mean()), "letta_accuracy": float(lv[k].mean()), **r}
            je["memswap_accuracy_on_the_empty_ones"] = float(mv[~keep].mean())
            je["letta_correct_on_the_empty_ones"] = int(lv[~keep].sum())
            entry[jname] = je
        out["datasets"][ds] = entry
    with open(os.path.join(os.path.dirname(HERE), "results", "letta_empty_answers.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for ds, e in out["datasets"].items():
        for j in ("judge1", "judge2"):
            for name in ("all_questions", "without_empty_answers"):
                x = e[j][name]
                print(ds, j, name, x["n"], "MemSwap %.2f Letta %.2f diff %+.2f p %.4g" % (100 * x["memswap_accuracy"], 100 * x["letta_accuracy"], x["diff_pp"], x["mcnemar_p"]))


if __name__ == "__main__":
    main()
