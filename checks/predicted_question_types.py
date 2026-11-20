"""The workload streams with predicted question types. Deterministic replay of logged outcomes, no model calls.

  python checks/predicted_question_types.py   -> results/predicted_question_types.json

The classifier (memswap/managed/memory_manager/question_classifier.py fit_predict) is trained on the first batch's
labels only. Streams 1 to 5, margin 1 point, cap 0, no budget, two routings: benchmark label and predicted question
type. A question type without a mechanism in R gets the default mechanism m0. Exact McNemar, both judges.
"""
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import evaluation as E                                                   # noqa: E402
from memswap.managed.memory_manager.question_classifier import LabelTypes, fit_predict      # noqa: E402
from memswap.knowledge.knowledge_base import AdaptationGoals           # noqa: E402
from memswap.loop import MemSwap                                         # noqa: E402

warnings.filterwarnings("ignore", message="The least populated class")
DECISIONS = {}      # setting -> list of (question_id, mechanism), for stats/verify_stats.py


def predict(train, y, held):
    classes, proba, _ = fit_predict(list(train), list(held), list(y))
    return classes[np.argmax(proba, 1)], proba.max(1)


class Typer:
    def __init__(self, types):
        self.types = types

    def classify(self, qid):
        return self.types[qid]


def replay(fb, sq, typer, labels, S, TOK):
    ms = MemSwap(AdaptationGoals(margin=E.MARGIN, cap=0), typer, lambda m, qid: (S.at[qid, m], TOK.at[qid, m]), S, TOK)
    default = ms.first_batch("fb", fb, labels.classify)          # the knowledge comes from the labelled first batch
    v, t, mech = np.empty(len(sq)), np.empty(len(sq)), []
    for i, qid in enumerate(sq):
        rec = ms.handle(qid, "fb", i + 1)
        v[i], t[i] = rec["outcome"], rec["tokens"]; mech.append(rec["mechanism"])
    return v, t, np.array(mech), default


def summary(v, t):
    return {"accuracy_pct": round(100 * float(v.mean()), 2), "mean_memory_tokens": round(float(t.mean()), 1)}


def cmp(a, b):
    r = E.mcnemar(a, b)
    return {"diff_pp": round(r["diff_pp"], 2), "mcnemar_p": r["mcnemar_p"], "first_only_correct": r["b_memswitch_only_correct"],
            "second_only_correct": r["c_baseline_only_correct"]}


def main():
    E.check_sums()
    a, s, q, S_by_judge, _, TOK = E.load()
    labels = LabelTypes(q)
    ref = json.load(open(os.path.join(HERE, "results", "replay.json")))["results.json"]["streams"]
    out = {"config": {"routing": "margin 1 point, cap 0, no budget; a question type without a mechanism in R gets m0",
                      "classifier": "memswap/managed/memory_manager/question_classifier.py fit_predict, trained on the first batch's labels only"},
           "streams": {}}
    for st in sorted(s.stream.unique()):
        g = s[s.stream == st]
        fb = g[g.role == "first_batch"].question_id.tolist()
        sq = g[g.role == "stream"].sort_values("position").question_id.tolist()
        y_fb = q.loc[fb, "type_key"].tolist()
        lab = q.loc[sq, "type_key"].to_numpy()
        known = sorted(set(y_fb))
        pred, _ = predict(fb, y_fb, sq)
        is_new = ~np.isin(lab, known)
        typ_pred = Typer(dict(zip(sq, pred)))
        row = {"name": g.stream_name.iloc[0], "n_first_batch": len(fb), "n_stream": len(sq), "known_question_types": known,
               "predicted_type_correct_on_known_pct": round(100 * float((pred[~is_new] == lab[~is_new]).mean()), 2) if (~is_new).any() else None}
        for jname, jk in (("judge1", "j1"), ("judge2", "j2")):
            S = S_by_judge[jk]
            vl, tl, ml, default = replay(fb, sq, labels, labels, S, TOK)
            vp, tp, mp, _ = replay(fb, sq, typ_pred, labels, S, TOK)
            DECISIONS[f"stream|stream_{st}|predicted|margin1|{jname}"] = list(zip(sq, mp.tolist()))
            r = ref[f"stream_{st}"]["configs"]["margin_pp=1,cap=0"][jname]
            assert abs(r["memswitch"]["accuracy"] - vl.mean()) < 1e-12, (st, jname)     # label routing = replay.json
            row[jname] = {"default_m0": default["chosen"] if isinstance(default, dict) else str(default),
                          "labels": summary(vl, tl), "predicted": summary(vp, tp), "predicted_vs_labels": cmp(vp, vl),
                          "new_type_questions_only": {"n": int(is_new.sum()),
                                                      "labels": summary(vl[is_new], tl[is_new]) if is_new.any() else None,
                                                      "predicted": summary(vp[is_new], tp[is_new]) if is_new.any() else None},
                          "mechanism_of_new_type_questions": {k: {m: int((mm[is_new] == m).sum()) for m in E.MECHS}
                                                              for k, mm in (("labels", ml), ("predicted", mp))}}
        out["streams"][f"stream_{st}"] = row

    path = os.path.join(HERE, "results", "predicted_question_types.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)); f.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
