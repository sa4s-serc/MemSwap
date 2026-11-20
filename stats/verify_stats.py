"""Independent check of the statistics of the paper.

Recomputes, directly from data/answers.csv, every exact McNemar p value and accuracy difference that the paper
reports for a comparison between systems, the Wilson lower bounds of Table I's selection, the selection rule itself
(leave-one-conversation-out, benchmark labels), and Cohen's kappa between the two judges.

It does NOT import the memswap package, evaluation.py or replay.py. It uses only pandas, numpy and scipy, and its own
implementation of each statistic. The only things it takes from our pipeline are
  - results/decisions.csv: which mechanism served each question in each experiment (written by reproduce.py), and
  - the values it checks (results/*.json, results/paper_numbers.csv).
For the main experiment (Table II, MemSwap row) it also re-derives the decisions with its own selection rule and
compares them with decisions.csv.

    python stats/verify_stats.py        -> results/stats_verification.csv ; exit code 0 only if everything matches
"""
import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MECHS = ["single_lookup", "planned_read", "iterative_search"]
FAMILY = {"locomo": "locomo", "longmemeval_s_star": "lme", "longmemeval_r": "lme"}
TOL = 0.005                     # the Plan Optimizer's tolerance on the budget


def pick(name):
    """the recomputed file if reproduce.py has run, else the shipped expected one."""
    p = os.path.join(RES, name)
    return p if os.path.exists(p) and os.path.getsize(p) else os.path.join(RES, "expected", name)


# ---------------------------------------------------------------- data
a = pd.read_csv(os.path.join(ROOT, "data", "answers.csv"))
V1 = a.pivot(index="question_id", columns="system", values="verdict_judge1").astype(float)
V2 = a.pivot(index="question_id", columns="system", values="verdict_judge2").astype(float)
V2 = V2.where(V2.notna(), V1)                    # an answer without a second-judge verdict keeps the primary verdict
V = {"1": V1, "2": V2}
TOK = a.pivot(index="question_id", columns="system", values="memory_tokens").astype(float)
Q = a.drop_duplicates("question_id").set_index("question_id")[["dataset", "conversation_id", "question_type"]]
streams = pd.read_csv(os.path.join(ROOT, "data", "streams.csv"))
dec = {}
for setting, qid, m in csv.reader(open(pick("decisions.csv"))):
    if setting != "setting":
        dec.setdefault(setting, []).append((qid, m))
J = {n: json.load(open(pick(f))) for n, f in (("R", "replay.json"), ("OS", "predicted_question_types.json"), ("LE", "letta_empty_answers.json"))}
PREFIX = {"W": "R:results.json/memswitch_within_dataset_leave_one_conversation_out", "ST": "R:results.json/streams",
          "BV": "R:budget_vs_fixed.json/cases", "BA": "R:budget_adaptation_p40c.json/cases",
          "TC": "R:type_classifier_nonllm.json/datasets", "OSS": "OS:streams", "LED": "LE:datasets", "PT": "R:results.json/per_type"}


def ours(key):
    f, _, path = key.partition(":")
    if f in PREFIX:
        return ours(PREFIX[f] + "/" + path)
    x = J[f]
    for k in path.split("/"):
        x = x[k]
    return x


# ---------------------------------------------------------------- statistics, written here from their definitions
def mcnemar_exact(x, y):
    """two-sided exact McNemar test on paired 0/1 outcomes: a binomial test on the discordant pairs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    n = b + c
    if n == 0:
        return 1.0, 0.0
    k = min(b, c)
    p = min(1.0, 2.0 * binom.cdf(k, n, 0.5)) if b != c else 1.0
    return float(p), 100.0 * (x.mean() - y.mean())


def wilson_lower(k, n, conf=0.95):
    """one-sided lower Wilson score bound."""
    if n == 0:
        return 0.0
    z = norm.ppf(conf)
    p = k / n
    return float((p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n))


def kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return (po - pe) / (1 - pe)


def select(ids, S, margin):
    """the selection rule: the cheapest mechanism whose Wilson lower bound is within `margin` of the best accuracy."""
    n = len(ids)
    k = {m: float(S.loc[ids, m].sum()) for m in MECHS}
    tok = {m: float(TOK.loc[ids, m].mean()) for m in MECHS}
    acc = {m: k[m] / n for m in MECHS}
    best = sorted(MECHS, key=lambda m: (-acc[m], tok[m], MECHS.index(m)))[0]
    ok = [m for m in MECHS if tok[m] < tok[best] and wilson_lower(k[m], n) >= acc[best] - margin - 1e-12]
    return sorted(ok, key=lambda m: (tok[m], MECHS.index(m)))[0] if ok else best


def own_decisions(ds, judge, margin=0.01):
    """leave-one-conversation-out with benchmark labels, derived here without our pipeline."""
    S = V[judge]
    dq = Q[Q.dataset == ds]
    out = {}
    for conv in sorted(dq.conversation_id.unique()):
        train, held = dq[dq.conversation_id != conv], dq[dq.conversation_id == conv]
        default = select(train.index.tolist(), S, margin)
        table = {t: select(g.index.tolist(), S, margin) for t, g in train.groupby("question_type")}
        for qid, t in held.question_type.items():
            out[qid] = table.get(t, default)
    return out


def verdicts(setting, judge):
    ids = [q for q, _ in dec[setting]]
    return ids, np.array([V[judge].at[q, m] for q, m in dec[setting]], dtype=float)


rows = []


def check(what, our, indep, tol=1e-9):
    good = (our == indep) if isinstance(our, str) else math.isclose(float(our), float(indep), rel_tol=tol, abs_tol=tol)
    rows.append((what, our, indep, "yes" if good else "NO"))


# ---------------------------------------------------------------- 1. the selection rule, re-derived
for ds in FAMILY:
    for j in ("1", "2"):
        mine = own_decisions(ds, j)
        theirs = dict(dec["within|%s|labels|margin1|judge%s" % (ds, j)])
        check("selection rule, %s, judge %s: questions with the same mechanism" % (ds, j), len(theirs), sum(mine[q] == theirs[q] for q in theirs))

# ---------------------------------------------------------------- 2. Wilson bounds of Table I's selection
for ds, types in J["R"]["results.json"]["per_type"].items():
    dq = Q[Q.dataset == ds]
    for t, e in types.items():
        ids = dq.index[dq.question_type == t]
        for j in ("1", "2"):
            for m in MECHS:
                check("Wilson lower bound, %s, %s, %s, judge %s" % (ds, t, m, j), e["judge" + j]["selection_by_margin_pp"]["1"]["wilson_lb"][m],
                      wilson_lower(V[j].loc[ids, m].sum(), len(ids)), tol=1e-6)
check("Wilson lower bound of 80% on 30 questions (paper: 65.7%)", "65.7", "%.1f" % (100 * wilson_lower(24, 30)))
check("Wilson lower bound of 80% on 300 questions (paper: 75.9%)", "75.9", "%.1f" % (100 * wilson_lower(240, 300)))

# ---------------------------------------------------------------- 3. every comparison between systems
empty = set()
for l in open(os.path.join(ROOT, "data", "answer_texts.jsonl")):
    r = json.loads(l)
    if r["system"] == "letta" and not r["answer"].strip():
        empty.add(r["question_id"])
done, skipped = set(), []
paper = list(csv.DictReader(open(os.path.join(RES, "paper_numbers.csv"))))
for r in paper:
    key = r["source_key"]
    if not (key.endswith("/mcnemar_p") or key.endswith("/diff_pp")):
        if "mcnemar_p_median" in key or "minus_random" in key:
            skipped.append(key)
        continue
    base, stat = key.rsplit("/", 1)
    if base in done:
        continue
    f, _, path = base.partition(":")
    p = path.split("/")
    x = y = None
    if f == "W" and p[3] == "vs_baselines":                       # ds/margin_pp=M/judgeJ/vs_baselines/system
        ds, m, j, sysname = p[0], p[1].split("=")[1], p[2][-1], p[4]
        ids, x = verdicts("within|%s|labels|margin%s|judge%s" % (ds, m, j), j)
        y = V[j].loc[ids, sysname].to_numpy()
    elif f == "TC":                                               # ds/replay/judgeJ/a_argmax/vs_...
        ds, j, other = p[0], p[2][-1], p[4]
        ids, x = verdicts("within|%s|predicted|margin1|judge%s" % (ds, j), j)
        if other == "vs_labels":
            ids2, y = verdicts("within|%s|labels|margin1|judge%s" % (ds, j), j)
            assert ids == ids2
        else:
            y = V[j].loc[ids, other.replace("vs_fixed_", "").replace("vs_", "")].to_numpy()
    elif f == "BV":                                               # ds/judgeJ/B=NNNN  (MemSwap against the best fixed mechanism that fits)
        ds, j, bk = p[0], p[1][-1], p[2]
        ids, x = verdicts("budget|%s|%s|judge%s" % (ds, bk, j), j)
        B = float(ours(base + "/budget"))
        fits = [m for m in MECHS if TOK.loc[ids, m].mean() <= B * (1 + TOL)]
        best = max(fits, key=lambda m: (V[j].loc[ids, m].mean(), -TOK.loc[ids, m].mean()))
        check("best fixed mechanism that fits, %s, %s, judge %s" % (ds, bk, j), ours(base + "/best_fixed"), best)
        y = V[j].loc[ids, best].to_numpy()
    elif f == "BA" and "budget_drop_scenario" in p:               # ds/judgeJ/budget_drop_scenario/vs_fixed_<m>
        ds, j = p[0], p[1][-1]
        ids, x = verdicts("budget|%s|drop|judge%s" % (ds, j), j)
        y = V[j].loc[ids, p[3].replace("vs_fixed_", "")].to_numpy()
    elif f == "ST":                                               # stream_K/configs/.../judgeJ/vs_baselines_on_stream_questions/system
        st, j, sysname = p[0], p[3][-1], p[5]
        ids, x = verdicts("stream|%s|labels|margin1|judge%s" % (st, j), j)
        y = V[j].loc[ids, sysname].to_numpy()
    elif f == "OSS" and p[2] == "predicted_vs_labels":            # stream_K/judgeJ/predicted_vs_labels
        st, j = p[0], p[1][-1]
        if "stream|%s|predicted|margin1|judge%s" % (st, j) not in dec:
            skipped.append(key)
            continue
        ids, x = verdicts("stream|%s|predicted|margin1|judge%s" % (st, j), j)
        ids2, y = verdicts("stream|%s|labels|margin1|judge%s" % (st, j), j)
        assert ids == ids2
    elif f == "LED":                                              # ds/judgeJ/without_empty_answers
        ds, j = p[0], p[1][-1]
        ids, x = verdicts("within|%s|labels|margin1|judge%s" % (ds, j), j)
        keep = np.array([q not in empty for q in ids])
        y = V[j].loc[ids, "letta"].to_numpy()[keep]
        x = x[keep]
    else:
        skipped.append(key)
        continue
    done.add(base)
    pv, diff = mcnemar_exact(x, y)
    rounded = f == "OSS"                                          # that file stores differences rounded to two decimals
    check("exact McNemar p: " + base, ours(base + "/mcnemar_p"), pv)
    check("difference in points: " + base, ours(base + "/diff_pp"), round(diff, 2) if rounded else diff)

# ---------------------------------------------------------------- 4. agreement of the two judges
kap = []
for ds in FAMILY:
    sub = a[(a.dataset == ds) & a.verdict_judge2.notna()]
    kap.append((ds, len(sub), kappa(sub.verdict_judge1.to_numpy(), sub.verdict_judge2.to_numpy().astype(int))))

out = os.path.join(RES, "stats_verification.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["number", "our_value", "independent_value", "match"])
    w.writerows(rows)
bad = [r for r in rows if r[3] != "yes"]
for r in rows:
    print("%-3s %s | ours %s | independent %s" % (r[3], r[0], r[1], r[2]))
print("\nCohen's kappa between the two judges (answers that both judges scored; the paper states no value):")
for ds, n, k in kap:
    print("  %-20s n = %5d  kappa = %.3f" % (ds, n, k))
print("\nnot checked here (they depend on 200 random draws; reproduce.py recomputes them with the fixed seed): %d keys" % len(set(skipped)))
print("checked %d values: %d match, %d do not. Table written to results/stats_verification.csv" % (len(rows), len(rows) - len(bad), len(bad)))
sys.exit(1 if bad else 0)
