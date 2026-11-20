"""Times two components of MemSwap on this machine, one question or one plan at a time. No model calls.

  python checks/time_components.py   -> results/time_components.json

(b) Plan Optimizer: exact_plan over six budget levels per dataset, 200 repetitions.
(c) Question Classifier, end to end: the bge-small-en-v1.5 embedding on the CPU, then tf-idf and logistic regression;
    10 questions of warm-up, then 200 questions timed one at a time per dataset.

The times depend on the machine. reproduce.py does not compare them; the values behind the paper are logged in
results/expected/logged/efficiency.json. The embedding model is downloaded from Hugging Face on the first run.
"""
import csv
import json
import os
import platform
import statistics
import time
from collections import Counter, defaultdict
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MECHS = ["single_lookup", "planned_read", "iterative_search"]
DS = ["locomo", "longmemeval_s_star", "longmemeval_r"]


def summ(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    return {"n": len(v), "mean": statistics.mean(v), "median": statistics.median(v), "p95": v[int(0.95 * (len(v) - 1))]}


ANS = defaultdict(dict)   # (dataset, mech) -> qid -> row
for r in csv.DictReader(open(os.path.join(DATA, "answers.csv"))):
    if r["system"] in MECHS:
        ANS[(r["dataset"], r["system"])][r["question_id"]] = r

out = {}

# (b) Plan Optimizer
TOL = 0.005


def exact_plan(kb, mix, B):
    total = sum(mix.get(t, 0) for t in kb) or 1
    w = {t: mix.get(t, 0) / total for t in kb}
    types = sorted(kb)
    best = None
    for asg in product(MECHS, repeat=len(types)):
        tok = sum(w[t] * kb[t]["tok"][m] for t, m in zip(types, asg))
        if tok > B * (1 + TOL):
            continue
        acc = sum(w[t] * kb[t]["acc"][m] for t, m in zip(types, asg))
        if best is None or (acc, -tok) > (best[0], -best[1]):
            best = (acc, tok, asg)
    if best is None:
        asg = tuple(min(MECHS, key=lambda m: kb[t]["tok"][m]) for t in types)
        best = (0, 0, asg)
    a = dict(zip(types, best[2]))
    start = {t: kb[t]["start"] for t in kb}
    moves = [{"type": t, "from": start[t], "to": a[t]} for t in types if a[t] != start[t]]
    return a, moves


out["b_budget_planner"] = {}
for ds in DS:
    acc, tok, n = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), Counter()
    for m in MECHS:
        for q, a in ANS[(ds, m)].items():
            acc[a["question_type"]][m].append(int(a["verdict_judge1"])); tok[a["question_type"]][m].append(float(a["memory_tokens"]))
            if m == MECHS[0]:
                n[a["question_type"]] += 1
    kb = {t: {"acc": {m: float(np.mean(acc[t][m])) for m in MECHS}, "tok": {m: float(np.mean(tok[t][m])) for m in MECHS},
              "start": "iterative_search"} for t in acc}
    lo_b, hi_b = min(min(v["tok"].values()) for v in kb.values()), max(max(v["tok"].values()) for v in kb.values())
    budgets = list(np.linspace(lo_b, hi_b, 6))
    times = []
    for _ in range(200):
        for B in budgets:
            t0 = time.perf_counter_ns(); exact_plan(kb, n, B); times.append((time.perf_counter_ns() - t0) / 1e6)
    out["b_budget_planner"][ds] = {"types": len(kb), "assignments": 3 ** len(kb), "ms": summ(times), "budgets_timed": len(budgets),
                                   "repetitions": 200}

# (c) Question Classifier, end to end, one question at a time
from scipy.sparse import csr_matrix, hstack                       # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer       # noqa: E402
from sklearn.linear_model import LogisticRegression               # noqa: E402
from sentence_transformers import SentenceTransformer             # noqa: E402
import torch                                                      # noqa: E402

TEXTS = json.load(open(os.path.join(DATA, "q_texts.json")))
_z = np.load(os.path.join(DATA, "q_emb.npz"), allow_pickle=True)
EMB = {str(i): e for i, e in zip(_z["ids"], _z["emb"])}
enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
out["c_classifier"] = {"torch_threads": torch.get_num_threads()}
for ds in DS:
    ids = sorted(ANS[(ds, "single_lookup")])
    y = [ANS[(ds, "single_lookup")][q]["question_type"] for q in ids]
    vw, vc = TfidfVectorizer(ngram_range=(1, 2)), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    txt = [TEXTS[q] for q in ids]
    X = hstack([csr_matrix(np.stack([EMB[q] for q in ids])), vw.fit_transform(txt), vc.fit_transform(txt)]).tocsr()
    clf = LogisticRegression(random_state=0, class_weight="balanced", max_iter=2000).fit(X, y)
    sample = ids[:200]
    for q in sample[:10]:   # warm-up
        enc.encode([TEXTS[q]], normalize_embeddings=True)
    t_emb, t_rest, cos = [], [], []
    for q in sample:
        t0 = time.perf_counter_ns(); e = enc.encode([TEXTS[q]], normalize_embeddings=True); t1 = time.perf_counter_ns()
        xh = hstack([csr_matrix(e), vw.transform([TEXTS[q]]), vc.transform([TEXTS[q]])]).tocsr(); clf.predict_proba(xh)
        t2 = time.perf_counter_ns()
        t_emb.append((t1 - t0) / 1e6); t_rest.append((t2 - t1) / 1e6); cos.append(float(e[0] @ EMB[q]))
    out["c_classifier"][ds] = {"embedding_ms": summ(t_emb), "tfidf_lr_ms": summ(t_rest),
                               "total_ms": summ([a + b for a, b in zip(t_emb, t_rest)]),
                               "min_cosine_to_cached_embedding": min(cos), "llm_tokens": 0}

out["machine"] = {"processor": platform.processor() or platform.machine(), "python": platform.python_version()}
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", "time_components.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items()}, indent=1)[:3000])
