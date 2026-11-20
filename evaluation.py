"""Evaluation helpers that are not MemSwap components: data loading, statistics and the baselines (fixed
mechanisms, random selection, random mixing, Mem0, Letta). Every block is copied unchanged from the scripts that
produced the paper's numbers; the header of each block names its source lines. Only the module paths changed:
select comes from memswap/plan/mechanism_selector.py (a copy of the same code), and the names R and BA, which the
budget scripts used for their imports, are bound below to this module's own copies."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from memswap.plan.mechanism_selector import select, EPS  # noqa: F401  (rule_on calls select)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

MECHS = ["single_lookup", "planned_read", "iterative_search"]
EXTRA_SYSTEMS = ["iterative_search_unpruned", "mem0", "letta"]
ALL_SYSTEMS = MECHS + EXTRA_SYSTEMS
DATASETS = ["locomo", "longmemeval_s_star", "longmemeval_r"]
FAMILY = {"locomo": "locomo", "longmemeval_s_star": "lme", "longmemeval_r": "lme"}

MARGINS_PP = [1, 2, 5, 10]          # accuracy margins in points
CAPS = [0, 30, 100, 300]            # profiling caps
RANDOM_DRAWS = 200
RANDOM_SEED = 0
Z_SPEC = 1.645                      # z of the one-sided 95% Wilson bound
Z_SM = float(norm.ppf(0.95))        # z used by statsmodels for alpha=0.10 two-sided
EPS = 1e-12                         # tolerance for ">=" comparisons on floats

# judge variants:
#   judge1          : selection and scoring with verdict_judge1
#   judge2          : selection and scoring with verdict_judge2 (filled from judge1 where empty)
#   judge2_rescored : selection with judge1, scoring with judge2 (filled)
JUDGE_VARIANTS = {
    "judge1": ("j1", "j1"),
    "judge2": ("j2", "j2"),
    "judge2_rescored": ("j1", "j2"),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sums():
    sums = {}
    with open(os.path.join(DATA, "SHA256SUMS")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(None, 1)
            name = name.lstrip("*")
            actual = sha256(os.path.join(DATA, name))
            if actual != digest:
                sys.exit(f"CHECKSUM MISMATCH for data/{name}: expected {digest}, got {actual}")
            sums[name] = digest
    return sums


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load():
    a = pd.read_csv(os.path.join(DATA, "answers.csv"),
                    dtype={"verdict_judge1": "int64", "verdict_judge2": "Int64", "steps": "Int64",
                           "memory_tokens": "int64"})
    s = pd.read_csv(os.path.join(DATA, "streams.csv"), dtype={"position": "Int64"})
    assert not a.duplicated(["question_id", "system"]).any()
    q = (a.drop_duplicates("question_id")[["question_id", "dataset", "conversation_id", "question_type"]]
         .set_index("question_id"))
    q["family"] = q["dataset"].map(FAMILY)
    q["type_key"] = q["family"] + "|" + q["question_type"]
    V1 = a.pivot(index="question_id", columns="system", values="verdict_judge1").astype("float64")
    V2raw = a.pivot(index="question_id", columns="system", values="verdict_judge2").astype("float64")
    TOK = a.pivot(index="question_id", columns="system", values="memory_tokens").astype("float64")
    # second judge: fill empties from judge 1 for every system (ambiguity A3/A4).
    V2 = V2raw.copy()
    for sysname in ALL_SYSTEMS:
        V2[sysname] = V2raw[sysname].where(V2raw[sysname].notna(), V1[sysname])
    for M in (V1, V2raw, V2, TOK):
        M.columns.name = None
    return a, s, q, {"j1": V1, "j2": V2}, V2raw, TOK

def rule_on(qids, S, TOK, margin, context=None):
    """Apply the selection rule to a set of questions using score matrix S."""
    n = len(qids)
    if n == 0:
        k = {m: 0 for m in MECHS}
        tok = {m: 0.0 for m in MECHS}
    else:
        sub = S.loc[qids, MECHS]
        k = {m: int(sub[m].sum()) for m in MECHS}
        tok = {m: float(TOK.loc[qids, m].mean()) for m in MECHS}
    return select(k, n, tok, margin, context)


def mcnemar(x, y):
    """Exact two-sided McNemar test on paired 0/1 arrays. b = x correct & y wrong."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    if b + c == 0:
        p = 1.0
    else:
        p = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return {"n": int(ok.sum()), "b_memswitch_only_correct": b, "c_baseline_only_correct": c,
            "mcnemar_p": p, "diff_pp": float((x.mean() - y.mean()) * 100.0) if ok.sum() else None,
            "significant_0_05": bool(p < 0.05)}


def accuracy_tokens(v, t):
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = ~np.isnan(v)
    return {"n": int(ok.sum()),
            "n_correct": int(np.nansum(v)),
            "accuracy": float(np.nanmean(v)) if ok.any() else None,
            "mean_memory_tokens": float(t[ok].mean()) if ok.any() else None,
            "total_memory_tokens": float(t[ok].sum()) if ok.any() else None}


# ----------------------------------------------------------------------------
# random selection
# ----------------------------------------------------------------------------

RANDOM_DRAW_CACHE = {}


def random_draws(qids):
    """200 draws of a uniform mechanism index per question; one generator per question set."""
    key = tuple(qids)
    if key not in RANDOM_DRAW_CACHE:
        rng = np.random.default_rng(RANDOM_SEED)
        RANDOM_DRAW_CACHE[key] = [rng.integers(0, 3, len(qids)) for _ in range(RANDOM_DRAWS)]
    return RANDOM_DRAW_CACHE[key]


def random_selection(qids, S, TOK, ms_verdicts):
    draws = random_draws(qids)
    SV = S.loc[qids, MECHS].to_numpy()
    TV = TOK.loc[qids, MECHS].to_numpy()
    rows = np.arange(len(qids))
    accs, ps, toks = [], [], []
    for d in draws:
        v = SV[rows, d]
        accs.append(float(np.mean(v)))
        toks.append(float(TV[rows, d].mean()))
        ps.append(mcnemar(ms_verdicts, v)["mcnemar_p"])
    mean_tok_of_means = float(np.mean([TV[:, i].mean() for i in range(3)]))
    return {"n": len(qids), "draws": RANDOM_DRAWS, "seed": RANDOM_SEED,
            "accuracy_mean_over_draws": float(np.mean(accs)),
            "accuracy_min_over_draws": float(np.min(accs)),
            "accuracy_max_over_draws": float(np.max(accs)),
            "mean_memory_tokens_spec": mean_tok_of_means,
            "mean_memory_tokens_mean_over_draws": float(np.mean(toks)),
            "mcnemar_p_median_vs_memswitch": float(np.median(ps)),
            "mcnemar_p_min_vs_memswitch": float(np.min(ps)),
            "mcnemar_p_max_vs_memswitch": float(np.max(ps)),
            "diff_pp_memswitch_minus_random": float((np.mean(ms_verdicts) - np.mean(accs)) * 100.0)}


def compare_to_baselines(qids, S, TOK, ms_v):
    out = {}
    for sysname in ALL_SYSTEMS:
        if sysname not in S.columns:
            continue
        bv = S.loc[qids, sysname].to_numpy(dtype=float)
        if np.isnan(bv).all():
            continue
        r = mcnemar(ms_v, bv)
        r["baseline_accuracy"] = float(np.nanmean(bv))
        r["baseline_mean_memory_tokens"] = float(TOK.loc[qids, sysname].to_numpy()[~np.isnan(bv)].mean())
        r["covers_all_questions"] = bool(r["n"] == len(qids))
        out[sysname] = r
    out["random_selection"] = random_selection(qids, S, TOK, ms_v)
    return out

def per_system(q, S_by_judge, TOK):
    res = {}
    for ds in DATASETS:
        qids = q.index[q.dataset == ds].tolist()
        res[ds] = {"n_questions": len(qids)}
        for jname, S in S_by_judge.items():
            res[ds][jname] = {}
            for sysname in ALL_SYSTEMS:
                v = S.loc[qids, sysname].to_numpy(dtype=float)
                if np.isnan(v).all():
                    res[ds][jname][sysname] = None
                    continue
                res[ds][jname][sysname] = accuracy_tokens(v, TOK.loc[qids, sysname].to_numpy())
    return res

def per_type(q, S_by_judge, TOK):
    res = {}
    for ds in DATASETS:
        res[ds] = {}
        for qt in sorted(q.question_type[q.dataset == ds].unique()):
            qids = q.index[(q.dataset == ds) & (q.question_type == qt)].tolist()
            entry = {"n_questions": len(qids)}
            for jname, S in S_by_judge.items():
                je = {"systems": {}}
                for sysname in ALL_SYSTEMS:
                    v = S.loc[qids, sysname].to_numpy(dtype=float)
                    if np.isnan(v).all():
                        continue
                    je["systems"][sysname] = accuracy_tokens(v, TOK.loc[qids, sysname].to_numpy())
                je["selection_by_margin_pp"] = {}
                for mpp in MARGINS_PP:
                    r = rule_on(qids, S, TOK, mpp / 100.0, context=f"per_type:{ds}:{qt}:{jname}:m{mpp}")
                    je["selection_by_margin_pp"][str(mpp)] = {
                        "best": r["best"], "cheapest_sufficient": r["chosen"],
                        "cheapest_sufficient_z1645": r["chosen_z1645"], "wilson_lb": r["wilson_lb"]}
                entry[jname] = je
            res[ds][qt] = entry
    return res

class R:
    MECHS = MECHS
    mcnemar = staticmethod(mcnemar)
    accuracy_tokens = staticmethod(accuracy_tokens)


MARGIN = 0.01
WIN = 50
N_LEVELS = 6
DRAWS, SEED = 200, 0

class BA:
    SEED, DRAWS, WIN = SEED, DRAWS, WIN


def settings_for(name, order, q, S, TOK):
    qids = [qid for qid, _ in order]
    fixed_tok = {m: float(TOK.loc[qids, m].mean()) for m in R.MECHS}
    lo, hi = fixed_tok["planned_read"], fixed_tok["iterative_search"]
    levels = [lo + k * (hi - lo) / (N_LEVELS - 1) for k in range(N_LEVELS)]
    return fixed_tok, levels

def non_dominated(fixed_tok, fixed_acc):
    keep = []
    for m in R.MECHS:
        dom = any(fixed_acc[o] >= fixed_acc[m] and fixed_tok[o] <= fixed_tok[m]
                  and (fixed_acc[o] > fixed_acc[m] or fixed_tok[o] < fixed_tok[m]) for o in R.MECHS if o != m)
        if not dom:
            keep.append(m)
    return keep


def random_mixing_nd(order, S, TOK, B, fixed_tok, ms_v):
    qids = [qid for qid, _ in order]
    fixed_acc = {m: float(S.loc[qids, m].mean()) for m in R.MECHS}
    nd = sorted(non_dominated(fixed_tok, fixed_acc), key=lambda m: fixed_tok[m])
    lo = max((m for m in nd if fixed_tok[m] <= B), key=lambda m: fixed_tok[m], default=nd[0])
    hi = min((m for m in nd if fixed_tok[m] >= B), key=lambda m: fixed_tok[m], default=nd[-1])
    pr = 0.0 if hi == lo else (B - fixed_tok[lo]) / (fixed_tok[hi] - fixed_tok[lo])
    Vlo = S.loc[qids, lo].to_numpy(dtype=float); Vhi = S.loc[qids, hi].to_numpy(dtype=float)
    Tlo = TOK.loc[qids, lo].to_numpy(); Thi = TOK.loc[qids, hi].to_numpy()
    rng = np.random.default_rng(BA.SEED)
    accs, toks, ps = [], [], []
    for _ in range(BA.DRAWS):
        pick = rng.random(len(qids)) < pr
        vv = np.where(pick, Vhi, Vlo)
        accs.append(float(vv.mean())); toks.append(float(np.where(pick, Thi, Tlo).mean()))
        ps.append(R.mcnemar(ms_v, vv)["mcnemar_p"])
    return {"mechanisms": [lo, hi], "non_dominated": nd, "p_expensive": pr,
            "accuracy_mean_over_draws": float(np.mean(accs)), "mean_tokens_over_draws": float(np.mean(toks)),
            "mcnemar_p_median_vs_memswitch": float(np.median(ps)),
            "diff_pp_memswitch_minus_random": float((np.mean(ms_v) - np.mean(accs)) * 100)}

