"""Replay MemSwap (memswap/loop.py) over the logged answers in data/. No model calls.

  python replay.py                        -> results/replay.json (the paper's tables). In the budget experiments the
                                             type shares w_t start from each fold's first labelled batch and are
                                             updated with the questions served from that fold ("first_batch_fold",
                                             the default)
  python replay.py --shares given         -> results/replay_shares_given.json (budget experiments only, with the shares
                                             started from the type counts of the whole dataset)
  python replay.py --shares first_batch   -> results/replay_shares_first_batch.json (budget experiments only; first
                                             batch, then every question served, from any fold)

Every mechanism answered every question once and both judges scored every answer, so the Router's read and the
MechanismEvaluator's trial runs are lookups of logged outcomes. The output mirrors the result files that hold the paper's tables,
under the same keys, so reproduce.py can compare them leaf by leaf:

  results.json                 Table I (per_type), Table II (per_system_per_dataset, within-dataset MemSwap and
                               its baselines), Table IV (streams: MemSwap at every margin and cap)
  budget_adaptation_p40c.json  Table III (budgets, random mixing, drop scenario)
  budget_vs_fixed.json         Table III, the most accurate fixed mechanism within each budget
  type_classifier_nonllm.json  Table II, "with predicted types" (variant a_argmax)
  full_loop.json               Table IV, fixed mechanisms on the streams

The loop is the same for all of them; only the protocol around it differs (which first batch, which questions, which
budget). """
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluation as E                                          # noqa: E402
from memswap.managed.memory_manager.question_classifier import LabelTypes, QuestionClassifier   # noqa: E402
from memswap.knowledge.knowledge_base import AdaptationGoals, MECHANISMS, TOLERANCE  # noqa: E402
from memswap.loop import MemSwap                                # noqa: E402

MECHS = MECHANISMS
DECISIONS = {}      # setting -> list of (question_id, mechanism); written by reproduce.py as results/decisions.csv
SHARES = "first_batch_fold"      # type shares w_t of the budget experiments (memswap/knowledge/knowledge_base.py)


def reader(S_score, TOK):
    """the Router's read in the replay: the logged verdict and memory tokens of mechanism m on question qid."""
    return lambda m, qid: (S_score.at[qid, m], TOK.at[qid, m])


# ------------------------------------------------------------------ Table II: within one dataset
def within_dataset(q, S_by_judge, TOK):
    """within-dataset evaluation, leave-one-conversation-out."""
    res = {}
    labels = LabelTypes(q)
    for ds in E.DATASETS:
        res[ds] = {}
        dq = q[q.dataset == ds]
        qids_all = dq.index.tolist()
        convs = sorted(dq.conversation_id.unique())
        for mpp in E.MARGINS_PP:
            res[ds][f"margin_pp={mpp}"] = {}
            for vname, (sel_j, score_j) in E.JUDGE_VARIANTS.items():
                S_sel, S_score = S_by_judge[sel_j], S_by_judge[score_j]
                ms = MemSwap(AdaptationGoals(margin=mpp / 100.0, cap=0), labels, reader(S_score, TOK), S_sel, TOK)
                ms_v = pd.Series(np.nan, index=qids_all)
                ms_t = pd.Series(np.nan, index=qids_all)
                ms_mech = pd.Series("", index=qids_all)
                selections, fallbacks = {}, []
                for conv in convs:
                    others = dq.index[dq.conversation_id != conv].tolist()
                    ms.first_batch(conv, others, labels.classify, context=f"loco:{ds}:{conv}:{vname}:m{mpp}")
                    held = dq[dq.conversation_id == conv]
                    selections[conv] = {}
                    for qt in sorted(held.question_type.unique()):
                        tk = E.FAMILY[ds] + "|" + qt
                        r = ms.kb.estimates.batch[conv].get(tk)
                        if r is None:                                   # type absent from the first batch: m0
                            r = ms.kb.estimates.batch_default[conv]
                            fallbacks.append({"conversation": conv, "type": qt})
                        selections[conv][qt] = {"chosen": r["chosen"], "best": r["best"], "n_profiled": r["n"],
                                                "chosen_z1645": r["chosen_z1645"]}
                    for i, qid in enumerate(held.index):
                        rec = ms.handle(qid, conv, i + 1)
                        ms_v.loc[qid] = rec["outcome"]; ms_t.loc[qid] = rec["tokens"]; ms_mech.loc[qid] = rec["mechanism"]
                assert not ms_v.isna().any()
                if vname in ("judge1", "judge2") and mpp in (1, 10):
                    DECISIONS[f"within|{ds}|labels|margin{mpp}|{vname}"] = list(zip(qids_all, ms_mech.tolist()))
                share = {m: int((ms_mech == m).sum()) for m in MECHS}
                res[ds][f"margin_pp={mpp}"][vname] = {
                    "memswitch": E.accuracy_tokens(ms_v.to_numpy(), ms_t.to_numpy()),
                    "questions_served_by_mechanism": share,
                    "selections_per_held_out_conversation": selections,
                    "fallbacks_type_absent_from_other_conversations": fallbacks,
                    "vs_baselines": E.compare_to_baselines(qids_all, S_score, TOK, ms_v.to_numpy())}
    return res


# ------------------------------------------------------------------ Table IV: streams
def run_stream(fb, sq, q, S_sel, S_score, TOK, margin, cap, context):
    """one workload stream, driven through the loop."""
    labels = LabelTypes(q)
    ms = MemSwap(AdaptationGoals(margin=margin, cap=cap), labels, reader(S_score, TOK), S_sel, TOK)
    default = ms.first_batch("fb", fb, labels.classify, context=context)
    fb_decisions = {t: {"chosen": r["chosen"], "best": r["best"], "n_profiled": r["n"]}
                    for t, r in sorted(ms.kb.estimates.batch["fb"].items())}
    verdicts = np.empty(len(sq)); tokens = np.empty(len(sq))
    served, runtime_decisions, extra_profiling_tokens = [], {}, []
    n_runtime_profiled = n_default_unprofiled = 0
    seen_in_dataset = {}
    for i, qid in enumerate(sq):
        ds_q = q.at[qid, "dataset"]
        seen_in_dataset[ds_q] = seen_in_dataset.get(ds_q, 0) + 1
        rec = ms.handle(qid, "fb", i + 1, context=context)
        verdicts[i] = rec["outcome"]; tokens[i] = rec["charged_tokens"]; served.append(rec["mechanism"])
        if rec["known"]:
            continue
        if not rec["profiled"]:
            n_default_unprofiled += 1
            continue
        extra_profiling_tokens.append(rec["charged_tokens"] - rec["tokens"])
        n_runtime_profiled += 1
        r = rec["decision"]
        if r is not None:
            runtime_decisions[rec["type"]] = {"chosen": r["chosen"], "best": r["best"], "n_profiled": rec["n_profiled"],
                                              "decided_by": "cheaper_sufficient_found" if r["chosen"] != r["best"] else "cap_reached",
                                              "stream_position": i + 1,
                                              "position_within_dataset_block": seen_in_dataset[ds_q],
                                              "question_id": qid}
    kb = ms.kb.routing.table["fb"]
    switched = {tk: v for tk, v in runtime_decisions.items() if v["decided_by"] == "cheaper_sufficient_found"}
    types_in_stream_not_in_fb = sorted(set(q.loc[sq, "type_key"]) - set(fb_decisions))
    undecided = [tk for tk in types_in_stream_not_in_fb if tk not in kb]
    return {
        "n_first_batch": len(fb), "n_stream": len(sq),
        "default_mechanism": default["chosen"], "default_rule": {k: default[k] for k in ("best", "n", "accuracy", "mean_tokens", "wilson_lb")},
        "first_batch_decisions": fb_decisions,
        "types_in_stream_not_in_first_batch": types_in_stream_not_in_fb,
        "runtime_decisions": runtime_decisions,
        "types_never_decided": undecided,
        "n_questions_runtime_profiled": n_runtime_profiled,
        "n_questions_default_unprofiled": n_default_unprofiled,
        "n_types_switched_to_cheaper_at_runtime": len(switched),
        "types_switched_to_cheaper_at_runtime": switched,
        "extra_tokens_per_runtime_profiled_question": (float(np.mean(extra_profiling_tokens))
                                                       if extra_profiling_tokens else None),
        "questions_served_by_mechanism": {m: int(sum(1 for s in served if s == m)) for m in MECHS},
        "memswitch": E.accuracy_tokens(verdicts, tokens),
        "_verdicts": verdicts, "_served": served,
    }


def streams(q, s, S_by_judge, TOK):
    """all workload streams, calling run_stream above."""
    res = {}
    for st in sorted(s.stream.unique()):
        g = s[s.stream == st]
        fb = g[g.role == "first_batch"].question_id.tolist()
        sq = g[g.role == "stream"].sort_values("position").question_id.tolist()
        name = g.stream_name.iloc[0]
        res[f"stream_{st}"] = {"name": name, "n_first_batch": len(fb), "n_stream": len(sq),
                               "stream_datasets_in_order": list(dict.fromkeys(q.loc[sq, "dataset"])),
                               "configs": {}}
        for mpp in E.MARGINS_PP:
            cap0_verdicts = {}
            for cap in E.CAPS:
                key = f"margin_pp={mpp},cap={cap}"
                res[f"stream_{st}"]["configs"][key] = {}
                for vname, (sel_j, score_j) in E.JUDGE_VARIANTS.items():
                    r = run_stream(fb, sq, q, S_by_judge[sel_j], S_by_judge[score_j], TOK, mpp / 100.0, cap,
                                   context=f"stream{st}:m{mpp}:cap{cap}:{vname}")
                    v = r.pop("_verdicts")
                    served = r.pop("_served")
                    if cap == 0 and mpp == 1 and vname in ("judge1", "judge2"):
                        DECISIONS[f"stream|stream_{st}|labels|margin1|{vname}"] = list(zip(sq, served))
                    r["vs_baselines_on_stream_questions"] = E.compare_to_baselines(sq, S_by_judge[score_j], TOK, v)
                    if cap == 0:
                        cap0_verdicts[vname] = v
                    r["vs_memswitch_cap_0_same_margin"] = E.mcnemar(v, cap0_verdicts[vname])
                    res[f"stream_{st}"]["configs"][key][vname] = r
    return res


# ------------------------------------------------------------------ Table III: budgets
def run_budget(order, q, S, TOK, folds_fb, mix0, budget_of, shares=SHARES):
    """one budget run, driven through the loop."""
    labels = LabelTypes(q)
    ms = MemSwap(AdaptationGoals(margin=E.MARGIN, cap=0), labels, reader(S, TOK), S, TOK, mix=mix0, shares=shares)
    for f, train in folds_fb.items():
        ms.first_batch(f, train, labels.classify)
    first_moves = ms.set_budget(budget_of(0))                 # the budget in force from the start
    v = np.empty(len(order)); t = np.empty(len(order)); served = Counter(); mechs = []
    for i, (qid, f) in enumerate(order):
        nb = budget_of(i)
        if nb != ms.kb.goals.budget:
            ms.set_budget(nb, at=i + 1)
        rec = ms.handle(qid, f, i + 1)
        v[i] = rec["outcome"]; t[i] = rec["tokens"]; served[rec["mechanism"]] += 1; mechs.append(rec["mechanism"])
    windows = ms.usage_tracker.windows
    kept = float(np.mean([w["kept"] for w in windows])) if windows else None
    return {"memswitch": E.accuracy_tokens(v, t), "windows_kept_share": kept, "n_windows": len(windows),
            "windows": windows,
            "questions_served_by_mechanism": dict(served), "initial_moves": first_moves,
            "replans": ms.replans, "_v": v, "_t": t, "_mechs": mechs}


def eval_case(name, order, q, S, TOK, folds_fb, mix0, jname, out, shares=SHARES):
    """all budget levels of one dataset and judge, with budget-matched mixing."""
    qids = [qid for qid, _ in order]
    fixed_tok, levels = E.settings_for(name, order, q, S, TOK)
    fixed = {m: {"accuracy": float(S.loc[qids, m].mean()), "mean_tokens": fixed_tok[m]} for m in MECHS}
    case = {"n": len(qids), "budget_levels": [round(b, 1) for b in levels], "fixed": fixed, "budgets": {}}
    for B in levels:
        r = run_budget(order, q, S, TOK, folds_fb, mix0, lambda i, B=B: B, shares)
        v = r.pop("_v"); r.pop("_t")
        DECISIONS[f"budget|{name}|B={B:.0f}|{jname}"] = list(zip(qids, r.pop("_mechs")))
        for m in MECHS:
            r[f"vs_fixed_{m}"] = E.mcnemar(v, S.loc[qids, m].to_numpy(dtype=float))
        r["random_mixing"] = E.random_mixing_nd(order, S, TOK, B, fixed_tok, v)
        case["budgets"][f"B={B:.0f}"] = r
    base = levels[3]
    n = len(order)

    def bf(i):
        return base * 0.6 if n // 2 <= i < 3 * n // 4 else base

    r = run_budget(order, q, S, TOK, folds_fb, mix0, bf, shares)
    v = r.pop("_v"); r.pop("_t")
    DECISIONS[f"budget|{name}|drop|{jname}"] = list(zip(qids, r.pop("_mechs")))
    for m in MECHS:
        r[f"vs_fixed_{m}"] = E.mcnemar(v, S.loc[qids, m].to_numpy(dtype=float))
    case["budget_drop_scenario"] = {"base_B": round(base, 1), "drop_B": round(base * 0.6, 1), **r}
    out[jname] = case


def budgets(q, S_by_judge, TOK, shares=SHARES):
    """every dataset, both judges.
    shares="first_batch_fold" (the default) starts a fold's type shares from its first labelled batch and updates
    them with the questions served from that fold; "first_batch" updates them with every question served.
    "given" starts them from the counts of the whole dataset (memswap/knowledge/knowledge_base.py)."""
    out = {"tolerance": TOLERANCE, "window": E.WIN, "cases": {}}
    for ds in E.DATASETS:
        dq = q[q.dataset == ds]
        case = {}
        for jname, jk in (("judge1", "j1"), ("judge2", "j2")):
            S = S_by_judge[jk]
            order = [(qid, dq.at[qid, "conversation_id"]) for qid in dq.index]
            folds_fb = {conv: dq.index[dq.conversation_id != conv].tolist() for conv in sorted(dq.conversation_id.unique())}
            mix0 = Counter(dq.type_key) if shares == "given" else None
            eval_case(ds, order, q, S, TOK, folds_fb, mix0, jname, case, shares)
            # label selection reference (budget_adaptation_exact.py:66 to :72): sigma_0 without a budget
            labels = LabelTypes(q)
            ms = MemSwap(AdaptationGoals(margin=E.MARGIN, cap=0), labels, reader(S, TOK), S, TOK)
            for f, train in folds_fb.items():
                ms.first_batch(f, train, labels.classify)
            vl, tl = [], []
            for i, (qid, f) in enumerate(order):
                rec = ms.handle(qid, f, i + 1)
                vl.append(rec["outcome"]); tl.append(rec["tokens"])
            case[jname]["label_selection"] = E.accuracy_tokens(np.array(vl, dtype=float), np.array(tl, dtype=float))
        out["cases"][ds] = case
    return out


def budget_vs_fixed(P):
    """the most accurate fixed mechanism within each budget."""
    tol = P["tolerance"]; out = {"tolerance": tol, "cases": {}}
    for ds, c in P["cases"].items():
        for jv in ("judge1", "judge2"):
            cj = c[jv]; fixed = cj["fixed"]
            for bkey, x in cj["budgets"].items():
                B = float(bkey.split("=")[1])
                fits = {m: f for m, f in fixed.items() if f["mean_tokens"] <= B * (1 + tol)}
                row = {"budget": B, "memswap_accuracy": x["memswitch"]["accuracy"], "memswap_tokens": x["memswitch"]["mean_memory_tokens"],
                       "fixed_that_fit": sorted(fits)}
                if fits:
                    best = max(fits, key=lambda m: (fits[m]["accuracy"], -fits[m]["mean_tokens"]))
                    v = x["vs_fixed_" + best]
                    row.update({"best_fixed": best, "best_fixed_accuracy": fits[best]["accuracy"], "best_fixed_tokens": fits[best]["mean_tokens"],
                                "diff_pp": v["diff_pp"], "mcnemar_p": v["mcnemar_p"]})
                out["cases"].setdefault(ds, {}).setdefault(jv, {})[bkey] = row
    return out


# ------------------------------------------------------------------ Table II, with predicted types
def predicted_types(q, S_by_judge, TOK):
    """within-dataset evaluation with predicted question types (variant a_argmax)."""
    out = {"datasets": {}}
    for ds in E.DATASETS:
        dq = q[q.dataset == ds]
        qids = dq.index.tolist()
        pos = {qid: j for j, qid in enumerate(qids)}
        clf = QuestionClassifier(prefix=E.FAMILY[ds] + "|")
        for conv in sorted(dq.conversation_id.unique()):                      # classify once per fold
            train = dq.index[dq.conversation_id != conv].tolist()
            held = dq.index[dq.conversation_id == conv].tolist()
            clf.fit_predict(train, held, dq.loc[train, "question_type"].tolist())
        per_type = {}
        for qid in qids:
            t_true, p = dq.at[qid, "question_type"], clf.pred[qid]
            e = per_type.setdefault(t_true, {"n": 0, "correct": 0})
            e["n"] += 1; e["correct"] += int(p == t_true)
        for e in per_type.values():
            e["accuracy"] = e["correct"] / e["n"]
        entry = {"n": len(qids), "accuracy": sum(e["correct"] for e in per_type.values()) / len(qids), "per_type": per_type,
                 "replay": {}}
        labels = LabelTypes(q)
        for jname, jk in (("judge1", "j1"), ("judge2", "j2")):
            S = S_by_judge[jk]
            runs = {}
            for vname, typer in (("labels", labels), ("a_argmax", clf)):
                ms = MemSwap(AdaptationGoals(margin=E.MARGIN, cap=0), typer, reader(S, TOK), S, TOK)   # confusion_routing.py:23, MARGIN = 0.01
                vv = np.full(len(qids), np.nan); tt = np.zeros(len(qids)); served = Counter(); mech_of = {}
                for conv in sorted(dq.conversation_id.unique()):
                    train = dq.index[dq.conversation_id != conv].tolist()
                    ms.first_batch(conv, train, labels.classify)                # the knowledge comes from labelled profiling
                    for i, qid in enumerate(dq.index[dq.conversation_id == conv]):
                        rec = ms.handle(qid, conv, i + 1)
                        vv[pos[qid]] = rec["outcome"]; tt[pos[qid]] = rec["tokens"]; served[rec["mechanism"]] += 1; mech_of[qid] = rec["mechanism"]
                runs[vname] = (vv, tt, served)
                if vname == "a_argmax":
                    DECISIONS[f"within|{ds}|predicted|margin1|{jname}"] = [(qid, mech_of[qid]) for qid in qids]
            vl, tl, _ = runs["labels"]
            fixed = S.loc[qids, "iterative_search"].to_numpy(dtype=float)
            je = {"labels": E.accuracy_tokens(vl, tl),
                  "fixed_iterative_search": E.accuracy_tokens(fixed, TOK.loc[qids, "iterative_search"].to_numpy())}
            mem0 = S.loc[qids, "mem0"].to_numpy(dtype=float)
            vv, tt, served = runs["a_argmax"]
            e = E.accuracy_tokens(vv, tt)
            e["questions_served_by_mechanism"] = dict(served)
            e["vs_labels"] = E.mcnemar(vv, vl)
            e["vs_fixed_iterative_search"] = E.mcnemar(vv, fixed)
            if not np.isnan(mem0).all():
                e["vs_mem0"] = E.mcnemar(vv, mem0)
            je["a_argmax"] = e
            entry["replay"][jname] = je
        out["datasets"][ds] = entry
    return out


# ------------------------------------------------------------------ Table IV, fixed mechanisms
def fixed_on_streams(q, s, S_by_judge, TOK):
    """copied from the original analysis script full_loop_tables.py, lines 28 to 43 (the fixed-mechanism rows)."""
    out = {"table3": {}}
    for st in sorted(s.stream.unique()):
        g = s[s.stream == st]
        sq = g[g.role == "stream"].sort_values("position").question_id.tolist()
        so = {"n_stream": len(sq)}
        for jname, jk in (("judge1", "j1"), ("judge2", "j2")):
            S = S_by_judge[jk]
            is_v = S.loc[sq, "iterative_search"].to_numpy(dtype=float)
            e = {}
            for m in MECHS:
                v = S.loc[sq, m].to_numpy(dtype=float)
                r = E.accuracy_tokens(v, TOK.loc[sq, m].to_numpy())
                r["vs_fixed_iterative_search"] = E.mcnemar(v, is_v)
                e[m] = r
            so[jname] = e
        out["table3"][f"stream_{st}"] = so
    return out


def main_shares(shares):
    """the budget experiments only, with the type shares of mode `shares` -> results/replay_shares_<shares>.json"""
    E.check_sums()
    a, s, q, S_by_judge, _, TOK = E.load()
    if shares not in ("given", "first_batch", "first_batch_fold"):
        raise SystemExit("unknown shares mode: " + shares)
    out = {"shares": shares, "budget_adaptation_p40c.json": budgets(q, S_by_judge, TOK, shares)}
    out["budget_vs_fixed.json"] = budget_vs_fixed(out["budget_adaptation_p40c.json"])
    path = os.path.join(HERE, "results", f"replay_shares_{shares}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, allow_nan=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        f.write("\n")
    print("wrote", path)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--shares":
        return main_shares(sys.argv[2])
    E.check_sums()
    a, s, q, S_by_judge, _, TOK = E.load()
    J = {"judge1": S_by_judge["j1"], "judge2": S_by_judge["j2"]}
    out = {"results.json": {"per_system_per_dataset": E.per_system(q, J, TOK),
                            "per_type": E.per_type(q, J, TOK),
                            "memswitch_within_dataset_leave_one_conversation_out": within_dataset(q, S_by_judge, TOK),
                            "streams": streams(q, s, S_by_judge, TOK)}}
    print("results.json parts done", flush=True)
    out["budget_adaptation_p40c.json"] = budgets(q, S_by_judge, TOK)
    out["budget_vs_fixed.json"] = budget_vs_fixed(out["budget_adaptation_p40c.json"])
    print("budgets done", flush=True)
    out["type_classifier_nonllm.json"] = predicted_types(q, S_by_judge, TOK)
    out["full_loop.json"] = fixed_on_streams(q, s, S_by_judge, TOK)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "replay.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, allow_nan=False, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        f.write("\n")
    print("wrote results/replay.json")


if __name__ == "__main__":
    main()
