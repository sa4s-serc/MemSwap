"""Rebuild every table and every number of the paper from data/. No model calls, no API keys.

    python reproduce.py            about 2 minutes on a 6-core desktop
    python reproduce.py --fast     skips the predicted question types check (about 1.5 minutes); its numbers are then
                                   read from results/expected/ and reported as SKIPPED

What it does
  1. checks the SHA-256 of the data files;
  2. replays MemSwap over the logged answers (replay.py)                    -> results/replay.json
  3. MemSwap against Letta with and without Letta's empty answers         -> results/letta_empty_answers.json
  4. the workload streams with predicted question types                   -> results/predicted_question_types.json
  5. writes which mechanism served each question in every experiment      -> results/decisions.csv
  6. writes Tables I to IV as CSV                                         -> results/tables/
  7. compares every number of the paper (results/paper_numbers.csv) with the recomputed value, PASS or FAIL,
     and every leaf of the recomputed result files with results/expected/ -> results/report.md

The timings and judge-token counts of RQ3 were measured once on our machine and from our logs. They cannot be
recomputed from data/, so they are read from results/expected/logged/ and reported as LOGGED.
Exit code 0 only if nothing failed.
"""
import csv
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "checks"))
RES = os.path.join(HERE, "results")
EXP = os.path.join(RES, "expected")


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, p + "/" + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, p + "[%d]" % i)
    else:
        yield p, x


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def compare_json(name):
    a = dict(flat(json.load(open(os.path.join(EXP, name)))))
    b = dict(flat(json.load(open(os.path.join(RES, name)))))
    diff = [k for k in a if k not in b or not same(a[k], b[k])] + [k for k in b if k not in a]
    return len(a), diff


def write_decisions(decisions):
    path = os.path.join(RES, "decisions.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "question_id", "mechanism"])
        for setting in sorted(decisions):
            for qid, m in decisions[setting]:
                w.writerow([setting, qid, m])
    return path


def write_tables(R):
    import paper_numbers as PN
    out = os.path.join(RES, "tables")
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(RES, "paper_numbers.csv"))))
    for tab in ("Table I", "Table II"):
        with open(os.path.join(out, tab.lower().replace(" ", "_") + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["what", "value", "judge", "source_key"])
            for r in rows:
                if r["where"] == tab:
                    w.writerow([r["what"], PN.value(R, r["source_key"], r["format"]), r["judge"], r["source_key"]])
    mech = ["single_lookup", "planned_read", "iterative_search"]
    with open(os.path.join(out, "table_iii_budgets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "judge", "budget", "best_fixed_that_fits", "best_fixed_accuracy", "memswap_accuracy", "memswap_tokens",
                    "memswap_minus_fixed_pp", "mcnemar_p", "mixing_accuracy", "memswap_minus_mixing_pp", "mixing_median_p",
                    "windows_within_budget", "windows"])
        for ds, c in R.get("R:budget_vs_fixed.json/cases").items():
            for j in ("judge1", "judge2"):
                for bk in sorted(c[j], key=lambda k: int(k[2:])):
                    x = c[j][bk]
                    b = R.get("R:budget_adaptation_p40c.json/cases/%s/%s/budgets/%s" % (ds, j, bk))
                    rm = b["random_mixing"]
                    w.writerow([ds, j, round(x["budget"]), x.get("best_fixed"), x.get("best_fixed_accuracy"), x["memswap_accuracy"], x["memswap_tokens"],
                                x.get("diff_pp"), x.get("mcnemar_p"), rm.get("accuracy_mean_over_draws"), rm.get("diff_pp_memswitch_minus_random"),
                                rm.get("mcnemar_p_median_vs_memswitch"), sum(1 for v in b["windows"] if v["kept"]), b["n_windows"]])
    with open(os.path.join(out, "table_iv_streams.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream", "name", "judge", "questions", "question_types", "accuracy", "memory_tokens"] + ["fixed_%s_accuracy" % m for m in mech])
        streams = R.get("R:results.json/streams")
        for st in sorted(streams):
            for j in ("judge1", "judge2"):
                c = streams[st]["configs"]["margin_pp=1,cap=0"][j]
                fx = [c["vs_baselines_on_stream_questions"][m]["baseline_accuracy"] for m in mech]
                w.writerow([st, streams[st]["name"], j, streams[st]["n_stream"], "labels", c["memswitch"]["accuracy"], c["memswitch"]["mean_memory_tokens"]] + fx)
                try:
                    o = R.get("OS:streams/%s/%s/predicted" % (st, j))
                    w.writerow([st, streams[st]["name"], j, streams[st]["n_stream"], "predicted", o["accuracy_pct"] / 100, o["mean_memory_tokens"]] + fx)
                except FileNotFoundError:
                    pass
    return out


def derived_checks(R):
    """arithmetic that ties the logged RQ3 file to the recomputed results."""
    out = []
    for ds in ("locomo", "longmemeval_s_star", "longmemeval_r"):
        ms = R.get("W:%s/margin_pp=1/judge1/memswitch/mean_memory_tokens" % ds)
        fx = R.get("W:%s/margin_pp=1/judge1/vs_baselines/iterative_search/baseline_mean_memory_tokens" % ds)
        ng = R.get("NG:datasets/%s" % ds)
        extra = R.get("EF:d_profiling/%s/extra_over_default_mean" % ds)
        out.append(("saving per question, %s" % ds, ng["saving_per_question"], fx - ms))
        out.append(("payback questions, %s" % ds, ng["a_payback_memory_tokens_only"], ng["first_batch_questions"] * extra / (fx - ms)))
        all3 = sum(R.get("R:results.json/per_system_per_dataset/%s/judge1/%s/mean_memory_tokens" % (ds, m)) for m in ("single_lookup", "planned_read", "iterative_search"))
        out.append(("profiling tokens of all three mechanisms, %s" % ds, R.get("EF:d_profiling/%s/memory_tokens_all_three_mean" % ds), all3))
    return [(n, a, b, math.isclose(a, b, rel_tol=1e-9)) for n, a, b in out]


def main():
    fast = "--fast" in sys.argv
    t0 = time.time()
    import evaluation as E
    import paper_numbers as PN
    import replay
    E.check_sums()
    print("[1/7] data checksums ok", flush=True)
    old_argv, sys.argv = sys.argv, [sys.argv[0]]
    replay.main()
    print("[2/7] replay done (%.0f s)" % (time.time() - t0), flush=True)
    import letta_empty_answers
    letta_empty_answers.main()
    print("[3/7] Letta check done", flush=True)
    decisions = dict(replay.DECISIONS)
    if fast:
        skipped_files = {"OS"}
        print("[4/7] predicted question types check SKIPPED (--fast)", flush=True)
    else:
        skipped_files = set()
        import predicted_question_types
        predicted_question_types.main()
        decisions.update(predicted_question_types.DECISIONS)
        print("[4/7] streams with predicted question types done (%.0f s)" % (time.time() - t0), flush=True)
    sys.argv = old_argv
    write_decisions(decisions)
    os.makedirs(os.path.join(RES, "logged"), exist_ok=True)
    for f in os.listdir(os.path.join(EXP, "logged")):
        shutil.copy2(os.path.join(EXP, "logged", f), os.path.join(RES, "logged", f))
    R = PN.Results(RES)
    write_tables(R)
    print("[5/7] decisions.csv and tables written", flush=True)

    rows = list(csv.DictReader(open(os.path.join(RES, "paper_numbers.csv"))))
    counts = {"PASS": 0, "FAIL": 0, "LOGGED": 0, "SKIPPED": 0}
    lines = []
    for r in rows:
        f = R.file_of(r["source_key"].split("  /  ")[0])
        if f in skipped_files:
            status, got = "SKIPPED", ""
        else:
            try:
                got = PN.value(R, r["source_key"], r["format"])
            except Exception as e:      # noqa: BLE001
                got = "error: %s" % e
            status = "PASS" if got == r["paper_value"] else "FAIL"
            if status == "PASS" and f in PN.LOGGED:
                status = "LOGGED"
        counts[status] += 1
        lines.append((r["id"], status, r["where"], r["what"], r["paper_value"], got, r["judge"], r["source_key"]))
        print("%-7s %s  %-9s paper %-14s recomputed %-14s %s" % (status, r["id"], r["where"], r["paper_value"], got, r["what"]))
    print("[6/7] numbers compared", flush=True)

    files = ["replay.json", "letta_empty_answers.json"] + ([] if fast else ["predicted_question_types.json"])
    leaf = {n: compare_json(n) for n in files}
    exp_dec = os.path.join(EXP, "decisions.csv")
    got_rows = set(map(tuple, csv.reader(open(os.path.join(RES, "decisions.csv")))))
    exp_rows = set(map(tuple, csv.reader(open(exp_dec))))
    if fast:
        exp_rows = {r for r in exp_rows if "|predicted|" not in r[0] or not r[0].startswith("stream|")}
    dec_diff = len(got_rows ^ exp_rows)
    derived = derived_checks(R)
    ok = counts["FAIL"] == 0 and all(not d for _, d in leaf.values()) and dec_diff == 0 and all(d[3] for d in derived)

    with open(os.path.join(RES, "report.md"), "w") as f:
        f.write("# Reproduction report\n\n")
        f.write("Result: **%s**. Run time %.0f s%s.\n\n" % ("ALL PASS" if ok else "FAILURES", time.time() - t0, " (--fast)" if fast else ""))
        f.write("| check | result |\n|---|---|\n")
        f.write("| numbers of the paper recomputed and equal | %d PASS, %d FAIL |\n" % (counts["PASS"], counts["FAIL"]))
        f.write("| numbers read from logged measurements (RQ3 timings, judge tokens) | %d LOGGED |\n" % counts["LOGGED"])
        if counts["SKIPPED"]:
            f.write("| numbers skipped (--fast) | %d SKIPPED |\n" % counts["SKIPPED"])
        for n, (total, d) in leaf.items():
            f.write("| every value of results/%s equals results/expected/%s | %d of %d equal |\n" % (n, n, total - len(d), total))
        f.write("| results/decisions.csv equals results/expected/decisions.csv | %s |\n" % ("yes" if dec_diff == 0 else "%d rows differ" % dec_diff))
        for n, a, b, good in derived:
            f.write("| logged %s = recomputed | %s (%.6g, %.6g) |\n" % (n, "yes" if good else "NO", a, b))
        f.write("\n## Every number\n\n| id | status | where | what | paper | recomputed | judge | source key |\n|---|---|---|---|---|---|---|---|\n")
        for l in lines:
            f.write("| " + " | ".join(str(c).replace("|", "\\|") for c in l) + " |\n")
    print("[7/7] results/report.md written")
    print("\nnumbers: %(PASS)d PASS, %(FAIL)d FAIL, %(LOGGED)d LOGGED, %(SKIPPED)d SKIPPED" % counts)
    for n, (total, d) in leaf.items():
        print("%s: %d of %d values equal to results/expected" % (n, total - len(d), total), *(d[:5]))
    print("decisions.csv: %s" % ("equal to expected" if dec_diff == 0 else "%d rows differ" % dec_diff))
    print("derived checks: %d of %d ok" % (sum(d[3] for d in derived), len(derived)))
    print("RESULT: %s (%.0f s)" % ("ALL PASS" if ok else "FAILURES", time.time() - t0))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
