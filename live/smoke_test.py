"""Smoke test of the live wiring with the real mechanisms and a real model. It makes model calls.

  MEMSWAP_BASE_URL=... MEMSWAP_API_KEY=... python live/smoke_test.py [--model deepseek-v4.1-flash]

It uses one LoCoMo conversation (conv-30, the smallest: 19 sessions) and five of its questions:
  1. builds the memory store of that conversation (dated facts, turn index) with live/store.py;
  2. profiles a labelled batch of three questions, one per question type (temporal, single-hop, multi-hop): each of
     the three reading mechanisms answers each question, and the judge scores each answer (9 answers, 9 verdicts);
  3. writes the Routing Table, and answers two further questions with the mechanism that it gives their type.
The question types are the benchmark labels of data/answers.csv. The test checks that every call returns an answer
and a token count, that the judge scores every profiled answer, and that each served question is answered by the
mechanism of its type. It does not measure accuracy: five questions say nothing about it. The result is written to
live/smoke_test_result.json. LoCoMo is downloaded first by live/download_locomo.py (CC BY-NC 4.0).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCOMO = ROOT / "data" / "benchmarks" / "locomo10.json"
CONV, SID = 1, "conv-30"
os.environ.update({"OWN_DATA": str(LOCOMO), "AGENT_DATA": str(LOCOMO), "OWN_STORE_DIR": "smoke_store_conv30",
                   "OWN_SKIP_CAT": "5", "AGENT_SKIP_CAT": "5", "AGENT_FACTS": "own", "AGENT_MODE": "prune",
                   "TOKENIZERS_PARALLELISM": "false", "MEM0_TELEMETRY": "False"})
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "live"))

from memswap.knowledge.knowledge_base import AdaptationGoals   # noqa: E402
from memswap.live import LiveMemSwap, Question                  # noqa: E402


def pick_questions():
    """the first labelled question of each of three types, and the next one of two of them (benchmark labels)."""
    lab = {r["question_id"]: r["question_type"] for r in csv.DictReader(open(ROOT / "data" / "answers.csv"))
           if r["dataset"] == "locomo" and r["system"] == "iterative_search" and r["conversation_id"] == SID}
    qa = json.load(open(LOCOMO))[CONV]["qa"]
    by_type = {}
    for qi, x in enumerate(qa):
        qid = f"{SID}_q{qi}"
        if qid in lab:
            by_type.setdefault(lab[qid], []).append(Question(qid, x["question"], lab[qid], str(x.get("answer", "")), {"qi": qi}))
    batch = [by_type[t][0] for t in ("temporal", "single-hop", "multi-hop")]
    served = [by_type["temporal"][1], by_type["single-hop"][1]]
    return batch, served


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="deepseek-v4.1-flash"); a = ap.parse_args()
    if not LOCOMO.exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "live" / "download_locomo.py")], check=True)
    import store
    t0 = time.time()
    store.extract_sessions([CONV], a.model, 6)
    store.index()
    import mechanisms as MX
    import judge as JD
    import tiktoken
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    client = OpenAI(api_key=os.environ["MEMSWAP_API_KEY"], base_url=os.environ["MEMSWAP_BASE_URL"], timeout=300, max_retries=2)
    data = json.load(open(LOCOMO)); sample = data[CONV]; conv_meta = sample["conversation"]
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    facts = store.OwnFacts(SID, emb)
    last = max(date for _, date, *_ in store.turns(sample))
    enc = tiktoken.get_encoding("o200k_base")
    conv = MX.Conversation(CONV, sample, emb)
    tools = MX.COND_TOOLS["agent_both"]
    calls = []

    def single_lookup(q):          # as store.onepass, for one question: facts up to 6,000 tokens, one answer call
        lines, used = [], 0
        for h in facts.search(q.text, 200):
            line = f"[{h['date']}] {h['fact']}"
            n = len(enc.encode(line))
            if used + n > 6000:
                break
            lines.append(line); used += n
        prompt = store.ANSWER_V2.format(a=conv_meta["speaker_a"], b=conv_meta["speaker_b"], last_date=last,
                                        memories="\n".join(sorted(lines)), question=q.text)
        r = client.chat.completions.create(model=a.model, temperature=0, seed=0, max_tokens=900, messages=[{"role": "user", "content": prompt}])
        raw = (r.choices[0].message.content or "").strip()
        pred = raw.rsplit("ANSWER:", 1)[1].strip() if "ANSWER:" in raw else raw.splitlines()[-1].strip()
        return pred, r.usage.prompt_tokens

    def planned_read(q):
        r = MX.run_plan(client, a.model, conv, q.text, tools)
        return r.get("pred", ""), r.get("in") or 0

    def iterative_search(q):       # AGENT_MODE=prune: context pruning, at most eight steps
        r = MX.run_question(client, a.model, conv, q.text, tools, 8)
        return r.get("pred", ""), r.get("in") or 0

    def logged(name, f):
        def call(q):
            t = time.time(); pred, tok = f(q)
            calls.append({"question_id": q.id, "question_type": q.qtype, "mechanism": name, "answer": pred,
                          "memory_tokens": tok, "seconds": round(time.time() - t, 2)})
            return pred, tok
        return call

    mechs = {"single_lookup": logged("single_lookup", single_lookup), "planned_read": logged("planned_read", planned_read),
             "iterative_search": logged("iterative_search", iterative_search)}
    verdicts = []

    def judge(q, answer):
        v = JD.call(client, a.model, [{"id": "x", "question": q.text, "gold": q.gold, "pred": answer}], 2000).get("x", "")
        verdicts.append({"question_id": q.id, "answer": answer, "verdict": v})
        return int(v == "CORRECT")

    batch, served = pick_questions()
    live = LiveMemSwap(mechs, judge, lambda q: q.qtype, AdaptationGoals(margin=0.01))
    S, TOK, _ = live.labelled_batch(batch)
    table, default = live.routing_table
    out_served = []
    for q in served:
        rec = live.ask(q)
        out_served.append({"question_id": q.id, "question_type": q.qtype, "mechanism": rec["mechanism"], "known": rec["known"],
                           "answer": rec["outcome"], "memory_tokens": rec["tokens"], "routing_table_says": table.get(q.qtype, default)})

    checks = {
        "store built (facts indexed)": len(facts.facts) > 0,
        "every profiling call returned an answer and a token count": len(calls) >= 9 and all(c["answer"] and c["memory_tokens"] > 0 for c in calls[:9]),
        "the judge scored every profiled answer": len(verdicts) == 9 and all(v["verdict"] in ("CORRECT", "WRONG") for v in verdicts),
        "the Routing Table has every profiled question type": set(table) == {q.qtype for q in batch},
        "each served question used the mechanism of its type": all(s["known"] and s["mechanism"] == s["routing_table_says"] and s["memory_tokens"] > 0 for s in out_served),
    }
    res = {"model": a.model, "conversation": SID, "facts_in_store": len(facts.facts), "seconds": round(time.time() - t0, 1),
           "labelled_batch": [{"question_id": q.id, "question_type": q.qtype, "question": q.text, "gold": q.gold} for q in batch],
           "profiling_calls": calls[:9], "verdicts": verdicts,
           "profiled_accuracy": {q.id: {m: int(S.at[q.id, m]) for m in S.columns} for q in batch},
           "profiled_tokens": {q.id: {m: int(TOK.at[q.id, m]) for m in TOK.columns} for q in batch},
           "routing_table": table, "default_mechanism": default, "served": out_served, "checks": checks,
           "result": "PASS" if all(checks.values()) else "FAIL"}
    (ROOT / "live" / "smoke_test_result.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print("RESULT:", res["result"], f"({res['seconds']} s)")
    return 0 if res["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
