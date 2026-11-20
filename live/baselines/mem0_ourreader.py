"""Mem0's store, OUR generic reader. The exact top-200 memories the Mem0 harness handed its own answerer
(runtime/mem0_dated/predicted_mem0_dated/conv*_q*.json, retrieval.search_results, dated by created_at = session date),
formatted "[date] memory" in date order, answered with store.ANSWER_V2 (the same prompt as our one-search reader).
Isolates the store from the benchmark-specific answer prompt. system = mem0_top200_ourreader; rows runtime/mem0_ourreader/predictions.jsonl"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from store import ANSWER_V2, DATA, client, turns  # noqa: E402

PRED = HERE / "runtime" / "mem0_dated" / "predicted_mem0_dated"
OUT = HERE / "runtime" / "mem0_ourreader"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    model = os.environ.get("MEMSWAP_MODEL", "deepseek-v4.1-flash"); workers = int(os.environ.get("WORKERS", "16"))
    data = json.load(open(DATA)); order = [s["sample_id"] for s in data]
    last = {s["sample_id"]: max(date for _, date, *_ in turns(s)) for s in data}
    speakers = {s["sample_id"]: (s["conversation"]["speaker_a"], s["conversation"]["speaker_b"]) for s in data}
    out = OUT / "predictions.jsonl"
    done = {json.loads(l)["qa_id"] for l in open(out)} if out.exists() else set()
    jobs = []
    for p in sorted(glob.glob(str(PRED / "conv*_q*.json"))):
        d = json.load(open(p)); ci, qi = d["question_id"][4:].split("_q"); sid = order[int(ci)]; qid = f"{sid}_q{qi}"
        if qid in done or d.get("category") == 5 or not d.get("cutoff_results"):
            continue
        n = (d["cutoff_results"].get("top_200") or {}).get("memories_evaluated") or 0
        mems = (d.get("retrieval") or {}).get("search_results") or []
        jobs.append((qid, sid, d, mems[:n] if n else mems))
    print(f"{len(jobs)} questions ({len(done)} done)", flush=True)
    c = client(); lock = threading.Lock()

    def one(job):
        qid, sid, d, mems = job; a, b = speakers[sid]
        lines = sorted(f"[{(m.get('created_at') or '')[:10]}] {m.get('memory') or ''}" for m in mems)
        prompt = ANSWER_V2.format(a=a, b=b, last_date=last[sid], memories="\n".join(lines), question=d["question"])
        t = time.time(); pred, err, usage, raw = "", None, None, ""
        for attempt in range(4):
            try:
                r = c.chat.completions.create(model=model, temperature=0, seed=0, max_tokens=900, messages=[{"role": "user", "content": prompt}])
                raw = (r.choices[0].message.content or "").strip(); usage = r.usage; err = None
                pred = raw.rsplit("ANSWER:", 1)[1].strip() if "ANSWER:" in raw else raw.splitlines()[-1].strip(); break
            except Exception as e:  # noqa: BLE001
                err = str(e)[:120]; time.sleep(3 * (attempt + 1))
        row = {"qa_id": qid, "sample_id": sid, "category": d.get("category"), "system": "mem0_top200_ourreader", "question": d["question"], "gold": str(d.get("ground_truth_answer", "")),
               "pred": pred, "memories": len(lines), "in": getattr(usage, "prompt_tokens", None), "out": getattr(usage, "completion_tokens", None), "seconds": round(time.time() - t, 2), "error": err}
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(one, jobs), 1):
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s last: {row['qa_id']} pred={row['pred'][:40]!r}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
