"""Reading mechanism m1 of the memory manager (managed system): single lookup. One similarity search over the dated facts, the best facts under a
6,000-token budget, one answer call. Live code, not run by replay.py. The store helpers it uses are imported from
memswap/managed/memory_store.py.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from memswap.managed.memory_store import ANSWER_V2, DATA, OUT, SKIP_CAT, DatedFacts, client, turns

def single_lookup(convs: list[int], model: str, budget: int, k: int, workers: int, reader: str = "v2"):
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    data = json.load(open(DATA)); c = client()
    odir = OUT / f"onepass_{reader}"; odir.mkdir(parents=True, exist_ok=True); out = odir / "predictions.jsonl"
    done = {json.loads(l)["qa_id"] for l in open(out)} if out.exists() else set()
    lock = threading.Lock(); stores = {}
    jobs = []; last = {}
    for ci in convs:
        s = data[ci]; sid = s["sample_id"]
        stores[sid] = DatedFacts(sid); last[sid] = max(date for _, date, *_ in turns(s))
        for qi, qa in enumerate(s["qa"]):
            if qa.get("category") == SKIP_CAT:
                continue
            qid = f"{sid}_q{qi}"
            if qid not in done:
                jobs.append((qid, sid, qi, qa, s["conversation"]["speaker_a"], s["conversation"]["speaker_b"]))
    print(f"{len(jobs)} questions ({len(done)} done)", flush=True)

    def one(job):
        qid, sid, qi, qa, a, b = job
        hits = stores[sid].search(qa["question"], k)
        lines, used = [], 0
        for h in hits:
            line = f"[{h['date']}] {h['fact']}"
            n = len(enc.encode(line))
            if used + n > budget:
                break
            lines.append(line); used += n
        # date order, reasoning steps, specific answer after ANSWER:
        prompt = ANSWER_V2.format(a=a, b=b, last_date=last[sid], memories="\n".join(sorted(lines)), question=qa["question"]); max_out = 900
        t = time.time(); pred, err, usage, raw = "", None, None, ""
        for attempt in range(4):
            try:
                r = c.chat.completions.create(model=model, temperature=0, seed=0, max_tokens=max_out, messages=[{"role": "user", "content": prompt}])
                raw = (r.choices[0].message.content or "").strip(); usage = r.usage; err = None
                pred = raw.rsplit("ANSWER:", 1)[1].strip() if "ANSWER:" in raw else raw.splitlines()[-1].strip()
                break
            except Exception as e:  # noqa: BLE001
                err = str(e)[:120]; time.sleep(3 * (attempt + 1))
        row = {"qa_id": qid, "sample_id": sid, "category": qa.get("category"), "system": "own_facts_onepass", "question": qa["question"], "gold": str(qa.get("answer", "")),
               "pred": pred, "reasoning": raw, "memory_tokens": used, "memories": len(lines), "in": getattr(usage, "prompt_tokens", None), "out": getattr(usage, "completion_tokens", None),
               "seconds": round(time.time() - t, 2), "error": err}
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(one, jobs), 1):
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s last: {row['qa_id']} pred={row['pred'][:40]!r}", flush=True)
    print("onepass done", flush=True)
