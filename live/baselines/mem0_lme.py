"""Mem0 on LongMemEval-S* (baseline of Table II). Mem0's own library (mem0 OSS 2.2.1, local Qdrant, dated extraction)
builds one store per conversation from the user and assistant turns, one add() per turn pair with the session
timestamp, as its LoCoMo harness does. Then one search (top 200) per question, answered with the shared answer prompt
(store.ANSWER_V2). Rows: runtime/mem0_lme/predictions.jsonl. Needs MEM0_PROXY_URL (an OpenAI-compatible endpoint)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("MEM0_DATED", "1"); os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("MEM0_PROXY_URL", "http://localhost:8000/v1"); os.environ.setdefault("MEM0_LLM_MODEL", "deepseek-v4.1-flash")
from mem0_local_client import Mem0LocalClient  # noqa: E402
from store import ANSWER_V2, client, session_date  # noqa: E402

DATA = HERE / "runtime" / "lme" / "lme_s_star_locomo.json"
OUT = HERE / "runtime" / "mem0_lme"; OUT.mkdir(parents=True, exist_ok=True)
DONE = OUT / "ingested.jsonl"; PRED = OUT / "predictions.jsonl"


def epoch(loco_date: str) -> int:
    return int(datetime.strptime(loco_date, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc).timestamp())


async def ingest(mem, sample, done):
    sid = sample["sample_id"]; conv = sample["conversation"]
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
    n = 0
    for k in keys:
        ts = epoch(conv[f"{k}_date_time"]); turns = conv[k]
        for i in range(0, len(turns), 2):   # one add() per user/assistant pair, as the harness's session chunking does for LoCoMo
            key = f"{sid}:{k}:{i}"
            if key in done:
                continue
            msgs = [{"role": "user" if t["speaker"] == "user" else "assistant", "content": t["text"]} for t in turns[i:i + 2] if t["text"].strip()]
            if not msgs:
                continue
            await mem.add(msgs, sid, timestamp=ts)
            with open(DONE, "a") as f:
                f.write(json.dumps({"key": key}) + "\n")
            n += 1
    print(f"  {sid}: {n} chunks added", flush=True)


async def main():
    data = json.load(open(DATA)); mem = Mem0LocalClient("lme_mem0")
    done = {json.loads(l)["key"] for l in open(DONE)} if DONE.exists() else set()
    t0 = time.time()
    await asyncio.gather(*(ingest(mem, s, done) for s in data))
    print(f"ingest done {time.time() - t0:.0f}s", flush=True)
    c = client(); model = os.environ.get("MEMSWAP_MODEL", "deepseek-v4.1-flash")
    have = {json.loads(l)["qa_id"] for l in open(PRED)} if PRED.exists() else set()
    for s in data:
        sid = s["sample_id"]; conv = s["conversation"]
        last = max(session_date(conv[k]) for k in conv if k.endswith("_date_time"))
        for qi, qa in enumerate(s["qa"]):
            qid = f"{sid}_q{qi}"
            if qid in have:
                continue
            hits = await mem.search(qa["question"], sid, top_k=200)
            lines = sorted(f"[{(h.get('created_at') or '')[:10]}] {h.get('memory') or ''}" for h in hits)
            prompt = ANSWER_V2.format(a="user", b="assistant", last_date=last, memories="\n".join(lines), question=qa["question"])
            t = time.time(); pred, raw, usage, err = "", "", None, None
            for attempt in range(4):
                try:
                    r = c.chat.completions.create(model=model, temperature=0, seed=0, max_tokens=900, messages=[{"role": "user", "content": prompt}])
                    raw = (r.choices[0].message.content or "").strip(); usage = r.usage
                    pred = raw.rsplit("ANSWER:", 1)[1].strip() if "ANSWER:" in raw else raw.splitlines()[-1].strip(); break
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:120]; await asyncio.sleep(3 * (attempt + 1))
            row = {"qa_id": qid, "sample_id": sid, "category": qa["category"], "system": "mem0_top200_ourreader", "question": qa["question"], "gold": qa["answer"], "pred": pred,
                   "memories": len(lines), "memory_tokens": sum(len(l) // 4 for l in lines), "in": getattr(usage, "prompt_tokens", None), "out": getattr(usage, "completion_tokens", None), "seconds": round(time.time() - t, 2), "error": err}
            with open(PRED, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"  {sid}: answered", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
