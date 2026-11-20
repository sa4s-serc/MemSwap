"""The memory store of MemSwap and its cheapest mechanism, single lookup.

  extract_sessions  one LLM call per session (the earlier facts as context): standalone facts with absolute dates (JSONL)
  index             bge-small-en-v1.5 embeddings of the facts (local, no tokens)
  onepass           single lookup: one similarity search, the best facts under a 6,000-token budget, one answer call

The paper's configuration (see live/run_mechanisms.sh):
  python live/store.py extract_sessions --conversations 0,...,9
  python live/store.py index
  python live/store.py onepass --reader v2 --budget 6000 --conversations 0,...,9

Endpoint: MEMSWAP_BASE_URL / MEMSWAP_API_KEY (any OpenAI-compatible endpoint), read from
the environment or from an optional .env file at the package root. Benchmark file: OWN_DATA (default
data/benchmarks/locomo10.json); store directory: runtime/<OWN_STORE_DIR>.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # package root
DATA = Path(os.environ["OWN_DATA"]) if os.environ.get("OWN_DATA") else ROOT / "data" / "benchmarks" / "locomo10.json"   # LME(S*) in LoCoMo shape via OWN_DATA
SKIP_CAT = int(os.environ.get("OWN_SKIP_CAT", "5"))   # LoCoMo's adversarial type; set to -1 for LME
OUT = ROOT / "runtime" / os.environ.get("OWN_STORE_DIR", "own_store")
ORDER = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]

EXTRACT_SESSION_SYSTEM = """You extract facts from one dated session of a conversation between two people, for a memory that will answer questions months later.
Write each fact as one standalone sentence that names the person and states the fact with absolute dates: resolve relative time words (yesterday, last weekend, next month, this summer, two years ago) against the session date.
Keep every fact a later question could ask about: events and when they happened, people, places, things, numbers, preferences, plans, feelings and their reasons, what someone shared or showed (images are described in brackets).
Merge what one speaker says across the session into complete facts; do not split one event into fragments and do not write a fact for greetings, thanks, or small talk.
Use the earlier facts only to resolve references (who "she" is, which trip); do not repeat them.
Return a JSON object {"facts": [{"turn": "<id of the turn the fact comes from>", "fact": "..."}]}."""

EXTRACT_SESSION_USER = """Session date: {date}
Speakers: {a} and {b}
Earlier facts about them (context only):
{earlier}

Session transcript (each line: turn id, speaker, message):
{transcript}"""

ANSWER_V2 = """You answer one question about a long conversation between {a} and {b}. Below are memories that a memory system extracted from that conversation, in date order; each line starts with the date of the session it came from. The last session took place on {last_date}.

Work in three short steps, then give the answer.
1. Collect: read every memory and list the ones that bear on the question, with their dates. Details are often spread over several memories, so do not stop at the first match.
2. Resolve: combine the collected memories. Check the memory is about the right person. For a list or a count, enumerate every distinct item. For a date, use the dates in the memories and resolve words like "last week" against the session date; for "how long" or "how long ago", compute from the dates. Prefer the most specific detail (a name, a number, a place) over a general one.
3. Answer: give one direct answer with the specific details in it. Do not say "not mentioned" or "not specified" when any memory bears on the question; answer from the best evidence. Do not invent names, places or dates that no memory states.

Memories:
{memories}

Question: {question}

Write steps 1 to 3 briefly, then the final answer on its own line after "ANSWER:"."""


def load_env():
    env = ROOT / ".env"                                  # optional: MEMSWAP_BASE_URL=..., MEMSWAP_API_KEY=... (any OpenAI-compatible endpoint)
    if not env.exists():
        return
    for line in open(env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def client():
    from openai import OpenAI
    load_env()
    return OpenAI(api_key=os.environ["MEMSWAP_API_KEY"], base_url=os.environ["MEMSWAP_BASE_URL"], timeout=300, max_retries=4)


def session_date(s: str) -> str:
    try:
        return datetime.strptime(s, "%I:%M %p on %d %B, %Y").strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return s


def turns(sample: dict):
    conv = sample["conversation"]
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
    for k in keys:
        date = session_date(conv.get(f"{k}_date_time", ""))
        prev = ""
        for t in conv[k]:
            text = t.get("text", "")
            if t.get("blip_caption"):
                text = f"{text} [shares an image: {t['blip_caption']}]"
            yield k, date, t.get("speaker", ""), t.get("dia_id", ""), text, prev
            prev = f"{t.get('speaker', '')}: {text}"[:400]


def extract_sessions(convs: list[int], model: str, workers: int):
    """one call per session, the previous session's facts as context; one output row per fact, with the turn it comes from."""
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.load(open(DATA)); c = client()
    out = OUT / "facts.jsonl"
    done = {(r["sid"], r["session"]) for r in map(json.loads, open(out)) if not r.get("error")} if out.exists() else set()
    if out.exists():
        out.write_text("".join(l for l in open(out) if not json.loads(l).get("error")))
    lock = threading.Lock()

    def one_conv(ci):
        s = data[ci]; sid = s["sample_id"]; conv = s["conversation"]; a, b = conv["speaker_a"], conv["speaker_b"]
        keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
        earlier = []
        for k in keys:
            date = session_date(conv.get(f"{k}_date_time", ""))
            lines = []
            for t in conv[k]:
                text = t.get("text", "")
                if t.get("blip_caption"):
                    text = f"{text} [shares an image: {t['blip_caption']}]"
                if text.strip():
                    lines.append(f"{t.get('dia_id', '')} | {t.get('speaker', '')}: {text}")
            if (sid, k) in done or not lines:
                continue
            msgs = [{"role": "system", "content": EXTRACT_SESSION_SYSTEM},
                    {"role": "user", "content": EXTRACT_SESSION_USER.format(date=date, a=a, b=b, earlier="\n".join(earlier[-60:]) or "(none)", transcript="\n".join(lines))}]
            facts, err = [], None
            for attempt in range(4):
                try:
                    r = c.chat.completions.create(model=model, temperature=0, seed=0, max_tokens=3000, messages=msgs, response_format={"type": "json_object"})
                    txt = r.choices[0].message.content or ""
                    m = re.search(r"\{.*\}", txt, re.S)
                    facts = [(str(f.get("turn", "")).strip(), str(f.get("fact", "")).strip()) for f in (json.loads(m.group(0)).get("facts") or []) if isinstance(f, dict) and str(f.get("fact", "")).strip()]
                    err = None; break
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:120]; time.sleep(3 * (attempt + 1))
            speakers = {t.get("dia_id", ""): t.get("speaker", "") for t in conv[k]}
            rows = [{"sid": sid, "session": k, "date": date, "speaker": speakers.get(turn, ""), "turn": turn, "facts": [fact], "error": err} for turn, fact in facts] or [{"sid": sid, "session": k, "date": date, "speaker": "", "turn": "", "facts": [], "error": err}]
            with lock:
                with open(out, "a") as f:
                    f.writelines(json.dumps(r) + "\n" for r in rows)
            earlier += [fact for _, fact in facts]
            print(f"  {sid} {k} {date}: {len(facts)} facts{' ERROR ' + err if err else ''}", flush=True)
        return sid
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one_conv, convs))
    print("extract_sessions done", flush=True)


def index():
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    rows = [json.loads(l) for l in open(OUT / "facts.jsonl")]
    by_sid = {}
    for r in rows:
        for f in r["facts"]:
            by_sid.setdefault(r["sid"], []).append({"date": r["date"], "session": r["session"], "speaker": r["speaker"], "turn": r["turn"], "fact": f})
    for sid, facts in by_sid.items():
        E = emb.encode([f["fact"] for f in facts], batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
        np.save(OUT / f"facts_{sid}.npy", E)
        (OUT / f"facts_{sid}.jsonl").write_text("".join(json.dumps(f) + "\n" for f in facts))
        print(f"{sid}: {len(facts)} facts from {sum(1 for r in rows if r['sid'] == sid)} turns", flush=True)
    print("index done", flush=True)


class OwnFacts:
    """search over our fact store for one conversation (used by the one-pass reader and by agent_reader with AGENT_FACTS=own)."""
    def __init__(self, sid: str, embedder=None):
        from sentence_transformers import SentenceTransformer
        self.emb = embedder or SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.E = np.load(OUT / f"facts_{sid}.npy")
        self.facts = [json.loads(l) for l in open(OUT / f"facts_{sid}.jsonl")]

    def search(self, query: str, k: int = 200):
        v = self.emb.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        order = np.argsort(-(self.E @ v)); out, seen = [], set()
        for i in order:   # the same fact can be extracted from two neighbouring turns; return each wording once
            key = self.facts[i]["fact"].strip().lower()
            if key in seen:
                continue
            seen.add(key); out.append(dict(self.facts[i], score=float(self.E[i] @ v)))
            if len(out) >= k:
                break
        return out


def onepass(convs: list[int], model: str, budget: int, k: int, workers: int, reader: str = "v2"):
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    data = json.load(open(DATA)); c = client()
    odir = OUT / f"onepass_{reader}"; odir.mkdir(parents=True, exist_ok=True); out = odir / "predictions.jsonl"
    done = {json.loads(l)["qa_id"] for l in open(out)} if out.exists() else set()
    lock = threading.Lock(); stores = {}
    jobs = []; last = {}
    for ci in convs:
        s = data[ci]; sid = s["sample_id"]
        stores[sid] = OwnFacts(sid); last[sid] = max(date for _, date, *_ in turns(s))
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["extract_sessions", "index", "onepass"])
    ap.add_argument("--conversations", default="0,1,2,3,4,5"); ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--budget", type=int, default=6000); ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--reader", default="v2", choices=["v2"], help="the answer prompt (ANSWER_V2); names the output folder onepass_v2")
    a = ap.parse_args(); convs = [int(x) for x in a.conversations.split(",")]
    if a.cmd == "extract_sessions":
        extract_sessions(convs, a.model, a.workers)
    elif a.cmd == "index":
        index()
    else:
        onepass(convs, a.model, a.budget, a.k, a.workers, a.reader)


if __name__ == "__main__":
    main()
