"""Reading mechanism m3 of the memory manager (managed system): iterative search. A tool loop over the memory store, with context pruning
(AGENT_MODE=prune). Also holds the benchmark harness (main) that ran planned read and iterative search on every
question. Live code, not run by replay.py.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memswap.managed.memory_manager.mechanisms.planned_read import planned_read  # noqa: F401  (was a function of this module)
from memswap.managed.memory_store import RUNTIME, MemoryStore  # noqa: F401  (was defined in this module)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent                       # package root (this file lives in memswap/managed/memory_manager/mechanisms/)

DATA = Path(os.environ["AGENT_DATA"]) if os.environ.get("AGENT_DATA") else ROOT / "data" / "benchmarks" / "locomo10.json"   # LME(S*) via AGENT_DATA
SKIP_CAT = int(os.environ.get("AGENT_SKIP_CAT", "5"))

ENV = ROOT / ".env"                                       # optional, see store.py
ORDER = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]

MODE = os.environ.get("AGENT_MODE", "full")   # full = re-send every tool result at every step; prune = keep only the latest step's tool results in full,
PRUNE_HEAD = int(os.environ.get("AGENT_PRUNE_HEAD", "300"))   # earlier ones cut to this many characters; plan = one planning call, tools run without the model, one answer call

SYSTEM = """You are answering questions about a long conversation between {a} and {b} using memory tools.
Memory of the conversation is available in two forms: raw turns (search_turns: exact messages with their dates) and
extracted facts (search_facts: short statements the memory system distilled, with the date of the session they came from).
You can also list the sessions (list_sessions) and open a whole session (open_session) to read it in full.
Search first; use several searches if needed; open a session when the exact wording or the surrounding context matters.
Note the dates: every turn and fact carries the date of its session. If the question asks when something happened,
resolve relative expressions ("yesterday", "last week") against the session date and answer with an absolute date.
If memories contradict each other, prefer the most recent. Do not make up information; if the memory does not
contain the answer, answer "not mentioned".
Finish by calling answer with a precise, concise answer (a few words, e.g. a date, a name, a short phrase)."""


TOOLS = {
    "search_turns": {"type": "function", "function": {"name": "search_turns", "description": "Search the raw conversation turns (exact messages with dates). Returns the 8 most similar turns.",
                                                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    "search_facts": {"type": "function", "function": {"name": "search_facts", "description": "Search the extracted facts the memory system distilled from the conversation (each with the date of its session). Returns the 8 best matches.",
                                                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    "list_sessions": {"type": "function", "function": {"name": "list_sessions", "description": "List the conversation's sessions: id, date and first line.",
                                                         "parameters": {"type": "object", "properties": {}}}},
    "open_session": {"type": "function", "function": {"name": "open_session", "description": "Read one whole session transcript by its id (e.g. 'session_3').",
                                                        "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}, "required": ["session_id"]}}},
    "answer": {"type": "function", "function": {"name": "answer", "description": "Give the final answer and stop.",
                                                  "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
}
COND_TOOLS = {"agent_both": ["search_turns", "search_facts", "list_sessions", "open_session", "answer"],
              "agent_turns": ["search_turns", "list_sessions", "open_session", "answer"],
              "agent_facts": ["search_facts", "list_sessions", "open_session", "answer"],
              "agent_sessions": ["list_sessions", "open_session", "answer"]}

def load_env():
    if not ENV.exists():
        return
    for line in open(ENV):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def iterative_search(client, model, conv: MemoryStore, question: str, tools: list[str], max_steps: int) -> dict:
    msgs = [{"role": "system", "content": SYSTEM.format(a=conv.a, b=conv.b)}, {"role": "user", "content": f"Question: {question}"}]
    spec = [TOOLS[t] for t in tools]
    trace, in_tok, out_tok, t0 = [], 0, 0, time.time()
    for step in range(max_steps + 1):
        force = step == max_steps
        if MODE == "prune" and step > 0:   # shrink every tool result except those of the latest step before re-sending the context
            last_assistant = max((i for i, mm in enumerate(msgs) if mm["role"] == "assistant" and mm.get("tool_calls")), default=-1)
            for i, mm in enumerate(msgs):
                if mm["role"] == "tool" and i < last_assistant and len(mm["content"]) > PRUNE_HEAD and not mm["content"].endswith("(earlier result, shortened)"):
                    mm["content"] = mm["content"][:PRUNE_HEAD] + "\n... (earlier result, shortened)"
        kw = dict(model=model, temperature=0, messages=msgs, max_tokens=700, extra_body={"think": False} if model.startswith("glm") else {"reasoning_effort": "none"})
        if not force:
            kw["tools"] = spec
            kw["tool_choice"] = "required" if step == 0 else "auto"
        else:
            msgs.append({"role": "user", "content": "You must answer now with a precise, concise answer (a few words)."})
        for attempt in range(4):
            try:
                r = client.chat.completions.create(**kw)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    return {"pred": "", "error": str(e)[:200], "trace": trace, "steps": step, "in": in_tok, "out": out_tok, "seconds": round(time.time() - t0, 1)}
                time.sleep(3 * (attempt + 1))
        in_tok += r.usage.prompt_tokens or 0; out_tok += r.usage.completion_tokens or 0
        m = r.choices[0].message
        calls = m.tool_calls or []
        if not calls:
            text = (m.content or "").strip()
            if force or text:
                return {"pred": text, "answered_by": "text", "trace": trace, "steps": step, "in": in_tok, "out": out_tok, "seconds": round(time.time() - t0, 1)}
            msgs.append({"role": "assistant", "content": text or "(continue)"})
            continue
        msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "answer":
                return {"pred": str(args.get("text", "")).strip(), "answered_by": "tool", "trace": trace, "steps": step + 1, "in": in_tok, "out": out_tok, "seconds": round(time.time() - t0, 1)}
            fn = getattr(conv, name, None)
            if not fn or name not in tools:
                out = f"(unknown tool {name})"
            else:
                try:
                    if name == "open_session":          # models vary the argument name; accept any single string value
                        args = {"session_id": str(next(iter(args.values()), ""))} if args else {"session_id": ""}
                    elif name in ("search_turns", "search_facts"):
                        args = {"query": str(args.get("query") or next(iter(args.values()), ""))} if args else {"query": ""}
                    else:
                        args = {}
                    out = fn(**args)
                except Exception as e:  # noqa: BLE001 - a bad tool call is reported to the model, never crashes the run
                    out = f"(tool error: {str(e)[:120]})"
            trace.append({"tool": name, "args": args, "result_head": out[:120]})
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
    return {"pred": "", "trace": trace, "steps": max_steps, "in": in_tok, "out": out_tok, "seconds": round(time.time() - t0, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(COND_TOOLS))
    ap.add_argument("--conversations", default="0,1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--qids", default=None)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--out", default="agent", help="runtime/<out>/<cond>/predictions.jsonl")
    a = ap.parse_args()
    load_env()
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    client = OpenAI(api_key=os.environ["MEMSWAP_API_KEY"], base_url=os.environ["MEMSWAP_BASE_URL"], timeout=300, max_retries=2)
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    data = json.load(open(DATA))
    out_dir = RUNTIME / a.out / a.cond; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "predictions.jsonl"
    done = {json.loads(l)["qa_id"] for l in open(out)} if out.exists() else set()
    keep = set(json.load(open(a.qids))) if a.qids else None
    convs, jobs = {}, []
    for ci in (int(x) for x in a.conversations.split(",")):
        convs[ci] = MemoryStore(ci, data[ci], embedder)
        qs = [(qi, qa) for qi, qa in enumerate(data[ci]["qa"]) if qa.get("category") != SKIP_CAT]
        if a.max_questions:
            qs = qs[: a.max_questions]
        for qi, qa in qs:
            qid = f"{data[ci]['sample_id']}_q{qi}"
            if (keep is None or qid in keep) and qid not in done:
                jobs.append((qid, ci, qa))
    print(f"{a.cond}: {len(jobs)} questions to run ({len(done)} done)", flush=True)
    lock = threading.Lock()
    tools = COND_TOOLS[a.cond]

    def run(job):
        qid, ci, qa = job
        r = planned_read(client, a.model, convs[ci], qa["question"], tools) if MODE == "plan" else iterative_search(client, a.model, convs[ci], qa["question"], tools, a.max_steps)
        row = {"qa_id": qid, "sample_id": data[ci]["sample_id"], "category": qa.get("category"), "system": a.cond, "question": qa["question"],
               "gold": str(qa.get("answer", "")), "pred": r.get("pred", ""), "memory_tokens": r.get("in"), "in": r.get("in"), "out": r.get("out"),
               "seconds": r.get("seconds"), "steps": r.get("steps"), "answered_by": r.get("answered_by"), "error": r.get("error"),
               "tools_used": [t["tool"] for t in r.get("trace", [])], "trace": r.get("trace", [])[:12]}
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(run, jobs), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s last: {row['qa_id']} tools={row['tools_used']} pred={row['pred'][:40]!r}", flush=True)
    print(f"{a.cond} done", flush=True)


if __name__ == "__main__":
    main()
