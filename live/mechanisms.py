"""Iterative search and planned read, the two mechanisms that use the model to choose what to read.

  python live/mechanisms.py --cond agent_both --model <model> [--conversations 0,1,2] [--workers 4]

tools over one conversation's memory types:
  search_turns(query)        raw turns (bge-small cosine, top 8, with date and session id)
  search_facts(query)        extracted facts (with AGENT_FACTS=own: the store of live/store.py, top 8)
  list_sessions()            session ids with their dates and opening line
  open_session(session_id)   the whole session transcript (truncated to 8,000 characters)
  answer(text)               final answer, ends the run
AGENT_MODE selects the mechanism: prune = iterative search with context pruning (earlier tool results cut to their first
AGENT_PRUNE_HEAD = 300 characters; the paper's iterative search); full = iterative search without pruning (the paper's
reference row); plan = planned read (one planning call, the reads without the model, one answer call with the shared
prompt). The paper uses --cond agent_both, AGENT_FACTS=own, at most 8 steps, temperature 0.
With AGENT_FACTS unset, search_facts reads a Mem0 store instead of our fact store; the paper sets AGENT_FACTS=own.
output: runtime/<out>/<cond>/predictions.jsonl, one row per question with the answer, memory tokens, steps and tool trace.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # package root
sys.path.insert(0, str(HERE))
DATA = Path(os.environ["AGENT_DATA"]) if os.environ.get("AGENT_DATA") else ROOT / "data" / "benchmarks" / "locomo10.json"   # LME(S*) via AGENT_DATA
SKIP_CAT = int(os.environ.get("AGENT_SKIP_CAT", "5"))
RUNTIME = ROOT / "runtime"
ENV = ROOT / ".env"                                       # optional, see store.py
ORDER = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"]
MEM0_PROJECT = os.environ.get("AGENT_MEM0_PROJECT", "mem0_store")   # the Mem0 stores that search_facts reads when AGENT_FACTS is unset
MEM0_PRED_DIR = RUNTIME / os.environ.get("AGENT_MEM0_PRED_DIR", "mem0/predicted")   # where the harness wrote _ingestion_*.json (user ids)
OWN_FACTS = os.environ.get("AGENT_FACTS") == "own"   # search_facts reads OUR fact store (own_store.py) instead of Mem0's
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


class Conversation:
    """one LoCoMo conversation with its three memory forms."""
    _emb = None

    def __init__(self, idx: int, sample: dict, embedder):
        self.idx, self.sid = idx, sample["sample_id"]
        conv = sample["conversation"]
        self.a, self.b = conv["speaker_a"], conv["speaker_b"]
        self.sessions = {}
        self.turns = []
        for k in sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1])):
            date = conv.get(f"{k}_date_time", "")
            lines = []
            for t in conv[k]:
                s = f"{t.get('speaker', '')}: {t.get('text', '')}" + (f" [Image: {t['blip_caption']}]" if t.get("img_url") and t.get("blip_caption") else "")
                lines.append(s)
                self.turns.append({"session": k, "date": date, "text": s})
            self.sessions[k] = {"date": date, "text": "\n".join(lines)}
        cache = RUNTIME / "agent" / "index" / f"{self.sid}.npy"   # the turn index is model-independent: shared
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            self.E = np.load(cache)
        else:
            self.E = embedder.encode([t["text"] for t in self.turns], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
            np.save(cache, self.E)
        self.embedder = embedder
        self._mem0 = None
        self._own = None
        self._lock = threading.Lock()   # the local Qdrant store is not thread safe
        self.user_id = None
        for f in MEM0_PRED_DIR.glob("_ingestion_*.json"):
            d = json.loads(f.read_text())
            if int(d.get("conversation_idx", -1)) == idx:
                self.user_id = d["user_id"]

    def mem0(self):
        if self._mem0 is None:
            from mem0 import Memory
            store = RUNTIME / "mem0_stores" / MEM0_PROJECT / self.user_id
            cfg = {"version": "v1.1",
                   "vector_store": {"provider": "qdrant", "config": {"path": str(store / "qdrant"), "on_disk": True, "collection_name": "mem", "embedding_model_dims": 384}},
                   "llm": {"provider": "openai", "config": {"api_key": "unused", "openai_base_url": os.environ.get("MEMSWAP_BASE_URL", ""), "model": "deepseek"}},
                   "embedder": {"provider": "huggingface", "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384}},
                   "history_db_path": str(store / "history.db")}
            os.environ.setdefault("MEM0_TELEMETRY", "False")
            self._mem0 = Memory.from_config(cfg)
        return self._mem0

    # ---- tools ----
    def search_turns(self, query: str) -> str:
        v = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        top = np.argsort(-(self.E @ v))[:8]
        return "\n".join(f"[{self.turns[i]['session']} | {self.turns[i]['date']}] {self.turns[i]['text'][:400]}" for i in top)

    def search_facts(self, query: str) -> str:
        if OWN_FACTS:
            if self._own is None:
                from store import OwnFacts
                self._own = OwnFacts(self.sid, self.embedder)
            hits = self._own.search(query, 8)
            return "\n".join(f"[{h['date']}] {h['fact']}" for h in hits) if hits else "(no facts found)"
        try:
            with self._lock:
                res = self.mem0().search(query=query, filters={"user_id": self.user_id}, top_k=8)
            items = res.get("results", res) if isinstance(res, dict) else res
        except Exception as e:  # noqa: BLE001
            return f"(fact search failed: {str(e)[:80]})"
        out = []
        for r in items or []:
            md = r.get("metadata") or {}
            date = (md.get("session_date") or "")[:10]
            out.append(f"[{date}] {r.get('memory', '')}")
        return "\n".join(out) if out else "(no facts found)"

    def list_sessions(self) -> str:
        return "\n".join(f"{k}: {v['date']} | {v['text'].splitlines()[0][:80] if v['text'] else ''}" for k, v in self.sessions.items())

    def open_session(self, session_id: str) -> str:
        s = self.sessions.get(session_id.strip())
        if not s:
            return f"(no session {session_id}; use list_sessions)"
        txt = f"Session {session_id} on {s['date']}:\n{s['text']}"
        return txt[:8000] + ("\n... (truncated)" if len(txt) > 8000 else "")


def run_question(client, model, conv: Conversation, question: str, tools: list[str], max_steps: int) -> dict:
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


PLAN_SYSTEM = """You plan how to answer one question about a long conversation between {a} and {b}, with the memory tools below, in one shot.
Tools: search_facts(query) returns the best matching extracted facts, each with its date; search_turns(query) returns the best matching raw turns, each with its date;
open_session(id) returns the full text of one session. The sessions are listed with their ids and dates.
Return a JSON object: {{"facts_query": "...", "turns_query": "...", "open_sessions": ["id", ...]}}. Give at most two session ids; give none if the searches should be enough.
Write queries that name the people, things and dates the question is about."""


def run_plan(client, model, conv: Conversation, question: str, tools: list[str]) -> dict:
    """planned read: one planning call picks the searches and the sessions; the tools run without the model; one answer call with the shared prompt."""
    from store import ANSWER_V2
    t0 = time.time(); in_tok = out_tok = 0; trace = []
    sessions_txt = conv.list_sessions() if "list_sessions" in tools else ""
    msgs = [{"role": "system", "content": PLAN_SYSTEM.format(a=conv.a, b=conv.b)}, {"role": "user", "content": f"Sessions:\n{sessions_txt}\n\nQuestion: {question}"}]
    plan = {}
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=model, temperature=0, messages=msgs, max_tokens=300, response_format={"type": "json_object"}, extra_body={"reasoning_effort": "none"})
            in_tok += r.usage.prompt_tokens or 0; out_tok += r.usage.completion_tokens or 0
            mm = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S); plan = json.loads(mm.group(0)) if mm else {}; break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return {"pred": "", "error": str(e)[:200], "trace": trace, "steps": 1, "in": in_tok, "out": out_tok, "seconds": round(time.time() - t0, 1)}
            time.sleep(3 * (attempt + 1))
    parts = []
    if "search_facts" in tools:
        q = str(plan.get("facts_query") or question); out = conv.search_facts(q); trace.append({"tool": "search_facts", "args": {"query": q}, "result_head": out[:120]}); parts.append("Facts:\n" + out)
    if "search_turns" in tools:
        q = str(plan.get("turns_query") or question); out = conv.search_turns(q); trace.append({"tool": "search_turns", "args": {"query": q}, "result_head": out[:120]}); parts.append("Raw turns:\n" + out)
    for sid in (plan.get("open_sessions") or [])[:2]:
        if "open_session" in tools:
            out = conv.open_session(str(sid))[:4000]; trace.append({"tool": "open_session", "args": {"session_id": str(sid)}, "result_head": out[:120]}); parts.append(out)
    last = max((v["date"] for v in conv.sessions.values()), default="")
    prompt = ANSWER_V2.format(a=conv.a, b=conv.b, last_date=last, memories="\n\n".join(parts), question=question)
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=model, temperature=0, seed=0, max_tokens=900, messages=[{"role": "user", "content": prompt}])
            in_tok += r.usage.prompt_tokens or 0; out_tok += r.usage.completion_tokens or 0
            raw = (r.choices[0].message.content or "").strip(); pred = raw.rsplit("ANSWER:", 1)[1].strip() if "ANSWER:" in raw else raw.splitlines()[-1].strip()
            return {"pred": pred, "answered_by": "plan", "trace": trace, "steps": 2, "in": in_tok, "out": out_tok, "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return {"pred": "", "error": str(e)[:200], "trace": trace, "steps": 2, "in": in_tok, "out": out_tok, "seconds": round(time.time() - t0, 1)}
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(COND_TOOLS))
    ap.add_argument("--conversations", default="0,1,2,3,4,5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--qids", default=None)
    ap.add_argument("--model", default="deepseek-v4.1-flash")
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
        convs[ci] = Conversation(ci, data[ci], embedder)
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
        r = run_plan(client, a.model, convs[ci], qa["question"], tools) if MODE == "plan" else run_question(client, a.model, convs[ci], qa["question"], tools, a.max_steps)
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
