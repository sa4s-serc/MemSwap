"""Letta's LoCoMo benchmark agent (letta-ai/letta-leaderboard, leaderboard/locomo/locomo_benchmark.py, "session"
chunking), ported to letta-client 1.x against a Letta 0.16.8 server.

  python letta_locomo.py --conversations 0,1,2,3,4,5 [--max-questions N] [--workers 4]

Kept from Letta: one file per session ("Timestamp: <date>" line, then "speaker: text [Image: caption]" lines),
uploaded to a folder attached to a fresh agent per question; the agent's system prompt (locomo_agent.txt);
tools = file search + answer_question; tool rules = must start with file search, may keep searching,
answer_question ends the run; adversarial questions excluded; the answer is the argument of the last
answer_question call. Changed, as the paper states: their `search_files` tool is `semantic_search_files` in 0.16.8;
the agent type is memgpt_v2_agent; the model is DeepSeek V4.1 Flash (reasoning off) instead of gpt-4o-mini;
embeddings are bge-small-en-v1.5 instead of text-embedding-3-large; sessions are sorted numerically.
Set LETTA_DATA to a benchmark file in LoCoMo shape to run it on LongMemEval-S*. Clone letta-ai/letta-leaderboard
at the commit we used into live/baselines/external/ first:

  git clone https://github.com/letta-ai/letta-leaderboard live/baselines/external/letta-leaderboard
  git -C live/baselines/external/letta-leaderboard checkout 802a794263279839f9384bfd518ca9da3d059d37

Output: runtime/letta/predictions.jsonl."""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from letta_client import Letta

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ["LETTA_DATA"]) if os.environ.get("LETTA_DATA") else HERE / "external" / "letta-leaderboard" / "leaderboard" / "locomo" / "locomo10.json"   # LongMemEval-S* in LoCoMo shape
SYSTEM = (HERE / "external" / "letta-leaderboard" / "leaderboard" / "locomo" / "locomo_agent.txt").read_text()
OUT = HERE / "runtime" / "letta"
MODEL = "openai-proxy/deepseek-v4.1-flash"   # overridden by --model
EMBED = "openai/text-embedding-3-small"       # overridden by --embedding
SEARCH_TOOL = "semantic_search_files"
ANSWER_SRC = '''def answer_question(answer: str) -> str:
    """
    Returns the provided answer to the user.
    Your answer should be as precise as possible. For example, "this month" is not precise, "June 2025" is precise.

    Args:
        answer (str): The answer to return.

    Returns:
        str: The provided answer.
    """
    return answer
'''


def session_files(sample: dict) -> list[tuple[str, str]]:
    conv = sample["conversation"]
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
    out = []
    for k in keys:
        ts = conv.get(f"{k}_date_time", "")
        lines = []
        for t in conv[k]:
            s = f"{t.get('speaker', '')}: {t.get('text', '')}"
            if t.get("img_url") and t.get("blip_caption"):
                s += f" [Image: {t['blip_caption']}]"
            lines.append(s)
        out.append((ts, "\n".join(lines)))
    return out


def write_files(sample: dict) -> list[Path]:
    d = OUT / "files" / f"sample_{sample['sample_id']}"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (ts, content) in enumerate(session_files(sample)):
        name = (ts.replace("/", "-").replace(":", "-") + ".txt") if ts else f"session_{i + 1:03d}.txt"
        p = d / name
        if not p.exists():
            p.write_text((f"Timestamp: {ts}\n" if ts else f"Session {i + 1}\n") + content, encoding="utf-8")
        paths.append(p)
    return sorted(paths)


def ensure_folder(client: Letta, sample: dict, log) -> str:
    reg = OUT / "folders.json"
    folders = json.loads(reg.read_text()) if reg.exists() else {}
    sid = sample["sample_id"]
    if sid in folders:
        return folders[sid]
    conv = sample["conversation"]
    name = f"Conversation Context for {sid}"
    existing = [x for x in client.folders.list(limit=200) if getattr(x, "name", "") == name]
    if existing:   # uploaded and embedded by an earlier run on this server: reuse
        folders[sid] = existing[0].id
        reg.write_text(json.dumps(folders, indent=1))
        log(f"{sid}: reusing folder {existing[0].id}")
        return existing[0].id
    f = client.folders.create(name=name, embedding=EMBED, embedding_chunk_size=100000,
                             description=f"The conversation history, containing timestamped messages from {conv['speaker_a']} and {conv['speaker_b']}")
    paths = write_files(sample)
    for p in paths:
        with open(p, "rb") as fh:
            client.folders.files.upload(folder_id=f.id, file=fh)
    for _ in range(600):
        files = list(client.folders.files.list(folder_id=f.id, limit=500))
        st = [getattr(x, "processing_status", "completed") for x in files]
        if len(files) >= len(paths) and all(s in ("completed", "error") for s in st):
            log(f"{sid}: {len(files)} files, status {dict((s, st.count(s)) for s in set(st))}")
            break
        time.sleep(2)
    folders[sid] = f.id
    reg.write_text(json.dumps(folders, indent=1))
    return f.id


def ask(client: Letta, folder_id: str, question: str, max_steps: int) -> dict:
    agent = client.agents.create(
        agent_type="memgpt_v2_agent", model=MODEL, embedding=EMBED, system=SYSTEM,
        include_base_tools=False, include_base_tool_rules=False,
        tools=[SEARCH_TOOL, "answer_question"], folder_ids=[folder_id],
        tool_rules=[{"type": "exit_loop", "tool_name": "answer_question"},
                    {"type": "continue_loop", "tool_name": SEARCH_TOOL},
                    {"type": "continue_loop", "tool_name": "grep_files"},
                    {"type": "run_first", "tool_name": SEARCH_TOOL}],
    )
    t0 = time.time()
    try:
        r = client.agents.messages.create(agent_id=agent.id, messages=[{"role": "user", "content": question}], max_steps=max_steps)
        msgs = list(r.messages)
        pred, calls, last_assistant = "", 0, ""
        for m in msgs:
            mt = getattr(m, "message_type", "")
            if mt == "tool_call_message":
                calls += 1
            if mt == "assistant_message":
                last_assistant = getattr(m, "content", "") or last_assistant
        for m in reversed(msgs):
            if getattr(m, "message_type", "") == "tool_return_message" and getattr(m, "name", "") == "answer_question":
                pred = getattr(m, "tool_return", "") or ""
                break
        if not pred:   # their extract returns "" here; keep the raw text as a fallback, flagged
            pred = last_assistant
        u = getattr(r, "usage", None)
        return {"pred": pred, "answered_by_tool": bool(pred) and pred == (pred), "steps": calls,
                "in": getattr(u, "prompt_tokens", None), "out": getattr(u, "completion_tokens", None), "seconds": round(time.time() - t0, 2),
                "trace": [{"type": getattr(m, "message_type", ""), "name": getattr(m, "name", None),
                           "text": (getattr(m, "content", None) or getattr(m, "tool_return", None) or str(getattr(m, "tool_call", "")))[:300]} for m in msgs][:30]}
    finally:
        try:
            client.agents.delete(agent.id)
        except Exception:  # noqa: BLE001
            pass


def main():
    global OUT, MODEL, EMBED
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", default="0,1,2,3,4,5")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--base-url", default="http://127.0.0.1:8283")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--embedding", default=EMBED)
    ap.add_argument("--out", default="letta", help="subdir of runtime/")
    ap.add_argument("--qids", default=None, help="json list of qa_ids to run")
    a = ap.parse_args()
    OUT = HERE / "runtime" / a.out; MODEL = a.model; EMBED = a.embedding
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "predictions.jsonl"
    done = {json.loads(l)["qa_id"] for l in open(out)} if out.exists() else set()
    lock = threading.Lock()
    log = lambda s: print(s, flush=True)  # noqa: E731
    client = Letta(base_url=a.base_url, timeout=600.0)
    client.tools.upsert(source_code=ANSWER_SRC)
    data = json.load(open(DATA))
    jobs = []
    for ci in (int(x) for x in a.conversations.split(",")):
        sample = data[ci]
        folder = ensure_folder(client, sample, log)
        qs = [(qi, qa) for qi, qa in enumerate(sample["qa"]) if qa.get("category") != (-1 if os.environ.get("LETTA_DATA") else 5)]   # LoCoMo: skip the adversarial type; LME: keep every type
        if a.max_questions:
            qs = qs[: a.max_questions]
        keep = set(json.load(open(a.qids))) if a.qids else None
        for qi, qa in qs:
            qid = f"{sample['sample_id']}_q{qi}"
            if keep is not None and qid not in keep:
                continue
            if qid not in done:
                jobs.append((qid, sample["sample_id"], folder, qi, qa))
    log(f"{len(jobs)} questions to run ({len(done)} done)")

    def run(job):
        qid, sid, folder, qi, qa = job
        try:
            r = ask(client, folder, qa["question"], a.max_steps)
        except Exception as e:  # noqa: BLE001
            r = {"pred": "", "error": str(e)[:300], "steps": 0, "in": None, "out": None, "seconds": 0, "trace": []}
        row = {"qa_id": qid, "sample_id": sid, "category": qa.get("category"), "system": "letta_filesystem",
               "question": qa["question"], "gold": str(qa.get("answer", "")), "pred": r["pred"], "memory_tokens": r.get("in"),
               "in": r.get("in"), "out": r.get("out"), "seconds": r.get("seconds"), "steps": r.get("steps"), "error": r.get("error"), "trace": r.get("trace")}
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(run, jobs), 1):
            if i % 25 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} {time.time() - t0:.0f}s last: {row['qa_id']} steps={row['steps']} pred={row['pred'][:40]!r}")
    log("letta done")


if __name__ == "__main__":
    main()
