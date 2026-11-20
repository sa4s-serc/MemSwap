"""Reading mechanism m2 of the memory manager (managed system): planned read. One planning call picks the searches and the sessions; the reads run
without the model; one answer call. Live code, not run by replay.py.
"""
from __future__ import annotations

import json
import re
import time

from memswap.managed.memory_store import MemoryStore  # noqa: F401  (type of `conv`)

PLAN_SYSTEM = """You plan how to answer one question about a long conversation between {a} and {b}, with the memory tools below, in one shot.
Tools: search_facts(query) returns the best matching extracted facts, each with its date; search_turns(query) returns the best matching raw turns, each with its date;
open_session(id) returns the full text of one session. The sessions are listed with their ids and dates.
Return a JSON object: {{"facts_query": "...", "turns_query": "...", "open_sessions": ["id", ...]}}. Give at most two session ids; give none if the searches should be enough.
Write queries that name the people, things and dates the question is about."""

def planned_read(client, model, conv: MemoryStore, question: str, tools: list[str]) -> dict:
    """planned read: one planning call picks the searches and the sessions; the tools run without the model; one answer call with the shared prompt."""
    from memswap.managed.memory_store import ANSWER_V2
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
