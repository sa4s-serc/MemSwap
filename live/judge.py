"""The Judge: scores each answer as correct or wrong against the gold answer (paper, Component 7).

The same prompt (live/prompts/judge_l1.txt), batching and parsing as the paper: 32 items per call, temperature 0, the
model returns a JSON list of {id, verdict} with CORRECT or WRONG; an item whose verdict does not parse is left
unscored and is sent again on the next run. Abstention questions (LongMemEval category 7) are scored by rule:
correct = the answer starts with "not mentioned". The paper replaces these rule verdicts with hand labels
(data/abstention_labels.csv; the labelling rule is in data/README.md). The primary judge used deepseek-v4.1-flash with max_tokens 2000; the
second judge used gpt-oss-120b with max_tokens 6000.

  python live/judge.py --model deepseek-v4.1-flash --max-tokens 2000 --pred runtime/<...>/predictions.jsonl [...]

Endpoint: MEMSWAP_BASE_URL / MEMSWAP_API_KEY (any OpenAI-compatible endpoint), from the environment or the package .env.
output: <pred>.verdicts.<model>.jsonl next to each predictions file, one row {qa_id, verdict, judge}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = (Path(__file__).resolve().parent / "prompts" / "judge_l1.txt").read_text()
BATCH = 32
ABSTAIN_CAT = 7


def norm(p: str) -> str:
    return re.sub(r"\s+", " ", (p or "").strip().lower())


def is_abstain(pred: str) -> bool:
    return norm(pred).startswith("not mentioned")


def call(client, model: str, items: list[dict], max_tokens: int) -> dict:
    text = "\n".join(json.dumps({"id": it["id"], "question": it["question"], "gold_answer": it["gold"], "generated_answer": it["pred"]}) for it in items)
    for attempt in range(6):
        try:
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                               messages=[{"role": "user", "content": PROMPT.format(items=text)}])
            out = r.choices[0].message.content or ""
            m = re.search(r"\[.*\]", out, re.S)
            res = json.loads(m.group(0)) if m else []
            return {str(v.get("id")): str(v.get("verdict", "")).upper() for v in res if isinstance(v, dict)}
        except json.JSONDecodeError:
            return {}
        except Exception as e:  # noqa: BLE001
            time.sleep(30 * (attempt + 1)); print("retry", type(e).__name__, str(e)[:120], flush=True)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--pred", nargs="+", required=True); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    from openai import OpenAI
    from store import load_env
    load_env()
    client = OpenAI(api_key=os.environ["MEMSWAP_API_KEY"], base_url=os.environ["MEMSWAP_BASE_URL"], timeout=300, max_retries=2)
    for p in a.pred:
        rows = [json.loads(l) for l in open(p)]
        out = Path(p).with_suffix(f".verdicts.{a.model.replace('/', '_')}.jsonl")
        done = {json.loads(l)["qa_id"] for l in open(out)} if out.exists() else set()
        todo, ruled = [], []
        for r in rows:
            if r["qa_id"] in done:
                continue
            if int(r.get("category") or 0) == ABSTAIN_CAT and str(r["qa_id"]).startswith("lme"):
                ruled.append({"qa_id": r["qa_id"], "verdict": "CORRECT" if is_abstain(r.get("pred")) else "WRONG", "judge": "rule: answer starts with 'not mentioned'"})
            else:
                todo.append({"id": r["qa_id"], "question": r["question"], "gold": str(r.get("gold", "")), "pred": r.get("pred") or ""})
        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        with ThreadPoolExecutor(a.workers) as ex:
            got = {}
            for part in ex.map(lambda b: call(client, a.model, b, a.max_tokens), batches):
                got |= part
        scored = [{"qa_id": it["id"], "verdict": got[it["id"]], "judge": a.model} for it in todo if got.get(it["id"]) in ("CORRECT", "WRONG")]
        with open(out, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in ruled + scored)
        print(f"{p}: {len(scored) + len(ruled)} scored, {len(todo) - len(scored)} unparsed (run again to retry) -> {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
