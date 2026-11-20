"""LongMemEval-S* (MemoryAgentBench's LongMemEval_S* subset: 5 dialogues, 300 questions, 7 question types) in the
LoCoMo file shape, so store.py and mechanisms.py run on it unchanged. categories: 1 single-session-user,
2 single-session-assistant, 3 single-session-preference, 4 multi-session, 5 temporal-reasoning, 6 knowledge-update, 7 abstention
(live/judge.py scores them by rule: correct = the answer starts with "not mentioned"; the paper uses hand labels instead,
see data/README.md). the question text carries LongMemEval's question
date, which temporal questions need. questions whose oracle evidence was not located (and that are not abstention questions) are dropped.
usage: python live/prepare_lme_s_star.py [input] [output]
  default: data/benchmarks/lme_s_star.json -> data/benchmarks/lme_s_star_locomo.json
  LongMemEval-R: python live/prepare_lme_s_star.py data/benchmarks/lme_r.json data/benchmarks/lme_r_locomo.json"""
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
B = HERE.parent / "data" / "benchmarks"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else B / "lme_s_star.json"           # LongMemEval-S* (or data/benchmarks/lme_r.json)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else B / "lme_s_star_locomo.json"
CAT = {"single-session-user": 1, "single-session-assistant": 2, "single-session-preference": 3, "multi-session": 4, "temporal-reasoning": 5, "knowledge-update": 6}


def loco_date(s):   # "2023/05/27 (Sat) 22:54" -> "10:54 PM on 27 May, 2023" (LoCoMo's format, which own_store.session_date parses)
    d = datetime.strptime(s.split(" (")[0] + " " + s.split(") ")[-1], "%Y/%m/%d %H:%M")
    return d.strftime("%I:%M %p on %d %B, %Y").lstrip("0")


out = []
for d in json.load(open(SRC)):
    conv = {"speaker_a": "user", "speaker_b": "assistant"}
    for s in d["sessions"]:
        k = f"session_{s['idx']}"
        conv[k] = [{"speaker": t["role"], "text": t["content"], "dia_id": f"S{s['idx']}:{j}"} for j, t in enumerate(s["turns"])]
        conv[f"{k}_date_time"] = loco_date(s["date"])
    qa = []
    for q in d["questions"]:
        if not q["evidence_sessions"] and not q["abstention"]:
            continue
        gold = (q["answers"][0] if isinstance(q["answers"], list) and q["answers"] else str(q["answers"])) if not q["abstention"] else "not mentioned"
        qa.append({"question": f"(question asked on {q['date']}) {q['question']}", "answer": gold, "category": 7 if q["abstention"] else CAT[q["type"]],
                   "evidence": list(q["evidence_turns"]), "question_type": q["type"]})
    out.append({"sample_id": d["dialogue"], "conversation": conv, "qa": qa})
json.dump(out, open(OUT, "w"))
import collections
print(len(out), "dialogues", sum(len(x["qa"]) for x in out), "questions", sum(1 for x in out for k in x["conversation"] if k.startswith("session_") and not k.endswith("_date_time")), "sessions;",
      dict(collections.Counter(q["category"] for x in out for q in x["qa"])), "->", OUT)
