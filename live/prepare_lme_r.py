"""LongMemEval-R: the 200 LongMemEval questions that are not in MemoryAgentBench's S* subset, packed into
dialogues the way S* is packed. The package ships its output (data/benchmarks/lme_r.json); this script documents how it was
built and needs LongMemEval's longmemeval_oracle.json (MIT, https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
Download it from the package root, and check that it is the file we used:

  curl -L -o data/benchmarks/longmemeval_oracle.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
  echo "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c  data/benchmarks/longmemeval_oracle.json" | sha256sum -c

Then run python live/prepare_lme_r.py. Convert the output to the LoCoMo shape with prepare_lme_s_star.py.
input: data/benchmarks/longmemeval_oracle.json (LongMemEval, 500 questions, evidence sessions only) and
       data/benchmarks/lme_s_star.json (the 300 S* questions, matched by question text)
output: data/benchmarks/lme_r.json (S* shape)
packing: questions that share a session are kept together (union-find); groups are ordered by their earliest session date
and filled into N_DIALOGUES dialogues of about equal session counts; inside a dialogue sessions are sorted by date and
numbered from 1; evidence_turns are the turns the oracle marks has_answer."""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent / "data" / "benchmarks"
N_DIALOGUES = 3

O = json.load(open(B / "longmemeval_oracle.json"))
S = json.load(open(B / "lme_s_star.json"))
star = {q["question"].strip().lower() for d in S for q in d["questions"]}
rest = [q for q in O if q["question"].strip().lower() not in star]
assert len(rest) == 200, len(rest)

# unique sessions and the questions that use them
sess = {}
for q in rest:
    for sid, date, turns in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
        sess.setdefault(sid, {"id": sid, "date": date, "turns": turns})
parent = {q["question_id"]: q["question_id"] for q in rest}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
owner = {}
for q in rest:
    for sid in q["haystack_session_ids"]:
        if sid in owner:
            parent[find(q["question_id"])] = find(owner[sid])
        else:
            owner[sid] = q["question_id"]
groups = defaultdict(list)
for q in rest:
    groups[find(q["question_id"])].append(q)
dt = lambda s: datetime.strptime(s.split(" (")[0] + " " + s.split(") ")[-1], "%Y/%m/%d %H:%M")
glist = sorted(groups.values(), key=lambda g: min(dt(d) for q in g for d in q["haystack_dates"]))
target = len(sess) / N_DIALOGUES
dialogues = [[] for _ in range(N_DIALOGUES)]; counts = [0] * N_DIALOGUES; k = 0
for g in glist:
    n = len({sid for q in g for sid in q["haystack_session_ids"]})
    if counts[k] + n > target * 1.05 and k < N_DIALOGUES - 1:
        k += 1
    dialogues[k].extend(g); counts[k] += n

out = []
for di, qs in enumerate(dialogues, 1):
    sids = sorted({sid for q in qs for sid in q["haystack_session_ids"]}, key=lambda sid: dt(sess[sid]["date"]))
    idx = {sid: i + 1 for i, sid in enumerate(sids)}
    sessions = [{"idx": idx[sid], "date": sess[sid]["date"], "turns": [{"role": t["role"], "content": t["content"]} for t in sess[sid]["turns"]]} for sid in sids]
    questions = []
    for q in qs:
        ev_s = sorted(idx[sid] for sid in q["haystack_session_ids"])
        ev_t = [f"S{idx[sid]}:{j}" for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"]) for j, t in enumerate(turns) if t.get("has_answer")]
        questions.append({"qid": q["question_id"], "type": q["question_type"], "date": q["question_date"], "question": q["question"],
                          "raw_question": f"Current Date: {q['question_date']},\n\n Now Answer the Question: {q['question']}", "answers": [q["answer"]],
                          "oracle_answer": q["answer"], "n_oracle_evidence_turns": len(ev_t), "evidence_turns": ev_t, "evidence_sessions": ev_s, "abstention": False})
    out.append({"dialogue": f"lmer_{di}", "n_sessions": len(sessions), "sessions": sessions, "questions": questions})
json.dump(out, open(B / "lme_r.json", "w"))
import collections
print("dialogues", [(d["dialogue"], d["n_sessions"], len(d["questions"]), sum(len(s["turns"]) for s in d["sessions"])) for d in out])
print("types", collections.Counter(q["type"] for d in out for q in d["questions"]))
print("unique sessions", len(sess), "shared by >1 question:", sum(1 for sid in sess if sum(sid in q["haystack_session_ids"] for q in rest) > 1))
