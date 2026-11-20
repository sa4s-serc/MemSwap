# Extending MemSwap

Everything in the replay goes through three small interfaces. If you keep them, `reproduce.py`, the tests and the
notebook work on your extension without further changes.

| you want to add | module to touch | interface |
|---|---|---|
| a reading mechanism | `memswap/knowledge/knowledge_base.py` (`MECHANISMS`), `data/answers.csv` | a `system` value with one row per question |
| a dataset | `data/answers.csv`, `data/streams.csv`, `evaluation.py` (`DATASETS`, `FAMILY`) | rows in the same columns |
| a selection rule | `memswap/plan/mechanism_selector.py` | `select(k, n, tok, margin) -> dict` |
| a planner | `memswap/plan/plan_optimizer.py` | `PlanOptimizer.plan(fold) -> (assignment, moves)` |
| a question classifier | `memswap/managed/memory_manager/question_classifier.py` | an object with `classify(question_id) -> question type` |

The replay never calls a model. A mechanism "runs" by looking up its logged outcome:

```python
read = lambda mechanism, question_id: (verdict, memory_tokens)     # what the Router's read returns
```

## 1. A new reading mechanism

1. Run your mechanism once on every question and have the judges score the answers (`live/judge.py` uses the paper's
   judge prompt). Add one row per question to `data/answers.csv` with a new value in `system`, for example
   `graph_walk`, and fill `verdict_judge1`, `verdict_judge2` and `memory_tokens`.
2. Add the name to `MECHANISMS` in `memswap/knowledge/knowledge_base.py` and to `MECHS` in `evaluation.py`. The order
   of the list breaks ties.
3. Update the checksum: `sha256sum data/answers.csv`, then edit `data/SHA256SUMS`.
4. `python reproduce.py`. The numbers of the paper will now FAIL where your mechanism changes a decision; that is the
   point. Read the new values in `results/replay.json` and `results/tables/`.

The Plan Optimizer tries all `len(MECHANISMS) ** number_of_known_question_types` plans. With four mechanisms and seven
question types that is 16,384 plans, still a few milliseconds.

Minimal check without touching the data files:

```python
import pandas as pd
from memswap.knowledge.knowledge_base import AdaptationGoals, MECHANISMS
from memswap.loop import MemSwap

MECHANISMS.append("graph_walk")                     # before any MemSwap object is built
ids = [f"q{i}" for i in range(100)]
S = pd.DataFrame({"single_lookup": 1, "planned_read": 0, "iterative_search": 1, "graph_walk": 1}, index=ids)
TOK = pd.DataFrame({"single_lookup": 6000.0, "planned_read": 2000.0, "iterative_search": 7000.0, "graph_walk": 900.0}, index=ids)

class Labels:
    def classify(self, qid): return "any"

ms = MemSwap(AdaptationGoals(margin=0.01), Labels(), lambda m, q: (S.at[q, m], TOK.at[q, m]), S, TOK)
ms.first_batch("f", ids[:50], Labels().classify)
print(ms.kb.routing.table["f"])                     # {'any': 'graph_walk'}: as accurate as the best, and cheapest
```

## 2. A new dataset

1. Add its rows to `data/answers.csv`: `question_id` (unique across datasets), `dataset`, `conversation_id` (the unit
   that leave-one-conversation-out holds out), `question_type`, and one row per system.
2. In `evaluation.py`, add the dataset to `DATASETS` and give it a family in `FAMILY`. Question types of different
   families are kept apart even when their names match; datasets of one family share their question types.
3. For the predicted-question-type experiments, add the question texts to `data/q_texts.json` and their embeddings
   (bge-small-en-v1.5, normalised) to `data/q_emb.npz` (`ids`, `emb`).
4. For a workload stream, add rows to `data/streams.csv`: `stream`, `stream_name`, `role` (`first_batch` or `stream`),
   `position`, `question_id`. No question of the first batch may be replayed (`tests/test_data_and_numbers.py` checks it).
5. Update `data/SHA256SUMS`, then `python reproduce.py`.

## 3. A different selection rule

`memswap/plan/mechanism_selector.py` holds the rule as one function:

```python
def select(k, n, tok, margin, context=None):
    """k[m]: correct answers of mechanism m on the n profiled questions; tok[m]: its mean memory tokens;
    margin: accuracy margin as a fraction. Must return a dict with at least
    best, chosen, chosen_z1645, n, accuracy, mean_tokens, wilson_lb."""
```

Example, a rule without the confidence bound (it trusts the measured accuracy):

```python
def select(k, n, tok, margin, context=None):
    acc = {m: (k[m] / n if n else 0.0) for m in MECHS}
    best = sorted(MECHS, key=lambda m: (-acc[m], tok[m], MECHS.index(m)))[0]
    ok = [m for m in MECHS if tok[m] < tok[best] and acc[m] >= acc[best] - margin]
    chosen = min(ok, key=lambda m: tok[m]) if ok else best
    return {"best": best, "chosen": chosen, "chosen_z1645": chosen, "n": int(n), "accuracy": acc, "mean_tokens": tok,
            "wilson_lb": acc}
```

`MechanismSelector.decide_batch` calls it once per question type of a labelled batch, and once over the whole batch
for the default mechanism. Run `pytest` (two tests state what the paper's rule must do; adjust them to your rule),
then `python reproduce.py`.

## 4. A different planner

`memswap/plan/plan_optimizer.py`:

```python
class PlanOptimizer:
    def plan(self, fold):
        """returns (assignment, moves): assignment maps each known question type to a mechanism; moves lists
        {"type", "from", "to"} for the question types that leave their selected mechanism."""
```

Inputs you can use: `self.planner_input(fold)` (per question type: `acc[m]`, `tok[m]`, `start`, `n`),
`self.kb.shares(fold)` (question counts per question type, the shares w_t) and `self.kb.goals.budget`. A greedy
planner, for example, starts from `start` and moves the question type with the smallest accuracy loss per token
saved until the expected cost fits. The loop (`memswap/loop.py`) calls `plan` on a new budget, after two windows
over budget, and after a labelled batch when a budget is set; the Plan Enactor writes the result.

## 5. Running live

`live/` holds the code that produced the logged answers. `live/run_mechanisms.sh` rebuilds the memory stores, runs
the three mechanisms and both judges, and needs an OpenAI-compatible endpoint (`MEMSWAP_BASE_URL`,
`MEMSWAP_API_KEY`). To connect a live mechanism to the loop, pass a `read` function that calls it and a judge
instead of the lookup above. The loop itself does not change.
