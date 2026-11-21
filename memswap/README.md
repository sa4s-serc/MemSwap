# `memswap/`: the architecture

One module per component of the architecture figure (Fig. 1 of the paper). The replay (`replay.py`) and the live
wiring (`live.py`) drive the same components through `loop.py`.

| paper component (Fig. 1) | module |
|---|---|
| **Managing system** | |
| Knowledge: Mechanism Profiles, Adaptation Goals, Routing Table | `knowledge/knowledge_base.py` |
| Monitor: Usage Tracker | `monitor/usage_tracker.py` |
| Monitor: Mechanism Evaluator | `monitor/mechanism_evaluator.py` |
| Analyze: Workload Analyzer | `analyze/workload_analyzer.py` |
| Analyze: Cost Analyzer | `analyze/cost_analyzer.py` |
| Plan: Mechanism Selector (selection rule, Eq. 2) | `plan/mechanism_selector.py` |
| Plan: Plan Optimizer (Eq. 3) | `plan/plan_optimizer.py` |
| Execute: Plan Enactor (the `update` interface) | `execute/plan_enactor.py` |
| the loop in the order of Algorithm 1 | `loop.py` |
| **Managed system: LLM agent** | |
| Memory manager: Question Classifier | `managed/memory_manager/question_classifier.py` |
| Memory manager: Router | `managed/memory_manager/router.py` |
| Memory manager: single lookup, planned read, iterative search | `managed/memory_manager/mechanisms/` |
| Memory store | `managed/memory_store.py` |
| **Live wiring** | |
| A live agent's mechanisms and judge, profiled and routed by the loop | `live.py` |

## The two ways the loop is driven

`MemSwap` in `loop.py` takes a `read(mechanism, question_id) -> (outcome, memory_tokens)` function, which the Router
calls on the request path, and the judged outcomes of the labelled batch, which the Mechanism Evaluator profiles.

- **Replay** (`replay.py`): `read` returns the logged outcome of a mechanism on a question, and the judged outcomes come
  from `data/answers.csv`. No model calls.
- **Live** (`live.py`): you pass one function per reading mechanism and a judge. `LiveMemSwap.labelled_batch` runs every
  mechanism on the labelled questions and has the judge score the answers; `LiveMemSwap.ask` answers a new question
  with the mechanism that the Routing Table gives its question type. See the home page, "Using MemSwap", path 3.

The mechanisms under `managed/memory_manager/mechanisms/` and the store under `managed/memory_store.py` are the code
that produced the logged answers. They need a model endpoint, and `replay.py` does not run them.
