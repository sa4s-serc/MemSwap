# `checks/`: further experiments and timings

| script | what it does | output | in the paper |
|---|---|---|---|
| `letta_empty_answers.py` | MemSwap against Letta with and without Letta's empty answers | `results/letta_empty_answers.json` | RQ1, Letta |
| `predicted_question_types.py` | the workload streams with question types predicted by the Question Classifier, against benchmark labels | `results/predicted_question_types.json` | RQ2, Table III (predicted question types) |
| `selector_time.py` | times each decision of the Mechanism Selector | prints the times | RQ3 |
| `time_components.py` | times the Question Classifier and the Plan Optimizer on your machine | `results/time_components.json` | RQ3 (optional) |

`reproduce.py` runs the first two and compares their outputs with `results/expected/`. The timings depend on the
machine, so `reproduce.py` reads the values behind the paper from `results/expected/logged/` and does not compare
times. `time_components.py` needs `pip install sentence-transformers==6.1.0 torch==2.14.1` and downloads the embedding
model on the first run.
