# Data

All files are UTF-8. `SHA256SUMS` holds their checksums; `reproduce.py` checks them before it runs
(`cd data && sha256sum -c SHA256SUMS`). Every value was copied from the logged runs. Nothing was aggregated.

## answers.csv

One row per (question, system): 11,598 rows, 2,033 questions. Each system answered each question once, and each
judge scored each answer once.

| column | meaning |
|---|---|
| `question_id` | unique across datasets: `conv-NN_qK` (LoCoMo), `lme_N_qK` (LongMemEval-S*), `lmer_N_qK` (LongMemEval-R) |
| `dataset` | `locomo`, `longmemeval_s_star` or `longmemeval_r` |
| `conversation_id` | the conversation that holds the question; leave-one-conversation-out splits on this column |
| `question_type` | the benchmark label. LoCoMo: `multi-hop`, `temporal`, `open-domain`, `single-hop`. LongMemEval: `ss-user` (about what the user said), `ss-assistant` (about what the assistant said), `ss-preference` (about the user's preferences), `multi-session` (across several sessions), `temporal` (about time), `k-update` (about updated knowledge), `abstention` (with no answer in memory) |
| `system` | `single_lookup`, `planned_read`, `iterative_search` (the three reading mechanisms), `iterative_search_unpruned` (reference without context pruning), `mem0`, `letta` |
| `verdict_judge1` | primary judge (DeepSeek V4.1 Flash): 1 = correct, 0 = wrong. For the LongMemEval-S* questions with no answer in memory, the verdict is a hand label against one rule (see below) |
| `verdict_judge2` | second judge (gpt-oss-120b, same prompt): 1 or 0. Empty for the hand-labelled questions and for two planned-read answers without a parsed verdict; the analysis then uses the primary verdict |
| `memory_tokens` | input tokens that the system sent to the model over all its calls for this question, as reported by the model endpoint |
| `steps` | iterative search only: number of loop steps (at most 8) |

Mem0 and Letta were not run on LongMemEval-R.

Empty answers. Some runs returned no answer text. Every empty answer counts as wrong under both judges:

| system | LoCoMo | LongMemEval-S* |
|---|---|---|
| iterative search | 1 | 1 |
| iterative search without context pruning | 6 | 0 |
| Letta | 51 | 46 |

Three Letta rows also have `memory_tokens` = 0 (two on LoCoMo, one on LongMemEval-S*). They are among Letta's empty
answers above, and all three are scored wrong under both judges. One LongMemEval-S* question of the 300 is left out because the primary judge gave no verdict for
Mem0's answer to it.

## answer_texts.jsonl

One JSON object per (question, system): `question_id`, `dataset`, `system`, `question`, `gold_answer`, `answer`.
The fields `question` and `gold_answer` are benchmark material (see `../DATA_LICENSES.md`).

## streams.csv

The five workload streams. Columns: `stream` (1 to 5), `stream_name`, `role` (`first_batch` = profiled before the
stream, `stream` = replayed), `position` (order of replay), `question_id`. No question of a first batch is replayed.

## folds.csv

The leave-one-conversation-out folds: per dataset and held-out conversation, the number of questions in the first
batch (all other conversations) and in the held-out conversation. Ten folds on LoCoMo, five on LongMemEval-S*, three
on LongMemEval-R.

## q_texts.json, q_emb.npz

The question texts and their embeddings (bge-small-en-v1.5, normalised; arrays `ids` and `emb`), which the Question
Classifier reads. Shipping the embeddings keeps the reproduction free of model downloads.

## abstention_labels.csv

The hand labels of the answers to the 13 LongMemEval-S* questions with no answer in memory. Rule: 1 if the
answer says the information is not available and gives no answer to the question (related context is allowed); 0 if
it gives any answer, including a partial one, or is empty. The label is the verdict under both judges. One row per
answer: `question_id`, `system`, `verdict`.

## benchmarks/

`lme_s_star.json` (MemoryAgentBench's LongMemEval-S* subset) and `lme_r.json` (the remaining 200 LongMemEval
questions, packed into three conversations by `live/prepare_lme_r.py`). Only the live code reads them.
