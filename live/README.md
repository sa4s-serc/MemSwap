# `live/`: running MemSwap and the mechanisms with a model

Everything in the rest of the package replays logged outcomes. This folder holds the code that talks to a model.

| file | what it does | needs a model endpoint |
|---|---|---|
| `example_mock.py` | **the live wiring with mock mechanisms and a mock judge**: profiling, routing, the default mechanism for an unprofiled question type, and a re-plan under a lower budget. Also run by `pytest` (`tests/test_live_wiring.py`). | no |
| `smoke_test.py` | **the live wiring with the real mechanisms and model** on five LoCoMo questions; writes `smoke_test_result.json` | yes |
| `smoke_test_result.json` | the result of our smoke-test run (see below) | |
| `run_mechanisms.sh` | the full rerun behind `data/`: builds the memory stores and runs the three mechanisms and both judges on all datasets | yes |
| `store.py` | builds the memory store (dated facts and their index) and runs single lookup | yes |
| `mechanisms.py` | planned read and iterative search | yes |
| `judge.py` | the LLM judge (the paper's prompt, batching and parsing) | yes |
| `download_locomo.py` | downloads LoCoMo at a pinned commit and checks its checksum (CC BY-NC 4.0) | no |
| `prepare_lme_s_star.py`, `prepare_lme_r.py` | convert the shipped LongMemEval files to the format of the mechanisms | no |
| `baselines/` | the Mem0 and Letta runners as we ran them | yes |
| `prompts/` | every prompt used, as plain text | |

## Setup

```bash
pip install -r requirements.txt -r live/requirements-live.txt
export MEMSWAP_BASE_URL=<an OpenAI-compatible endpoint>  MEMSWAP_API_KEY=<your key>   # or put them in .env
```

The Letta baseline needs its own environment (`live/baselines/requirements-letta.txt`), because Letta 0.16.8 ran with
other versions of openai and numpy.

## Model endpoints and ports

Every script talks to the answer model through an OpenAI-compatible endpoint. We ran all model calls through a small
local proxy that forwarded them to OpenRouter (DeepSeek V4.1 Flash), mapped model names and logged token usage and
cost. **That proxy is not shipped**: it held our API keys and spending limits. Any OpenAI-compatible endpoint that
serves the answer model works in its place. Point each script at it:

| script | how it finds the endpoint | default in the code | what we used |
|---|---|---|---|
| `store.py`, `mechanisms.py`, `judge.py`, `smoke_test.py` | `MEMSWAP_BASE_URL`, `MEMSWAP_API_KEY` | none (must be set) | the proxy on port 8789 |
| `baselines/mem0_lme.py`, `baselines/mem0_local_client.py` | `MEM0_PROXY_URL`, `MEM0_LLM_MODEL` | `http://localhost:8000/v1` (`mem0_lme.py`), `http://127.0.0.1:8787/v1` (`mem0_local_client.py`) | the proxy on port 8789 |
| `baselines/letta_server.py` | `--proxy` (the model endpoint), `--port` (the Letta server) | `--proxy http://localhost:8000/v1`, `--port 8283` | `--proxy http://127.0.0.1:8789/v1 --port 8285` |
| `baselines/letta_locomo.py` | `--base-url` (the Letta server) | `http://127.0.0.1:8283` | `--base-url http://127.0.0.1:8285`, the server started by `letta_server.py` |

The ports 8000, 8787 and 8789 are only placeholders for your own model endpoint; nothing in the package listens on
them. Port 8283 (or 8285, as we ran it) is the Letta server that `letta_server.py` starts.

## Smoke test

`python live/smoke_test.py` builds the memory store of one LoCoMo conversation (conv-30, 19 sessions). It profiles a
labelled batch of three questions, one each of temporal, single-hop and multi-hop, with all three mechanisms and the
judge. It then writes the Routing Table and answers two further questions with the mechanism of their question type.
It checks the wiring, not accuracy: five questions say nothing about accuracy.

**Our run (21 Nov 2026, DeepSeek V4.1 Flash, 107 s, USD 0.01): PASS.**

| step | result |
|---|---|
| Memory store | 112 dated facts from conv-30 |
| Profiling | 9 answers and 9 verdicts. On the temporal and single-hop questions all three mechanisms were correct; on the multi-hop question only iterative search was |
| Routing Table | temporal and single-hop to planned read (about 1,800 tokens), multi-hop to iterative search (7,878 tokens); default iterative search |
| Served questions | the temporal and the single-hop question were answered by planned read, as the Routing Table gives |

The answers, verdicts and token counts of every call are in `smoke_test_result.json`.

Model outputs are not bit-identical between runs, so a rerun gives similar but not equal answers and token counts.
