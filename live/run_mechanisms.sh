#!/usr/bin/env bash
# Full reproduction: rebuild the memory stores and rerun the three mechanisms and the judges as in the paper.
# Needs an OpenAI-compatible endpoint serving the answer model (the paper: deepseek-v4.1-flash, reasoning off, temperature 0)
# and the judges (deepseek-v4.1-flash; second judge openai/gpt-oss-120b). Set MEMSWAP_BASE_URL and MEMSWAP_API_KEY (or put them
# in a .env file at the package root). The LongMemEval files are shipped in data/benchmarks/; LoCoMo is downloaded below.
# Model outputs can differ slightly from the logged ones in data/answer_texts.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODEL:-deepseek-v4.1-flash}
export AGENT_FACTS=own MEM0_TELEMETRY=False TOKENIZERS_PARALLELISM=false

run_dataset() {   # $1 = benchmark file in LoCoMo shape, $2 = store dir, $3 = conversations, $4 = skip category (5 for LoCoMo, -1 for LongMemEval), $5 = output prefix, $6 = also run the unpruned reference (1/0)
  export OWN_DATA=$1 AGENT_DATA=$1 OWN_STORE_DIR=$2 OWN_SKIP_CAT=$4 AGENT_SKIP_CAT=$4
  python live/store.py extract_sessions --conversations "$3" --model "$M" --workers 10
  python live/store.py index
  python live/store.py onepass --reader v2 --budget 6000 --conversations "$3" --model "$M" --workers 12                          # single lookup
  AGENT_MODE=prune python live/mechanisms.py --cond agent_both --model "$M" --conversations "$3" --workers 10 --out "$5_prune"   # iterative search
  AGENT_MODE=plan  python live/mechanisms.py --cond agent_both --model "$M" --conversations "$3" --workers 10 --out "$5_plan"    # planned read
  if [ "$6" = 1 ]; then
    AGENT_MODE=full python live/mechanisms.py --cond agent_both --model "$M" --conversations "$3" --workers 10 --out "$5_full"   # reference: no pruning
  fi
}

python live/download_locomo.py                                   # LoCoMo (CC BY-NC 4.0), checksum-verified
(cd data/benchmarks && sha256sum -c SHA256SUMS)                       # shipped LongMemEval-S* and LongMemEval-R (MIT)
python live/prepare_lme_s_star.py data/benchmarks/lme_s_star.json data/benchmarks/lme_s_star_locomo.json
python live/prepare_lme_s_star.py data/benchmarks/lme_r.json      data/benchmarks/lme_r_locomo.json

run_dataset data/benchmarks/locomo10.json          own_store_locomo 0,1,2,3,4,5,6,7,8,9 5  locomo 1
run_dataset data/benchmarks/lme_s_star_locomo.json own_store_lme_s  0,1,2,3,4           -1 lme_s  1
run_dataset data/benchmarks/lme_r_locomo.json      own_store_lme_r  0,1,2               -1 lme_r  0

P="runtime/*/onepass_v2/predictions.jsonl runtime/*_prune/agent_both/predictions.jsonl runtime/*_plan/agent_both/predictions.jsonl runtime/*_full/agent_both/predictions.jsonl"
python live/judge.py --model "$M" --max-tokens 2000 --pred $P
python live/judge.py --model openai/gpt-oss-120b --max-tokens 6000 --pred $P
