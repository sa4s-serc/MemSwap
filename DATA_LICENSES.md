# Data licenses

The code in this package is under the MIT license (`LICENSE`).

The following are our own work, also released under the MIT license:

- the answers of our reading mechanisms and of the baselines as we ran them;
- the verdicts of both judges, the token counts and the step counts;
- `data/answers.csv`, `data/streams.csv`, `data/folds.csv`, `data/abstention_labels.csv`, `data/q_emb.npz`;
- the field `answer` of `data/answer_texts.jsonl`;
- everything under `results/`.

The files listed below contain material from third-party benchmarks. That material stays under the benchmarks' own
licenses. We checked the three licenses on 17 November 2026. All three allow redistribution; LoCoMo allows it for
non-commercial purposes only.

## LoCoMo (CC BY-NC 4.0)

`data/answer_texts.jsonl` (fields `question` and `gold_answer` of the rows with `"dataset": "locomo"`) and `data/q_texts.json` (the entries whose id starts with `conv-`) contain question texts and gold answers from LoCoMo:

> A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, Y. Fang. "Evaluating Very Long-Term Conversational Memory of LLM Agents." ACL 2024. https://github.com/snap-research/locomo

LoCoMo is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/. The LoCoMo material in this package is a derived version.

We changed it as follows:
- we kept the question, the gold answer and the question type of the questions we evaluated;
- we left out the adversarial category and the conversations;
- we added our systems' answers and the judges' verdicts.

**This material may be used for non-commercial purposes only.**

The package does not ship the LoCoMo conversations. `live/download_locomo.py` downloads them from the original repository, pinned to commit `cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc`, and checks their SHA-256.

## LongMemEval (MIT)

LongMemEval material appears in two places:
- `data/benchmarks/lme_s_star.json` and `data/benchmarks/lme_r.json` contain LongMemEval sessions, questions and answers, repacked into conversations;
- `data/answer_texts.jsonl` (fields `question` and `gold_answer` of the rows with `"dataset"` `longmemeval_s_star` or `longmemeval_r`) and `data/q_texts.json` contain its question texts and gold answers.

Source: D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, D. Yu. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." ICLR 2025. https://github.com/xiaowu0162/LongMemEval

```
MIT License

Copyright (c) 2024 Di Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MemoryAgentBench (MIT)

`data/benchmarks/lme_s_star.json` is MemoryAgentBench's LongMemEval_S* subset: its selection and packing of LongMemEval questions into five conversations.

Source: Y. Hu, Y. Wang, J. McAuley. "Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions." 2025. https://github.com/HUST-AI-HYZ/MemoryAgentBench, data at https://huggingface.co/datasets/ai-hyz/MemoryAgentBench

We added the evidence turns of each question, located in the LongMemEval oracle file. `data/benchmarks/lme_r.json` packs the remaining 200 LongMemEval questions the same way (`live/prepare_lme_r.py`).

```
MIT License

Copyright (c) 2026 Yuanzhe Hu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The MemoryAgentBench repository is under the MIT license. Its README states no separate license for the copy of the data on Hugging Face.
