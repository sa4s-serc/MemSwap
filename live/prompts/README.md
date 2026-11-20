# Prompts

Copies of the prompts used in the paper. The code in `live/` holds the same strings; `judge_l1.txt` is read from this folder by `live/judge.py`.

| file | used by | in code |
|---|---|---|
| `answer_prompt.txt` | single lookup and planned read (the shared answer prompt) | `live/store.py`, `ANSWER_V2` |
| `planned_read_planner_prompt.txt` | planned read, the planning call | `live/mechanisms.py`, `PLAN_SYSTEM` |
| `iterative_search_system_prompt.txt` | iterative search, system message; the final answer comes through the `answer` tool | `live/mechanisms.py`, `SYSTEM` |
| `iterative_search_tools.json` | iterative search, the five tool definitions (OpenAI function-calling format) | `live/mechanisms.py`, `TOOLS` |
| `fact_extraction_prompt.txt` | building the store, one call per session | `live/store.py`, `EXTRACT_SESSION_SYSTEM`, `EXTRACT_SESSION_USER` |
| `judge_l1.txt` | both judges | `live/judge.py` |

Placeholders in braces (`{a}`, `{b}`, `{memories}`, `{question}`, `{last_date}`, `{items}`, ...) are filled with Python `str.format`.
