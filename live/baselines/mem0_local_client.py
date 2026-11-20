"""In-process replacement for the Mem0 harness's REST client.

the harness's Mem0Client (benchmarks/common/mem0_client.py) wraps the OSS server's /memories and /search, and the
server (mem0/server/main.py) wraps mem0.Memory.add / .search. this class calls the same library methods directly,
with the same arguments, so the memory logic is Mem0 OSS 2.2.1 unchanged; only the HTTP hop, the Postgres session
store and the server auth are skipped. differences, stated: (1) the OSS server's MemoryCreate drops the harness's
`timestamp` field (not in its schema) while the cloud path keeps it; this client passes it to Memory.add, which
does NOT accept it either ("not supported by the OSS Memory SDK"), so this client stores the session timestamp in
the memory's metadata and reports it as `created_at` on search, which is what the cloud path does with it (and
what the harness's answer prompt reads for its chronological ordering and dates). (2) embeddings are local bge-small (384 d) instead of
text-embedding-3-small, and the extraction LLM is DeepSeek V4.1 Flash through an OpenAI-compatible endpoint, because the paper's runs use
zero dollars. config is otherwise the server's DEFAULT_CONFIG (temperature 0.2) with a Qdrant on-disk store.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# --- dated ingestion ---
# mem0 OSS 2.2.1 builds its extraction prompt with generate_additive_extraction_prompt(...) and never passes the
# `timestamp` that sets the prompt's "Observation Date", so every relative date ("yesterday") is resolved against
# the machine's clock. with MEM0_DATED=1 this client hands the session date to
# that builder through a context variable, which the prompt's own rules then use as the temporal anchor. nothing
# else in mem0 changes. MEM0_DATED unset = the undated behaviour.
import contextvars as _cv
import mem0.memory.main as _mm
_OBS_DATE: _cv.ContextVar = _cv.ContextVar("mem0_observation_date", default=None)
_orig_gen = _mm.generate_additive_extraction_prompt


def _dated_gen(*a, **kw):
    d = _OBS_DATE.get()
    if d and kw.get("timestamp") is None:
        kw["timestamp"] = d
    return _orig_gen(*a, **kw)


if os.environ.get("MEM0_DATED") == "1":
    _mm.generate_additive_extraction_prompt = _dated_gen


class Mem0LocalClient:
    """one mem0.Memory (own Qdrant store) per user_id = per conversation, so conversations ingest in parallel."""

    def __init__(self, project: str, proxy_url: str = "http://127.0.0.1:8787/v1", llm_model: str = "deepseek", **_: Any):
        self.root = HERE / "runtime" / "mem0_stores" / project
        self.proxy_url, self.llm_model = os.environ.get("MEM0_PROXY_URL", proxy_url), os.environ.get("MEM0_LLM_MODEL", llm_model)   # MEM0_PROXY_URL: an OpenAI-compatible endpoint on 8789   # e.g. glm-5.3-flash through the proxy
        self.mems: dict[str, Any] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        self.mode = "local"

    def _mem(self, user_id: str):
        if user_id not in self.mems:
            from mem0 import Memory
            store = self.root / user_id
            store.mkdir(parents=True, exist_ok=True)
            cfg = {
                "version": "v1.1",
                "vector_store": {"provider": "qdrant", "config": {"path": str(store / "qdrant"), "on_disk": True, "collection_name": "mem",
                                                                  "embedding_model_dims": 384}},
                "llm": {"provider": "openai", "config": {"api_key": "proxy", "openai_base_url": self.proxy_url, "model": self.llm_model, "temperature": 0.2}},
                "embedder": {"provider": "huggingface", "config": {"model": EMBED_MODEL, "embedding_dims": 384}},
                "history_db_path": str(store / "history.db"),
            }
            self.mems[user_id] = Memory.from_config(cfg)
            self.locks[user_id] = asyncio.Lock()
        return self.mems[user_id]

    async def close(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def add(self, messages, user_id, observation_date=None, timestamp=None, custom_instructions=None, metadata=None):
        kw: dict[str, Any] = {"user_id": user_id, "metadata": dict(metadata or {})}
        if timestamp is not None:
            kw["metadata"]["session_ts"] = int(timestamp)
            kw["metadata"]["session_date"] = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
            _OBS_DATE.set(datetime.fromtimestamp(int(timestamp), tz=timezone.utc).date().isoformat())
        if custom_instructions:
            kw["prompt"] = custom_instructions
        mem = self._mem(user_id)
        for attempt in range(5):
            try:
                async with self.locks[user_id]:
                    return await asyncio.to_thread(lambda: mem.add(messages, **kw))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ADD attempt %d/5 failed (user=%s): %s", attempt + 1, user_id, str(exc)[:200])
                await asyncio.sleep(2 * (attempt + 1))
        return None

    async def search(self, query, user_id, top_k=200, rerank=False, score_debug=False):
        for attempt in range(5):
            try:
                mem = self._mem(user_id)
                data = await asyncio.to_thread(lambda: mem.search(query=query, filters={"user_id": user_id}, top_k=top_k))
                results = data.get("results", data) if isinstance(data, dict) else data
                out = []
                for r in results or []:
                    e = {"memory": r.get("memory", r.get("data", "")), "score": r.get("score", 0), "id": r.get("id", "")}
                    md = r.get("metadata") or {}
                    e["created_at"] = md.get("session_date") or r.get("created_at") or ""   # the session date, as the cloud path reports it
                    if r.get("updated_at"):
                        e["updated_at"] = r["updated_at"]
                    out.append(e)
                out.sort(key=lambda x: x.get("score", 0), reverse=True)
                return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("SEARCH attempt %d/5 failed (user=%s): %s", attempt + 1, user_id, str(exc)[:200])
                await asyncio.sleep(2 * (attempt + 1))
        return []

    async def delete_user(self, user_id):
        try:
            await asyncio.to_thread(lambda: self._mem(user_id).delete_all(user_id=user_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_all failed for %s: %s", user_id, str(exc)[:200])
