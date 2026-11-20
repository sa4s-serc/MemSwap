"""start the embedded Postgres (if not running) and exec the Letta 0.16.8 server against the proxy.

  .venv-letta/bin/python letta_server.py [--port 8283] [--proxy http://localhost:8000/v1]

the server's OpenAI provider reads OPENAI_API_KEY / OPENAI_API_BASE; through the proxy every chat call is the
DeepSeek V4.1 Flash (reasoning off) and embeddings are local bge-small padded to the OpenAI widths Letta expects."""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8283)
ap.add_argument("--proxy", default="http://localhost:8000/v1", help="OpenAI-compatible base url; 'openai' = the real OpenAI API, with OPENAI_API_KEY from the environment")
ap.add_argument("--pgdata", default="pgdata")
a = ap.parse_args()

import pgserver  # noqa: E402

db = pgserver.get_server(str(HERE / "runtime" / a.pgdata), cleanup_mode=None)
os.environ["LETTA_PG_URI"] = db.get_uri().replace("@/", "@localhost/")   # letta drops the user from "user:@/db?host=sock"
if a.proxy == "openai":
    assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY"
else:
    os.environ["OPENAI_API_KEY"] = "proxy"
    os.environ["OPENAI_API_BASE"] = a.proxy
os.environ.setdefault("LETTA_DEBUG", "false")
os.environ.setdefault("LETTA_PG_POOL_SIZE", "25")
os.environ.setdefault("LETTA_PG_MAX_OVERFLOW", "10")
os.execv(str(HERE / ".venv-letta" / "bin" / "letta"), ["letta", "server", "--port", str(a.port)])
