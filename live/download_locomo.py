"""Downloads the LoCoMo conversations (locomo10.json) from the original repository and checks them against the file used in
the paper. LoCoMo is licensed CC BY-NC 4.0 (see DATA_LICENSES.md); the package links to it instead of shipping it.

  python live/download_locomo.py      -> data/benchmarks/locomo10.json"""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/snap-research/locomo/cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc/data/locomo10.json"
SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
OUT = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "locomo10.json"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(URL, timeout=120).read()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        raise SystemExit(f"checksum mismatch: expected {SHA256}, got {got}")
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data):,} bytes, sha256 ok)")
