"""How each number of the paper is read from the result files.

results/paper_numbers.csv has one row per number: its value as printed in the paper, the key of the result file it
comes from, and the name of the formatter that turns the stored value into the printed one. reproduce.py recomputes
the result files, formats each key again with FORMATTERS and compares the string with the paper's.

Key syntax: <FILE>:<path/with/slashes>. FILE is one of FILES below. PREFIXES are shorthands for long paths.
Key names that start with memswitch refer to MemSwap.
"""
import json
import math
import os

FILES = {"R": "replay.json", "OS": "predicted_question_types.json", "LE": "letta_empty_answers.json",
         "EF": "logged/efficiency.json", "NG": "logged/efficiency_netgain.json"}
LOGGED = {"EF", "NG"}        # measured once (timings, judge usage); shipped, not recomputed
PREFIXES = {"W": "R:results.json/memswitch_within_dataset_leave_one_conversation_out", "ST": "R:results.json/streams",
            "PT": "R:results.json/per_type", "PS": "R:results.json/per_system_per_dataset",
            "BV": "R:budget_vs_fixed.json/cases", "BA": "R:budget_adaptation_p40c.json/cases",
            "TC": "R:type_classifier_nonllm.json/datasets", "OSS": "OS:streams", "LED": "LE:datasets"}


class Results:
    def __init__(self, folder):
        self.folder, self.cache = folder, {}

    def file_of(self, key):
        f = key.split(":", 1)[0]
        return self.file_of(PREFIXES[f]) if f in PREFIXES else f

    def get(self, key):
        f, _, path = key.partition(":")
        if f in PREFIXES:
            return self.get(PREFIXES[f] + "/" + path)
        if f not in self.cache:
            self.cache[f] = json.load(open(os.path.join(self.folder, FILES[f])))
        x = self.cache[f]
        for k in path.split("/"):
            if k == "windows[kept]":
                return "%d/%d" % (sum(1 for w in x["windows"] if w["kept"]), len(x["windows"]))
            x = x[int(k)] if isinstance(x, list) else x[k]
        return x


def _p(v, mode):
    if v < 0.001:
        return "< 0.001"
    if v >= 0.9995:
        return "= 1.0"
    d = 2 if v >= 0.1 else 3
    q = 10 ** d
    r = {"round": round(v * q) / q, "floor": math.floor(v * q) / q, "ceil": math.ceil(v * q) / q}[mode]
    return "= " + f"{r:.{d}f}"


def _signed(v, d):
    s = ("+" if v >= 0 else "-") + f"{abs(v):.{d}f}"
    return s[1:] if float(s) == 0 else s


FORMATTERS = {
    "text": lambda v: str(v),
    "int": lambda v: f"{int(v):,}",
    "round_int": lambda v: f"{round(v):,}",
    "pct1": lambda v: f"{100 * v:.1f}", "pct2": lambda v: f"{100 * v:.2f}",
    "num0": lambda v: f"{v:.0f}", "num1": lambda v: f"{v:.1f}", "num2": lambda v: f"{v:.2f}",
    "abs1": lambda v: f"{abs(v):.1f}", "abs2": lambda v: f"{abs(v):.2f}",
    "signed1": lambda v: _signed(v, 1), "signed2": lambda v: _signed(v, 2),
    "signed1_flipped": lambda v: _signed(-v, 1), "signed2_flipped": lambda v: _signed(-v, 2),
    "p": lambda v: "p " + _p(v, "round"),
    "p_ge": lambda v: "p \\ge " + _p(v, "floor").replace("= ", ""),
    "p_le": lambda v: "p < 0.001" if v < 0.001 else "p \\le " + _p(v, "ceil").replace("= ", ""),
    "share_pct2": lambda v: f"{100 * v:.2f}",
}


def value(results, key, fmt):
    """the printed value of one row, from the result files in `results`."""
    if fmt in ("ratio_fewer_pct", "ratio_pct"):
        k1, k2 = [k.strip() for k in key.split("  /  ")]
        r = results.get(k1) / results.get(k2)
        return str(round(100 * (1 - r)) if fmt == "ratio_fewer_pct" else round(100 * r))
    if fmt == "payback_range":
        ys = sorted(results.get(key.replace("*", d)) for d in ("locomo", "longmemeval_s_star", "longmemeval_r"))
        return f"{round(ys[0]):,} to {round(ys[2]):,}"
    return FORMATTERS[fmt](results.get(key))
