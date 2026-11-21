"""Checks of the shipped data and of the expected results. No replay, a few seconds."""
import csv
import os
import re

import pandas as pd

import evaluation as E
import paper_numbers as PN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_checksums():
    assert len(E.check_sums()) >= 4


def test_answers_table():
    a = pd.read_csv(os.path.join(ROOT, "data", "answers.csv"))
    assert len(a) == 11598 and not a.duplicated(["question_id", "system"]).any()
    assert a.groupby("dataset").question_id.nunique().to_dict() == {"locomo": 1534, "longmemeval_r": 200, "longmemeval_s_star": 299}
    assert set(a.verdict_judge1.unique()) == {0, 1}
    assert (a.memory_tokens >= 0).all() and (a[a.system != "letta"].memory_tokens > 0).all()


def test_no_question_of_a_first_batch_is_replayed():
    s = pd.read_csv(os.path.join(ROOT, "data", "streams.csv"))
    for _, g in s.groupby("stream"):
        assert not set(g[g.role == "first_batch"].question_id) & set(g[g.role == "stream"].question_id)


def test_every_paper_number_follows_from_the_expected_results():
    R = PN.Results(os.path.join(ROOT, "results", "expected"))
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "paper_numbers.csv"))))
    assert len(rows) > 400
    bad = [r["id"] for r in rows if PN.value(R, r["source_key"], r["format"]) != r["paper_value"]]
    assert bad == []


FORBIDDEN = [r"/mnt/", r"/home/", r"/Users/", r"\bsk-[A-Za-z0-9]{16,}", r"api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]"]


def test_no_paths_or_secrets_in_the_package():
    hits = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in (".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "runtime")]
        for f in files:
            if not f.endswith((".py", ".md", ".txt", ".sh", ".ipynb", ".csv", ".json", ".jsonl", ".cfg", ".toml")):
                continue
            p = os.path.join(d, f)
            if os.path.getsize(p) > 20_000_000 or os.path.abspath(p) == os.path.abspath(__file__):
                continue
            text = open(p, errors="ignore").read()
            for pat in FORBIDDEN:
                if re.search(pat, text):
                    hits.append((os.path.relpath(p, ROOT), pat))
    assert hits == []
