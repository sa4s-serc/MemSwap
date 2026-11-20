"""Execute: Question Classifier, classify : Q -> T. A logistic regression over the question's tf-idf n-grams and its
embedding, trained on the labelled profiling questions.

fit_predict is copied unchanged from the original analysis script type_classifier_nonllm.py, lines 36 to 49 (the classifier behind
the "with predicted types" row of Table II). The question embeddings (bge-small-en-v1.5) and texts are read from
data/q_emb.npz and data/q_texts.json, as that script read them from its own folder.

For replays that route by the benchmark label (the paper's main results), LabelTypes stands in for the classifier.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
_z = np.load(os.path.join(DATA, "q_emb.npz"), allow_pickle=True)
EMB = {str(i): e for i, e in zip(_z["ids"], _z["emb"])}
TEXTS = json.load(open(os.path.join(DATA, "q_texts.json")))


def fit_predict(train_ids, held_ids, y_train):
    vw = TfidfVectorizer(ngram_range=(1, 2))
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    tr_txt = [TEXTS[i] for i in train_ids]
    Xw, Xc = vw.fit_transform(tr_txt), vc.fit_transform(tr_txt)
    Xe = csr_matrix(np.stack([EMB[i] for i in train_ids]))
    clf = LogisticRegression(random_state=0, class_weight="balanced", max_iter=2000)
    clf.fit(hstack([Xe, Xw, Xc]).tocsr(), y_train)
    t0 = time.perf_counter()
    he_txt = [TEXTS[i] for i in held_ids]
    Xh = hstack([csr_matrix(np.stack([EMB[i] for i in held_ids])), vw.transform(he_txt), vc.transform(he_txt)]).tocsr()
    proba = clf.predict_proba(Xh)
    dt = (time.perf_counter() - t0) / len(held_ids)
    return clf.classes_, proba, dt


class QuestionClassifier:
    """trained on labelled profiling questions; classify returns the most probable trained type (argmax)."""

    def __init__(self, prefix=""):
        self.prefix = prefix          # type keys carry the dataset family, e.g. "locomo|temporal"
        self.pred = {}

    def fit_predict(self, train_ids, held_ids, y_train):
        classes, proba, _ = fit_predict(train_ids, held_ids, y_train)
        for qid, p in zip(held_ids, proba):
            self.pred[qid] = classes[int(np.argmax(p))]

    def classify(self, qid):
        return self.prefix + self.pred[qid]


class LabelTypes:
    """the benchmark label stands in for the classifier."""

    def __init__(self, q):
        self.type_key = q["type_key"]

    def classify(self, qid):
        return self.type_key.at[qid]
