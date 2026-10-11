"""Tasks 1/2: train-only features, validation selection, held-out test."""
import argparse
import gzip
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ARTIFACTS, ROOT, SEED, load_data, save_result, sha256, tokenize, stable_hash


class EpochProgress(CallbackAny2Vec):
    def __init__(self, name):
        self.name, self.epoch = name, 0

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"{self.name}: epoch {self.epoch}/10 complete", flush=True)


def vectorizer(method):
    options = dict(tokenizer=tokenize, token_pattern=None, lowercase=False, dtype=np.float64)
    if method == "tfidf":
        return TfidfVectorizer(**options)
    return CountVectorizer(binary=method == "binary", **options)


def mean_embeddings(documents, vectors, dim=100):
    matrix = np.zeros((len(documents), dim), dtype=np.float32)
    covered = total = empty = 0
    for i, tokens in enumerate(documents):
        valid = [vectors[t] for t in tokens if t in vectors]
        total += len(tokens)
        covered += len(valid)
        if valid:
            matrix[i] = np.mean(valid, axis=0)
        else:
            empty += 1
    return matrix, {"tokens": total, "covered_tokens": covered, "coverage": covered / max(total, 1), "empty_documents": empty}


def load_glove(path, vocabulary):
    """Read the original text or Gensim's lossless word2vec-text conversion."""
    result = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = 0
    with opener(path, "rt", encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            fields = line.rstrip().split()
            if i == 0 and len(fields) == 2 and all(s.isdigit() for s in fields):
                if int(fields[1]) != 100:
                    raise ValueError("Expected 100-dimensional GloVe vectors.")
                continue
            if len(fields) != 101:
                raise ValueError(f"Malformed GloVe row {i + 1}")
            rows += 1
            if fields[0] in vocabulary:
                result[fields[0]] = np.asarray(fields[1:], dtype=np.float32)
    if rows != 400000:
        raise ValueError(f"Expected the complete glove.6B vocabulary (400000), got {rows}.")
    return result


def select_lr(features, y_train, y_val, labels):
    candidates = []
    best = None
    for c in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, random_state=SEED)
        with threadpool_limits(limits=4):
            model.fit(features["train"], y_train)
        if int(model.n_iter_.max()) >= model.max_iter:
            raise RuntimeError("Logistic regression failed to converge; increase max_iter.")
        pred = model.predict(features["validation"])
        score = f1_score(y_val, pred, labels=labels, average="macro", zero_division=0)
        candidates.append({"C": c, "validation_macro_f1": float(score), "iterations": int(model.n_iter_.max())})
        if best is None or score > best[0]:
            best = score, model, pred
    return best[1], best[2], candidates


def run(method, glove_path=None):
    started = time.perf_counter()
    df, split, labels = load_data()
    texts = {s: df.iloc[ids].text.tolist() for s, ids in split.items()}
    details = {"selection": "highest validation Macro-F1; ties prefer lower C", "fit_scope": "NYT train only"}
    bundle = {}
    if method in ("binary", "frequency", "tfidf"):
        vec = vectorizer(method)
        features = {"train": vec.fit_transform(texts["train"])}
        features.update({s: vec.transform(texts[s]) for s in ("validation", "test")})
        details["vocabulary_size"] = len(vec.vocabulary_)
        bundle["vectorizer"] = vec
    else:
        tokens = {s: [tokenize(t) for t in ts] for s, ts in texts.items()}
        if method == "glove":
            if not glove_path or not Path(glove_path).is_file():
                raise FileNotFoundError("Provide --glove artifacts/glove.6B.100d.txt (or .gz conversion).")
            # Reading known pretrained vectors is a fixed lookup, not fitting to held-out data.
            vocab = set(t for docs in tokens.values() for doc in docs for t in doc)
            vectors = load_glove(glove_path, vocab)
            details.update({"embedding_source": "Stanford glove.6B.100d", "embedding_sha256": sha256(glove_path)})
        else:
            corpus = tokens["train"] if method == "w2v_nyt" else [tokenize(t) for t in pd.read_csv(ROOT / "HW-1" / "ag.csv").text]
            params = dict(vector_size=100, window=5, min_count=5, sg=1, negative=5, epochs=10, workers=1, seed=SEED, hashfxn=stable_hash)
            print(f"Training {method}: {len(corpus)} documents", flush=True)
            w2v = Word2Vec(sentences=corpus, callbacks=[EpochProgress(method)], **params)
            ARTIFACTS.mkdir(exist_ok=True)
            w2v.save(str(ARTIFACTS / f"{method}.model"))
            vectors = w2v.wv
            details.update({"embedding_source": "NYT train" if method == "w2v_nyt" else "AG News", "embedding_vocabulary": len(vectors),
                            "word2vec": {k: v for k, v in params.items() if k != "hashfxn"}, "corpus_documents": len(corpus)})
        details["coverage"] = {}
        features = {}
        for s in split:
            features[s], details["coverage"][s] = mean_embeddings(tokens[s], vectors)
        scaler = StandardScaler().fit(features["train"])
        features = {s: scaler.transform(x) for s, x in features.items()}
        bundle.update({"scaler": scaler, "vectors": vectors})
        details["dimensions"] = 100
    model, val_pred, candidates = select_lr(features, df.iloc[split["train"]].label, df.iloc[split["validation"]].label, labels)
    # Test is evaluated only after all validation-based choices have been made.
    test_pred = model.predict(features["test"])
    details.update({"C_candidates": candidates, "selected_C": model.C, "seconds": time.perf_counter() - started})
    bundle.update({"classifier": model, "method": method})
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(bundle, ARTIFACTS / f"{method}.joblib")
    save_result(method, df, split, labels, val_pred, test_pred, details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["binary", "frequency", "tfidf", "w2v_ag", "w2v_nyt", "glove"],
                        choices=["binary", "frequency", "tfidf", "w2v_ag", "w2v_nyt", "glove"])
    parser.add_argument("--glove", type=Path)
    args = parser.parse_args()
    for method in args.methods:
        run(method, args.glove)


if __name__ == "__main__":
    main()
