"""Shared split, preprocessing and auditable evaluation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
SEED = 42
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def make_split(labels, seed=SEED):
    indices = np.arange(len(labels))
    train, rest = train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=np.asarray(labels)[rest], random_state=seed)
    return {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}


def validate_split(split, n):
    combined = [i for name in ("train", "validation", "test") for i in split[name]]
    if len(combined) != n or sorted(combined) != list(range(n)):
        raise ValueError("Split must contain each source row exactly once.")


def load_data():
    path = ROOT / "HW-1" / "nyt.csv"
    df = pd.read_csv(path, keep_default_na=False)
    if not {"text", "label"}.issubset(df.columns) or (df.text.str.strip() == "").any() or (df.label == "").any():
        raise ValueError("NYT requires nonempty text and label columns.")
    digest = sha256(path)
    split_path = RESULTS / "split.json"
    if split_path.exists():
        saved = json.loads(split_path.read_text(encoding="utf-8"))
        if saved["source_sha256"] != digest or saved["seed"] != SEED:
            raise ValueError("Source/seed changed: do not mix existing experiment results.")
        split = saved["indices"]
    else:
        split = make_split(df.label)
        write_json(split_path, {"seed": SEED, "source_sha256": digest, "indices": split})
    validate_split(split, len(df))
    return df, split, sorted(df.label.unique().tolist())


def scores(truth, predicted, labels):
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def environment():
    names = ["numpy", "pandas", "scikit-learn", "gensim", "torch", "transformers"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in names}}


def save_result(name, df, split, labels, val_pred, test_pred, details):
    output = {
        "method": name, "seed": SEED, "labels": labels,
        "source_sha256": sha256(ROOT / "HW-1" / "nyt.csv"),
        "split_sha256": sha256(RESULTS / "split.json"),
        "validation": scores(df.iloc[split["validation"]].label, val_pred, labels),
        "test": scores(df.iloc[split["test"]].label, test_pred, labels),
        "details": details, "environment": environment(),
    }
    write_json(RESULTS / f"{name}.json", output)
    pd.DataFrame({"row_id": split["test"], "true_label": df.iloc[split["test"]].label.to_numpy(),
                  "predicted_label": test_pred}).to_csv(RESULTS / f"{name}_predictions.csv", index=False)
    print(name, json.dumps({k: output["test"][k] for k in ("accuracy", "macro_f1")}), flush=True)


def prepare():
    df, split, labels = load_data()
    text_sets = {s: set(df.iloc[ids].text) for s, ids in split.items()}
    stats = {"rows": len(df), "labels": labels, "seed": SEED,
             "source_sha256": sha256(ROOT / "HW-1" / "nyt.csv"),
             "duplicate_rows": int(df.text.duplicated().sum()),
             "split_counts": {s: df.iloc[ids].label.value_counts().to_dict() for s, ids in split.items()},
             "split_sizes": {s: len(ids) for s, ids in split.items()},
             "identical_text_overlap": {f"{a}/{b}": len(text_sets[a] & text_sets[b])
                 for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}}
    ag = pd.read_csv(ROOT / "HW-1" / "ag.csv", keep_default_na=False)
    stats["ag_rows"] = len(ag)
    stats["ag_sha256"] = sha256(ROOT / "HW-1" / "ag.csv")
    stats["ag_nyt_exact_text_overlap"] = len(set(ag.text) & set(df.text))
    write_json(RESULTS / "data_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
