"""Independently recompute every reported test score from saved predictions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.common import RESULTS, load_data, scores, sha256, write_json

METHODS = ["binary", "frequency", "tfidf", "glove", "w2v_ag", "w2v_nyt", "bert"]


def main():
    df, split, labels = load_data()
    rows = []
    details = {}
    train_text = set(df.iloc[split["train"]].text)
    unseen = ~df.iloc[split["test"]].text.isin(train_text).to_numpy()
    for name in METHODS:
        result = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
        pred = pd.read_csv(RESULTS / f"{name}_predictions.csv")
        assert result["split_sha256"] == sha256(RESULTS / "split.json"), name
        assert result["source_sha256"] == sha256(Path(__file__).resolve().parents[1] / "HW-1" / "nyt.csv"), name
        assert pred.row_id.tolist() == split["test"], name
        assert pred.true_label.tolist() == df.iloc[split["test"]].label.tolist(), name
        assert set(pred.predicted_label).issubset(labels), name
        actual = scores(pred.true_label, pred.predicted_label, labels)
        for metric in ("accuracy", "macro_f1"):
            assert abs(actual[metric] - result["test"][metric]) < 1e-12, (name, metric)
        assert actual["confusion_matrix"] == result["test"]["confusion_matrix"], name
        assert np.sum(actual["confusion_matrix"]) == len(split["test"]), name
        rows.append({"method": name, "validation_accuracy": result["validation"]["accuracy"],
                     "validation_macro_f1": result["validation"]["macro_f1"],
                     "test_accuracy": actual["accuracy"], "test_macro_f1": actual["macro_f1"],
                     "seconds": result["details"]["seconds"]})
        details[name] = scores(pred.loc[unseen, "true_label"], pred.loc[unseen, "predicted_label"], labels)
    bert = json.loads((RESULTS / "bert.json").read_text())
    assert bert["details"]["max_length"] == 64
    assert bert["details"]["epochs"] == 3
    assert [x["epoch"] for x in bert["details"]["history"]] == [1, 2, 3]
    pd.DataFrame(rows).to_csv(RESULTS / "summary.csv", index=False)
    write_json(RESULTS / "audit.json", {"status": "passed", "methods": METHODS,
               "test_rows": len(split["test"]), "test_rows_without_exact_train_duplicates": int(unseen.sum()),
               "metrics_without_exact_train_duplicates": details})
    print(pd.DataFrame(rows).to_string(index=False))
    print("All seven results match their saved predictions and shared split.")


if __name__ == "__main__":
    main()
