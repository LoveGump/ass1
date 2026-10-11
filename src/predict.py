"""Classify one text using a locally trained experiment artifact."""
import argparse

from .common import ARTIFACTS, tokenize


def predict(method, text):
    if method == "bert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = ARTIFACTS / "bert_best"
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        model.eval()
        encoded = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
        with torch.inference_mode():
            label = model(**encoded).logits.argmax(-1).item()
        return model.config.id2label[label]
    import joblib
    from .classical import mean_embeddings
    artifact = joblib.load(ARTIFACTS / f"{method}.joblib")
    if "vectorizer" in artifact:
        features = artifact["vectorizer"].transform([text])
    else:
        features, _ = mean_embeddings([tokenize(text)], artifact["vectors"])
        features = artifact["scaler"].transform(features)
    return artifact["classifier"].predict(features)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["binary", "frequency", "tfidf", "glove", "w2v_ag", "w2v_nyt", "bert"], required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(predict(args.method, args.text))
