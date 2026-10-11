"""Task 3: full BERT-base fine-tuning for exactly three epochs."""
import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .common import ARTIFACTS, RESULTS, SEED, load_data, save_result, write_json

MODEL_ID = "google-bert/bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
MAX_LENGTH = 64
EPOCHS = 3


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.encodings.items()}, "labels": self.labels[i]}


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predictions = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        predictions.extend(model(**batch).logits.argmax(-1).cpu().tolist())
    return np.asarray(predictions)


def run(args):
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df, split, labels = load_data()
    label2id = {label: i for i, label in enumerate(labels)}
    source = str(args.model_dir) if args.model_dir else MODEL_ID
    load_options = {"local_files_only": True} if args.model_dir else {"revision": REVISION}
    tokenizer = AutoTokenizer.from_pretrained(source, **load_options)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id,
        attn_implementation="eager", **load_options).to(device)
    loaders = {}
    generator = torch.Generator().manual_seed(SEED)
    for s, ids in split.items():
        encoded = tokenizer(df.iloc[ids].text.tolist(), truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        targets = torch.tensor([label2id[x] for x in df.iloc[ids].label], dtype=torch.long)
        loaders[s] = DataLoader(EncodedDataset(encoded, targets), batch_size=args.batch_size,
                               shuffle=s == "train", generator=generator if s == "train" else None,
                               num_workers=0, pin_memory=device.type == "cuda")
    # Biases and LayerNorm weights are conventionally excluded from weight decay.
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in ("bias", "LayerNorm.weight"))], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in ("bias", "LayerNorm.weight"))], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=2e-5)
    steps_per_epoch = math.ceil(len(loaders["train"]) / args.accumulation_steps)
    total_steps = steps_per_epoch * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(0.1 * total_steps), total_steps)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score = -1.0
    history = []
    checkpoint = ARTIFACTS / "bert_best"
    checkpoint.mkdir(parents=True, exist_ok=True)
    print(f"BERT on {device}: {len(split['train'])} training rows, {EPOCHS} epochs", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        seen = 0
        for step, batch in enumerate(loaders["train"], start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            size = batch["labels"].shape[0]
            # Weight by examples in this accumulation group, including a short final batch.
            group_start = ((step - 1) // args.accumulation_steps) * args.accumulation_steps * args.batch_size
            group_size = min(args.accumulation_steps * args.batch_size, len(split["train"]) - group_start)
            context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
            with context:
                output = model(**batch)
                loss = output.loss * size / group_size
            scaler.scale(loss).backward()
            loss_sum += output.loss.item() * size
            seen += size
            if step % args.accumulation_steps == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"epoch={epoch} batch={step}/{len(loaders['train'])} mean_loss={loss_sum / seen:.4f}", flush=True)
        val_ids = predict(model, loaders["validation"], device)
        val_pred = np.asarray(labels)[val_ids]
        score = float(f1_score(df.iloc[split["validation"]].label, val_pred, labels=labels, average="macro", zero_division=0))
        record = {"epoch": epoch, "train_loss": loss_sum / seen, "validation_macro_f1": score,
                  "elapsed_seconds": time.perf_counter() - started}
        history.append(record)
        print(record, flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
        write_json(RESULTS / "bert_history.json", history)
    # Select checkpoint on validation only; all three training epochs are completed.
    del model, optimizer
    if amp:
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, attn_implementation="eager").to(device)
    val_pred = np.asarray(labels)[predict(model, loaders["validation"], device)]
    test_pred = np.asarray(labels)[predict(model, loaders["test"], device)]
    details = {"model_id": MODEL_ID, "revision": REVISION, "max_length": MAX_LENGTH, "epochs": EPOCHS,
               "batch_size": args.batch_size, "gradient_accumulation_steps": args.accumulation_steps,
               "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.1, "gradient_clip": 1.0,
               "best_epoch": best_epoch, "history": history, "device": str(device), "amp_fp16": amp,
               "gpu": torch.cuda.get_device_name(device) if amp else None, "seconds": time.perf_counter() - started,
               "selection": "highest validation Macro-F1; ties prefer earlier epoch", "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    save_result("bert", df, split, labels, val_pred, test_pred, details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        parser.error("Batch size and accumulation steps must be positive.")
    run(args)
