import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F

from train_scratch_lstm import (
    BucketBatcher,
    EncodedDataset,
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    split_indices,
    subset_dataset,
    tokenize,
)
from train_ulmfit_scratch import ULMFiTClassifier, evaluate_classifier


HASH_BASE = 1000003


def stable_hash_ids(parts: Sequence[int], buckets: int) -> int:
    h = 1469598103934665603
    for part in parts:
        h ^= part + 0x9E3779B97F4A7C15
        h = (h * HASH_BASE) & 0xFFFFFFFFFFFFFFFF
    return 1 + (h % (buckets - 1))


def ngram_ids_for_text(
    text: str,
    word2idx: dict,
    buckets: int,
    max_ngrams: int,
    max_tokens: int,
    orders: Tuple[int, ...],
) -> List[int]:
    token_ids = [word2idx.get(token, 1) for token in tokenize(text)[:max_tokens]]
    ids = set()
    for order in orders:
        if len(token_ids) < order:
            continue
        for i in range(len(token_ids) - order + 1):
            ids.add(stable_hash_ids(token_ids[i : i + order], buckets))
    ids = sorted(ids)
    if len(ids) > max_ngrams:
        # Keep a deterministic spread over the full document instead of only the beginning.
        step = len(ids) / max_ngrams
        ids = [ids[int(i * step)] for i in range(max_ngrams)]
    return ids


@dataclass
class HybridDataset:
    seq: EncodedDataset
    ngrams: torch.Tensor
    ngram_counts: torch.Tensor

    def __len__(self):
        return len(self.seq)


class HybridBatcher:
    def __init__(self, dataset: HybridDataset, batch_size: int, shuffle: bool, seed: int):
        self.seq_batcher = BucketBatcher(dataset.seq, batch_size, shuffle, seed)
        self.dataset = dataset

    def __len__(self):
        return len(self.seq_batcher)

    def __iter__(self):
        for batch in self.seq_batcher:
            x = batch[0]
            # Recover selected rows by matching sequence rows is unsafe; use a direct local batcher instead.
            raise RuntimeError("HybridBatcher should not be used")


class HybridIndexBatcher:
    def __init__(self, dataset: HybridDataset, batch_size: int, shuffle: bool, seed: int, bucket_size: int = 2048):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = bucket_size
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        if self.shuffle:
            rng.shuffle(indices)
            buckets = [indices[i : i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
            for bucket in buckets:
                bucket.sort(key=lambda idx: int(self.dataset.seq.lengths[idx]), reverse=True)
            rng.shuffle(buckets)
            indices = [idx for bucket in buckets for idx in bucket]
        else:
            indices.sort(key=lambda idx: int(self.dataset.seq.lengths[idx]), reverse=True)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            idx = torch.tensor(batch_idx, dtype=torch.long)
            fields = [
                self.dataset.seq.x.index_select(0, idx),
                self.dataset.seq.lengths.index_select(0, idx),
                self.dataset.ngrams.index_select(0, idx),
                self.dataset.ngram_counts.index_select(0, idx),
            ]
            if self.dataset.seq.y is not None:
                fields.append(self.dataset.seq.y.index_select(0, idx))
            yield tuple(fields)


def make_hybrid_dataset(
    texts: Sequence[str],
    labels: Optional[Sequence[float]],
    word2idx: dict,
    seq_len: int,
    buckets: int,
    max_ngrams: int,
    max_ngram_tokens: int,
    orders: Tuple[int, ...],
) -> HybridDataset:
    seq = make_dataset(texts, labels, word2idx, seq_len)
    rows = []
    counts = []
    for text in texts:
        ids = ngram_ids_for_text(
            text, word2idx, buckets, max_ngrams, max_ngram_tokens, orders
        )
        counts.append(max(1, len(ids)))
        if len(ids) < max_ngrams:
            ids = ids + [0] * (max_ngrams - len(ids))
        rows.append(ids)
    return HybridDataset(
        seq=seq,
        ngrams=torch.tensor(rows, dtype=torch.long),
        ngram_counts=torch.tensor(counts, dtype=torch.float32),
    )


def subset_hybrid(dataset: HybridDataset, indices: Sequence[int]) -> HybridDataset:
    idx = torch.tensor(indices, dtype=torch.long)
    return HybridDataset(
        seq=subset_dataset(dataset.seq, indices),
        ngrams=dataset.ngrams.index_select(0, idx),
        ngram_counts=dataset.ngram_counts.index_select(0, idx),
    )


class ULMFiTNgramClassifier(nn.Module):
    def __init__(
        self,
        base: ULMFiTClassifier,
        buckets: int,
        ngram_dropout: float,
        ngram_scale: float,
    ):
        super().__init__()
        self.base = base
        self.ngram_weight = nn.Embedding(buckets, 1, padding_idx=0)
        self.ngram_bias = nn.Parameter(torch.zeros(1))
        self.mix = nn.Parameter(torch.tensor([1.0, ngram_scale]))
        self.ngram_dropout = ngram_dropout
        nn.init.zeros_(self.ngram_weight.weight)

    def forward(self, x, lengths, ngrams, ngram_counts):
        base_logit = self.base(x, lengths)
        weights = self.ngram_weight(ngrams).squeeze(2)
        mask = (ngrams != 0).to(weights.dtype)
        if self.training and self.ngram_dropout > 0:
            keep = (torch.rand_like(mask) > self.ngram_dropout).to(weights.dtype)
            mask = mask * keep
        ngram_logit = (weights * mask).sum(dim=1) / ngram_counts.sqrt().to(weights.dtype).clamp_min(1.0)
        ngram_logit = ngram_logit + self.ngram_bias
        return self.mix[0] * base_logit + self.mix[1] * ngram_logit


def evaluate(model, batcher, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in batcher:
            x, lengths, ngrams, counts, labels = [item.to(device, non_blocking=True) for item in batch]
            logits = model(x, lengths, ngrams, counts)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * labels.numel()
            total_correct += (pred == labels).sum().item()
            total += labels.numel()
    return total_loss / total, total_correct / total


def collect_probs(model, dataset: HybridDataset, batch_size: int, device):
    batcher = HybridIndexBatcher(dataset, batch_size, False, 1234)
    model.eval()
    indexed = []
    # Reimplement without losing order.
    indices = list(range(len(dataset)))
    indices.sort(key=lambda idx: int(dataset.seq.lengths[idx]), reverse=True)
    probs = [None] * len(dataset)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            idx = torch.tensor(batch_idx, dtype=torch.long)
            x = dataset.seq.x.index_select(0, idx).to(device)
            lengths = dataset.seq.lengths.index_select(0, idx).to(device)
            ngrams = dataset.ngrams.index_select(0, idx).to(device)
            counts = dataset.ngram_counts.index_select(0, idx).to(device)
            batch_probs = torch.sigmoid(model(x, lengths, ngrams, counts)).cpu().tolist()
            for original, prob in zip(batch_idx, batch_probs):
                probs[original] = prob
    return [float(p) for p in probs]


def calibrate(model, valid_data, batch_size, device):
    probs = collect_probs(model, valid_data, batch_size, device)
    labels = valid_data.seq.y.tolist()
    best = (0.0, 0.5)
    for i in range(1, 1000):
        th = i / 1000
        acc = sum((p >= th) == (y >= 0.5) for p, y in zip(probs, labels)) / len(labels)
        if acc > best[0]:
            best = (acc, th)
    return best


def write_submissions(model, test_ids, test_data, args, device):
    probs = collect_probs(model, test_data, args.batch_size, device)
    for suffix, labels in [
        ("", [int(p >= 0.5) for p in probs]),
        ("_balanced", None),
    ]:
        path = args.submission.replace(".csv", f"{suffix}.csv")
        if labels is None:
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            labels = [0] * len(probs)
            for i in order[: len(probs) // 2]:
                labels[i] = 1
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            for sample_id, label in zip(test_ids, labels):
                writer.writerow([sample_id, label])
        print(f"wrote {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs_ulmfit_continue/ulmfit_clf_best.pt")
    parser.add_argument("--base-final", default="runs_ulmfit_full/final_model.pt")
    parser.add_argument("--out-dir", default="runs_ulmfit_ngram")
    parser.add_argument("--submission", default="submission_ulmfit_ngram.csv")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-lr-mult", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--buckets", type=int, default=1048576)
    parser.add_argument("--max-ngrams", type=int, default=900)
    parser.add_argument("--max-ngram-tokens", type=int, default=1400)
    parser.add_argument("--orders", default="1,2,3")
    parser.add_argument("--ngram-dropout", type=float, default=0.08)
    parser.add_argument("--ngram-scale", type=float, default=1.0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--pilot-train-size", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--skip-predict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_final = torch.load(args.base_final, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["args"])
    word2idx = base_final["word2idx"]
    idx2word = base_final["idx2word"]
    train_texts, train_labels = read_train(args.train, args.max_train)
    test_ids, test_texts = read_test(args.test)
    orders = tuple(int(x) for x in args.orders.split(",") if x)
    print("building hybrid datasets ...")
    split_seed = args.split_seed if args.split_seed is not None else cfg.get("seed", args.seed)
    train_idx, valid_idx = split_indices(len(train_texts), args.valid_ratio, split_seed)
    if args.pilot_train_size is not None:
        rng = random.Random(args.seed)
        train_idx = list(train_idx)
        rng.shuffle(train_idx)
        train_idx = sorted(train_idx[: args.pilot_train_size])
        pilot_train_texts = [train_texts[i] for i in train_idx]
        pilot_train_labels = [train_labels[i] for i in train_idx]
        valid_texts = [train_texts[i] for i in valid_idx]
        valid_labels = [train_labels[i] for i in valid_idx]
        train_data = make_hybrid_dataset(
            pilot_train_texts,
            pilot_train_labels,
            word2idx,
            cfg["seq_len"],
            args.buckets,
            args.max_ngrams,
            args.max_ngram_tokens,
            orders,
        )
        valid_data = make_hybrid_dataset(
            valid_texts,
            valid_labels,
            word2idx,
            cfg["seq_len"],
            args.buckets,
            args.max_ngrams,
            args.max_ngram_tokens,
            orders,
        )
        print(
            f"clean pilot train={len(train_data)} valid={len(valid_data)} split_seed={split_seed}"
        )
    else:
        full = make_hybrid_dataset(
            train_texts,
            train_labels,
            word2idx,
            cfg["seq_len"],
            args.buckets,
            args.max_ngrams,
            args.max_ngram_tokens,
            orders,
        )
        train_data = subset_hybrid(full, train_idx)
        valid_data = subset_hybrid(full, valid_idx)
        print(f"full split train={len(train_data)} valid={len(valid_data)} split_seed={split_seed}")
    test_data = None
    if not args.skip_predict:
        test_data = make_hybrid_dataset(
            test_texts,
            None,
            word2idx,
            cfg["seq_len"],
            args.buckets,
            args.max_ngrams,
            args.max_ngram_tokens,
            orders,
        )

    base_model = ULMFiTClassifier(
        len(idx2word),
        cfg["emb_dim"],
        cfg["hidden_dim"],
        cfg["layers"],
        cfg["dropout"],
        word_dropout=cfg.get("word_dropout", 0.04),
    )
    base_model.load_state_dict(checkpoint["model"])
    model = ULMFiTNgramClassifier(base_model, args.buckets, args.ngram_dropout, args.ngram_scale).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.base.parameters(), "lr": args.lr * args.base_lr_mult},
            {"params": model.ngram_weight.parameters(), "lr": args.lr},
            {"params": [model.ngram_bias, model.mix], "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    train_batcher = HybridIndexBatcher(train_data, args.batch_size, True, args.seed)
    total_steps = len(train_batcher) * args.epochs
    warmup = max(1, int(total_steps * 0.08))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_acc = -1.0
    best_path = os.path.join(args.out_dir, "ulmfit_ngram_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in train_batcher:
            x, lengths, ngrams, counts, labels = [item.to(device, non_blocking=True) for item in batch]
            targets = labels * (1 - 2 * args.label_smoothing) + args.label_smoothing
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x, lengths, ngrams, counts)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (pred == labels).sum().item()
                total += labels.numel()
                total_loss += loss.item() * labels.numel()
        valid_batcher = HybridIndexBatcher(valid_data, args.batch_size, False, args.seed)
        val_loss, val_acc = evaluate(model, valid_batcher, device)
        cal_acc, cal_th = calibrate(model, valid_data, args.batch_size, device)
        print(
            f"ngram epoch {epoch:02d} train_loss={total_loss/max(1,total):.5f} "
            f"train_acc={total_correct/max(1,total):.4f} val_acc={val_acc:.4f} "
            f"cal_acc={cal_acc:.4f} cal_th={cal_th:.3f} mix={model.mix.detach().cpu().tolist()}"
        )
        if cal_acc > best_acc:
            best_acc = cal_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "base_cfg": cfg,
                    "word2idx": word2idx,
                    "idx2word": idx2word,
                    "calibrated_val_acc": best_acc,
                    "val_acc": val_acc,
                },
                best_path,
            )
            print(f"saved {best_path} cal_acc={best_acc:.4f}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if not args.skip_predict:
        write_submissions(model, test_ids, test_data, args, device)
    torch.save(
        {
            "model": model.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": vars(args),
            "base_cfg": cfg,
            "calibrated_val_acc": best_acc,
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
