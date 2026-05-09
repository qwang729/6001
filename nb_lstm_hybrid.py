import argparse
import csv
import math
import os
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from train_scratch_lstm import (
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    split_indices,
    subset_dataset,
    tokenize,
)
from train_ulmfit_scratch import ULMFiTClassifier


HASH_BASE = 1000003


def stable_hash_ids(parts: Sequence[int], buckets: int) -> int:
    h = 1469598103934665603
    for part in parts:
        h ^= part + 0x9E3779B97F4A7C15
        h = (h * HASH_BASE) & 0xFFFFFFFFFFFFFFFF
    return h % buckets


def doc_features(text, word2idx, buckets, max_tokens, orders):
    ids = [word2idx.get(token, 1) for token in tokenize(text)[:max_tokens]]
    feats = set()
    for order in orders:
        if len(ids) < order:
            continue
        for i in range(len(ids) - order + 1):
            feats.add(stable_hash_ids(ids[i : i + order], buckets))
    return feats


def build_nb_ratio(texts, labels, word2idx, indices, buckets, max_tokens, orders, alpha):
    pos = [alpha] * buckets
    neg = [alpha] * buckets
    pos_total = alpha * buckets
    neg_total = alpha * buckets
    for idx in indices:
        feats = doc_features(texts[idx], word2idx, buckets, max_tokens, orders)
        if labels[idx] == 1:
            for feat in feats:
                pos[feat] += 1.0
            pos_total += len(feats)
        else:
            for feat in feats:
                neg[feat] += 1.0
            neg_total += len(feats)
    return [math.log((p / pos_total) / (n / neg_total)) for p, n in zip(pos, neg)]


def nb_scores(texts, word2idx, ratio, buckets, max_tokens, orders):
    scores = []
    for text in texts:
        feats = doc_features(text, word2idx, buckets, max_tokens, orders)
        if feats:
            scores.append(sum(ratio[feat] for feat in feats) / math.sqrt(len(feats)))
        else:
            scores.append(0.0)
    return scores


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def grid_search(base_logits, nb, labels):
    best = (0.0, 1.0, 0.0, 0.0)
    for nb_scale in [-2, -1.5, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.5, 2]:
        for base_scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
            scores = [base_scale * b + nb_scale * n for b, n in zip(base_logits, nb)]
            for th_i in range(1, 1000, 2):
                th = (th_i - 500) / 100.0
                acc = sum((s >= th) == (y == 1) for s, y in zip(scores, labels)) / len(labels)
                if acc > best[0]:
                    best = (acc, base_scale, nb_scale, th)
    return best


def write_submission(path, ids, scores, threshold):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, score in zip(ids, scores):
            writer.writerow([sample_id, int(score >= threshold)])


def write_balanced(path, ids, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    labels = [0] * len(scores)
    for i in order[: len(scores) // 2]:
        labels[i] = 1
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, label in zip(ids, labels):
            writer.writerow([sample_id, label])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs_ulmfit_continue/ulmfit_clf_best.pt")
    parser.add_argument("--base-final", default="runs_ulmfit_full/final_model.pt")
    parser.add_argument("--submission", default="submission_nb_lstm.csv")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--buckets", type=int, default=1048576)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--orders", default="1,2")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=2029)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = torch.load(args.base_final, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    word2idx = base["word2idx"]
    idx2word = base["idx2word"]
    orders = tuple(int(x) for x in args.orders.split(",") if x)

    train_texts, train_labels = read_train(args.train)
    train_idx, valid_idx = split_indices(len(train_texts), args.valid_ratio, args.split_seed)
    print(f"train_split={len(train_idx)} valid={len(valid_idx)}")
    ratio = build_nb_ratio(
        train_texts,
        train_labels,
        word2idx,
        train_idx,
        args.buckets,
        args.max_tokens,
        orders,
        args.alpha,
    )
    valid_texts = [train_texts[i] for i in valid_idx]
    valid_labels = [train_labels[i] for i in valid_idx]
    valid_nb = nb_scores(valid_texts, word2idx, ratio, args.buckets, args.max_tokens, orders)

    valid_data = make_dataset(valid_texts, valid_labels, word2idx, cfg["seq_len"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ULMFiTClassifier(
        len(idx2word),
        cfg["emb_dim"],
        cfg["hidden_dim"],
        cfg["layers"],
        cfg["dropout"],
        word_dropout=cfg.get("word_dropout", 0.04),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    valid_probs = predict_probs(model, valid_data, cfg["batch_size"], device)
    valid_logits = [logit(p) for p in valid_probs]

    nb_best = (0.0, 0.0)
    for th_i in range(-500, 501):
        th = th_i / 100.0
        acc = sum((s >= th) == (y == 1) for s, y in zip(valid_nb, valid_labels)) / len(valid_labels)
        if acc > nb_best[0]:
            nb_best = (acc, th)
    base_acc = sum((p >= 0.5) == (y == 1) for p, y in zip(valid_probs, valid_labels)) / len(valid_labels)
    mix_best = grid_search(valid_logits, valid_nb, valid_labels)
    print(f"base_acc@0.5={base_acc:.4f}")
    print(f"nb_best_acc={nb_best[0]:.4f} nb_threshold={nb_best[1]:.3f}")
    print(
        f"mix_best_acc={mix_best[0]:.4f} base_scale={mix_best[1]} "
        f"nb_scale={mix_best[2]} threshold={mix_best[3]:.3f}"
    )

    test_ids, test_texts = read_test(args.test)
    test_data = make_dataset(test_texts, None, word2idx, cfg["seq_len"])
    test_probs = predict_probs(model, test_data, cfg["batch_size"], device)
    test_logits = [logit(p) for p in test_probs]
    test_nb = nb_scores(test_texts, word2idx, ratio, args.buckets, args.max_tokens, orders)
    _, base_scale, nb_scale, threshold = mix_best
    test_scores = [base_scale * b + nb_scale * n for b, n in zip(test_logits, test_nb)]
    write_submission(args.submission, test_ids, test_scores, threshold)
    write_balanced(args.submission.replace(".csv", "_balanced.csv"), test_ids, test_scores)
    print(f"wrote {args.submission}")
    print(f"wrote {args.submission.replace('.csv', '_balanced.csv')}")


if __name__ == "__main__":
    main()
