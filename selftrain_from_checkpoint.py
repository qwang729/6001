import argparse
import os
from argparse import Namespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from train_scratch_lstm import (
    TextLSTMClassifier,
    build_vocab,
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    read_unlabel,
    split_indices,
    subset_dataset,
    tokenize,
    train_model,
    write_submission,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="runs_selftrain_fixed")
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--threshold", type=float, default=0.995)
    parser.add_argument("--per-class-limit", type=int, default=8000)
    parser.add_argument("--pseudo-weight", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    train_texts, train_labels = read_train(cfg.get("train", "train.csv"), cfg.get("max_train"))
    unlabel_texts = read_unlabel(cfg.get("unlabel", "train_unlabel.csv"), cfg.get("max_unlabel"))
    test_ids, test_texts = read_test(cfg.get("test", "test.csv"))

    vocab_texts = train_texts + unlabel_texts
    if cfg.get("include_test_vocab", False):
        vocab_texts += test_texts
    word2idx, idx2word = build_vocab(
        [tokenize(text) for text in vocab_texts],
        cfg.get("min_count", 2),
        cfg.get("max_vocab", 90000),
    )

    full_train = make_dataset(train_texts, train_labels, word2idx, cfg["seq_len"])
    train_idx, valid_idx = split_indices(
        len(full_train), cfg.get("valid_ratio", 0.1), cfg.get("seed", 2026)
    )
    valid_data = subset_dataset(full_train, valid_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextLSTMClassifier(
        vocab_size=len(idx2word),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    print("predicting unlabeled for fixed self-training ...")
    unlabel_data = make_dataset(unlabel_texts, None, word2idx, cfg["seq_len"])
    probs = predict_probs(model, unlabel_data, cfg["batch_size"], device)

    pos = [(i, p) for i, p in enumerate(probs) if p >= args.threshold]
    neg = [(i, p) for i, p in enumerate(probs) if p <= 1.0 - args.threshold]
    pos.sort(key=lambda item: item[1], reverse=True)
    neg.sort(key=lambda item: item[1])
    pos = pos[: args.per_class_limit]
    neg = neg[: args.per_class_limit]
    selected = pos + neg
    print(
        f"pseudo_selected={len(selected)} pos={len(pos)} neg={len(neg)} "
        f"threshold={args.threshold}"
    )

    pseudo_texts = [unlabel_texts[i] for i, _ in selected]
    pseudo_labels = [1.0 if p >= 0.5 else 0.0 for _, p in selected]
    combined_texts = [train_texts[i] for i in train_idx] + pseudo_texts
    combined_labels = [float(train_labels[i]) for i in train_idx] + pseudo_labels
    combined_weights = [1.0] * len(train_idx) + [args.pseudo_weight] * len(pseudo_texts)
    train_plus_pseudo = make_dataset(
        combined_texts,
        combined_labels,
        word2idx,
        cfg["seq_len"],
        combined_weights,
    )

    train_args = Namespace(**cfg)
    train_args.out_dir = args.out_dir
    train_args.epochs = args.epochs
    train_args.lr = cfg.get("lr", 0.0015)
    train_args.pseudo_weight = args.pseudo_weight

    print("fine-tuning with fixed pseudo labels ...")
    self_acc = train_model(
        model,
        train_plus_pseudo,
        valid_data,
        train_args,
        device,
        "selftrain_fixed",
        lr=args.lr,
    )
    best_path = os.path.join(args.out_dir, "selftrain_fixed_best.pt")
    best_acc = checkpoint.get("val_acc", 0.0)
    if self_acc >= best_acc:
        final_path = best_path
        final_acc = self_acc
    else:
        final_path = args.checkpoint
        final_acc = best_acc
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])

    print(f"using checkpoint={final_path} val_acc={final_acc}")
    test_data = make_dataset(test_texts, None, word2idx, cfg["seq_len"])
    test_probs = predict_probs(model, test_data, cfg["batch_size"], device)
    write_submission(test_ids, test_probs, args.submission)
    torch.save(
        {
            "model": model.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": cfg,
            "val_acc": final_acc,
            "source_checkpoint": final_path,
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"wrote {args.submission}")
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
