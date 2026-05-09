import argparse
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from train_scratch_lstm import (
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    split_indices,
    subset_dataset,
)
from train_ulmfit_scratch import ULMFiTClassifier, train_classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs_ulmfit_full/final_model.pt")
    parser.add_argument("--out-dir", default="runs_ulmfit_continue")
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--clf-epochs", type=int, default=4)
    parser.add_argument("--clf-lr", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    return parser.parse_args()


def calibrate_and_write(model, valid_data, test_ids, test_texts, word2idx, cfg, path, device):
    valid_probs = predict_probs(model, valid_data, cfg["batch_size"], device)
    labels = valid_data.y.tolist()
    best = (0.0, 0.5)
    for i in range(1, 1000):
        th = i / 1000
        acc = sum((p >= th) == (y >= 0.5) for p, y in zip(valid_probs, labels)) / len(labels)
        if acc > best[0]:
            best = (acc, th)

    test_data = make_dataset(test_texts, None, word2idx, cfg["seq_len"])
    test_probs = predict_probs(model, test_data, cfg["batch_size"], device)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, prob in zip(test_ids, test_probs):
            writer.writerow([sample_id, int(prob >= best[1])])
    print(f"calibrated_val_acc={best[0]:.4f} threshold={best[1]:.3f}")
    print(f"wrote {path}")
    return best


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    cfg = dict(cfg)
    cfg["out_dir"] = args.out_dir
    cfg["clf_epochs"] = args.clf_epochs
    cfg["clf_lr"] = args.clf_lr
    cfg["label_smoothing"] = args.label_smoothing

    train_texts, train_labels = read_train(cfg.get("train", "train.csv"), cfg.get("max_train"))
    test_ids, test_texts = read_test(cfg.get("test", "test.csv"))
    word2idx = checkpoint["word2idx"]
    idx2word = checkpoint["idx2word"]

    full_train = make_dataset(train_texts, train_labels, word2idx, cfg["seq_len"])
    train_idx, valid_idx = split_indices(len(full_train), cfg.get("valid_ratio", 0.1), cfg.get("seed", 2029))
    train_data = subset_dataset(full_train, train_idx)
    valid_data = subset_dataset(full_train, valid_idx)

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

    from argparse import Namespace

    print("continuing ULMFiT classifier fine-tune ...")
    best_acc = train_classifier(model, train_data, valid_data, Namespace(**cfg), device)
    print(f"continued_best_val_acc={best_acc:.4f}")
    best_path = os.path.join(args.out_dir, "ulmfit_clf_best.pt")
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    cal_acc, threshold = calibrate_and_write(
        model, valid_data, test_ids, test_texts, word2idx, cfg, args.submission, device
    )
    torch.save(
        {
            "model": model.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": cfg,
            "val_acc": best_acc,
            "calibrated_val_acc": cal_acc,
            "threshold": threshold,
            "source_checkpoint": args.checkpoint,
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
