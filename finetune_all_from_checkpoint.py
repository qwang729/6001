import argparse
import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F

from train_scratch_lstm import (
    BucketBatcher,
    EncodedDataset,
    TextLSTMClassifier,
    build_vocab,
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    read_unlabel,
    set_seed,
    tokenize,
    weighted_bce,
    write_submission,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--out-checkpoint", default=None)
    return parser.parse_args()


def train_all(model, dataset: EncodedDataset, cfg: dict, epochs: int, lr: float, device):
    batcher = BucketBatcher(dataset, cfg["batch_size"], True, cfg["seed"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.get("weight_decay", 0.0),
        betas=(0.9, 0.999),
    )
    total_steps = max(1, len(batcher) * epochs)
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for batch in batcher:
            x, lengths, labels = tuple(item.to(device, non_blocking=True) for item in batch)
            weights = torch.ones_like(labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x, lengths)
                loss = weighted_bce(logits, labels, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (pred == labels).sum().item()
                total_seen += labels.numel()
                total_loss += loss.item() * labels.numel()

        print(
            f"full epoch {epoch:02d} "
            f"train_loss={total_loss / max(1, total_seen):.5f} "
            f"train_acc={total_correct / max(1, total_seen):.4f}"
        )


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]
    set_seed(cfg["seed"])

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextLSTMClassifier(
        vocab_size=len(idx2word),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    dataset = make_dataset(train_texts, train_labels, word2idx, cfg["seq_len"])
    train_all(model, dataset, cfg, args.epochs, args.lr, device)

    test_data = make_dataset(test_texts, None, word2idx, cfg["seq_len"])
    probs = predict_probs(model, test_data, cfg["batch_size"], device)
    write_submission(test_ids, probs, args.submission)

    out_checkpoint = args.out_checkpoint or os.path.join(cfg.get("out_dir", "."), "full_finetuned_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": cfg,
            "source_checkpoint": args.checkpoint,
            "full_finetune_epochs": args.epochs,
            "full_finetune_lr": args.lr,
        },
        out_checkpoint,
    )
    print(f"wrote {args.submission}")
    print(f"saved {out_checkpoint}")


if __name__ == "__main__":
    main()
