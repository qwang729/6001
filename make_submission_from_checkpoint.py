import argparse
import os

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
    tokenize,
    write_submission,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--submission", default="submission.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["args"]

    train_texts, _ = read_train(cfg.get("train", "train.csv"), cfg.get("max_train"))
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

    test_data = make_dataset(test_texts, None, word2idx, cfg["seq_len"])
    probs = predict_probs(model, test_data, cfg["batch_size"], device)
    write_submission(test_ids, probs, args.submission)

    out_dir = cfg.get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": cfg,
            "val_acc": checkpoint.get("val_acc"),
            "source_checkpoint": args.checkpoint,
        },
        os.path.join(out_dir, "final_model.pt"),
    )
    print(f"wrote {args.submission}")
    print(f"saved {os.path.join(out_dir, 'final_model.pt')}")
    print(f"checkpoint_val_acc={checkpoint.get('val_acc')}")


if __name__ == "__main__":
    main()
