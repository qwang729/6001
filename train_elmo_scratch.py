import argparse
import csv
import math
import os
import random
from typing import List, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F

from train_scratch_lstm import (
    BucketBatcher,
    EncodedDataset,
    encode,
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    read_unlabel,
    set_seed,
    split_indices,
    subset_dataset,
    tokenize,
    write_submission,
)
from train_ulmfit_scratch import (
    EOS,
    ScratchLanguageModel,
    UniLSTMEncoder,
    batchify,
    train_language_model,
)


def build_reverse_lm_stream(texts: Sequence[str], word2idx: dict, seq_len: int) -> List[int]:
    eos_id = word2idx[EOS]
    ids: List[int] = []
    for text in texts:
        tokens = tokenize(text)
        if len(tokens) > seq_len:
            head = seq_len // 2
            tail = seq_len - head
            tokens = tokens[:head] + tokens[-tail:]
        doc_ids = [word2idx.get(token, 1) for token in reversed(tokens)]
        ids.extend(doc_ids)
        ids.append(eos_id)
    return ids


class ELMoStyleClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        word_dropout: float,
        pad_idx: int = 0,
        unk_idx: int = 1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout = word_dropout
        self.fwd_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.bwd_embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.fwd_encoder = UniLSTMEncoder(emb_dim, hidden_dim, layers, dropout)
        self.bwd_encoder = UniLSTMEncoder(emb_dim, hidden_dim, layers, dropout)
        self.emb_dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * 2
        self.layer_mix = nn.Parameter(torch.zeros(layers))
        self.attn = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.conv3 = nn.Conv1d(out_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        feature_dim = out_dim * 4 + hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1),
        )

    def load_lms(self, fwd_lm: ScratchLanguageModel, bwd_lm: ScratchLanguageModel) -> None:
        self.fwd_embedding.load_state_dict(fwd_lm.embedding.state_dict())
        self.bwd_embedding.load_state_dict(bwd_lm.embedding.state_dict())
        self.fwd_encoder.load_state_dict(fwd_lm.encoder.state_dict())
        self.bwd_encoder.load_state_dict(bwd_lm.encoder.state_dict())

    def _word_dropout(self, x):
        if not self.training or self.word_dropout <= 0:
            return x
        mask = (torch.rand(x.shape, device=x.device) < self.word_dropout) & (x != self.pad_idx)
        return x.masked_fill(mask, self.unk_idx)

    def _mix_layers(self, fwd_raw, bwd_raw):
        weights = F.softmax(self.layer_mix, dim=0)
        mixed = None
        for i, weight in enumerate(weights):
            bwd = torch.flip(bwd_raw[i], dims=[1])
            layer = torch.cat([fwd_raw[i], bwd], dim=2)
            mixed = layer * weight if mixed is None else mixed + layer * weight
        return mixed

    def forward(self, x, lengths):
        x = self._word_dropout(x)
        fwd_emb = self.emb_dropout(self.fwd_embedding(x))
        fwd_out, _, fwd_raw = self.fwd_encoder(fwd_emb, lengths, None)

        bwd_x = torch.flip(x, dims=[1])
        bwd_lengths = lengths
        bwd_emb = self.emb_dropout(self.bwd_embedding(bwd_x))
        bwd_out, _, bwd_raw = self.bwd_encoder(bwd_emb, bwd_lengths, None)
        out = self._mix_layers(fwd_raw, bwd_raw)

        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        last_idx = (lengths - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last = out.gather(1, last_idx).squeeze(1)
        first = out[:, 0, :]
        max_pool = out.masked_fill(~mask.unsqueeze(2), -1e4).max(dim=1).values
        mean_pool = (
            out.masked_fill(~mask.unsqueeze(2), 0.0).sum(dim=1)
            / lengths.clamp_min(1).to(out.dtype).unsqueeze(1)
        )
        attn_score = self.attn(out).squeeze(2).masked_fill(~mask, -1e4)
        attn = F.softmax(attn_score, dim=1).unsqueeze(2)
        attn_pool = (out * attn).sum(dim=1)
        conv_pool = F.adaptive_max_pool1d(F.gelu(self.conv3(out.transpose(1, 2))), 1).squeeze(2)
        return self.head(torch.cat([first + last, max_pool, mean_pool, attn_pool, conv_pool], dim=1)).squeeze(1)


def evaluate(model, batcher, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in batcher:
            x, lengths, labels = [item.to(device, non_blocking=True) for item in batch]
            logits = model(x, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * labels.numel()
            total_correct += (pred == labels).sum().item()
            total += labels.numel()
    return total_loss / total, total_correct / total


def train_classifier(model, train_data: EncodedDataset, valid_data: EncodedDataset, args, device):
    train_batcher = BucketBatcher(train_data, args.batch_size, True, args.seed)
    valid_batcher = BucketBatcher(valid_data, args.batch_size, False, args.seed)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.fwd_embedding.parameters()) + list(model.bwd_embedding.parameters()), "lr": args.clf_lr * 0.2},
            {"params": list(model.fwd_encoder.parameters()) + list(model.bwd_encoder.parameters()), "lr": args.clf_lr * 0.45},
            {"params": list(model.head.parameters()) + list(model.attn.parameters()) + list(model.conv3.parameters()) + [model.layer_mix], "lr": args.clf_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_batcher) * args.clf_epochs
    warmup = max(1, int(total_steps * 0.08))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_acc = -1.0
    best_path = os.path.join(args.out_dir, "elmo_clf_best.pt")
    for epoch in range(1, args.clf_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in train_batcher:
            x, lengths, labels = [item.to(device, non_blocking=True) for item in batch]
            targets = labels * (1 - 2 * args.label_smoothing) + args.label_smoothing
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x, lengths)
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
        val_loss, val_acc = evaluate(model, valid_batcher, device)
        print(
            f"elmo clf epoch {epoch:02d} train_loss={total_loss/max(1,total):.5f} "
            f"train_acc={total_correct/max(1,total):.4f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "val_acc": best_acc}, best_path)
            print(f"saved {best_path} val_acc={best_acc:.4f}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return best_acc


def calibrate_and_write(model, valid_data, test_ids, test_texts, word2idx, args, device):
    valid_probs = predict_probs(model, valid_data, args.batch_size, device)
    labels = valid_data.y.tolist()
    best = (0.0, 0.5)
    for i in range(1, 1000):
        th = i / 1000
        acc = sum((p >= th) == (y >= 0.5) for p, y in zip(valid_probs, labels)) / len(labels)
        if acc > best[0]:
            best = (acc, th)
    test_data = make_dataset(test_texts, None, word2idx, args.seq_len)
    test_probs = predict_probs(model, test_data, args.batch_size, device)
    write_submission(test_ids, test_probs, args.submission)
    with open(args.submission.replace(".csv", "_calibrated.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, prob in zip(test_ids, test_probs):
            writer.writerow([sample_id, int(prob >= best[1])])
    order = sorted(range(len(test_probs)), key=lambda i: test_probs[i], reverse=True)
    labels_balanced = [0] * len(test_probs)
    for i in order[: len(test_probs) // 2]:
        labels_balanced[i] = 1
    with open(args.submission.replace(".csv", "_balanced.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, label in zip(test_ids, labels_balanced):
            writer.writerow([sample_id, label])
    print(f"calibrated_val_acc={best[0]:.4f} threshold={best[1]:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-final", default="runs_ulmfit_full/final_model.pt")
    parser.add_argument("--fwd-lm", default="runs_ulmfit_full/lm_best.pt")
    parser.add_argument("--out-dir", default="runs_elmo_scratch")
    parser.add_argument("--submission", default="submission_elmo.csv")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--unlabel", default="train_unlabel.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--bptt", type=int, default=80)
    parser.add_argument("--lm-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bwd-lm-epochs", type=int, default=2)
    parser.add_argument("--clf-epochs", type=int, default=5)
    parser.add_argument("--bwd-lm-lr", type=float, default=0.0015)
    parser.add_argument("--clf-lr", type=float, default=0.0008)
    parser.add_argument("--dropout", type=float, default=0.38)
    parser.add_argument("--word-dropout", type=float, default=0.04)
    parser.add_argument("--weight-decay", type=float, default=0.00025)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--grad-clip", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-lm-docs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-unlabel", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = torch.load(args.base_final, map_location="cpu", weights_only=False)
    base_args = base["args"]
    word2idx = base["word2idx"]
    idx2word = base["idx2word"]
    emb_dim = base_args["emb_dim"]
    hidden_dim = base_args["hidden_dim"]
    layers = base_args["layers"]
    print(f"device={device} vocab={len(idx2word)} emb={emb_dim} hidden={hidden_dim} layers={layers}")

    train_texts, train_labels = read_train(args.train, args.max_train)
    unlabel_texts = read_unlabel(args.unlabel, args.max_unlabel)
    test_ids, test_texts = read_test(args.test)

    fwd_lm = ScratchLanguageModel(len(idx2word), emb_dim, hidden_dim, layers, base_args.get("dropout", 0.35)).to(device)
    fwd_ckpt = torch.load(args.fwd_lm, map_location=device, weights_only=False)
    fwd_lm.load_state_dict(fwd_ckpt["model"])

    bwd_lm = ScratchLanguageModel(len(idx2word), emb_dim, hidden_dim, layers, base_args.get("dropout", 0.35)).to(device)
    lm_docs = train_texts + unlabel_texts
    random.Random(args.seed).shuffle(lm_docs)
    if args.max_lm_docs:
        lm_docs = lm_docs[: args.max_lm_docs]
    print("building backward LM stream ...")
    bwd_ids = build_reverse_lm_stream(lm_docs, word2idx, args.seq_len)
    bwd_data = batchify(bwd_ids, args.lm_batch_size, device)
    print(f"bwd_lm_tokens={len(bwd_ids)} batches_per_epoch={(bwd_data.size(1)-1)//args.bptt}")
    lm_args = argparse.Namespace(**vars(args))
    lm_args.lm_lr = args.bwd_lm_lr
    lm_args.lm_epochs = args.bwd_lm_epochs
    print("pretraining backward LM ...")
    train_language_model(bwd_lm, bwd_data, lm_args, device)

    full_train = make_dataset(train_texts, train_labels, word2idx, args.seq_len)
    train_idx, valid_idx = split_indices(len(full_train), args.valid_ratio, args.seed)
    train_data = subset_dataset(full_train, train_idx)
    valid_data = subset_dataset(full_train, valid_idx)
    classifier = ELMoStyleClassifier(
        len(idx2word), emb_dim, hidden_dim, layers, args.dropout, args.word_dropout
    ).to(device)
    classifier.load_lms(fwd_lm, bwd_lm)
    print("training ELMo-style classifier ...")
    best_acc = train_classifier(classifier, train_data, valid_data, args, device)
    print(f"best_elmo_val_acc={best_acc:.4f}")
    calibrate_and_write(classifier, valid_data, test_ids, test_texts, word2idx, args, device)
    torch.save(
        {
            "model": classifier.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": vars(args),
            "val_acc": best_acc,
            "model_class": "ELMoStyleClassifier",
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
