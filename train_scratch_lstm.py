import argparse
import csv
import html
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# Some Windows PyTorch installs load duplicate OpenMP runtimes. This needs to be
# set before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.IGNORECASE)
PAD = "<pad>"
UNK = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = text.replace("\x85", " ")
    return text.lower()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(text))


def read_train(path: str, max_rows: Optional[int] = None) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0 and len(row) == 2 and row[0] == "0" and row[1] in {"0", "1"}:
                continue
            if len(row) != 2:
                continue
            texts.append(row[0])
            labels.append(int(row[1]))
            if max_rows and len(texts) >= max_rows:
                break
    return texts, labels


def read_unlabel(path: str, max_rows: Optional[int] = None) -> List[str]:
    texts: List[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0 and len(row) == 1 and row[0] == "0":
                continue
            if len(row) != 1:
                continue
            texts.append(row[0])
            if max_rows and len(texts) >= max_rows:
                break
    return texts


def read_test(path: str) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            ids.append(row[0])
            texts.append(row[1])
    return ids, texts


def build_vocab(
    tokenized_texts: Iterable[Sequence[str]],
    min_count: int,
    max_vocab: int,
) -> Tuple[dict, List[str]]:
    counter: Counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    idx2word = [PAD, UNK]
    for word, count in counter.most_common():
        if count < min_count:
            break
        if len(idx2word) >= max_vocab:
            break
        idx2word.append(word)

    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    return word2idx, idx2word


def encode(tokens: Sequence[str], word2idx: dict, seq_len: int) -> Tuple[List[int], int]:
    if len(tokens) > seq_len:
        # Keep the beginning and ending. Movie reviews often summarize sentiment at both ends.
        head = seq_len // 2
        tail = seq_len - head
        tokens = list(tokens[:head]) + list(tokens[-tail:])
    ids = [word2idx.get(token, 1) for token in tokens]
    length = len(ids)
    if length < seq_len:
        ids.extend([0] * (seq_len - length))
    return ids, max(1, length)


@dataclass
class EncodedDataset:
    x: torch.Tensor
    lengths: torch.Tensor
    y: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.x.size(0)


class BucketBatcher:
    def __init__(
        self,
        dataset: EncodedDataset,
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_size: int = 2048,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = bucket_size
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        if self.shuffle:
            rng.shuffle(indices)
            buckets = [
                indices[i : i + self.bucket_size]
                for i in range(0, len(indices), self.bucket_size)
            ]
            for bucket in buckets:
                bucket.sort(key=lambda idx: int(self.dataset.lengths[idx]), reverse=True)
            rng.shuffle(buckets)
            indices = [idx for bucket in buckets for idx in bucket]
        else:
            indices.sort(key=lambda idx: int(self.dataset.lengths[idx]), reverse=True)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            idx_tensor = torch.tensor(batch_idx, dtype=torch.long)
            fields = [
                self.dataset.x.index_select(0, idx_tensor),
                self.dataset.lengths.index_select(0, idx_tensor),
            ]
            if self.dataset.y is not None:
                fields.append(self.dataset.y.index_select(0, idx_tensor))
            if self.dataset.weights is not None:
                fields.append(self.dataset.weights.index_select(0, idx_tensor))
            yield tuple(fields)


def make_dataset(
    texts: Sequence[str],
    labels: Optional[Sequence[float]],
    word2idx: dict,
    seq_len: int,
    weights: Optional[Sequence[float]] = None,
) -> EncodedDataset:
    ids: List[List[int]] = []
    lengths: List[int] = []
    for text in texts:
        encoded, length = encode(tokenize(text), word2idx, seq_len)
        ids.append(encoded)
        lengths.append(length)

    y_tensor = None
    if labels is not None:
        y_tensor = torch.tensor(labels, dtype=torch.float32)

    w_tensor = None
    if weights is not None:
        w_tensor = torch.tensor(weights, dtype=torch.float32)

    return EncodedDataset(
        x=torch.tensor(ids, dtype=torch.long),
        lengths=torch.tensor(lengths, dtype=torch.long),
        y=y_tensor,
        weights=w_tensor,
    )


def split_indices(n: int, valid_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    valid_n = int(n * valid_ratio)
    valid_idx = sorted(indices[:valid_n])
    train_idx = sorted(indices[valid_n:])
    return train_idx, valid_idx


def subset_dataset(dataset: EncodedDataset, indices: Sequence[int]) -> EncodedDataset:
    idx = torch.tensor(indices, dtype=torch.long)
    return EncodedDataset(
        x=dataset.x.index_select(0, idx),
        lengths=dataset.lengths.index_select(0, idx),
        y=dataset.y.index_select(0, idx) if dataset.y is not None else None,
        weights=dataset.weights.index_select(0, idx)
        if dataset.weights is not None
        else None,
    )


class ScratchLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.x2h = nn.Linear(input_dim, 4 * hidden_dim)
        self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        return self.forward_from_input(self.x2h(x_t), state)

    def forward_from_input(
        self, x_gate_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ):
        h, c = state
        gates = x_gate_t + self.h2h(h)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + 1.0)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ScratchBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fw = ScratchLSTMCell(input_dim, hidden_dim)
        self.bw = ScratchLSTMCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _run_direction(
        self,
        cell: ScratchLSTMCell,
        x: torch.Tensor,
        lengths: torch.Tensor,
        reverse: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        h = x.new_zeros(batch, self.hidden_dim)
        c = x.new_zeros(batch, self.hidden_dim)
        x_gates = cell.x2h(x)
        outputs = []
        steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for t in steps:
            h_new, c_new = cell.forward_from_input(x_gates[:, t, :], (h, c))
            active = (lengths > t).to(x.dtype).unsqueeze(1)
            h = active * h_new + (1.0 - active) * h
            c = active * c_new + (1.0 - active) * c
            outputs.append(h)

        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1), h

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        fw_out, fw_last = self._run_direction(self.fw, x, lengths, reverse=False)
        bw_out, bw_last = self._run_direction(self.bw, x, lengths, reverse=True)
        out = torch.cat([fw_out, bw_out], dim=2)
        out = self.dropout(out)
        last = torch.cat([fw_last, bw_last], dim=1)
        return out, last


class TextLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = ScratchBiLSTM(emb_dim, hidden_dim, dropout)
        out_dim = hidden_dim * 2
        self.attn = nn.Linear(out_dim, 1)
        self.fc = nn.Sequential(
            nn.LayerNorm(out_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.emb_dropout(self.embedding(x))
        out, last = self.lstm(emb, lengths)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        attn_score = self.attn(out).squeeze(2).masked_fill(~mask, -1e4)
        attn_weight = F.softmax(attn_score, dim=1).unsqueeze(2)
        attn_pool = torch.sum(out * attn_weight, dim=1)

        masked_out = out.masked_fill(~mask.unsqueeze(2), -1e4)
        max_pool = masked_out.max(dim=1).values

        features = torch.cat([last, attn_pool, max_pool], dim=1)
        return self.fc(features).squeeze(1)


def move_batch(batch, device):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def weighted_bce(logits, labels, weights):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def smooth_binary_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return labels
    return labels * (1.0 - 2.0 * smoothing) + smoothing


def evaluate(model, batcher, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in batcher:
            x, lengths, labels = move_batch(batch, device)
            logits = model(x, lengths)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * labels.numel()
            total_correct += (pred == labels).sum().item()
            total += labels.numel()
    return total_loss / total, total_correct / total


def train_model(
    model,
    train_data: EncodedDataset,
    valid_data: EncodedDataset,
    args,
    device,
    stage_name: str,
    lr: Optional[float] = None,
) -> float:
    train_batcher = BucketBatcher(train_data, args.batch_size, True, args.seed)
    valid_batcher = BucketBatcher(valid_data, args.batch_size, False, args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr if lr is None else lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = max(1, len(train_batcher) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_acc = -1.0
    best_path = os.path.join(args.out_dir, f"{stage_name}_best.pt")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch in train_batcher:
            if len(batch) == 4:
                x, lengths, labels, weights = move_batch(batch, device)
            else:
                x, lengths, labels = move_batch(batch, device)
                weights = torch.ones_like(labels)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x, lengths)
                targets = smooth_binary_labels(
                    labels, getattr(args, "label_smoothing", 0.0)
                )
                loss = weighted_bce(logits, targets, weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (pred == labels).sum().item()
                total_seen += labels.numel()
                total_loss += loss.item() * labels.numel()

        val_loss, val_acc = evaluate(model, valid_batcher, device)
        train_acc = total_correct / max(1, total_seen)
        print(
            f"{stage_name} epoch {epoch:02d} "
            f"train_loss={total_loss / max(1, total_seen):.5f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.5f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "val_acc": best_acc,
                    "stage": stage_name,
                },
                best_path,
            )
            print(f"saved {best_path} val_acc={best_acc:.4f}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return best_acc


def predict_probs(model, dataset: EncodedDataset, batch_size: int, device) -> List[float]:
    model.eval()
    probs: List[Optional[float]] = [None] * len(dataset)
    indices = list(range(len(dataset)))
    indices.sort(key=lambda idx: int(dataset.lengths[idx]), reverse=True)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            idx_tensor = torch.tensor(batch_idx, dtype=torch.long)
            x = dataset.x.index_select(0, idx_tensor).to(device, non_blocking=True)
            lengths = dataset.lengths.index_select(0, idx_tensor).to(device, non_blocking=True)
            logits = model(x, lengths)
            batch_probs = torch.sigmoid(logits).detach().cpu().tolist()
            for original_idx, prob in zip(batch_idx, batch_probs):
                probs[original_idx] = prob
    return [float(prob) for prob in probs if prob is not None]


def write_submission(ids: Sequence[str], probs: Sequence[float], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for sample_id, prob in zip(ids, probs):
            writer.writerow([sample_id, int(prob >= 0.5)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--unlabel", default="train_unlabel.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--out-dir", default="runs_scratch_lstm")
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=90000)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--self-train", action="store_true")
    parser.add_argument("--self-threshold", type=float, default=0.985)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    parser.add_argument("--pseudo-limit", type=int, default=30000)
    parser.add_argument("--self-lr-mult", type=float, default=0.25)
    parser.add_argument("--include-test-vocab", action="store_true", default=True)
    parser.add_argument("--no-include-test-vocab", dest="include_test_vocab", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-unlabel", type=int, default=None)
    parser.add_argument("--skip-predict", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Works around duplicate OpenMP runtimes in some Windows PyTorch installs.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_texts, train_labels = read_train(args.train, args.max_train)
    unlabel_texts = read_unlabel(args.unlabel, args.max_unlabel)
    test_ids, test_texts = read_test(args.test)
    print(
        f"loaded train={len(train_texts)} unlabel={len(unlabel_texts)} test={len(test_texts)}"
    )

    print("tokenizing for vocabulary ...")
    vocab_texts = train_texts + unlabel_texts
    if args.include_test_vocab:
        vocab_texts = vocab_texts + test_texts
    vocab_source = [tokenize(text) for text in vocab_texts]
    word2idx, idx2word = build_vocab(vocab_source, args.min_count, args.max_vocab)
    print(f"vocab={len(idx2word)} min_count={args.min_count}")

    full_train = make_dataset(train_texts, train_labels, word2idx, args.seq_len)
    train_idx, valid_idx = split_indices(len(full_train), args.valid_ratio, args.seed)
    train_data = subset_dataset(full_train, train_idx)
    valid_data = subset_dataset(full_train, valid_idx)

    model = TextLSTMClassifier(
        vocab_size=len(idx2word),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print("training supervised model ...")
    best_acc = train_model(model, train_data, valid_data, args, device, "supervised")
    best_overall_acc = best_acc
    best_overall_path = os.path.join(args.out_dir, "supervised_best.pt")
    print(f"best supervised val_acc={best_acc:.4f}")

    if args.self_train and unlabel_texts:
        print("selecting pseudo labels ...")
        unlabel_data = make_dataset(unlabel_texts, None, word2idx, args.seq_len)
        probs = predict_probs(model, unlabel_data, args.batch_size, device)
        selected = [
            (i, prob)
            for i, prob in enumerate(probs)
            if prob >= args.self_threshold or prob <= 1.0 - args.self_threshold
        ]
        selected.sort(key=lambda item: abs(item[1] - 0.5), reverse=True)
        selected = selected[: args.pseudo_limit]
        print(f"pseudo_selected={len(selected)} threshold={args.self_threshold}")

        if selected:
            pseudo_texts = [unlabel_texts[i] for i, _ in selected]
            pseudo_labels = [1.0 if prob >= 0.5 else 0.0 for _, prob in selected]
            combined_texts = [train_texts[i] for i in train_idx] + pseudo_texts
            combined_labels = [float(train_labels[i]) for i in train_idx] + pseudo_labels
            combined_weights = [1.0] * len(train_idx) + [args.pseudo_weight] * len(
                pseudo_texts
            )
            train_plus_pseudo = make_dataset(
                combined_texts,
                combined_labels,
                word2idx,
                args.seq_len,
                combined_weights,
            )
            print("fine-tuning with pseudo labels ...")
            self_acc = train_model(
                model,
                train_plus_pseudo,
                valid_data,
                args,
                device,
                "selftrain",
                lr=args.lr * args.self_lr_mult,
            )
            print(f"best self-train val_acc={self_acc:.4f}")
            if self_acc > best_overall_acc:
                best_overall_acc = self_acc
                best_overall_path = os.path.join(args.out_dir, "selftrain_best.pt")

    best_checkpoint = torch.load(best_overall_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    best_acc = best_overall_acc
    print(f"using checkpoint={best_overall_path} val_acc={best_acc:.4f}")

    if not args.skip_predict:
        print("predicting test ...")
        test_data = make_dataset(test_texts, None, word2idx, args.seq_len)
        test_probs = predict_probs(model, test_data, args.batch_size, device)
        write_submission(test_ids, test_probs, args.submission)
        print(f"wrote {args.submission}")

    torch.save(
        {
            "model": model.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": vars(args),
            "val_acc": best_acc,
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
