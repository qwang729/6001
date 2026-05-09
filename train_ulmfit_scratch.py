import argparse
import math
import os
import random
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F

from train_scratch_lstm import (
    BucketBatcher,
    EncodedDataset,
    ScratchLSTMCell,
    build_vocab,
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


EOS = "<eos>"


class UniLSTMLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.cell = ScratchLSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        batch, seq_len, _ = x.shape
        if state is None:
            h = x.new_zeros(batch, self.hidden_dim)
            c = x.new_zeros(batch, self.hidden_dim)
        else:
            h, c = state
        x_gates = self.cell.x2h(x)
        outputs = []
        for t in range(seq_len):
            h_new, c_new = self.cell.forward_from_input(x_gates[:, t, :], (h, c))
            if lengths is not None:
                active = (lengths > t).to(x.dtype).unsqueeze(1)
                h = active * h_new + (1.0 - active) * h
                c = active * c_new + (1.0 - active) * c
            else:
                h, c = h_new, c_new
            outputs.append(h)
        out = torch.stack(outputs, dim=1)
        return self.dropout(out), (h, c)


class UniLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        current = input_dim
        for _ in range(layers):
            self.layers.append(UniLSTMLayer(current, hidden_dim, dropout))
            current = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        new_states = []
        raw_outputs = []
        for i, layer in enumerate(self.layers):
            state = None if states is None else states[i]
            x, state = layer(x, lengths, state)
            raw_outputs.append(x)
            new_states.append(state)
        return x, new_states, raw_outputs


class ScratchLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.encoder = UniLSTMEncoder(emb_dim, hidden_dim, layers, dropout)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        if emb_dim == hidden_dim:
            self.decoder.weight = self.embedding.weight

    def forward(self, x: torch.Tensor, states=None):
        emb = self.emb_dropout(self.embedding(x))
        out, states, raw_outputs = self.encoder(emb, None, states)
        logits = self.decoder(out)
        return logits, states, raw_outputs


class ULMFiTClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        pad_idx: int = 0,
        unk_idx: int = 1,
        word_dropout: float = 0.03,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout = word_dropout
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.encoder = UniLSTMEncoder(emb_dim, hidden_dim, layers, dropout)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        feature_dim = hidden_dim * 4
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def load_lm(self, lm: ScratchLanguageModel) -> None:
        self.embedding.load_state_dict(lm.embedding.state_dict())
        self.encoder.load_state_dict(lm.encoder.state_dict())

    def _word_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.word_dropout <= 0.0:
            return x
        mask = (
            torch.rand(x.shape, device=x.device) < self.word_dropout
        ) & (x != self.pad_idx)
        return x.masked_fill(mask, self.unk_idx)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self._word_dropout(x)
        emb = self.emb_dropout(self.embedding(x))
        out, _, _ = self.encoder(emb, lengths, None)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        last_idx = (lengths - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, out.size(2))
        last = out.gather(1, last_idx).squeeze(1)
        max_pool = out.masked_fill(~mask.unsqueeze(2), -1e4).max(dim=1).values
        mean_pool = (
            out.masked_fill(~mask.unsqueeze(2), 0.0).sum(dim=1)
            / lengths.clamp_min(1).to(out.dtype).unsqueeze(1)
        )
        attn_score = self.attn(out).squeeze(2).masked_fill(~mask, -1e4)
        attn = F.softmax(attn_score, dim=1).unsqueeze(2)
        attn_pool = (out * attn).sum(dim=1)
        return self.head(torch.cat([last, max_pool, mean_pool, attn_pool], dim=1)).squeeze(1)


def build_lm_stream(texts: Sequence[str], word2idx: dict, seq_len: int) -> List[int]:
    eos_id = word2idx[EOS]
    ids = []
    for text in texts:
        doc_ids, _ = encode(tokenize(text), word2idx, seq_len=10_000)
        while doc_ids and doc_ids[-1] == 0:
            doc_ids.pop()
        ids.extend(doc_ids)
        ids.append(eos_id)
    return ids


def batchify(ids: Sequence[int], batch_size: int, device: torch.device) -> torch.Tensor:
    n_batch = len(ids) // batch_size
    data = torch.tensor(ids[: n_batch * batch_size], dtype=torch.long, device=device)
    return data.view(batch_size, n_batch).contiguous()


def detach_states(states):
    if states is None:
        return None
    return [(h.detach(), c.detach()) for h, c in states]


def train_language_model(lm, data, args, device):
    optimizer = torch.optim.AdamW(lm.parameters(), lr=args.lm_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_loss = float("inf")
    best_path = os.path.join(args.out_dir, "lm_best.pt")
    steps_per_epoch = max(1, (data.size(1) - 1) // args.bptt)
    total_steps = steps_per_epoch * args.lm_epochs
    warmup = max(1, int(total_steps * 0.05))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0
    for epoch in range(1, args.lm_epochs + 1):
        lm.train()
        states = None
        total_loss = 0.0
        total_tokens = 0
        positions = list(range(0, data.size(1) - 1, args.bptt))
        for step, i in enumerate(positions, 1):
            seq_len = min(args.bptt, data.size(1) - 1 - i)
            x = data[:, i : i + seq_len]
            y = data[:, i + 1 : i + 1 + seq_len]
            states = detach_states(states)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits, states, raw_outputs = lm(x, states)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                if args.alpha:
                    loss = loss + args.alpha * raw_outputs[-1].pow(2).mean()
                if args.beta and raw_outputs[-1].size(1) > 1:
                    diff = raw_outputs[-1][:, 1:, :] - raw_outputs[-1][:, :-1, :]
                    loss = loss + args.beta * diff.pow(2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(lm.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            if step % args.log_every == 0:
                print(
                    f"lm epoch {epoch:02d} step {step}/{len(positions)} "
                    f"loss={total_loss / max(1,total_tokens):.4f} "
                    f"ppl={math.exp(min(20,total_loss / max(1,total_tokens))):.2f}"
                )
        epoch_loss = total_loss / max(1, total_tokens)
        print(f"lm epoch {epoch:02d} loss={epoch_loss:.4f} ppl={math.exp(min(20, epoch_loss)):.2f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": lm.state_dict(), "args": vars(args), "lm_loss": best_loss}, best_path)
            print(f"saved {best_path} lm_loss={best_loss:.4f}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    lm.load_state_dict(ckpt["model"])


def train_classifier(model, train_data: EncodedDataset, valid_data: EncodedDataset, args, device):
    train_batcher = BucketBatcher(train_data, args.batch_size, True, args.seed)
    valid_batcher = BucketBatcher(valid_data, args.batch_size, False, args.seed)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.embedding.parameters(), "lr": args.clf_lr * 0.25},
            {"params": model.encoder.parameters(), "lr": args.clf_lr * 0.5},
            {"params": model.head.parameters(), "lr": args.clf_lr},
            {"params": model.attn.parameters(), "lr": args.clf_lr},
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
    best_path = os.path.join(args.out_dir, "ulmfit_clf_best.pt")
    for epoch in range(1, args.clf_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in train_batcher:
            x, lengths, labels = [item.to(device, non_blocking=True) for item in batch]
            targets = labels * (1.0 - 2.0 * args.label_smoothing) + args.label_smoothing
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
        val_loss, val_acc = evaluate_classifier(model, valid_batcher, device)
        print(
            f"clf epoch {epoch:02d} train_loss={total_loss/max(1,total):.5f} "
            f"train_acc={total_correct/max(1,total):.4f} "
            f"val_loss={val_loss:.5f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "val_acc": best_acc}, best_path)
            print(f"saved {best_path} val_acc={best_acc:.4f}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return best_acc


def evaluate_classifier(model, batcher, device):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--unlabel", default="train_unlabel.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--out-dir", default="runs_ulmfit_scratch")
    parser.add_argument("--submission", default="submission_ulmfit.csv")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--bptt", type=int, default=80)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-vocab", type=int, default=60000)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--word-dropout", type=float, default=0.04)
    parser.add_argument("--lm-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lm-epochs", type=int, default=2)
    parser.add_argument("--clf-epochs", type=int, default=6)
    parser.add_argument("--lm-lr", type=float, default=0.0015)
    parser.add_argument("--clf-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0002)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-unlabel", type=int, default=None)
    parser.add_argument("--max-lm-docs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--lm-checkpoint", default=None)
    parser.add_argument("--skip-predict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    train_texts, train_labels = read_train(args.train, args.max_train)
    unlabel_texts = read_unlabel(args.unlabel, args.max_unlabel)
    test_ids, test_texts = read_test(args.test)
    print(f"loaded train={len(train_texts)} unlabel={len(unlabel_texts)} test={len(test_texts)}")

    vocab_docs = train_texts + unlabel_texts
    tokenized = [tokenize(text) for text in vocab_docs]
    word2idx, idx2word = build_vocab(tokenized + [[EOS]], args.min_count, args.max_vocab)
    if EOS not in word2idx:
        word2idx[EOS] = len(idx2word)
        idx2word.append(EOS)
    print(f"vocab={len(idx2word)}")

    lm_docs = train_texts + unlabel_texts
    random.Random(args.seed).shuffle(lm_docs)
    if args.max_lm_docs:
        lm_docs = lm_docs[: args.max_lm_docs]
    print("building LM stream ...")
    lm_ids = build_lm_stream(lm_docs, word2idx, args.seq_len)
    lm_data = batchify(lm_ids, args.lm_batch_size, device)
    print(f"lm_tokens={len(lm_ids)} batches_per_epoch={(lm_data.size(1)-1)//args.bptt}")

    lm = ScratchLanguageModel(
        len(idx2word), args.emb_dim, args.hidden_dim, args.layers, args.dropout
    ).to(device)
    if args.lm_checkpoint:
        ckpt = torch.load(args.lm_checkpoint, map_location=device, weights_only=False)
        lm.load_state_dict(ckpt["model"])
    if not args.skip_lm:
        print("pretraining scratch language model ...")
        train_language_model(lm, lm_data, args, device)

    full_train = make_dataset(train_texts, train_labels, word2idx, args.seq_len)
    train_idx, valid_idx = split_indices(len(full_train), args.valid_ratio, args.seed)
    train_data = subset_dataset(full_train, train_idx)
    valid_data = subset_dataset(full_train, valid_idx)
    classifier = ULMFiTClassifier(
        len(idx2word),
        args.emb_dim,
        args.hidden_dim,
        args.layers,
        args.dropout,
        word_dropout=args.word_dropout,
    ).to(device)
    classifier.load_lm(lm)
    print("fine-tuning classifier from LM ...")
    best_acc = train_classifier(classifier, train_data, valid_data, args, device)
    print(f"best ulmfit val_acc={best_acc:.4f}")

    if not args.skip_predict:
        test_data = make_dataset(test_texts, None, word2idx, args.seq_len)
        probs = predict_probs(classifier, test_data, args.batch_size, device)
        write_submission(test_ids, probs, args.submission)
        print(f"wrote {args.submission}")
    torch.save(
        {
            "model": classifier.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": vars(args),
            "val_acc": best_acc,
            "model_class": "ULMFiTClassifier",
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
