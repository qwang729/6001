import argparse
import math
import os
from argparse import Namespace
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.nn import functional as F

from train_scratch_lstm import (
    ScratchBiLSTM,
    build_vocab,
    evaluate,
    make_dataset,
    predict_probs,
    read_test,
    read_train,
    read_unlabel,
    set_seed,
    split_indices,
    subset_dataset,
    tokenize,
    train_model,
    write_submission,
)


class EmbeddingChannelDropout(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(keep) / keep
        return x * mask


class StackedScratchBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        current_dim = input_dim
        for _ in range(layers):
            self.layers.append(ScratchBiLSTM(current_dim, hidden_dim, dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * 2))
            current_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        last = None
        for i, layer in enumerate(self.layers):
            residual = x
            x, last = layer(x, lengths)
            x = self.norms[i](x)
            if residual.shape == x.shape:
                x = x + residual
            x = self.dropout(x)
        return x, last


class StrongTextLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        layers: int,
        conv_channels: int,
        word_dropout: float,
        emb_channel_dropout: float,
        pad_idx: int = 0,
        unk_idx: int = 1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout = word_dropout
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_channel_dropout = EmbeddingChannelDropout(emb_channel_dropout)
        self.encoder = StackedScratchBiLSTM(emb_dim, hidden_dim, layers, dropout)
        out_dim = hidden_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        kernels = [2, 3, 4, 5]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(emb_dim, conv_channels, kernel_size=k, padding=0, bias=False)
                for k in kernels
            ]
        )
        feature_dim = out_dim * 4 + conv_channels * len(kernels)
        self.fc = nn.Sequential(
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
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

    def _word_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.word_dropout <= 0.0:
            return x
        mask = (
            torch.rand(x.shape, device=x.device) < self.word_dropout
        ) & (x != self.pad_idx)
        return x.masked_fill(mask, self.unk_idx)

    def _conv_features(self, emb: torch.Tensor) -> torch.Tensor:
        conv_in = emb.transpose(1, 2)
        pooled: List[torch.Tensor] = []
        for conv in self.convs:
            feat = F.gelu(conv(conv_in))
            pooled.append(F.adaptive_max_pool1d(feat, 1).squeeze(2))
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self._word_dropout(x)
        emb = self.embedding(x)
        emb = self.emb_channel_dropout(self.emb_dropout(emb))
        conv_pool = self._conv_features(emb)
        out, last = self.encoder(emb, lengths)

        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_score = self.attn(out).squeeze(2).masked_fill(~mask, -1e4)
        attn_weight = F.softmax(attn_score, dim=1).unsqueeze(2)
        attn_pool = torch.sum(out * attn_weight, dim=1)
        max_pool = out.masked_fill(~mask.unsqueeze(2), -1e4).max(dim=1).values
        mean_pool = (
            out.masked_fill(~mask.unsqueeze(2), 0.0).sum(dim=1)
            / lengths.clamp_min(1).to(out.dtype).unsqueeze(1)
        )
        features = torch.cat([last, attn_pool, max_pool, mean_pool, conv_pool], dim=1)
        return self.fc(features).squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--unlabel", default="train_unlabel.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--out-dir", default="runs_strong_lstm")
    parser.add_argument("--submission", default="submission_strong.csv")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-vocab", type=int, default=100000)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--conv-channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.42)
    parser.add_argument("--word-dropout", type=float, default=0.04)
    parser.add_argument("--emb-channel-dropout", type=float, default=0.18)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--include-test-vocab", action="store_true", default=True)
    parser.add_argument("--no-include-test-vocab", dest="include_test_vocab", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-unlabel", type=int, default=None)
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

    vocab_texts = train_texts + unlabel_texts
    if args.include_test_vocab:
        vocab_texts += test_texts
    print("tokenizing for vocabulary ...")
    word2idx, idx2word = build_vocab(
        [tokenize(text) for text in vocab_texts], args.min_count, args.max_vocab
    )
    print(f"vocab={len(idx2word)}")

    full_train = make_dataset(train_texts, train_labels, word2idx, args.seq_len)
    train_idx, valid_idx = split_indices(len(full_train), args.valid_ratio, args.seed)
    train_data = subset_dataset(full_train, train_idx)
    valid_data = subset_dataset(full_train, valid_idx)

    model = StrongTextLSTMClassifier(
        vocab_size=len(idx2word),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        layers=args.layers,
        conv_channels=args.conv_channels,
        word_dropout=args.word_dropout,
        emb_channel_dropout=args.emb_channel_dropout,
    ).to(device)
    print("training strong LSTM-like model ...")
    best_acc = train_model(model, train_data, valid_data, args, device, "strong")
    print(f"best strong val_acc={best_acc:.4f}")

    if not args.skip_predict:
        test_data = make_dataset(test_texts, None, word2idx, args.seq_len)
        probs = predict_probs(model, test_data, args.batch_size, device)
        write_submission(test_ids, probs, args.submission)
        print(f"wrote {args.submission}")

    torch.save(
        {
            "model": model.state_dict(),
            "word2idx": word2idx,
            "idx2word": idx2word,
            "args": vars(args),
            "val_acc": best_acc,
            "model_class": "StrongTextLSTMClassifier",
        },
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"saved {os.path.join(args.out_dir, 'final_model.pt')}")


if __name__ == "__main__":
    main()
