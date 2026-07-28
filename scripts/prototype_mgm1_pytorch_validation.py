#!/usr/bin/env python3
"""
Prototype: MGM-1 PyTorch validation + FedAvg/UsefulPoW simulator.

Question being answered:
    Does local training on small, biased genome slices followed by FedAvg
    weight averaging explain the observed 100% local training accuracy but
    ~29.8% global evaluation accuracy in the Rust/Candle implementation?

This is throwaway code. Run it with a Python that has PyTorch installed, e.g.:

    /tmp/venv_mgm1/bin/python scripts/prototype_mgm1_pytorch_validation.py

Or with options:

    /tmp/venv_mgm1/bin/python scripts/prototype_mgm1_pytorch_validation.py \
        --n-clients 5 --local-samples 64 --local-epochs 3 \
        --local-at-rich 0.75 --test-at-rich 0.55
"""

import argparse
import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Token ids match the Rust `mini-genome-model` layout:
# 0=A, 1=C, 2=G, 3=T, 4=[MASK], 5=PAD, 6=[CLS], 7=[SEP]
DNA_A = 0
DNA_C = 1
DNA_G = 2
DNA_T = 3
MASK = 4
PAD = 5

ID_TO_BASE = {DNA_A: "A", DNA_C: "C", DNA_G: "G", DNA_T: "T"}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def generate_sequence(length: int, probs: Optional[List[float]] = None, seed: Optional[int] = None) -> str:
    """Generate a random DNA string with an optional base distribution."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    if probs is None:
        probs = [0.25, 0.25, 0.25, 0.25]
    bases = ["A", "C", "G", "T"]
    return "".join(rng.choices(bases, weights=probs, k=length))


def generate_sequences(
    n: int,
    length: int,
    at_fraction: float = 0.5,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate n DNA strings with a controllable AT/CG ratio.

    `at_fraction` is the combined probability of A+T (and therefore 1 - at_fraction
    is C+G). Within AT, A and T are split evenly; within CG, C and G are split evenly.
    """
    if seed is not None:
        base_rng = random.Random(seed)
    else:
        base_rng = random

    half_at = at_fraction / 2.0
    half_cg = (1.0 - at_fraction) / 2.0
    probs = [half_at, half_cg, half_cg, half_at]  # A, C, G, T
    bases = ["A", "C", "G", "T"]

    return ["".join(base_rng.choices(bases, weights=probs, k=length)) for _ in range(n)]


class MGM1Config:
    """Matches the Rust `MiniGenomeConfig` defaults for MGM-1 v2."""

    vocab_size = 8
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 512
    max_seq_len = 512
    dropout = 0.2
    grad_clip_norm = 1.0
    weight_decay = 0.01
    label_smoothing = 0.1
    class_weights = [1.5, 1.5, 1.5, 1.0]  # A, C, G, T


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, matching the Rust implementation."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with ReLU FFN, matching `mini-genome-model`."""

    def __init__(self, config: MGM1Config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Pre-norm self-attention with residual
        h = self.norm1(x)
        attn_out, _ = self.attention(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)

        # Pre-norm FFN with residual
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        return x


class MGM1(nn.Module):
    """Mini Genome Model in PyTorch, as close to the Rust/Candle version as practical."""

    def __init__(self, config: MGM1Config = None):
        super().__init__()
        self.config = config or MGM1Config()

        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.pos_encoding = PositionalEncoding(self.config.d_model, self.config.max_seq_len)
        self.dropout = nn.Dropout(self.config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )

        self.final_norm = nn.LayerNorm(self.config.d_model, eps=1e-5)
        self.head = nn.Linear(self.config.d_model, self.config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Default Xavier uniform for most linear layers (close to Candle default).
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Scaled token embeddings to stabilise the residual stream.
                module.weight.data.normal_(0.0, 1.0 / math.sqrt(self.config.d_model))

        # Output head: small random init + zero bias for near-uniform initial logits.
        nn.init.normal_(self.head.weight, 0.0, 0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.head(x)

    def compute_mlm_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLM loss restricted to the first four DNA logits.

        Implements the same label-smoothing + class-weighting formula used in
        `mini-genome-model/src/lib.rs`.
        """
        logits = self.forward(input_ids)  # [batch, seq_len, vocab]
        logits_dna = logits[..., :4]  # mask out special-token predictions

        # Flatten and select only the masked positions
        mask = mask_positions.reshape(-1)
        labels_flat = labels.reshape(-1)[mask]
        dna_logits_flat = logits_dna.reshape(-1, 4)[mask]

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

        log_probs = F.log_softmax(dna_logits_flat, dim=-1)
        target = log_probs.gather(1, labels_flat.unsqueeze(1))

        smoothing = self.config.label_smoothing
        if smoothing > 0.0:
            sum_log_probs = log_probs.sum(dim=-1, keepdim=True)
            other = sum_log_probs - target
            nll = target * (smoothing - 1.0) - other * (smoothing / 3.0)
        else:
            nll = -target

        nll = nll.squeeze(1)

        # Class weighting over the four DNA bases
        cw = torch.tensor(self.config.class_weights, device=logits.device, dtype=torch.float)
        cw_per_token = cw[labels_flat]
        nll = nll * cw_per_token

        loss = nll.sum() / mask.sum()

        # Accuracy on masked positions only
        preds = dna_logits_flat.argmax(dim=-1)
        correct = (preds == labels_flat).float().sum()
        accuracy = correct / mask.sum()

        return loss, accuracy


def per_base_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_positions: torch.Tensor,
) -> Dict[str, float]:
    """Per-base accuracy over masked positions."""
    logits_dna = logits[..., :4]
    mask = mask_positions.reshape(-1)
    labels_flat = labels.reshape(-1)[mask]
    preds = logits_dna.reshape(-1, 4).argmax(dim=-1)[mask]

    counts = Counter()
    correct = Counter()
    for p, t in zip(preds.tolist(), labels_flat.tolist()):
        base = ID_TO_BASE.get(t, "X")
        counts[base] += 1
        if p == t:
            correct[base] += 1

    return {base: (correct[base] / counts[base] if counts[base] else 0.0) for base in ID_TO_BASE.values()}


class GenomeDataset(Dataset):
    """Minimal char-level DNA MLM dataset."""

    def __init__(
        self,
        sequences: List[str],
        max_seq_len: int = 512,
        mask_ratio: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][: self.max_seq_len]
        token_ids = [ORD_BASE_MAP.get(ord(c), PAD) for c in seq]

        # Pad
        n_pad = self.max_seq_len - len(token_ids)
        token_ids = token_ids + [PAD] * n_pad

        # Mask random non-pad positions
        input_ids = token_ids.copy()
        labels = token_ids.copy()
        valid_positions = [i for i, t in enumerate(token_ids) if t != PAD]
        n_mask = max(1, int(len(valid_positions) * self.mask_ratio))
        if n_mask > len(valid_positions):
            n_mask = len(valid_positions)

        # Deterministic-ish but reproducible per-epoch via worker seed.
        mask_idx = self.rng.sample(valid_positions, n_mask)
        for i in mask_idx:
            input_ids[i] = MASK

        # Non-masked labels are -100 (ignored if accidentally used)
        for i in range(self.max_seq_len):
            if i not in mask_idx:
                labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mask_positions": torch.tensor([i in mask_idx for i in range(self.max_seq_len)], dtype=torch.bool),
        }


ORD_BASE_MAP = {
    ord("A"): DNA_A,
    ord("C"): DNA_C,
    ord("G"): DNA_G,
    ord("T"): DNA_T,
    ord("a"): DNA_A,
    ord("c"): DNA_C,
    ord("g"): DNA_G,
    ord("t"): DNA_T,
}


class MGM1Trainer:
    """AdamW trainer with weight decay and gradient clipping."""

    def __init__(self, model: MGM1, lr: float = 1e-3, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=model.config.weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask_positions = batch["mask_positions"].to(self.device)

            loss, acc = self.model.compute_mlm_loss(input_ids, labels, mask_positions)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.config.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    def fit(self, dataloader: DataLoader, epochs: int) -> Tuple[float, float]:
        for epoch in range(epochs):
            loss, acc = self.train_epoch(dataloader)
            print(f"    epoch {epoch + 1}/{epochs}: loss={loss:.4f}, acc={acc:.2%}")
        return loss, acc

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        all_logits = []
        all_labels = []
        all_masks = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                mask_positions = batch["mask_positions"].to(self.device)

                loss, acc = self.model.compute_mlm_loss(input_ids, labels, mask_positions)
                logits = self.model(input_ids)

                total_loss += loss.item()
                total_acc += acc.item()
                n_batches += 1

                all_logits.append(logits)
                all_labels.append(labels)
                all_masks.append(mask_positions)

        per_base = per_base_accuracy(
            torch.cat(all_logits, dim=0),
            torch.cat(all_labels, dim=0),
            torch.cat(all_masks, dim=0),
        )

        return total_loss / n_batches, total_acc / n_batches, per_base


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """FedAvg: simple uniform average of model weights."""
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def run_fedavg_simulation(args: argparse.Namespace):
    """Simulate N local miners, each training on a biased slice, then FedAvg."""
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed)

    # Shared global test set (can be different distribution from local data)
    print(f"\n[Global test set] n={args.test_samples}, AT-fraction={args.test_at_rich}")
    test_seqs = generate_sequences(
        args.test_samples,
        args.seq_len,
        at_fraction=args.test_at_rich,
        seed=args.seed + 9000,
    )
    test_dataset = GenomeDataset(test_seqs, args.seq_len, args.mask_ratio, seed=args.seed + 9001)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Central starting model
    base_config = MGM1Config()
    base_model = MGM1(base_config)

    # Evaluate the untrained global model as a UsefulPoW-style baseline
    base_trainer = MGM1Trainer(base_model, lr=args.lr, device=device)
    base_loss, base_acc, base_per_base = base_trainer.evaluate(test_loader)
    print(f"  Untrained global: loss={base_loss:.4f}, acc={base_acc:.2%}")

    client_states = []
    client_metrics = []

    # Each client trains on its own biased slice
    for client_id in range(args.n_clients):
        at_frac = args.local_at_rich
        # Slightly vary each client's AT bias to mimic real miner diversity
        if args.local_bias_jitter > 0:
            at_frac = min(0.95, max(0.05, at_frac + random.uniform(-args.local_bias_jitter, args.local_bias_jitter)))

        print(f"\n[Client {client_id + 1}/{args.n_clients}] local AT-fraction={at_frac:.2f}")
        local_seqs = generate_sequences(
            args.local_samples,
            args.seq_len,
            at_fraction=at_frac,
            seed=args.seed + client_id * 1000,
        )
        local_dataset = GenomeDataset(local_seqs, args.seq_len, args.mask_ratio, seed=args.seed + client_id * 1001)
        local_loader = DataLoader(local_dataset, batch_size=args.batch_size, shuffle=True)

        client_model = MGM1(base_config)
        # All clients start from the same random init so the average is meaningful
        client_model.load_state_dict(base_model.state_dict())
        client_trainer = MGM1Trainer(client_model, lr=args.lr, device=device)

        client_trainer.fit(local_loader, args.local_epochs)

        # Client train set evaluation (local overfitting signal)
        local_train_loss, local_train_acc, local_train_per_base = client_trainer.evaluate(local_loader)
        print(f"  → train eval: loss={local_train_loss:.4f}, acc={local_train_acc:.2%}, per-base={local_train_per_base}")

        # Client on global test (generalisation signal)
        local_test_loss, local_test_acc, local_test_per_base = client_trainer.evaluate(test_loader)
        print(f"  → global eval:  loss={local_test_loss:.4f}, acc={local_test_acc:.2%}, per-base={local_test_per_base}")

        client_states.append(client_model.state_dict())
        client_metrics.append(
            {
                "client": client_id + 1,
                "at_fraction": at_frac,
                "local_train_acc": local_train_acc,
                "local_test_acc": local_test_acc,
            }
        )

    # FedAvg aggregation
    print("\n[FedAvg] averaging client weights...")
    global_state = average_state_dicts(client_states)
    global_model = MGM1(base_config)
    global_model.load_state_dict(global_state)
    global_trainer = MGM1Trainer(global_model, lr=args.lr, device=device)

    global_loss, global_acc, global_per_base = global_trainer.evaluate(test_loader)
    print(f"\n[Global model]")
    print(f"  loss={global_loss:.4f}")
    print(f"  accuracy={global_acc:.2%}")
    print(f"  per-base accuracy={global_per_base}")

    local_train_accs = [m["local_train_acc"] for m in client_metrics]
    local_test_accs = [m["local_test_acc"] for m in client_metrics]
    print(f"\n[Summary]")
    print(f"  n_clients={args.n_clients}, local_samples={args.local_samples}, local_epochs={args.local_epochs}")
    print(f"  mean client train acc = {sum(local_train_accs) / len(local_train_accs):.2%}")
    print(f"  mean client global acc = {sum(local_test_accs) / len(local_test_accs):.2%}")
    print(f"  FedAvg global acc     = {global_acc:.2%}")
    print(f"  gap (train → global)  = {sum(local_train_accs) / len(local_train_accs) - global_acc:.2%}")


def main():
    parser = argparse.ArgumentParser(description="MGM-1 PyTorch validation / FedAvg prototype")
    parser.add_argument("--n-clients", type=int, default=3, help="Number of simulated local miners")
    parser.add_argument("--local-samples", type=int, default=64, help="Sequences per client")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local training epochs")
    parser.add_argument("--local-at-rich", type=float, default=0.70, help="Client AT fraction (0.5 = balanced)")
    parser.add_argument("--local-bias-jitter", type=float, default=0.05, help="Jitter around local-at-rich per client")
    parser.add_argument("--test-samples", type=int, default=128, help="Global test set size")
    parser.add_argument("--test-at-rich", type=float, default=0.55, help="Global test AT fraction")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="MLM mask ratio")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="PyTorch device")
    args = parser.parse_args()

    run_fedavg_simulation(args)


if __name__ == "__main__":
    main()
