"""
Training loop for the chess transformer.

Trains on prepared data (board tensors + value targets) using:
  - AdamW optimizer with cosine learning rate schedule
  - Mixed precision training (fp16) for GPU efficiency
  - TensorBoard logging
  - Periodic checkpointing
  - Gradient clipping for stability

Usage:
    # Single data source:
    python train.py --data data/round_0/ --output models/round_0.pt
    python train.py --data data/round_0/ --checkpoint models/round_0.pt --output models/round_1.pt
    python train.py --data data/round_0/ --model-size tiny --epochs 5

    # Mixed data sources (proportional sampling per epoch):
    python train.py --ccrl data/round_0/ 0.8 --synthetic data/synthetic/ 0.2 \\
        --output models/round_0.pt --epoch-size 5000000
"""

import argparse
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import create_model, ChessTransformer
from config import TrainingConfig, MODEL_CONFIGS


class ChessDataset(Dataset):
    """
    Load training data from .npz chunks.

    Supports two modes:
    - Eager (default): loads all chunks into RAM. Fast but memory-heavy.
    - Lazy: LRU cache of N chunks in memory, loads on demand.
      Uses bisect for O(log n) chunk lookup instead of linear scan.
      Suitable for 12M+ positions with <4GB RAM.
    """

    def __init__(self, data_dir: str, lazy: bool = False, chunk_cache_size: int = 3):
        self.data_dir = data_dir
        self.lazy = lazy
        self.chunk_cache_size = chunk_cache_size

        # Load metadata
        meta_path = os.path.join(data_dir, "metadata.npz")
        if os.path.exists(meta_path):
            meta = dict(np.load(meta_path))
            self.num_chunks = int(meta["num_chunks"])
            self.has_policy = bool(meta.get("has_policy", False))
        else:
            self.num_chunks = len(list(Path(data_dir).glob("chunk_*.npz")))
            self.has_policy = False

        # Discover actual chunk files and sizes
        self.chunk_paths = []
        self.chunk_sizes = []
        self.cumulative_sizes = []
        total = 0

        for i in range(self.num_chunks):
            path = os.path.join(data_dir, f"chunk_{i:04d}.npz")
            if not os.path.exists(path):
                break
            self.chunk_paths.append(path)
            with np.load(path) as data:
                sz = len(data["values"])
            self.chunk_sizes.append(sz)
            total += sz
            self.cumulative_sizes.append(total)

        self.total_size = total
        self.num_chunks = len(self.chunk_paths)

        if lazy:
            # LRU cache: OrderedDict of chunk_idx -> (boards, values, policies)
            self._chunk_cache: OrderedDict[int, tuple] = OrderedDict()
            print(f"Lazy dataset: {self.total_size:,} positions in {self.num_chunks} chunks "
                  f"(cache={chunk_cache_size}, policy: {'yes' if self.has_policy else 'no'})")
        else:
            # Eager mode: load everything into memory
            print(f"Loading {self.num_chunks} chunks from {data_dir}...")
            boards, values, policies = [], [], []
            for path in self.chunk_paths:
                data = np.load(path)
                boards.append(data["boards"])
                values.append(data["values"])
                if "policies" in data:
                    policies.append(data["policies"])

            self.boards = np.concatenate(boards, axis=0)
            self.values = np.concatenate(values, axis=0)
            self.policies = np.concatenate(policies, axis=0) if policies else None
            print(f"Loaded {self.total_size:,} positions "
                  f"(policy: {'yes' if self.policies is not None else 'no'})")

    def _load_chunk(self, chunk_idx: int) -> tuple:
        """Load a chunk into the LRU cache and return (boards, values, policies)."""
        if chunk_idx in self._chunk_cache:
            # Move to end (most recently used)
            self._chunk_cache.move_to_end(chunk_idx)
            return self._chunk_cache[chunk_idx]

        # Evict oldest if cache is full
        while len(self._chunk_cache) >= self.chunk_cache_size:
            self._chunk_cache.popitem(last=False)

        data = np.load(self.chunk_paths[chunk_idx])
        entry = (
            data["boards"],
            data["values"],
            data["policies"] if "policies" in data else None,
        )
        self._chunk_cache[chunk_idx] = entry
        return entry

    def _find_chunk(self, idx: int) -> tuple[int, int]:
        """Find which chunk contains global index idx using bisect (O(log n))."""
        import bisect
        ci = bisect.bisect_right(self.cumulative_sizes, idx)
        if ci >= self.num_chunks:
            raise IndexError(f"Index {idx} out of range (total: {self.total_size})")
        local = idx - (self.cumulative_sizes[ci - 1] if ci > 0 else 0)
        return ci, local

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.lazy:
            ci, local = self._find_chunk(idx)
            boards, values, policies = self._load_chunk(ci)
            board = torch.from_numpy(boards[local].copy())
            value = torch.tensor(values[local])
            pol = policies[local] if policies is not None else 0
            policy = torch.tensor(pol, dtype=torch.long)
        else:
            board = torch.from_numpy(self.boards[idx])
            value = torch.tensor(self.values[idx])
            pol = self.policies[idx] if self.policies is not None else 0
            policy = torch.tensor(pol, dtype=torch.long)
        return board, value, policy


def train(
    model: ChessTransformer,
    train_loader: DataLoader,
    config: TrainingConfig,
    output_path: str,
    device: torch.device,
    log_dir: str | None = None,
    val_loader: DataLoader | None = None,
    sampler=None,
) -> None:
    """Main training loop with warmup, validation, and per-loss tracking.

    Args:
        sampler: Optional ProportionalSampler; if provided, set_epoch() is
                 called each epoch for reproducible shuffling.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Linear warmup then cosine decay
    total_steps = len(train_loader) * config.epochs

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    model.train()
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_samples = 0
        t0 = time.time()

        for batch_idx, (boards, values, policies) in enumerate(train_loader):
            boards = boards.to(device)
            values = values.to(device)
            policies = policies.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_value, pred_policy = model(boards)
                pred_value = pred_value.squeeze(-1)

                value_loss = F.mse_loss(pred_value, values)
                policy_loss = F.cross_entropy(pred_policy, policies)

                loss = (config.value_loss_weight * value_loss +
                        config.policy_loss_weight * policy_loss)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = boards.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_value_loss += value_loss.item() * batch_size
            epoch_policy_loss += policy_loss.item() * batch_size
            epoch_samples += batch_size
            global_step += 1

            if writer and global_step % 100 == 0:
                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/value", value_loss.item(), global_step)
                writer.add_scalar("loss/policy", policy_loss.item(), global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)

        # Epoch summary
        avg_loss = epoch_loss / epoch_samples
        avg_value_loss = epoch_value_loss / epoch_samples
        avg_policy_loss = epoch_policy_loss / epoch_samples
        elapsed = time.time() - t0
        samples_per_sec = epoch_samples / elapsed

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"loss: {avg_loss:.6f} | "
            f"value: {avg_value_loss:.6f} | "
            f"policy: {avg_policy_loss:.4f} | "
            f"lr: {scheduler.get_last_lr()[0]:.2e} | "
            f"{samples_per_sec:.0f} pos/s | "
            f"{elapsed:.1f}s"
        )

        if writer:
            writer.add_scalar("epoch/loss", avg_loss, epoch)
            writer.add_scalar("epoch/value_loss", avg_value_loss, epoch)
            writer.add_scalar("epoch/policy_loss", avg_policy_loss, epoch)

        # Validation
        val_loss = None
        if val_loader is not None and (epoch + 1) % config.eval_every == 0:
            val_loss = _evaluate(model, val_loader, config, device)
            print(f"  val_loss: {val_loss:.6f}")
            if writer:
                writer.add_scalar("epoch/val_loss", val_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": model.config.__dict__,
                "loss": avg_loss,
            }
            torch.save(checkpoint, output_path)
            print(f"  Saved checkpoint to {output_path}")

            # Best model: prefer val loss if available, else train loss
            compare_loss = val_loss if val_loss is not None else avg_loss
            if compare_loss < best_val_loss:
                best_val_loss = compare_loss
                best_path = output_path.replace(".pt", "_best.pt")
                torch.save(checkpoint, best_path)
                print(f"  New best model: {best_path} "
                      f"({'val' if val_loss is not None else 'train'}_loss={compare_loss:.6f})")

    if writer:
        writer.close()

    print(f"\nTraining complete. Best loss: {best_val_loss:.6f}")


@torch.no_grad()
def _evaluate(
    model: ChessTransformer,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for boards, values, policies in val_loader:
        boards = boards.to(device)
        values = values.to(device)
        policies = policies.to(device)

        pred_value, pred_policy = model(boards)
        pred_value = pred_value.squeeze(-1)

        value_loss = F.mse_loss(pred_value, values)
        policy_loss = F.cross_entropy(pred_policy, policies)
        loss = (config.value_loss_weight * value_loss +
                config.policy_loss_weight * policy_loss)

        total_loss += loss.item() * boards.size(0)
        total_samples += boards.size(0)

    model.train()
    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train chess transformer")
    parser.add_argument("--data", default=None, help="Training data directory (single source)")
    parser.add_argument("--output", default="models/model.pt", help="Output model path")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--model-size", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lazy", action="store_true", help="Lazy data loading (low RAM)")
    parser.add_argument("--log-dir", default="runs/", help="TensorBoard log dir")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction of data for validation (default: 0.05)")
    # Multi-source mixing
    parser.add_argument("--ccrl", nargs=2, metavar=("DIR", "RATIO"),
                        help="CCRL data directory and ratio (e.g., data/round_0/ 0.8)")
    parser.add_argument("--synthetic", nargs=2, metavar=("DIR", "RATIO"),
                        help="Synthetic data directory and ratio (e.g., data/synthetic/ 0.2)")
    parser.add_argument("--endgame", nargs=2, metavar=("DIR", "RATIO"),
                        help="Endgame data directory and ratio")
    parser.add_argument("--opening", nargs=2, metavar=("DIR", "RATIO"),
                        help="Opening data directory and ratio")
    parser.add_argument("--epoch-size", type=int, default=None,
                        help="Positions per epoch when mixing (default: total dataset size)")
    args = parser.parse_args()

    # Determine data mode: mixed sources vs single directory
    mix_sources = {}
    for name in ["ccrl", "synthetic", "endgame", "opening"]:
        val = getattr(args, name)
        if val:
            mix_sources[name] = (val[0], float(val[1]))

    use_mixed = len(mix_sources) > 0
    if not use_mixed and args.data is None:
        parser.error("Either --data or at least one source (--ccrl, --synthetic, etc.) is required")
    if use_mixed and args.data:
        parser.error("Cannot use --data together with --ccrl/--synthetic/--endgame/--opening")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # Create or load model
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        from config import ModelConfig
        config = ModelConfig(**checkpoint["config"])
        model = ChessTransformer(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"  Resumed from epoch {checkpoint['epoch']}")
    else:
        model = create_model(args.model_size).to(device)

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
    )

    sampler = None

    if use_mixed:
        from data_mixer import DataMixer
        print(f"\nMixed-source training:")
        mixer = DataMixer(mix_sources)
        dataset = mixer.dataset
        epoch_size = args.epoch_size or dataset.total_size

        # For mixed sources, split validation off from the largest source
        # by reserving a fraction of its indices, then use proportional
        # sampling on the training portion.
        # Simpler approach: no val split for mixed mode (eval on separate data).
        val_loader = None
        if train_config.val_fraction > 0:
            print(f"  Note: validation uses {train_config.val_fraction:.0%} of epoch samples")

        from data_mixer import ProportionalSampler
        sampler = ProportionalSampler(dataset, epoch_size=epoch_size)

        num_workers = 0  # mixed mode uses LRU caching, workers hurt cache hits
        train_loader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        print(f"  Epoch size: {epoch_size:,} positions")
        print(f"  Batches per epoch: {len(train_loader)}")
    else:
        # Single-source mode (original behavior)
        dataset = ChessDataset(args.data, lazy=args.lazy)

        # Train/val split
        val_loader = None
        if train_config.val_fraction > 0:
            val_size = max(1, int(len(dataset) * train_config.val_fraction))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
            print(f"Split: {train_size:,} train, {val_size:,} val")
        else:
            train_dataset = dataset

        num_workers = 0 if args.lazy else train_config.num_workers

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        if train_config.val_fraction > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )

        print(f"Batches per epoch: {len(train_loader)}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Train
    train(model, train_loader, train_config, args.output, device,
          args.log_dir, val_loader, sampler=sampler)


if __name__ == "__main__":
    main()
