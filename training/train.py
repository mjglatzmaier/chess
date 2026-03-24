"""
Training loop for the chess transformer (v2 — simplified, seg-fault-free).

Trains on HDF5 data with all positions preloaded into RAM as torch tensors
before training begins. No HDF5 file handles remain open during training,
no shared numpy/torch memory, no ctypes hacks.

Usage:
    # Single HDF5 source:
    python train.py --data data/round_0.h5 --output models/round_0.pt

    # Mixed HDF5 sources (proportional blending):
    python train.py --ccrl data/round_0.h5 0.8 --synthetic data/synthetic.h5 0.2 \
        --output models/round_0.pt --epoch-size 5000000

    # Resume from checkpoint:
    python train.py --data data/round_0.h5 --checkpoint models/round_0.pt \
        --output models/round_1.pt
"""

import argparse
import gc
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import create_model, ChessTransformer
from config import TrainingConfig, MODEL_CONFIGS

# Board encoding constants
HALFMOVE_TOKEN = 64
HALFMOVE_FEAT = 26
HALFMOVE_SCALE = 255.0


class PreloadedDataset(Dataset):
    """
    In-memory dataset loaded from one or more HDF5 files.

    All data is read into RAM at construction time. HDF5 files are opened,
    read, and immediately closed — no file handles remain during training.

    Boards are stored as uint8 tensors (1 byte per feature) and converted
    to float32 on-the-fly in __getitem__. This keeps resident memory at
    ~1× raw data size instead of 4× for float32 storage.
    """

    def __init__(
        self,
        sources: dict[str, tuple[str, float]],
        epoch_size: int | None = None,
        seed: int = 42,
    ):
        """
        Args:
            sources: {"name": (h5_path, ratio)} — ratios are normalized.
            epoch_size: Cap on positions per epoch. None = use everything.
            seed: RNG seed for subsampling when epoch_size < total.
        """
        super().__init__()

        # Normalize ratios
        raw_ratios = {name: ratio for name, (_, ratio) in sources.items()}
        total_ratio = sum(raw_ratios.values())
        ratios = {k: v / total_ratio for k, v in raw_ratios.items()} if total_ratio > 0 else raw_ratios

        # Probe all sources first to estimate memory before loading
        source_info: list[tuple[str, str, float, int, np.dtype]] = []
        total_available = 0
        any_uint8 = False

        for name, (path, _) in sources.items():
            if not os.path.exists(path):
                print(f"  Warning: '{name}' not found: {path}, skipping")
                continue
            ratio = ratios[name]
            with h5py.File(path, "r") as f:
                n = int(f.attrs["total_positions"])
                dtype = f["boards"].dtype
                if dtype == np.uint8:
                    any_uint8 = True
                source_info.append((name, path, ratio, n, dtype))
                total_available += n

        effective_size = epoch_size if epoch_size is not None else total_available
        # Memory estimate: boards (uint8) + values (f32) + policies (i64)
        bytes_per_pos = 65 * 27 * 1 + 4 + 8  # uint8 boards + f32 value + i64 policy
        est_bytes = effective_size * bytes_per_pos
        est_gb = est_bytes / (1024 ** 3)

        # Check available memory (best-effort, non-fatal)
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            print(f"Memory estimate: {est_gb:.1f} GB needed, {avail_gb:.1f} GB available")
            if est_gb > avail_gb * 0.85:
                print(f"  ⚠ WARNING: Dataset may exceed available RAM. "
                      f"Use --epoch-size to limit.")
        except ImportError:
            print(f"Memory estimate: ~{est_gb:.1f} GB (install psutil for availability check)")

        # Read all sources into numpy lists, then concatenate once
        all_boards: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        all_policies: list[np.ndarray] = []
        self._has_uint8 = any_uint8

        print("Loading HDF5 data into RAM...")
        for name, path, ratio, total_positions, dtype in source_info:
            t0 = time.time()

            with h5py.File(path, "r") as f:
                n_target = min(int((epoch_size or total_positions) * ratio), total_positions)

                if n_target >= total_positions:
                    boards = f["boards"][:]
                    values = f["values"][:]
                    policies = f["policies"][:]
                else:
                    chunk_size = f["boards"].chunks[0]
                    n_chunks_total = math.ceil(total_positions / chunk_size)
                    n_chunks_needed = min(
                        math.ceil(n_target / chunk_size),
                        n_chunks_total,
                    )
                    rng = np.random.default_rng(seed)
                    chosen = np.sort(rng.choice(n_chunks_total, size=n_chunks_needed, replace=False))

                    chunk_boards, chunk_values, chunk_policies = [], [], []
                    for ci in chosen:
                        start = int(ci) * chunk_size
                        end = min(start + chunk_size, total_positions)
                        chunk_boards.append(f["boards"][start:end])
                        chunk_values.append(f["values"][start:end])
                        chunk_policies.append(f["policies"][start:end])

                    boards = np.concatenate(chunk_boards)
                    values = np.concatenate(chunk_values)
                    policies = np.concatenate(chunk_policies)
                    del chunk_boards, chunk_values, chunk_policies

                    if len(values) > n_target:
                        perm = rng.choice(len(values), size=n_target, replace=False)
                        boards = boards[perm]
                        values = values[perm]
                        policies = policies[perm]

                # Normalize boards to uint8 if source is float32
                if dtype != np.uint8:
                    boards = np.clip(np.round(boards * 255 if boards.max() <= 1.0 else boards),
                                     0, 255).astype(np.uint8)

            elapsed = time.time() - t0
            print(f"  {name}: {len(values):,} positions loaded in {elapsed:.1f}s "
                  f"(ratio: {ratio:.0%}, dtype: {dtype})")

            all_boards.append(boards)
            all_values.append(values)
            all_policies.append(policies)
            del boards, values, policies

        # Concatenate and free source lists incrementally to reduce peak memory
        boards_np = np.concatenate(all_boards)
        del all_boards
        gc.collect()

        values_np = np.concatenate(all_values)
        del all_values

        policies_np = np.concatenate(all_policies)
        del all_policies
        gc.collect()

        # Subsample to epoch_size if we loaded more than requested
        total_loaded = len(values_np)
        if epoch_size is not None and total_loaded > epoch_size:
            rng = np.random.default_rng(seed + 1)
            perm = rng.choice(total_loaded, size=epoch_size, replace=False)
            boards_np = boards_np[perm]
            values_np = values_np[perm]
            policies_np = policies_np[perm]
            del perm
            total_loaded = epoch_size

        # Validate policy range
        max_pol = int(policies_np.max())
        if max_pol >= 4096:
            raise ValueError(
                f"Policy index {max_pol} >= 4096. Data may be corrupt."
            )

        # Store boards as uint8 tensor — 4× less memory than float32.
        # Conversion to float32 happens per-batch in __getitem__.
        self.boards = torch.from_numpy(np.ascontiguousarray(boards_np))
        del boards_np

        self.values = torch.tensor(values_np, dtype=torch.float32)
        del values_np

        # Convert directly to int64 without intermediate numpy copy
        self.policies = torch.from_numpy(policies_np).long()
        del policies_np
        gc.collect()

        print(f"  Total: {len(self.values):,} positions in memory "
              f"({self.boards.nbytes / 1e9:.1f} GB boards [uint8] + "
              f"{self.values.nbytes / 1e6:.0f} MB values + "
              f"{self.policies.nbytes / 1e6:.0f} MB policies)")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = self.boards[idx].float()
        if self._has_uint8:
            board[HALFMOVE_TOKEN, HALFMOVE_FEAT] /= HALFMOVE_SCALE
        return board, self.values[idx], self.policies[idx]


def train(
    model: ChessTransformer,
    train_loader: DataLoader,
    config: TrainingConfig,
    output_path: str,
    device: torch.device,
    log_dir: str | None = None,
    val_loader: DataLoader | None = None,
) -> None:
    """Main training loop with warmup, cosine decay, mixed precision, and checkpointing."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None

    model.train()
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_samples = 0
        t0 = time.time()

        for boards, values, policies in train_loader:
            boards = boards.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)

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
            f"{elapsed:.1f}s",
            flush=True,
        )

        if writer:
            writer.add_scalar("epoch/loss", avg_loss, epoch)
            writer.add_scalar("epoch/value_loss", avg_value_loss, epoch)
            writer.add_scalar("epoch/policy_loss", avg_policy_loss, epoch)

        # Validation
        val_loss = None
        if val_loader is not None and (epoch + 1) % config.eval_every == 0:
            val_loss = _evaluate(model, val_loader, config, device)
            print(f"  val_loss: {val_loss:.6f}", flush=True)
            if writer:
                writer.add_scalar("epoch/val_loss", val_loss, epoch)

        # Checkpoint
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
            print(f"  Saved checkpoint to {output_path}", flush=True)

            compare_loss = val_loss if val_loss is not None else avg_loss
            if compare_loss < best_val_loss:
                best_val_loss = compare_loss
                best_path = output_path.replace(".pt", "_best.pt")
                torch.save(checkpoint, best_path)
                print(f"  New best model: {best_path} "
                      f"({'val' if val_loss is not None else 'train'}_loss="
                      f"{compare_loss:.6f})", flush=True)

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
        boards = boards.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)

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
    parser = argparse.ArgumentParser(description="Train chess transformer (v2)")
    parser.add_argument("--data", default=None,
                        help="Single HDF5 file (e.g., data/round_0.h5)")
    parser.add_argument("--output", default="models/model.pt",
                        help="Output model path")
    parser.add_argument("--checkpoint", default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--model-size", default="small",
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-dir", default="runs/",
                        help="TensorBoard log dir")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction of data for validation (default: 0.05)")
    parser.add_argument("--epoch-size", type=int, default=None,
                        help="Cap positions per epoch (default: use all data)")

    # Multi-source mixing (HDF5 files only)
    parser.add_argument("--ccrl", nargs=2, metavar=("H5_FILE", "RATIO"),
                        help="CCRL HDF5 file and ratio (e.g., data/round_0.h5 0.8)")
    parser.add_argument("--synthetic", nargs=2, metavar=("H5_FILE", "RATIO"),
                        help="Synthetic HDF5 file and ratio")
    parser.add_argument("--endgame", nargs=2, metavar=("H5_FILE", "RATIO"),
                        help="Endgame HDF5 file and ratio")
    parser.add_argument("--opening", nargs=2, metavar=("H5_FILE", "RATIO"),
                        help="Opening HDF5 file and ratio")

    args = parser.parse_args()

    # Build sources dict
    sources = {}
    for name in ["ccrl", "synthetic", "endgame", "opening"]:
        val = getattr(args, name)
        if val:
            sources[name] = (val[0], float(val[1]))

    if args.data:
        if sources:
            parser.error("Cannot use --data with --ccrl/--synthetic/--endgame/--opening")
        sources["data"] = (args.data, 1.0)

    if not sources:
        parser.error("Provide --data or at least one source (--ccrl, --synthetic, etc.)")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # Load model
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    # Load dataset — all HDF5 I/O happens here, before training
    dataset = PreloadedDataset(
        sources, epoch_size=args.epoch_size, seed=42,
    )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    if train_config.val_fraction > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    print(f"Batches per epoch: {len(train_loader)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Train — no open file handles, no shared memory, just tensors
    train(model, train_loader, train_config, args.output, device,
          args.log_dir, val_loader)


if __name__ == "__main__":
    main()
