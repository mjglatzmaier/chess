"""
Multi-source dataset mixer for blending training data.

Samples from multiple data sources (CCRL, synthetic, endgame, opening) with
configurable ratios per batch, enabling balanced training across data types.

Usage:
    python data_mixer.py \\
        --ccrl data/round_0/ 0.6 \\
        --synthetic data/synthetic/ 0.2 \\
        --output data/mixed/

    # Or use in code:
    mixer = DataMixer({"ccrl": ("data/round_0/", 0.6), "synthetic": ("data/synthetic/", 0.2)})
    loader = mixer.get_dataloader(batch_size=256)
"""

import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from encoding import NUM_FEATURES


# Source ID mapping (stored in the sources field of chunks)
SOURCE_IDS = {
    "ccrl": 0,
    "synthetic": 1,
    "endgame": 2,
    "opening": 3,
}


class MixedChessDataset(Dataset):
    """
    Dataset that wraps multiple chunk directories into a unified view.

    Tracks which source each position comes from, enabling per-source
    loss tracking and proportional sampling.
    """

    def __init__(
        self,
        sources: dict[str, tuple[str, float]],
        chunk_cache_size: int = 4,
    ):
        """
        Args:
            sources: {"name": (data_dir, ratio)} where ratios should sum to ~1.0.
            chunk_cache_size: Number of chunks to keep in LRU cache.
        """
        self.source_names = list(sources.keys())
        self.source_dirs = {name: path for name, (path, _) in sources.items()}
        self.source_ratios = {name: ratio for name, (_, ratio) in sources.items()}
        self.chunk_cache_size = chunk_cache_size

        # Normalize ratios
        total_ratio = sum(self.source_ratios.values())
        if total_ratio > 0:
            self.source_ratios = {
                k: v / total_ratio for k, v in self.source_ratios.items()
            }

        # Build unified index: list of (source_name, chunk_path, chunk_offset, chunk_size)
        self.entries: list[tuple[str, str, int, int]] = []  # per-chunk entries
        self.chunk_cumulative: list[int] = []
        # Store source ranges as (start, end) for memory-efficient sampling
        self.source_ranges: dict[str, list[tuple[int, int]]] = {n: [] for n in self.source_names}
        total = 0

        for name in self.source_names:
            data_dir = self.source_dirs[name]
            if not os.path.isdir(data_dir):
                print(f"Warning: source '{name}' directory not found: {data_dir}")
                continue

            chunk_paths = sorted(Path(data_dir).glob("chunk_*.npz"))

            # Fast path: use metadata to avoid scanning every chunk
            meta_path = os.path.join(data_dir, "metadata.npz")
            if os.path.exists(meta_path):
                meta = np.load(meta_path)
                total_positions = int(meta["total_positions"])
                num_chunks = int(meta["num_chunks"])
                chunk_size = int(meta["chunk_size"])
                # All chunks are chunk_size except possibly the last
                for i, cpath in enumerate(chunk_paths[:num_chunks]):
                    if i < num_chunks - 1:
                        sz = chunk_size
                    else:
                        sz = total_positions - chunk_size * (num_chunks - 1)
                    start_idx = total
                    self.entries.append((name, str(cpath), start_idx, sz))
                    total += sz
                    self.chunk_cumulative.append(total)
                    self.source_ranges[name].append((start_idx, total))
            else:
                for cpath in chunk_paths:
                    with np.load(str(cpath)) as data:
                        sz = len(data["values"])
                    start_idx = total
                    self.entries.append((name, str(cpath), start_idx, sz))
                    total += sz
                    self.chunk_cumulative.append(total)
                    self.source_ranges[name].append((start_idx, total))

        self.total_size = total
        self._source_sizes: dict[str, int] = {
            name: sum(end - start for start, end in ranges)
            for name, ranges in self.source_ranges.items()
        }

        # LRU chunk cache
        self._chunk_cache: OrderedDict[str, tuple] = OrderedDict()

        # Print summary
        print(f"MixedChessDataset: {self.total_size:,} total positions")
        for name in self.source_names:
            count = self._source_sizes.get(name, 0)
            ratio = self.source_ratios.get(name, 0)
            print(f"  {name}: {count:,} positions (target ratio: {ratio:.1%})")

    def _load_chunk(self, chunk_path: str) -> tuple:
        """Load chunk with LRU caching."""
        if chunk_path in self._chunk_cache:
            self._chunk_cache.move_to_end(chunk_path)
            return self._chunk_cache[chunk_path]

        while len(self._chunk_cache) >= self.chunk_cache_size:
            self._chunk_cache.popitem(last=False)

        data = np.load(chunk_path)
        entry = (
            data["boards"],
            data["values"],
            data["policies"] if "policies" in data else None,
            data["sources"] if "sources" in data else None,
            data["weights"] if "weights" in data else None,
        )
        self._chunk_cache[chunk_path] = entry
        return entry

    def _find_chunk(self, idx: int) -> tuple[str, str, int]:
        """Find chunk containing global index. Returns (source, chunk_path, local_idx)."""
        import bisect
        ci = bisect.bisect_right(self.chunk_cumulative, idx)
        if ci >= len(self.entries):
            raise IndexError(f"Index {idx} out of range")
        source, cpath, start, size = self.entries[ci]
        local = idx - start
        return source, cpath, local

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, cpath, local = self._find_chunk(idx)
        boards, values, policies, sources_arr, weights_arr = self._load_chunk(cpath)

        board = torch.from_numpy(boards[local].copy())
        value = torch.tensor(values[local])
        pol = policies[local] if policies is not None else 0
        policy = torch.tensor(pol, dtype=torch.long)

        return board, value, policy


class ProportionalSampler(Sampler):
    """
    Chunk-aware sampler that draws from each source proportionally per epoch.

    Instead of fully random indices (which thrash the chunk cache), this
    sampler selects random chunks from each source, then yields all positions
    from each chunk before moving to the next. Chunks are shuffled for
    inter-chunk randomness; positions within each chunk are shuffled for
    intra-chunk randomness. This keeps LRU cache hit rate near 100%.
    """

    def __init__(
        self,
        dataset: MixedChessDataset,
        epoch_size: int | None = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.epoch_size = epoch_size or dataset.total_size
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        # For each source, select random chunks to fill the target count
        chunk_groups: list[tuple[int, int, int]] = []  # (start_idx, chunk_size, take)

        for name, ratio in self.dataset.source_ratios.items():
            ranges = self.dataset.source_ranges[name]
            if not ranges:
                continue
            n_target = int(self.epoch_size * ratio)

            remaining = n_target
            while remaining > 0:
                shuffled = list(ranges)
                rng.shuffle(shuffled)
                for start, end in shuffled:
                    if remaining <= 0:
                        break
                    chunk_size = end - start
                    take = min(chunk_size, remaining)
                    chunk_groups.append((start, chunk_size, take))
                    remaining -= take

        # Shuffle chunk order for inter-chunk randomness
        rng.shuffle(chunk_groups)

        # Yield indices: within each chunk, sample and shuffle
        indices: list[int] = []
        for start, chunk_size, take in chunk_groups:
            if take >= chunk_size:
                chunk_indices = list(range(start, start + chunk_size))
            else:
                chunk_indices = rng.sample(range(start, start + chunk_size), take)
            rng.shuffle(chunk_indices)
            indices.extend(chunk_indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self._epoch = epoch


class DataMixer:
    """
    High-level interface for creating mixed-source DataLoaders.

    Example:
        mixer = DataMixer({
            "ccrl": ("data/round_0/", 0.6),
            "synthetic": ("data/synthetic/", 0.2),
        })
        loader = mixer.get_dataloader(batch_size=256)
    """

    def __init__(self, sources: dict[str, tuple[str, float]]):
        self.sources = sources
        self.dataset = MixedChessDataset(sources)

    def get_dataloader(
        self,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        epoch_size: int | None = None,
    ) -> DataLoader:
        """
        Returns a DataLoader that samples from sources according to ratios.

        Args:
            batch_size: Batch size
            num_workers: Data loading workers
            pin_memory: Pin memory for GPU transfer
            epoch_size: Override epoch size (default: total dataset size)
        """
        sampler = ProportionalSampler(
            self.dataset, epoch_size=epoch_size,
        )
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    def get_sampler(self) -> ProportionalSampler:
        """Get the sampler for manual epoch management."""
        return ProportionalSampler(self.dataset)


def merge_to_chunks(
    sources: dict[str, tuple[str, float]],
    output_dir: str,
    chunk_size: int = 100_000,
) -> None:
    """
    Merge multiple source directories into a single output directory
    with proportional sampling pre-applied. Useful for offline preparation.

    Works chunk-by-chunk for efficiency: loads each input chunk once,
    takes all (or a random subsample of) its positions, and streams
    into output chunks.
    """
    os.makedirs(output_dir, exist_ok=True)
    from tqdm import tqdm

    # Normalize ratios
    total_ratio = sum(r for _, r in sources.values())
    ratios = {name: r / total_ratio for name, (_, r) in sources.items()}

    # Discover chunks and sizes per source
    source_chunks: dict[str, list[tuple[str, int]]] = {}
    source_totals: dict[str, int] = {}
    for name, (data_dir, _) in sources.items():
        chunks = []
        total = 0
        for cpath in sorted(Path(data_dir).glob("chunk_*.npz")):
            with np.load(str(cpath)) as data:
                sz = len(data["values"])
            chunks.append((str(cpath), sz))
            total += sz
        source_chunks[name] = chunks
        source_totals[name] = total
        print(f"  {name}: {total:,} positions in {len(chunks)} chunks")

    # Determine how many positions to take from each source.
    # Use the largest source to anchor, then scale others to match ratios.
    # E.g., 80/20 with 271M ccrl → take all 271M ccrl + 67.8M synthetic
    # But if synthetic only has 552K, oversample it to 67.8M.
    # To keep output size reasonable, cap at the largest source's full size
    # divided by its ratio.
    anchor_name = max(source_totals, key=source_totals.get)
    anchor_total = source_totals[anchor_name]
    anchor_ratio = ratios[anchor_name]
    dataset_size = int(anchor_total / anchor_ratio)

    target_counts = {name: int(dataset_size * ratios[name]) for name in sources}
    print(f"\nTarget mix ({dataset_size:,} total):")
    for name, count in target_counts.items():
        avail = source_totals[name]
        repeats = count / avail if avail > 0 else 0
        print(f"  {name}: {count:,} positions ({ratios[name]:.0%})"
              f" — {repeats:.1f}x of available {avail:,}")

    # Build the work list: (chunk_path, source_name, sample_fraction)
    # For sources needing oversampling, repeat chunks multiple times.
    rng = np.random.default_rng(42)
    work: list[tuple[str, str, float]] = []

    for name in sources:
        chunks = source_chunks[name]
        avail = source_totals[name]
        target = target_counts[name]
        if avail == 0:
            continue

        if target <= avail:
            # Subsample: take fraction of each chunk
            frac = target / avail
            for cpath, sz in chunks:
                work.append((cpath, name, frac))
        else:
            # Oversample: full passes + a fractional remainder pass
            full_passes = target // avail
            remainder_frac = (target % avail) / avail
            for _ in range(full_passes):
                for cpath, sz in chunks:
                    work.append((cpath, name, 1.0))
            if remainder_frac > 0:
                for cpath, sz in chunks:
                    work.append((cpath, name, remainder_frac))

    # Shuffle work order for good mixing
    rng.shuffle(work)

    # Process chunks and stream into output
    boards_buf = []
    values_buf = []
    policy_buf = []
    sources_buf = []
    weights_buf = []
    buf_len = 0
    chunk_idx = 0
    total_written = 0

    for cpath, name, frac in tqdm(work, desc="Merging chunks"):
        data = np.load(cpath)
        n = len(data["values"])
        src_id = SOURCE_IDS.get(name, 0)

        if frac >= 1.0:
            sel = np.arange(n)
        else:
            k = max(1, int(n * frac))
            sel = rng.choice(n, size=k, replace=False)
            sel.sort()

        boards_buf.append(data["boards"][sel])
        values_buf.append(data["values"][sel])
        if "policies" in data:
            policy_buf.append(data["policies"][sel])
        else:
            policy_buf.append(np.zeros(len(sel), dtype=np.int64))
        sources_buf.append(np.full(len(sel), src_id, dtype=np.uint8))
        if "weights" in data:
            weights_buf.append(data["weights"][sel])
        else:
            weights_buf.append(np.ones(len(sel), dtype=np.float32))
        buf_len += len(sel)

        # Flush when buffer exceeds chunk_size
        while buf_len >= chunk_size:
            b = np.concatenate(boards_buf)
            v = np.concatenate(values_buf)
            p = np.concatenate(policy_buf)
            s = np.concatenate(sources_buf)
            w = np.concatenate(weights_buf)

            # Shuffle within the output chunk
            perm = rng.permutation(len(b))
            b, v, p, s, w = b[perm], v[perm], p[perm], s[perm], w[perm]

            # Write chunk_size positions, keep remainder
            _save_mixed_chunk_arrays(
                output_dir, chunk_idx,
                b[:chunk_size], v[:chunk_size], p[:chunk_size],
                s[:chunk_size], w[:chunk_size],
            )
            chunk_idx += 1
            total_written += chunk_size

            # Keep leftovers
            b, v, p, s, w = b[chunk_size:], v[chunk_size:], p[chunk_size:], s[chunk_size:], w[chunk_size:]
            boards_buf = [b] if len(b) > 0 else []
            values_buf = [v] if len(v) > 0 else []
            policy_buf = [p] if len(p) > 0 else []
            sources_buf = [s] if len(s) > 0 else []
            weights_buf = [w] if len(w) > 0 else []
            buf_len = len(b)

    # Final partial chunk
    if buf_len > 0:
        b = np.concatenate(boards_buf)
        v = np.concatenate(values_buf)
        p = np.concatenate(policy_buf)
        s = np.concatenate(sources_buf)
        w = np.concatenate(weights_buf)
        perm = rng.permutation(len(b))
        _save_mixed_chunk_arrays(
            output_dir, chunk_idx,
            b[perm], v[perm], p[perm], s[perm], w[perm],
        )
        chunk_idx += 1
        total_written += len(b)

    metadata = {
        "total_positions": total_written,
        "total_games": 0,
        "skipped_games": 0,
        "num_chunks": chunk_idx,
        "chunk_size": chunk_size,
        "num_features": NUM_FEATURES,
        "skip_moves": 0,
        "source": "mixed",
        "has_policy": True,
    }
    np.savez(os.path.join(output_dir, "metadata.npz"), **metadata)
    print(f"\nDone: {total_written:,} positions in {chunk_idx} chunks → {output_dir}/")


def _save_mixed_chunk(
    output_dir: str,
    chunk_idx: int,
    boards: list,
    values: list,
    policies: list,
    sources: list,
    weights: list,
) -> None:
    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        boards=np.stack(boards, axis=0),
        values=np.array(values, dtype=np.float32),
        policies=np.array(policies, dtype=np.int64),
        sources=np.array(sources, dtype=np.uint8),
        weights=np.array(weights, dtype=np.float32),
    )


def _save_mixed_chunk_arrays(
    output_dir: str,
    chunk_idx: int,
    boards: np.ndarray,
    values: np.ndarray,
    policies: np.ndarray,
    sources: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Save pre-assembled numpy arrays directly (no stacking needed)."""
    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        boards=boards,
        values=values.astype(np.float32),
        policies=policies.astype(np.int64),
        sources=sources.astype(np.uint8),
        weights=weights.astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Mix multiple training data sources")
    parser.add_argument(
        "--ccrl", nargs=2, metavar=("DIR", "RATIO"),
        help="CCRL data directory and ratio (e.g., data/round_0/ 0.6)",
    )
    parser.add_argument(
        "--synthetic", nargs=2, metavar=("DIR", "RATIO"),
        help="Synthetic data directory and ratio",
    )
    parser.add_argument(
        "--endgame", nargs=2, metavar=("DIR", "RATIO"),
        help="Endgame data directory and ratio",
    )
    parser.add_argument(
        "--opening", nargs=2, metavar=("DIR", "RATIO"),
        help="Opening data directory and ratio",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for mixed chunks",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="Positions per output chunk (default: 100000)",
    )
    args = parser.parse_args()

    sources = {}
    for name in ["ccrl", "synthetic", "endgame", "opening"]:
        val = getattr(args, name)
        if val:
            sources[name] = (val[0], float(val[1]))

    if not sources:
        parser.error("At least one data source is required")

    merge_to_chunks(sources, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
