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
        self.source_indices: dict[str, list[int]] = {n: [] for n in self.source_names}
        total = 0

        for name in self.source_names:
            data_dir = self.source_dirs[name]
            if not os.path.isdir(data_dir):
                print(f"Warning: source '{name}' directory not found: {data_dir}")
                continue

            chunk_paths = sorted(Path(data_dir).glob("chunk_*.npz"))
            for cpath in chunk_paths:
                with np.load(str(cpath)) as data:
                    sz = len(data["values"])
                start_idx = total
                self.entries.append((name, str(cpath), start_idx, sz))
                total += sz
                self.chunk_cumulative.append(total)
                # Record which global indices belong to this source
                for i in range(start_idx, total):
                    self.source_indices[name].append(i)

        self.total_size = total

        # LRU chunk cache
        self._chunk_cache: OrderedDict[str, tuple] = OrderedDict()

        # Print summary
        print(f"MixedChessDataset: {self.total_size:,} total positions")
        for name in self.source_names:
            count = len(self.source_indices[name])
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
    Sampler that draws from each source proportionally per epoch.

    Instead of uniform random sampling (which would over-represent larger
    sources), this sampler ensures each batch contains approximately the
    target ratio of each source.
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

        indices = []
        for name, ratio in self.dataset.source_ratios.items():
            source_indices = self.dataset.source_indices[name]
            if not source_indices:
                continue
            # Number of samples from this source for this epoch
            n_samples = int(self.epoch_size * ratio)
            # Sample with replacement if source is smaller than needed
            if n_samples <= len(source_indices):
                sampled = rng.sample(source_indices, n_samples)
            else:
                sampled = rng.choices(source_indices, k=n_samples)
            indices.extend(sampled)

        rng.shuffle(indices)
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
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = MixedChessDataset(sources)
    sampler = ProportionalSampler(dataset)
    indices = list(sampler)

    boards_buf, values_buf, policy_buf = [], [], []
    sources_buf, weights_buf = [], []
    chunk_idx = 0
    total = 0

    print(f"Merging {len(indices):,} positions into {output_dir}/")
    from tqdm import tqdm

    for idx in tqdm(indices, desc="Merging"):
        source, cpath, local = dataset._find_chunk(idx)
        boards, values, policies, src_arr, wt_arr = dataset._load_chunk(cpath)

        boards_buf.append(boards[local])
        values_buf.append(values[local])
        pol = policies[local] if policies is not None else 0
        policy_buf.append(pol)
        sources_buf.append(SOURCE_IDS.get(source, 0))
        weights_buf.append(wt_arr[local] if wt_arr is not None else 1.0)
        total += 1

        if len(boards_buf) >= chunk_size:
            _save_mixed_chunk(
                output_dir, chunk_idx,
                boards_buf, values_buf, policy_buf, sources_buf, weights_buf,
            )
            chunk_idx += 1
            boards_buf, values_buf, policy_buf = [], [], []
            sources_buf, weights_buf = [], []

    if boards_buf:
        _save_mixed_chunk(
            output_dir, chunk_idx,
            boards_buf, values_buf, policy_buf, sources_buf, weights_buf,
        )
        chunk_idx += 1

    metadata = {
        "total_positions": total,
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
    print(f"\nDone: {total:,} positions in {chunk_idx} chunks → {output_dir}/")


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
