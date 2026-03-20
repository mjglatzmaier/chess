"""
HDF5-backed dataset for multi-source chess training.

Replaces the .npz chunk-based MixedChessDataset with HDF5 files that support:
  - uint8 board storage (4x smaller than float32)
  - Efficient partial reads (no full-chunk decompression)
  - No LRU cache needed (h5py manages its own chunk cache)
  - Single file per source (no thousands of chunk files)

Usage:
    dataset = HDF5ChessDataset({
        "ccrl": ("data/round_0.h5", 0.8),
        "synthetic": ("data/synthetic.h5", 0.2),
    })
    sampler = HDF5ProportionalSampler(dataset, epoch_size=5_000_000)
    loader = DataLoader(dataset, batch_size=256, sampler=sampler)
"""

import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# Halfmove clock location in the board encoding
HALFMOVE_TOKEN = 64
HALFMOVE_FEAT = 26
HALFMOVE_SCALE = 255.0

# HDF5 chunk cache: 128 MB per file (controls h5py's internal read cache)
H5_CHUNK_CACHE_BYTES = 128 * 1024 * 1024


class HDF5ChessDataset(Dataset):
    """
    Dataset backed by HDF5 files with uint8 boards.

    Each source is a single .h5 file containing boards, values, policies,
    etc. as chunked+compressed datasets. h5py handles caching internally,
    so no LRU cache is needed.
    """

    def __init__(self, sources: dict[str, tuple[str, float]]):
        """
        Args:
            sources: {"name": (h5_path, ratio)} where ratios sum to ~1.0.
        """
        self.source_names = list(sources.keys())
        self.source_paths = {name: path for name, (path, _) in sources.items()}
        self.source_ratios = {name: ratio for name, (_, ratio) in sources.items()}

        # Normalize ratios
        total_ratio = sum(self.source_ratios.values())
        if total_ratio > 0:
            self.source_ratios = {
                k: v / total_ratio for k, v in self.source_ratios.items()
            }

        # Open HDF5 files (kept open for process lifetime)
        self._files: dict[str, h5py.File] = {}
        self.source_sizes: dict[str, int] = {}
        self.source_offsets: dict[str, int] = {}  # global offset per source
        total = 0

        for name in self.source_names:
            path = self.source_paths[name]
            if not os.path.exists(path):
                print(f"Warning: source '{name}' file not found: {path}")
                continue

            f = h5py.File(
                path, "r",
                rdcc_nbytes=H5_CHUNK_CACHE_BYTES,
                rdcc_nslots=10007,  # prime number, recommended by h5py docs
            )
            self._files[name] = f

            n = f.attrs["total_positions"]
            self.source_sizes[name] = int(n)
            self.source_offsets[name] = total
            total += int(n)

        self.total_size = total

        # Detect encoding version (uint8 vs float32 boards)
        for name in self.source_names:
            if name in self._files:
                f = self._files[name]
                self._encoding_version = int(f.attrs.get("encoding_version", 1))
                self._board_dtype = f["boards"].dtype
                break

        # Print summary
        print(f"HDF5ChessDataset: {self.total_size:,} total positions")
        for name in self.source_names:
            count = self.source_sizes.get(name, 0)
            ratio = self.source_ratios.get(name, 0)
            print(f"  {name}: {count:,} positions (target ratio: {ratio:.1%})")
        print(f"  Board dtype: {self._board_dtype}, "
              f"encoding v{self._encoding_version}")

    def _find_source(self, idx: int) -> tuple[str, int]:
        """Map global index to (source_name, local_index)."""
        for name in self.source_names:
            offset = self.source_offsets.get(name, 0)
            size = self.source_sizes.get(name, 0)
            if offset <= idx < offset + size:
                return name, idx - offset
        raise IndexError(f"Index {idx} out of range (total: {self.total_size})")

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name, local = self._find_source(idx)
        f = self._files[name]

        # Read directly from HDF5 — h5py decompresses only the relevant chunk
        board_raw = f["boards"][local]    # uint8 [65, 27] or float32
        value = f["values"][local]        # float32 scalar
        policy = f["policies"][local]     # uint16 or int64 scalar

        # Convert uint8 board to float32 for the model
        if self._board_dtype == np.uint8:
            board = torch.from_numpy(board_raw.astype(np.float32))
            # Reconstruct halfmove clock from quantized uint8
            board[HALFMOVE_TOKEN, HALFMOVE_FEAT] /= HALFMOVE_SCALE
        else:
            board = torch.from_numpy(board_raw.copy())

        value_t = torch.tensor(value, dtype=torch.float32)
        policy_t = torch.tensor(int(policy), dtype=torch.long)

        return board, value_t, policy_t

    def close(self):
        """Close all HDF5 file handles."""
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __del__(self):
        self.close()


class HDF5ProportionalSampler(Sampler):
    """
    Proportional sampler for HDF5 datasets.

    Draws from each source according to its ratio per epoch. Indices are
    generated in blocks sorted by source for read locality (sequential
    HDF5 reads are much faster than random), then shuffled at the block
    level for inter-source mixing.
    """

    def __init__(
        self,
        dataset: HDF5ChessDataset,
        epoch_size: int | None = None,
        seed: int = 42,
        block_size: int = 10_000,
    ):
        """
        Args:
            dataset: The HDF5ChessDataset to sample from.
            epoch_size: Positions per epoch (default: total dataset size).
            seed: Random seed for reproducibility.
            block_size: Number of indices per block. Blocks are sorted
                internally for read locality, then shuffled across blocks.
        """
        self.dataset = dataset
        self.epoch_size = epoch_size or dataset.total_size
        self.seed = seed
        self.block_size = block_size
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        np_rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        # For each source, sample the target number of random indices
        blocks: list[np.ndarray] = []

        for name, ratio in self.dataset.source_ratios.items():
            source_size = self.dataset.source_sizes.get(name, 0)
            if source_size == 0:
                continue
            offset = self.dataset.source_offsets[name]
            n_target = int(self.epoch_size * ratio)

            if n_target <= source_size:
                # Subsample: draw n_target unique indices
                local_indices = np_rng.choice(
                    source_size, size=n_target, replace=False
                )
            else:
                # Oversample: full passes + remainder
                full_passes = n_target // source_size
                remainder = n_target % source_size
                parts = []
                for _ in range(full_passes):
                    parts.append(np_rng.permutation(source_size))
                if remainder > 0:
                    parts.append(np_rng.choice(
                        source_size, size=remainder, replace=False
                    ))
                local_indices = np.concatenate(parts)

            # Convert to global indices
            global_indices = local_indices + offset

            # Split into blocks, sort within each block for read locality
            for i in range(0, len(global_indices), self.block_size):
                block = global_indices[i:i + self.block_size]
                block.sort()
                blocks.append(block)

        # Shuffle blocks for inter-source mixing
        rng.shuffle(blocks)

        # Yield indices block by block
        for block in blocks:
            yield from block.tolist()

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self._epoch = epoch
