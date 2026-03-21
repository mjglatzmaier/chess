"""
HDF5-backed dataset for multi-source chess training.

Provides two dataset implementations:

1. HDF5ChunkIterableDataset (recommended for training):
   Reads full HDF5 chunks sequentially, amortizing gzip decompression
   across all positions in each chunk. A shuffle buffer mixes positions
   across multiple chunks for training randomness.

2. HDF5ChessDataset (for validation / spot-checks):
   Map-style dataset with per-sample random access. Fine for small
   validation sets but causes catastrophic I/O on large gzip-compressed
   files (each 1.7 KB sample decompresses a ~17 MB chunk).

Usage:
    dataset = HDF5ChunkIterableDataset({
        "ccrl": ("data/round_0.h5", 0.8),
        "synthetic": ("data/synthetic.h5", 0.2),
    }, epoch_size=5_000_000)
    loader = DataLoader(dataset, batch_size=256)
"""

import gc
import math
import os
import random
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

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


class HDF5ChunkIterableDataset(IterableDataset):
    """
    Chunk-aligned iterable dataset for efficient HDF5 training reads.

    Instead of reading individual positions (which decompresses entire gzip
    chunks for single 1.7 KB samples — 0.01% efficiency), this reads full
    HDF5 chunks sequentially and buffers multiple chunks for cross-chunk
    shuffling before yielding samples.

    I/O pattern:
        - Sequential full-chunk reads (gzip decompression amortized over 10K positions)
        - Shuffle buffer mixes positions across `buffer_chunks` chunks (~80K positions)
        - Chunks from different sources are interleaved proportionally

    Memory usage:
        buffer_chunks × chunk_size × 1,755 bytes (uint8) ≈ 140 MB for 8 chunks,
        plus ~560 MB for float32 conversion. ~700 MB total peak.
    """

    def __init__(
        self,
        sources: dict[str, tuple[str, float]],
        epoch_size: int | None = None,
        seed: int = 42,
        buffer_chunks: int = 8,
    ):
        """
        Args:
            sources: {"name": (h5_path, ratio)} where ratios sum to ~1.0.
            epoch_size: Positions per epoch. Defaults to total dataset size.
            seed: Random seed for reproducibility.
            buffer_chunks: Number of HDF5 chunks to accumulate before
                shuffling and yielding. Larger = better randomness, more RAM.
        """
        super().__init__()
        self.seed = seed
        self.buffer_chunks = buffer_chunks
        self._epoch = 0

        self.source_names = list(sources.keys())
        raw_ratios = {name: ratio for name, (_, ratio) in sources.items()}

        # Normalize ratios
        total_ratio = sum(raw_ratios.values())
        self.source_ratios = (
            {k: v / total_ratio for k, v in raw_ratios.items()}
            if total_ratio > 0 else raw_ratios
        )

        # Open HDF5 files and discover chunk layout
        self._files: dict[str, h5py.File] = {}
        self._source_meta: dict[str, dict] = {}
        self.total_positions = 0

        for name, (path, _) in sources.items():
            if not os.path.exists(path):
                print(f"Warning: source '{name}' file not found: {path}")
                continue

            f = h5py.File(
                path, "r",
                rdcc_nbytes=H5_CHUNK_CACHE_BYTES,
                rdcc_nslots=10007,
            )
            self._files[name] = f

            n = int(f.attrs["total_positions"])
            cs = f["boards"].chunks[0]
            self._source_meta[name] = {
                "n_positions": n,
                "chunk_size": cs,
                "n_chunks": math.ceil(n / cs),
            }
            self.total_positions += n

        self.epoch_size = epoch_size or self.total_positions

        # Detect board encoding
        self._board_dtype = np.float32
        for name in self._files:
            self._board_dtype = self._files[name]["boards"].dtype
            break

        # Summary
        print(f"HDF5ChunkIterableDataset: {self.total_positions:,} total positions")
        for name in self._files:
            meta = self._source_meta[name]
            ratio = self.source_ratios.get(name, 0)
            print(f"  {name}: {meta['n_positions']:,} positions, "
                  f"{meta['n_chunks']} chunks of {meta['chunk_size']:,} "
                  f"(ratio: {ratio:.1%})")
        print(f"  Epoch size: {self.epoch_size:,}, "
              f"buffer: {buffer_chunks} chunks")

    def _build_chunk_schedule(
        self, rng: np.random.Generator,
    ) -> list[tuple[str, int, int]]:
        """Build a shuffled list of (source_name, start_idx, end_idx) for one epoch."""
        schedule: list[tuple[str, int, int]] = []

        for name, ratio in self.source_ratios.items():
            if name not in self._source_meta:
                continue
            meta = self._source_meta[name]

            n_target = int(self.epoch_size * ratio)
            n_chunks_needed = min(
                math.ceil(n_target / meta["chunk_size"]),
                meta["n_chunks"],
            )

            if n_chunks_needed >= meta["n_chunks"]:
                chunk_indices = np.arange(meta["n_chunks"])
            else:
                chunk_indices = rng.choice(
                    meta["n_chunks"], size=n_chunks_needed, replace=False,
                )

            for ci in chunk_indices:
                start = int(ci) * meta["chunk_size"]
                end = min(start + meta["chunk_size"], meta["n_positions"])
                schedule.append((name, start, end))

        rng.shuffle(schedule)
        return schedule

    def _flush_buffer(self, buf_boards, buf_values, buf_policies, rng, max_yield):
        """Concatenate buffered chunks, shuffle, yield samples. Returns count."""
        boards = np.concatenate(buf_boards)
        values = np.concatenate(buf_values)
        policies = np.concatenate(buf_policies)

        n = len(values)
        limit = min(n, max_yield)

        # Validate policy range before it hits cross_entropy on GPU
        max_pol = int(policies.max())
        if max_pol >= 4096:
            raise ValueError(
                f"Policy index {max_pol} >= 4096 in buffer. "
                f"HDF5 data may be corrupt or use a different encoding."
            )

        # Vectorized float32 conversion (entire buffer at once)
        boards_f32 = boards.astype(np.float32)
        if self._board_dtype == np.uint8:
            boards_f32[:, HALFMOVE_TOKEN, HALFMOVE_FEAT] /= HALFMOVE_SCALE

        # torch.tensor() copies data so tensors own their memory, decoupled
        # from the numpy arrays. torch.from_numpy() shares memory, which can
        # cause use-after-free when the generator's locals are freed while
        # the DataLoader still holds view references via pin_memory.
        boards_t = torch.tensor(boards_f32, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        policies_t = torch.tensor(policies, dtype=torch.long)

        # Free numpy arrays now that tensors own their own memory
        del buf_boards[:], buf_values[:], buf_policies[:]
        del boards, values, policies, boards_f32

        perm = rng.permutation(n)

        count = 0
        for i in range(limit):
            yield boards_t[perm[i]], values_t[perm[i]], policies_t[perm[i]]
            count += 1
        return count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        schedule = self._build_chunk_schedule(rng)

        buf_boards: list[np.ndarray] = []
        buf_values: list[np.ndarray] = []
        buf_policies: list[np.ndarray] = []
        buf_count = 0
        yielded = 0

        for name, start, end in schedule:
            if yielded >= self.epoch_size:
                break

            f = self._files[name]

            # Read full chunk slice — one decompression, 100% efficient
            buf_boards.append(f["boards"][start:end])
            buf_values.append(f["values"][start:end])
            buf_policies.append(f["policies"][start:end])
            buf_count += 1

            if buf_count >= self.buffer_chunks:
                n = yield from self._flush_buffer(
                    buf_boards, buf_values, buf_policies,
                    rng, self.epoch_size - yielded,
                )
                yielded += n
                buf_boards, buf_values, buf_policies = [], [], []
                buf_count = 0

        # Flush remaining buffer
        if buf_boards and yielded < self.epoch_size:
            n = yield from self._flush_buffer(
                buf_boards, buf_values, buf_policies,
                rng, self.epoch_size - yielded,
            )
            yielded += n

    def __len__(self) -> int:
        return self.epoch_size

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self._epoch = epoch

    def close(self):
        """Close all HDF5 file handles."""
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __del__(self):
        self.close()


class HDF5EpochDataset(Dataset):
    """
    Preloads one epoch of HDF5 data into RAM as a standard map-style Dataset.

    Completely separates HDF5 I/O from GPU training: load_epoch() reads all
    needed chunks upfront into contiguous numpy arrays, then __getitem__
    operates purely on in-memory data with no h5py calls. This avoids any
    interaction between HDF5's C library and CUDA's runtime during training.

    Memory: epoch_size × 1,755 bytes (uint8 boards) ≈ 8.8 GB for 5M positions.
    """

    def __init__(
        self,
        sources: dict[str, tuple[str, float]],
        epoch_size: int | None = None,
        seed: int = 42,
    ):
        """
        Args:
            sources: {"name": (h5_path, ratio)} where ratios sum to ~1.0.
            epoch_size: Positions per epoch. Defaults to total dataset size.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.seed = seed

        self.source_names = list(sources.keys())
        raw_ratios = {name: ratio for name, (_, ratio) in sources.items()}

        # Normalize ratios
        total_ratio = sum(raw_ratios.values())
        self.source_ratios = (
            {k: v / total_ratio for k, v in raw_ratios.items()}
            if total_ratio > 0 else raw_ratios
        )

        # Open HDF5 files and discover chunk layout
        self._files: dict[str, h5py.File] = {}
        self._source_meta: dict[str, dict] = {}
        self.total_positions = 0

        for name, (path, _) in sources.items():
            if not os.path.exists(path):
                print(f"Warning: source '{name}' file not found: {path}")
                continue

            f = h5py.File(path, "r")
            self._files[name] = f

            n = int(f.attrs["total_positions"])
            cs = f["boards"].chunks[0]
            self._source_meta[name] = {
                "n_positions": n,
                "chunk_size": cs,
                "n_chunks": math.ceil(n / cs),
            }
            self.total_positions += n

        self.epoch_size = epoch_size or self.total_positions

        # Detect board encoding
        self._board_dtype = np.float32
        for name in self._files:
            self._board_dtype = self._files[name]["boards"].dtype
            break

        # In-memory epoch data (populated by load_epoch)
        self._boards: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._policies: np.ndarray | None = None
        self._size = self.epoch_size  # For len() before first load_epoch

        # Summary
        print(f"HDF5EpochDataset: {self.total_positions:,} total positions")
        for name in self._files:
            meta = self._source_meta[name]
            ratio = self.source_ratios.get(name, 0)
            print(f"  {name}: {meta['n_positions']:,} positions, "
                  f"{meta['n_chunks']} chunks of {meta['chunk_size']:,} "
                  f"(ratio: {ratio:.1%})")
        est_gb = self.epoch_size * 1755 / 1e9
        print(f"  Epoch size: {self.epoch_size:,} (~{est_gb:.1f} GB in RAM)")

    def load_epoch(self, epoch: int) -> None:
        """Read proportional chunks from HDF5 into RAM for this epoch.

        After this call, __getitem__ operates purely on in-memory numpy
        arrays — no h5py interaction during training.
        """
        # Free previous epoch's data before allocating new
        self._boards = None
        self._values = None
        self._policies = None
        gc.collect()

        t0 = time.time()
        rng = np.random.default_rng(self.seed + epoch)

        all_boards: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        all_policies: list[np.ndarray] = []

        for name, ratio in self.source_ratios.items():
            if name not in self._source_meta:
                continue
            meta = self._source_meta[name]
            f = self._files[name]

            n_target = int(self.epoch_size * ratio)
            n_chunks = min(
                math.ceil(n_target / meta["chunk_size"]),
                meta["n_chunks"],
            )

            if n_chunks >= meta["n_chunks"]:
                chunk_indices = np.arange(meta["n_chunks"])
            else:
                chunk_indices = rng.choice(
                    meta["n_chunks"], size=n_chunks, replace=False,
                )

            chunk_indices.sort()  # Sequential disk reads

            for ci in chunk_indices:
                start = int(ci) * meta["chunk_size"]
                end = min(start + meta["chunk_size"], meta["n_positions"])
                all_boards.append(f["boards"][start:end])
                all_values.append(f["values"][start:end])
                all_policies.append(f["policies"][start:end])

        boards = np.concatenate(all_boards)
        values = np.concatenate(all_values)
        policies = np.concatenate(all_policies)
        del all_boards, all_values, all_policies

        # Validate policy range
        max_pol = int(policies.max())
        if max_pol >= 4096:
            raise ValueError(
                f"Policy index {max_pol} >= 4096. HDF5 data may be corrupt."
            )

        # Subsample to epoch_size if we loaded more (due to full-chunk reads)
        n_loaded = len(values)
        if n_loaded > self.epoch_size:
            perm = rng.choice(n_loaded, size=self.epoch_size, replace=False)
            boards = boards[perm]
            values = values[perm]
            policies = policies[perm]

        # Store as contiguous uint8 arrays (4x less RAM than float32)
        self._boards = np.ascontiguousarray(boards)
        self._values = np.ascontiguousarray(values)
        self._policies = policies.astype(np.int64)
        self._size = len(self._values)

        del boards, values, policies
        gc.collect()

        elapsed = time.time() - t0
        print(f"  Loaded {self._size:,} positions from HDF5 in {elapsed:.1f}s "
              f"(epoch {epoch})")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board_raw = self._boards[idx]  # uint8 [65, 27]

        if self._board_dtype == np.uint8:
            board = board_raw.astype(np.float32)
            board[HALFMOVE_TOKEN, HALFMOVE_FEAT] /= HALFMOVE_SCALE
        else:
            board = board_raw.astype(np.float32)

        return (
            torch.from_numpy(board),
            torch.tensor(self._values[idx], dtype=torch.float32),
            torch.tensor(self._policies[idx], dtype=torch.long),
        )

    def __len__(self) -> int:
        return self._size

    def close(self):
        """Close all HDF5 file handles."""
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __del__(self):
        self.close()