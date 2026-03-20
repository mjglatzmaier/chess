"""
Convert training data from compressed .npz chunks to HDF5 format.

Reads existing chunk directories (round_0/, synthetic/, etc.) and produces
a single .h5 file per source with:
  - boards:   uint8   [N, 65, 27]  (4x smaller than float32)
  - values:   float32 [N]
  - policies: uint16  [N]          (4x smaller than int64)
  - sources:  uint8   [N]
  - weights:  float32 [N]

The halfmove clock (boards[:, 64, 26]) is the only non-binary board feature.
It is quantized from float32 [0, 1] to uint8 [0, 255] and can be
reconstructed at load time as: value / 255.0

Usage:
    # Convert a single source:
    python convert_to_hdf5.py --input data/round_0/ --output data/round_0.h5

    # Convert with verification (spot-checks random positions):
    python convert_to_hdf5.py --input data/round_0/ --output data/round_0.h5 --verify 1000

    # Dry run (report sizes, don't write):
    python convert_to_hdf5.py --input data/round_0/ --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


# HDF5 chunk size: positions per HDF5 chunk (not the same as .npz chunk size).
# 10K positions keeps chunks small (~1.8 MB for uint8 boards) for granular reads.
H5_CHUNK_POSITIONS = 10_000

# Halfmove clock location in board encoding
HALFMOVE_TOKEN = 64   # global context token index
HALFMOVE_FEAT = 26    # feature index within the token (last of 27 features)
HALFMOVE_SCALE = 255.0


def convert_boards_to_uint8(boards_f32: np.ndarray) -> np.ndarray:
    """Convert float32 boards to uint8.

    All features are binary (0.0 or 1.0) except the halfmove clock at
    [64, 26] which is float in [0, 1]. The halfmove clock is scaled to
    [0, 255] for uint8 storage.
    """
    # Scale halfmove clock: [0, 1] → [0, 255]
    boards_f32[:, HALFMOVE_TOKEN, HALFMOVE_FEAT] *= HALFMOVE_SCALE

    # Round and clip to uint8 range
    return np.clip(np.round(boards_f32), 0, 255).astype(np.uint8)


def scan_chunks(data_dir: str) -> tuple[list[Path], int, dict]:
    """Discover chunks and total size. Returns (chunk_paths, total_positions, metadata)."""
    meta_path = os.path.join(data_dir, "metadata.npz")
    chunk_paths = sorted(Path(data_dir).glob("chunk_*.npz"))

    if not chunk_paths:
        print(f"Error: no chunk_*.npz files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    metadata = {}
    if os.path.exists(meta_path):
        meta = np.load(meta_path)
        metadata = {k: meta[k].item() if meta[k].ndim == 0 else str(meta[k])
                     for k in meta.files}
        total_positions = int(meta["total_positions"])
        num_chunks = int(meta["num_chunks"])
        chunk_paths = chunk_paths[:num_chunks]
    else:
        # No metadata — scan all chunks
        total_positions = 0
        for cpath in tqdm(chunk_paths, desc="Scanning chunks"):
            with np.load(str(cpath)) as data:
                total_positions += len(data["values"])
        num_chunks = len(chunk_paths)
        metadata = {"num_chunks": num_chunks}

    metadata["total_positions"] = total_positions
    return chunk_paths, total_positions, metadata


def convert(
    input_dir: str,
    output_path: str,
    compression: str = "gzip",
    compression_level: int = 1,
    verify_count: int = 0,
    dry_run: bool = False,
) -> None:
    """Convert .npz chunk directory to a single .h5 file."""

    print(f"Scanning {input_dir}...")
    chunk_paths, total_positions, metadata = scan_chunks(input_dir)

    # Peek at first chunk to get shapes and confirm dtypes
    with np.load(str(chunk_paths[0])) as sample:
        board_shape = sample["boards"].shape[1:]  # e.g., (65, 27)
        has_policies = "policies" in sample
        has_sources = "sources" in sample
        has_weights = "weights" in sample
        orig_board_dtype = sample["boards"].dtype
        sample_size = len(sample["values"])

    print(f"  Source: {input_dir}")
    print(f"  Chunks: {len(chunk_paths)}")
    print(f"  Total positions: {total_positions:,}")
    print(f"  Board shape: {board_shape}, dtype: {orig_board_dtype}")
    print(f"  Sample chunk size: {sample_size:,}")

    # Size estimates
    boards_new = total_positions * np.prod(board_shape) * 1  # uint8
    boards_old = total_positions * np.prod(board_shape) * 4  # float32
    policies_new = total_positions * 2  # uint16
    policies_old = total_positions * 8  # int64
    values_size = total_positions * 4   # float32
    sources_size = total_positions * 1  # uint8
    weights_size = total_positions * 4  # float32

    total_new = boards_new + values_size + policies_new + sources_size + weights_size
    total_old = boards_old + values_size + policies_old + sources_size + weights_size

    print(f"\n  Uncompressed size estimate:")
    print(f"    Old (float32/int64): {total_old / 1e9:.1f} GB")
    print(f"    New (uint8/uint16):  {total_new / 1e9:.1f} GB")
    print(f"    Reduction: {total_old / total_new:.1f}x")

    if dry_run:
        print("\n  (dry run — not writing)")
        return

    # Set up compression kwargs
    comp_kwargs = {}
    if compression == "gzip":
        comp_kwargs = {"compression": "gzip", "compression_opts": compression_level}
    elif compression == "lzf":
        comp_kwargs = {"compression": "lzf"}
    elif compression == "none":
        comp_kwargs = {}
    else:
        print(f"Warning: unknown compression '{compression}', using gzip")
        comp_kwargs = {"compression": "gzip", "compression_opts": compression_level}

    print(f"\n  Writing {output_path} (compression={compression})...")
    t0 = time.time()

    with h5py.File(output_path, "w") as hf:
        # Create datasets with final shape
        ds_boards = hf.create_dataset(
            "boards",
            shape=(total_positions, *board_shape),
            dtype=np.uint8,
            chunks=(min(H5_CHUNK_POSITIONS, total_positions), *board_shape),
            **comp_kwargs,
        )
        ds_values = hf.create_dataset(
            "values",
            shape=(total_positions,),
            dtype=np.float32,
            chunks=(min(H5_CHUNK_POSITIONS, total_positions),),
            **comp_kwargs,
        )
        ds_policies = hf.create_dataset(
            "policies",
            shape=(total_positions,),
            dtype=np.uint16,
            chunks=(min(H5_CHUNK_POSITIONS, total_positions),),
            **comp_kwargs,
        )
        ds_sources = hf.create_dataset(
            "sources",
            shape=(total_positions,),
            dtype=np.uint8,
            chunks=(min(H5_CHUNK_POSITIONS, total_positions),),
            **comp_kwargs,
        )
        ds_weights = hf.create_dataset(
            "weights",
            shape=(total_positions,),
            dtype=np.float32,
            chunks=(min(H5_CHUNK_POSITIONS, total_positions),),
            **comp_kwargs,
        )

        # Set metadata as HDF5 attributes
        hf.attrs["total_positions"] = total_positions
        hf.attrs["encoding_version"] = 2  # v2 = uint8 boards
        hf.attrs["halfmove_scale"] = HALFMOVE_SCALE
        hf.attrs["board_shape"] = board_shape
        for k, v in metadata.items():
            if k not in ("total_positions",):
                try:
                    hf.attrs[k] = v
                except TypeError:
                    hf.attrs[k] = str(v)

        # Convert chunk by chunk
        offset = 0
        for cpath in tqdm(chunk_paths, desc="Converting"):
            with np.load(str(cpath)) as data:
                n = len(data["values"])
                boards = data["boards"].copy()  # float32, need mutable copy
                values = data["values"]
                policies = data["policies"] if has_policies else np.zeros(n, dtype=np.int64)
                sources = data["sources"] if has_sources else np.zeros(n, dtype=np.uint8)
                weights = data["weights"] if has_weights else np.ones(n, dtype=np.float32)

                # Convert boards: float32 → uint8
                boards_u8 = convert_boards_to_uint8(boards)

                # Write to HDF5
                end = offset + n
                ds_boards[offset:end] = boards_u8
                ds_values[offset:end] = values
                ds_policies[offset:end] = policies.astype(np.uint16)
                ds_sources[offset:end] = sources.astype(np.uint8)
                ds_weights[offset:end] = weights.astype(np.float32)

                offset += n

        assert offset == total_positions, (
            f"Position count mismatch: wrote {offset}, expected {total_positions}"
        )

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path)
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Output: {output_path} ({file_size / 1e9:.2f} GB)")
    print(f"  Compression ratio: {total_new / file_size:.1f}x")

    # Verification
    if verify_count > 0:
        verify(input_dir, output_path, chunk_paths, verify_count)


def verify(
    input_dir: str,
    h5_path: str,
    chunk_paths: list[Path],
    n_checks: int = 1000,
) -> None:
    """Spot-check random positions for round-trip accuracy."""
    print(f"\n  Verifying {n_checks} random positions...")
    rng = np.random.default_rng(42)

    with h5py.File(h5_path, "r") as hf:
        total = hf.attrs["total_positions"]

        # Build chunk offset map for the original data
        chunk_offsets = []  # (start, end, path)
        offset = 0
        for cpath in chunk_paths:
            with np.load(str(cpath)) as data:
                n = len(data["values"])
            chunk_offsets.append((offset, offset + n, cpath))
            offset += n

        # Check random positions
        indices = rng.choice(total, size=min(n_checks, total), replace=False)
        indices.sort()

        errors = 0
        max_halfmove_err = 0.0

        for idx in indices:
            # Find original chunk
            for start, end, cpath in chunk_offsets:
                if start <= idx < end:
                    local = idx - start
                    break
            else:
                print(f"    ERROR: index {idx} not found in any chunk")
                errors += 1
                continue

            # Load original
            with np.load(str(cpath)) as data:
                orig_board = data["boards"][local]   # float32
                orig_value = data["values"][local]
                orig_policy = data["policies"][local] if "policies" in data else 0

            # Load converted
            h5_board_u8 = hf["boards"][idx]          # uint8
            h5_value = hf["values"][idx]
            h5_policy = hf["policies"][idx]

            # Reconstruct float32 board from uint8
            h5_board_f32 = h5_board_u8.astype(np.float32)
            h5_board_f32[HALFMOVE_TOKEN, HALFMOVE_FEAT] /= HALFMOVE_SCALE

            # Check binary features (all except halfmove clock)
            mask = np.ones_like(orig_board, dtype=bool)
            mask[HALFMOVE_TOKEN, HALFMOVE_FEAT] = False
            if not np.array_equal(orig_board[mask].astype(np.uint8),
                                  h5_board_u8[mask]):
                print(f"    ERROR at idx {idx}: binary feature mismatch")
                errors += 1

            # Check halfmove clock precision
            hm_err = abs(float(orig_board[HALFMOVE_TOKEN, HALFMOVE_FEAT]) -
                         float(h5_board_f32[HALFMOVE_TOKEN, HALFMOVE_FEAT]))
            max_halfmove_err = max(max_halfmove_err, hm_err)
            if hm_err > 0.005:
                print(f"    WARNING at idx {idx}: halfmove error {hm_err:.4f}")

            # Check value
            if abs(float(orig_value) - float(h5_value)) > 1e-6:
                print(f"    ERROR at idx {idx}: value mismatch "
                      f"{orig_value} vs {h5_value}")
                errors += 1

            # Check policy
            if int(orig_policy) != int(h5_policy):
                print(f"    ERROR at idx {idx}: policy mismatch "
                      f"{orig_policy} vs {h5_policy}")
                errors += 1

        if errors == 0:
            print(f"  ✓ All {len(indices)} positions verified OK")
            print(f"    Max halfmove clock error: {max_halfmove_err:.6f}")
        else:
            print(f"  ✗ {errors} errors found in {len(indices)} positions")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert .npz training chunks to HDF5 format with uint8 boards"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory containing chunk_*.npz files",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .h5 file path (default: <input>.h5)",
    )
    parser.add_argument(
        "--compression", default="gzip", choices=["gzip", "lzf", "none"],
        help="HDF5 compression filter (default: gzip)",
    )
    parser.add_argument(
        "--compression-level", type=int, default=1,
        help="Gzip compression level 1-9 (default: 1, fastest)",
    )
    parser.add_argument(
        "--verify", type=int, default=1000, metavar="N",
        help="Number of positions to spot-check after conversion (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report sizes without writing",
    )
    args = parser.parse_args()

    input_dir = args.input.rstrip("/")
    output_path = args.output or f"{input_dir}.h5"

    if os.path.exists(output_path) and not args.dry_run:
        print(f"Error: output file already exists: {output_path}", file=sys.stderr)
        print("  Delete it first or choose a different output path.", file=sys.stderr)
        sys.exit(1)

    convert(
        input_dir=input_dir,
        output_path=output_path,
        compression=args.compression,
        compression_level=args.compression_level,
        verify_count=args.verify,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
