"""
Data validation and statistics for training datasets.

Reports position counts, material distributions, value/policy distributions,
game phase breakdown, and anomaly detection across data sources.

Usage:
    python data_stats.py data/round_0/
    python data_stats.py data/mixed/ --detailed
    python data_stats.py data/synthetic/ data/round_0/ --compare
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import chess
import numpy as np
from tqdm import tqdm

from encoding import board_to_tensor, NUM_FEATURES


# Piece plane indices in the encoding (see encoding.py)
PIECE_PLANES = {
    0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K",    # white
    6: "p", 7: "n", 8: "b", 9: "r", 10: "q", 11: "k",   # black
}

SOURCE_NAMES = {0: "ccrl", 1: "synthetic", 2: "endgame", 3: "opening"}


def count_material(board_tensor: np.ndarray) -> str:
    """
    Extract material signature from a board tensor.
    Works with both [64, 25] (old) and [65, 27] (new) format.
    Returns string like 'KQR vs KRR'.
    """
    white_pieces = []
    black_pieces = []

    # Only look at first 64 rows (square tokens)
    num_squares = min(64, board_tensor.shape[0])

    for sq in range(num_squares):
        for plane in range(12):
            if board_tensor[sq, plane] > 0.5:
                piece_char = PIECE_PLANES[plane]
                if plane < 6:
                    white_pieces.append(piece_char)
                else:
                    black_pieces.append(piece_char.upper())

    # Sort: K first, then Q, R, B, N, P
    order = "KQRBNP"
    white_pieces.sort(key=lambda c: order.index(c))
    black_pieces.sort(key=lambda c: order.index(c))

    return "".join(white_pieces) + "v" + "".join(black_pieces)


def estimate_phase(board_tensor: np.ndarray) -> str:
    """
    Estimate game phase from material on board.
    Heuristic based on total non-pawn material.
    """
    # Count major/minor pieces (planes 1-4 white, 7-10 black)
    minor_major_planes = [1, 2, 3, 4, 7, 8, 9, 10]
    num_squares = min(64, board_tensor.shape[0])
    piece_count = 0
    for sq in range(num_squares):
        for plane in minor_major_planes:
            if board_tensor[sq, plane] > 0.5:
                piece_count += 1

    if piece_count >= 12:
        return "opening"
    elif piece_count >= 6:
        return "middlegame"
    else:
        return "endgame"


def analyze_directory(
    data_dir: str,
    max_samples: int = 0,
    detailed: bool = False,
) -> dict:
    """
    Analyze a training data directory and return statistics.

    Returns dict with keys: total_positions, num_chunks, value_stats,
    material_counts, phase_counts, source_counts, anomalies.
    """
    chunk_paths = sorted(Path(data_dir).glob("chunk_*.npz"))
    if not chunk_paths:
        print(f"No chunks found in {data_dir}")
        return {}

    # Accumulators
    total_positions = 0
    values_all = []
    policies_all = []
    material_counter: Counter = Counter()
    phase_counter: Counter = Counter()
    source_counter: Counter = Counter()
    samples_analyzed = 0

    print(f"\nAnalyzing {data_dir}/ ({len(chunk_paths)} chunks)...")

    for cpath in tqdm(chunk_paths, desc="Chunks"):
        data = np.load(str(cpath))
        boards = data["boards"]
        values = data["values"]
        policies = data["policies"] if "policies" in data else None
        sources = data["sources"] if "sources" in data else None

        n = len(values)
        total_positions += n
        values_all.append(values)
        if policies is not None:
            policies_all.append(policies)

        # Source counts
        if sources is not None:
            for sid in sources:
                source_counter[SOURCE_NAMES.get(int(sid), f"unknown_{sid}")] += 1
        else:
            source_counter["unknown"] += n

        # Detailed analysis on a sample of positions
        if detailed:
            sample_size = min(n, 1000)
            sample_idx = np.random.choice(n, sample_size, replace=False)
            for idx in sample_idx:
                material_counter[count_material(boards[idx])] += 1
                phase_counter[estimate_phase(boards[idx])] += 1
                samples_analyzed += 1

        if max_samples > 0 and total_positions >= max_samples:
            break

    # Aggregate values
    values_all = np.concatenate(values_all)
    policies_concat = np.concatenate(policies_all) if policies_all else None

    # Value statistics
    value_stats = {
        "mean": float(np.mean(values_all)),
        "std": float(np.std(values_all)),
        "min": float(np.min(values_all)),
        "max": float(np.max(values_all)),
        "median": float(np.median(values_all)),
        "pct_negative": float((values_all < 0).mean()),
        "pct_zero": float((np.abs(values_all) < 0.01).mean()),
        "pct_positive": float((values_all > 0).mean()),
        "pct_extreme": float((np.abs(values_all) > 0.9).mean()),
    }

    # Policy statistics
    policy_stats = {}
    if policies_concat is not None:
        unique, counts = np.unique(policies_concat, return_counts=True)
        top_10 = sorted(zip(counts, unique), reverse=True)[:10]
        policy_stats = {
            "unique_moves": len(unique),
            "top_10_moves": [(int(idx), int(cnt)) for cnt, idx in top_10],
        }

    # Anomaly detection
    anomalies = []
    if value_stats["std"] < 0.05:
        anomalies.append("WARNING: Very low value std — all positions have similar eval")
    if value_stats["pct_extreme"] > 0.5:
        anomalies.append("WARNING: >50% positions have extreme eval (|v| > 0.9)")
    if value_stats["pct_zero"] > 0.8:
        anomalies.append("WARNING: >80% positions eval near zero — possible all-draw source")
    if abs(value_stats["mean"]) > 0.3:
        anomalies.append(f"WARNING: Mean value {value_stats['mean']:.3f} is biased")

    result = {
        "data_dir": data_dir,
        "total_positions": total_positions,
        "num_chunks": len(chunk_paths),
        "value_stats": value_stats,
        "policy_stats": policy_stats,
        "source_counts": dict(source_counter),
        "material_counts": dict(material_counter.most_common(30)),
        "phase_counts": dict(phase_counter),
        "samples_analyzed": samples_analyzed,
        "anomalies": anomalies,
    }
    return result


def print_report(stats: dict) -> None:
    """Pretty-print analysis results."""
    if not stats:
        return

    print(f"\n{'=' * 60}")
    print(f"Dataset: {stats['data_dir']}")
    print(f"{'=' * 60}")
    print(f"Total positions: {stats['total_positions']:,}")
    print(f"Chunks: {stats['num_chunks']}")

    vs = stats["value_stats"]
    print(f"\n--- Value Distribution ---")
    print(f"  Mean:     {vs['mean']:+.4f}")
    print(f"  Std:      {vs['std']:.4f}")
    print(f"  Min:      {vs['min']:+.4f}")
    print(f"  Max:      {vs['max']:+.4f}")
    print(f"  Median:   {vs['median']:+.4f}")
    print(f"  Negative: {vs['pct_negative']:.1%}")
    print(f"  Near 0:   {vs['pct_zero']:.1%}")
    print(f"  Positive: {vs['pct_positive']:.1%}")
    print(f"  Extreme:  {vs['pct_extreme']:.1%}")

    if stats["policy_stats"]:
        ps = stats["policy_stats"]
        print(f"\n--- Policy Distribution ---")
        print(f"  Unique moves: {ps['unique_moves']:,}")
        print(f"  Top 10 moves (index, count):")
        for idx, cnt in ps["top_10_moves"]:
            from_sq = idx // 64
            to_sq = idx % 64
            move_uci = chess.square_name(from_sq) + chess.square_name(to_sq)
            print(f"    {move_uci:<8} (idx={idx:>4d})  {cnt:>8,}")

    if stats["source_counts"]:
        print(f"\n--- Source Breakdown ---")
        for name, count in sorted(stats["source_counts"].items(), key=lambda x: -x[1]):
            pct = count / stats["total_positions"]
            print(f"  {name:<15} {count:>10,}  ({pct:.1%})")

    if stats["phase_counts"]:
        total_sampled = sum(stats["phase_counts"].values())
        print(f"\n--- Game Phase Distribution (sampled {total_sampled:,}) ---")
        for phase in ["opening", "middlegame", "endgame"]:
            count = stats["phase_counts"].get(phase, 0)
            pct = count / total_sampled if total_sampled > 0 else 0
            print(f"  {phase:<15} {count:>8,}  ({pct:.1%})")

    if stats["material_counts"]:
        print(f"\n--- Top Material Configurations (sampled) ---")
        for mat, count in list(stats["material_counts"].items())[:20]:
            print(f"  {mat:<25} {count:>6,}")

    if stats["anomalies"]:
        print(f"\n--- Anomalies ---")
        for a in stats["anomalies"]:
            print(f"  ⚠ {a}")
    else:
        print(f"\n  ✓ No anomalies detected")


def main():
    parser = argparse.ArgumentParser(description="Training data statistics and validation")
    parser.add_argument(
        "data_dirs", nargs="+",
        help="One or more training data directories to analyze",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Run detailed analysis (material, phase breakdown — slower)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max positions to analyze (0 = all)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print comparison table across directories",
    )
    args = parser.parse_args()

    all_stats = []
    for data_dir in args.data_dirs:
        stats = analyze_directory(data_dir, args.max_samples, args.detailed)
        if stats:
            print_report(stats)
            all_stats.append(stats)

    if args.compare and len(all_stats) > 1:
        print(f"\n{'=' * 60}")
        print(f"Comparison")
        print(f"{'=' * 60}")
        header = f"{'Dataset':<30} {'Positions':>12} {'Mean':>8} {'Std':>8} {'Extreme':>8}"
        print(header)
        print("-" * len(header))
        for s in all_stats:
            name = os.path.basename(s["data_dir"].rstrip("/"))
            vs = s["value_stats"]
            print(f"{name:<30} {s['total_positions']:>12,} "
                  f"{vs['mean']:>+8.4f} {vs['std']:>8.4f} {vs['pct_extreme']:>8.1%}")


if __name__ == "__main__":
    main()
