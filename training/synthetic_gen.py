"""
Synthetic position generator with Stockfish evaluation.

Generates random legal positions with controlled material configurations,
evaluates them with Stockfish, and saves as training data chunks. This fills
gaps in the CCRL data — material imbalances, rare endgames, edge cases.

Usage:
    python synthetic_gen.py --stockfish /path/to/stockfish --output data/synthetic/
    python synthetic_gen.py --stockfish stockfish --depth 20 --threads 8
    python synthetic_gen.py --stockfish stockfish --configs KQvK KRvK --num 500
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np
from tqdm import tqdm

from encoding import board_to_tensor, move_to_index, NUM_FEATURES


# ---------------------------------------------------------------------------
# Material configurations to generate
# ---------------------------------------------------------------------------

PIECE_MAP = {
    "K": chess.KING,
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
    "P": chess.PAWN,
}


@dataclass
class MaterialConfig:
    """Defines a material configuration to generate positions for."""
    name: str
    white_pieces: str  # e.g., "KQP"
    black_pieces: str  # e.g., "KP"
    num_positions: int = 1000
    category: str = "general"

    def white_list(self) -> list[int]:
        return [PIECE_MAP[c] for c in self.white_pieces]

    def black_list(self) -> list[int]:
        return [PIECE_MAP[c] for c in self.black_pieces]


# Standard set of material configurations covering key gaps
DEFAULT_CONFIGS = [
    # Queen odds — magnitude calibration
    MaterialConfig("KQvK", "KQ", "K", 1000, "queen_odds"),
    MaterialConfig("KQPvKP", "KQP", "KP", 1000, "queen_odds"),
    MaterialConfig("KQvKR", "KQ", "KR", 1000, "queen_odds"),

    # Rook odds — endgame eval
    MaterialConfig("KRvK", "KR", "K", 1000, "rook_odds"),
    MaterialConfig("KRPvKP", "KRP", "KP", 1000, "rook_odds"),
    MaterialConfig("KRRvKR", "KRR", "KR", 1000, "rook_odds"),

    # Minor piece odds — draw recognition
    MaterialConfig("KBvK", "KB", "K", 1000, "minor_odds"),
    MaterialConfig("KNvK", "KN", "K", 1000, "minor_odds"),
    MaterialConfig("KBNvK", "KBN", "K", 1000, "minor_odds"),
    MaterialConfig("KBBvK", "KBB", "K", 1000, "minor_odds"),

    # Pawn endgames
    MaterialConfig("KPvK", "KP", "K", 1000, "pawn_endgame"),
    MaterialConfig("KPPvK", "KPP", "K", 1000, "pawn_endgame"),
    MaterialConfig("KPPPvKN", "KPPP", "KN", 1000, "pawn_endgame"),
    MaterialConfig("KPvKP", "KP", "KP", 1000, "pawn_endgame"),

    # Piece vs pawns — material tradeoffs
    MaterialConfig("KRvKPP", "KR", "KPP", 1000, "piece_vs_pawns"),
    MaterialConfig("KBvKPP", "KB", "KPP", 1000, "piece_vs_pawns"),
    MaterialConfig("KNvKPP", "KN", "KPP", 1000, "piece_vs_pawns"),

    # Mutual imbalance — piece coordination
    MaterialConfig("KRBvKRN", "KRB", "KRN", 1000, "mutual_imbalance"),
    MaterialConfig("KQvKRR", "KQ", "KRR", 1000, "mutual_imbalance"),
    MaterialConfig("KRNvKRB", "KRN", "KRB", 1000, "mutual_imbalance"),
    MaterialConfig("KQvKRB", "KQ", "KRB", 1000, "mutual_imbalance"),
    MaterialConfig("KQvKRN", "KQ", "KRN", 1000, "mutual_imbalance"),

    # Extreme — promotions, multiple queens
    MaterialConfig("KQQvKQ", "KQQ", "KQ", 500, "extreme"),
    MaterialConfig("KQRvK", "KQR", "K", 500, "extreme"),
]


def get_config_by_name(name: str) -> Optional[MaterialConfig]:
    """Look up a config by name (e.g., 'KQvK')."""
    for cfg in DEFAULT_CONFIGS:
        if cfg.name == name:
            return cfg
    return None


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

# Squares where pawns can be placed (ranks 2-7, not rank 1 or 8)
PAWN_SQUARES = [sq for sq in range(64) if 1 <= chess.square_rank(sq) <= 6]
ALL_SQUARES = list(range(64))


def generate_random_position(
    white_pieces: list[int],
    black_pieces: list[int],
    max_attempts: int = 100,
) -> Optional[chess.Board]:
    """
    Generate a random legal position with the specified material.

    Places pieces on random squares, ensuring:
    - Kings are not adjacent
    - Pawns are not on ranks 1 or 8
    - The position is legal (side to move not giving check to opponent)
    - No two pieces share a square

    Returns None if no legal position found after max_attempts.
    """
    for _ in range(max_attempts):
        board = chess.Board.empty()
        occupied: set[int] = set()

        # Place white king first
        wk_sq = random.choice(ALL_SQUARES)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        occupied.add(wk_sq)

        # Place black king (must not be adjacent to white king)
        adjacent = set(chess.SQUARES[s] for s in chess.SquareSet(chess.BB_KING_ATTACKS[wk_sq]))
        adjacent.add(wk_sq)
        bk_candidates = [sq for sq in ALL_SQUARES if sq not in adjacent]
        if not bk_candidates:
            continue
        bk_sq = random.choice(bk_candidates)
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        occupied.add(bk_sq)

        # Place remaining white pieces (skip king, already placed)
        ok = True
        for piece_type in white_pieces:
            if piece_type == chess.KING:
                continue
            candidates = PAWN_SQUARES if piece_type == chess.PAWN else ALL_SQUARES
            candidates = [sq for sq in candidates if sq not in occupied]
            if not candidates:
                ok = False
                break
            sq = random.choice(candidates)
            board.set_piece_at(sq, chess.Piece(piece_type, chess.WHITE))
            occupied.add(sq)

        if not ok:
            continue

        # Place remaining black pieces (skip king, already placed)
        for piece_type in black_pieces:
            if piece_type == chess.KING:
                continue
            candidates = PAWN_SQUARES if piece_type == chess.PAWN else ALL_SQUARES
            candidates = [sq for sq in candidates if sq not in occupied]
            if not candidates:
                ok = False
                break
            sq = random.choice(candidates)
            board.set_piece_at(sq, chess.Piece(piece_type, chess.BLACK))
            occupied.add(sq)

        if not ok:
            continue

        # No castling rights, no en passant in synthetic positions
        board.set_castling_fen("-")
        board.ep_square = None

        # Pick side to move randomly
        board.turn = random.choice([chess.WHITE, chess.BLACK])

        # Validate position
        if not board.is_valid():
            continue

        # Ensure side to move is not in checkmate or stalemate (boring positions)
        if board.is_game_over():
            continue

        return board

    return None


def generate_positions_for_config(
    config: MaterialConfig,
) -> list[tuple[str, None]]:
    """
    Generate random legal positions for a material config.
    Returns list of (FEN, None) — eval is filled in later by Stockfish.
    """
    white_pieces = config.white_list()
    black_pieces = config.black_list()
    results: list[tuple[str, None]] = []
    attempts = 0
    max_total = config.num_positions * 50  # safety limit

    while len(results) < config.num_positions and attempts < max_total:
        attempts += 1
        board = generate_random_position(white_pieces, black_pieces)
        if board is not None:
            results.append((board.fen(), None))

    return results


# ---------------------------------------------------------------------------
# Stockfish evaluation
# ---------------------------------------------------------------------------

class StockfishEvaluator:
    """UCI interface to Stockfish for position evaluation."""

    def __init__(self, stockfish_path: str, depth: int = 20, hash_mb: int = 64):
        self.depth = depth
        self.process = subprocess.Popen(
            [stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name Threads value 1")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str) -> None:
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _wait_for(self, token: str) -> list[str]:
        """Read lines until one contains the token."""
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            lines.append(line)
            if token in line:
                return lines

    def evaluate(self, fen: str) -> Optional[float]:
        """
        Evaluate a position and return centipawns from side-to-move perspective.
        Returns None if evaluation fails. Mate scores converted to ±10000.
        """
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")

        score_cp = None
        lines = self._wait_for("bestmove")

        for line in lines:
            if "score cp" in line:
                parts = line.split()
                try:
                    idx = parts.index("cp")
                    score_cp = int(parts[idx + 1])
                except (ValueError, IndexError):
                    pass
            elif "score mate" in line:
                parts = line.split()
                try:
                    idx = parts.index("mate")
                    mate_in = int(parts[idx + 1])
                    score_cp = 10000 if mate_in > 0 else -10000
                except (ValueError, IndexError):
                    pass

        return score_cp

    def evaluate_batch(self, fens: list[str]) -> list[Optional[float]]:
        """Evaluate a batch of positions sequentially."""
        return [self.evaluate(fen) for fen in fens]

    def close(self):
        try:
            self._send("quit")
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def cp_to_value(cp: float) -> float:
    """Convert centipawns to [-1, +1] using tanh scaling."""
    return math.tanh(cp / 600.0)


# ---------------------------------------------------------------------------
# Worker function for parallel evaluation
# ---------------------------------------------------------------------------

def _evaluate_worker(
    fens: list[str],
    stockfish_path: str,
    depth: int,
) -> list[tuple[str, float]]:
    """Worker process: evaluate a batch of FENs with its own Stockfish instance."""
    results = []
    with StockfishEvaluator(stockfish_path, depth=depth) as sf:
        for fen in fens:
            cp = sf.evaluate(fen)
            if cp is not None:
                results.append((fen, cp_to_value(cp)))
    return results


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class SyntheticGenerator:
    """
    Generates synthetic chess positions with controlled material and
    Stockfish evaluations. Outputs training data in .npz chunk format.
    """

    def __init__(
        self,
        stockfish_path: str,
        depth: int = 20,
        threads: int = 4,
        chunk_size: int = 100_000,
    ):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.threads = threads
        self.chunk_size = chunk_size

    def generate_material_config(
        self,
        config: MaterialConfig,
        progress: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Generate random positions with specific material and evaluate with SF.
        Returns list of (FEN, value) pairs where value is in [-1, +1].
        """
        # Generate positions (fast, no Stockfish needed)
        positions = generate_positions_for_config(config)
        fens = [fen for fen, _ in positions]

        if not fens:
            print(f"  Warning: Could not generate positions for {config.name}")
            return []

        # Split FENs across worker processes for parallel evaluation
        results: list[tuple[str, float]] = []
        batch_size = max(1, len(fens) // self.threads)
        batches = [fens[i:i + batch_size] for i in range(0, len(fens), batch_size)]

        desc = f"  {config.name}" if progress else None
        pbar = tqdm(total=len(fens), desc=desc, disable=not progress)

        if self.threads <= 1:
            # Single-threaded evaluation
            worker_results = _evaluate_worker(fens, self.stockfish_path, self.depth)
            results.extend(worker_results)
            pbar.update(len(fens))
        else:
            with ProcessPoolExecutor(max_workers=self.threads) as executor:
                futures = {
                    executor.submit(
                        _evaluate_worker, batch, self.stockfish_path, self.depth
                    ): len(batch)
                    for batch in batches
                }
                for future in as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)
                    pbar.update(futures[future])

        pbar.close()
        return results

    def generate_all(
        self,
        output_dir: str,
        configs: Optional[list[MaterialConfig]] = None,
    ) -> dict:
        """
        Generate the full synthetic dataset.

        Args:
            output_dir: Directory to write .npz chunks
            configs: Material configs to generate (default: DEFAULT_CONFIGS)

        Returns:
            Statistics dict with counts per config and totals.
        """
        if configs is None:
            configs = DEFAULT_CONFIGS

        os.makedirs(output_dir, exist_ok=True)

        all_fens: list[str] = []
        all_values: list[float] = []
        stats: dict[str, int] = {}
        source_id = 1  # synthetic source identifier

        print(f"Generating synthetic positions (depth={self.depth}, threads={self.threads})")
        print(f"  Configs: {len(configs)}, target positions: "
              f"{sum(c.num_positions for c in configs):,}")
        print()

        for config in configs:
            results = self.generate_material_config(config)
            stats[config.name] = len(results)
            for fen, value in results:
                all_fens.append(fen)
                all_values.append(value)

        # Shuffle all positions together
        combined = list(zip(all_fens, all_values))
        random.shuffle(combined)
        all_fens, all_values = zip(*combined) if combined else ([], [])

        # Encode and save as chunks
        total_positions = len(all_fens)
        chunk_idx = 0
        boards_buf = []
        values_buf = []
        policy_buf = []
        sources_buf = []
        weights_buf = []

        print(f"\nEncoding {total_positions:,} positions into chunks...")
        for fen, value in tqdm(zip(all_fens, all_values), total=total_positions):
            board = chess.Board(fen)
            features = board_to_tensor(board)

            # Pick a random legal move as policy target
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                policy_idx = move_to_index(move)
            else:
                policy_idx = 0

            boards_buf.append(features)
            values_buf.append(value)
            policy_buf.append(policy_idx)
            sources_buf.append(source_id)
            weights_buf.append(1.0)

            if len(boards_buf) >= self.chunk_size:
                _save_synthetic_chunk(
                    output_dir, chunk_idx,
                    boards_buf, values_buf, policy_buf, sources_buf, weights_buf,
                )
                chunk_idx += 1
                boards_buf, values_buf, policy_buf = [], [], []
                sources_buf, weights_buf = [], []

        # Final partial chunk
        if boards_buf:
            _save_synthetic_chunk(
                output_dir, chunk_idx,
                boards_buf, values_buf, policy_buf, sources_buf, weights_buf,
            )
            chunk_idx += 1

        # Save metadata
        metadata = {
            "total_positions": total_positions,
            "total_games": 0,
            "skipped_games": 0,
            "num_chunks": chunk_idx,
            "chunk_size": self.chunk_size,
            "num_features": NUM_FEATURES,
            "skip_moves": 0,
            "source": "synthetic",
            "has_policy": True,
            "stockfish_depth": self.depth,
        }
        np.savez(os.path.join(output_dir, "metadata.npz"), **metadata)

        # Print summary
        print(f"\nDone: {total_positions:,} positions in {chunk_idx} chunks")
        print(f"Output: {output_dir}/")
        print("\nPer-config breakdown:")
        for name, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {name:<20} {count:>6,}")

        stats["total"] = total_positions
        stats["chunks"] = chunk_idx
        return stats


def _save_synthetic_chunk(
    output_dir: str,
    chunk_idx: int,
    boards: list,
    values: list,
    policies: list,
    sources: list,
    weights: list,
) -> None:
    """Save a chunk with extended fields (sources, weights)."""
    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        boards=np.stack(boards, axis=0),
        values=np.array(values, dtype=np.float32),
        policies=np.array(policies, dtype=np.int64),
        sources=np.array(sources, dtype=np.uint8),
        weights=np.array(weights, dtype=np.float32),
    )
    tqdm.write(f"  Saved chunk {chunk_idx}: {len(boards):,} positions")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic chess positions with Stockfish evaluation"
    )
    parser.add_argument(
        "--stockfish", required=True,
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--output", default="data/synthetic/",
        help="Output directory for .npz chunks (default: data/synthetic/)",
    )
    parser.add_argument(
        "--depth", type=int, default=20,
        help="Stockfish search depth (default: 20)",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of parallel Stockfish instances (default: 4)",
    )
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="Specific configs to generate (e.g., KQvK KRvK). Default: all",
    )
    parser.add_argument(
        "--num", type=int, default=None,
        help="Override num_positions per config",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="Positions per chunk (default: 100000)",
    )
    args = parser.parse_args()

    # Verify Stockfish exists
    sf_path = args.stockfish
    try:
        result = subprocess.run(
            [sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=5, input="quit\n", text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"Error: Cannot find/run Stockfish at '{sf_path}'")
        sys.exit(1)

    # Select configs
    if args.configs:
        configs = []
        for name in args.configs:
            cfg = get_config_by_name(name)
            if cfg is None:
                print(f"Unknown config: {name}")
                print(f"Available: {[c.name for c in DEFAULT_CONFIGS]}")
                sys.exit(1)
            configs.append(cfg)
    else:
        configs = list(DEFAULT_CONFIGS)

    # Override num_positions if specified
    if args.num is not None:
        for cfg in configs:
            cfg.num_positions = args.num

    generator = SyntheticGenerator(
        stockfish_path=sf_path,
        depth=args.depth,
        threads=args.threads,
        chunk_size=args.chunk_size,
    )
    generator.generate_all(args.output, configs)


if __name__ == "__main__":
    main()
