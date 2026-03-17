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
import multiprocessing
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
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


# ---------------------------------------------------------------------------
# Systematic config generation
# ---------------------------------------------------------------------------

# Base imbalances: (stronger_pieces, weaker_pieces, category)
# These define the *piece* advantage without pawns. Kings are implicit.
_BASE_IMBALANCES = [
    # Queen advantage
    ("Q", "",     "queen_up"),
    ("Q", "R",    "queen_up"),
    ("Q", "B",    "queen_up"),
    ("Q", "N",    "queen_up"),
    ("Q", "BB",   "queen_vs_pieces"),
    ("Q", "BN",   "queen_vs_pieces"),
    ("Q", "NN",   "queen_vs_pieces"),
    ("Q", "RB",   "queen_vs_pieces"),
    ("Q", "RN",   "queen_vs_pieces"),
    ("Q", "RR",   "queen_vs_pieces"),
    ("Q", "BBN",  "queen_vs_pieces"),
    ("Q", "BNN",  "queen_vs_pieces"),
    ("Q", "RBN",  "queen_vs_pieces"),

    # Rook advantage
    ("R", "",     "rook_up"),
    ("R", "B",    "rook_vs_minor"),
    ("R", "N",    "rook_vs_minor"),
    ("RR", "R",   "rook_up"),
    ("R", "BN",   "rook_vs_pieces"),
    ("R", "NN",   "rook_vs_pieces"),
    ("R", "BB",   "rook_vs_pieces"),
    ("RR", "BN",  "rook_vs_pieces"),
    ("RR", "BB",  "rook_vs_pieces"),
    ("RR", "NN",  "rook_vs_pieces"),

    # Minor piece advantage
    ("B", "",     "minor_up"),
    ("N", "",     "minor_up"),
    ("BN", "",    "minor_pair_up"),
    ("BB", "",    "minor_pair_up"),
    ("NN", "",    "minor_pair_up"),
    ("BN", "B",   "minor_up"),
    ("BN", "N",   "minor_up"),
    ("BB", "N",   "minor_up"),

    # Piece vs pawns (weaker side has only pawns as compensation)
    ("Q", "PPP",  "piece_vs_pawns"),
    ("Q", "PPPP", "piece_vs_pawns"),
    ("Q", "PPPPP","piece_vs_pawns"),
    ("R", "PP",   "piece_vs_pawns"),
    ("R", "PPP",  "piece_vs_pawns"),
    ("R", "PPPP", "piece_vs_pawns"),
    ("B", "PP",   "piece_vs_pawns"),
    ("B", "PPP",  "piece_vs_pawns"),
    ("N", "PP",   "piece_vs_pawns"),
    ("N", "PPP",  "piece_vs_pawns"),
]

# Pawn imbalance overlays: (white_extra_pawns, black_extra_pawns)
# Applied on top of base imbalances to vary pawn structure
_PAWN_OVERLAYS = [
    (0, 0),   # pure piece imbalance
    (1, 0),   # stronger side +1P
    (0, 1),   # weaker side +1P (partial compensation)
    (1, 1),   # both +1P
    (2, 0),   # stronger side +2P
    (0, 2),   # weaker side +2P
    (2, 1),   # asymmetric
    (1, 2),   # weaker side has pawn compensation
]

# Background piece sets to simulate game phases
# Added to BOTH sides equally (no material change, just density)
_PHASE_BACKGROUNDS = [
    ("",    "",    "endgame"),       # pure endgame (no extras)
    ("R",   "R",   "late_middle"),   # late middlegame density
    ("RB",  "RB",  "middlegame"),    # middlegame density
    ("RBN", "RBN", "early_middle"),  # early middlegame density
]

# Max total pieces per side (king + pieces + pawns) before position
# generation becomes unreliable due to board crowding
_MAX_PIECES_PER_SIDE = 8


def _build_config(
    w_pieces: str, b_pieces: str, category: str, num_positions: int,
) -> MaterialConfig:
    """Build a MaterialConfig with auto-generated name."""
    w_sorted = "K" + "".join(sorted(w_pieces.replace("K", ""), key="QRBNP".index))
    b_sorted = "K" + "".join(sorted(b_pieces.replace("K", ""), key="QRBNP".index))
    name = f"{w_sorted}v{b_sorted}"
    return MaterialConfig(name, w_sorted, b_sorted, num_positions, category)


def generate_default_configs(
    positions_per_config: int = 500,
    include_phases: bool = True,
    max_pawn_overlay: int = 2,
) -> list[MaterialConfig]:
    """
    Systematically generate material configurations covering:
    - All base imbalances (piece advantages, piece-vs-pawns)
    - Pawn count variations (0-2 extra pawns per side)
    - Phase simulation (endgame through middlegame density)

    This produces ~300-500 configs targeting broad imbalance coverage.
    """
    seen: set[str] = set()
    configs: list[MaterialConfig] = []

    for stronger, weaker, base_cat in _BASE_IMBALANCES:
        # Filter pawn overlays by max_pawn_overlay
        overlays = [(wp, bp) for wp, bp in _PAWN_OVERLAYS
                     if wp <= max_pawn_overlay and bp <= max_pawn_overlay]

        phases = _PHASE_BACKGROUNDS if include_phases else [("", "", "endgame")]

        for extra_wp, extra_bp in overlays:
            for bg_w, bg_b, phase in phases:
                # Build piece strings (without king — added by _build_config)
                w_pieces = stronger + bg_w + "P" * extra_wp
                b_pieces = weaker + bg_b + "P" * extra_bp

                # Check piece count limits
                if len(w_pieces) + 1 > _MAX_PIECES_PER_SIDE:
                    continue
                if len(b_pieces) + 1 > _MAX_PIECES_PER_SIDE:
                    continue

                category = f"{base_cat}/{phase}"
                cfg = _build_config(w_pieces, b_pieces, category, positions_per_config)

                # Deduplicate
                if cfg.name not in seen:
                    seen.add(cfg.name)
                    configs.append(cfg)

    return configs


# Default configs: systematic coverage
DEFAULT_CONFIGS = generate_default_configs(
    positions_per_config=500,
    include_phases=True,
    max_pawn_overlay=2,
)


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
        self._send("ucinewgame")
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
    work_queue: "multiprocessing.Queue",
    result_queue: "multiprocessing.Queue",
    stockfish_path: str,
    depth: int,
    hash_mb: int,
) -> None:
    """
    Persistent worker process: pulls FEN batches from work_queue, evaluates
    with a single long-lived Stockfish instance, pushes results to result_queue.
    Sentinel value None signals shutdown.
    """
    with StockfishEvaluator(stockfish_path, depth=depth, hash_mb=hash_mb) as sf:
        while True:
            item = work_queue.get()
            if item is None:
                break
            batch_id, fens = item
            results = []
            for fen in fens:
                cp = sf.evaluate(fen)
                if cp is not None:
                    results.append((fen, cp_to_value(cp)))
            result_queue.put((batch_id, results))


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class SyntheticGenerator:
    """
    Generates synthetic chess positions with controlled material and
    Stockfish evaluations. Outputs training data in .npz chunk format.

    Uses persistent worker processes, each with a long-lived Stockfish
    instance, to avoid per-config process spawn and SF startup overhead.
    """

    def __init__(
        self,
        stockfish_path: str,
        depth: int = 20,
        threads: int = 4,
        chunk_size: int = 100_000,
        hash_mb: int = 64,
    ):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.threads = threads
        self.chunk_size = chunk_size
        self.hash_mb = hash_mb

    def _evaluate_fens(self, fens: list[str]) -> list[tuple[str, float]]:
        """
        Evaluate a list of FENs using persistent worker pool.
        Splits work into small batches across N Stockfish processes.
        """
        if not fens:
            return []

        if self.threads <= 1:
            # Single-threaded: direct evaluation, no IPC overhead
            results = []
            with StockfishEvaluator(self.stockfish_path, depth=self.depth,
                                     hash_mb=self.hash_mb) as sf:
                for fen in fens:
                    cp = sf.evaluate(fen)
                    if cp is not None:
                        results.append((fen, cp_to_value(cp)))
            return results

        # Multi-threaded: persistent workers with queues
        ctx = multiprocessing.get_context("spawn")
        work_queue: multiprocessing.Queue = ctx.Queue()
        result_queue: multiprocessing.Queue = ctx.Queue()

        # Start persistent workers
        workers = []
        for _ in range(self.threads):
            p = ctx.Process(
                target=_evaluate_worker,
                args=(work_queue, result_queue, self.stockfish_path,
                      self.depth, self.hash_mb),
            )
            p.start()
            workers.append(p)

        # Split FENs into small batches (~50 per batch for good load balancing)
        batch_size = max(1, min(50, len(fens) // (self.threads * 2)))
        batches = [fens[i:i + batch_size] for i in range(0, len(fens), batch_size)]
        for i, batch in enumerate(batches):
            work_queue.put((i, batch))

        # Collect results
        results: list[tuple[str, float]] = []
        for _ in range(len(batches)):
            batch_id, batch_results = result_queue.get()
            results.extend(batch_results)

        # Shutdown workers
        for _ in workers:
            work_queue.put(None)
        for p in workers:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()

        return results

    def generate_material_config(
        self,
        config: MaterialConfig,
        progress: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Generate random positions with specific material and evaluate with SF.
        Returns list of (FEN, value) pairs where value is in [-1, +1].
        """
        positions = generate_positions_for_config(config)
        fens = [fen for fen, _ in positions]

        if not fens:
            print(f"  Warning: Could not generate positions for {config.name}")
            return []

        results = self._evaluate_fens(fens)
        return results

    def _load_checkpoint(self, output_dir: str) -> tuple[set[str], dict[str, int], int, int]:
        """Load generation checkpoint if it exists. Returns (done_configs, stats, chunk_idx, total)."""
        ckpt_path = os.path.join(output_dir, "_checkpoint.json")
        if os.path.exists(ckpt_path):
            import json
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            done = set(ckpt.get("completed_configs", []))
            stats = ckpt.get("stats", {})
            chunk_idx = ckpt.get("next_chunk_idx", 0)
            total = ckpt.get("total_positions", 0)
            print(f"Resuming from checkpoint: {len(done)} configs done, "
                  f"{total:,} positions in {chunk_idx} chunks")
            return done, stats, chunk_idx, total
        return set(), {}, 0, 0

    def _save_checkpoint(
        self, output_dir: str, done: set[str], stats: dict, chunk_idx: int, total: int,
    ) -> None:
        """Save generation checkpoint for resume support."""
        import json
        ckpt = {
            "completed_configs": sorted(done),
            "stats": stats,
            "next_chunk_idx": chunk_idx,
            "total_positions": total,
            "depth": self.depth,
        }
        ckpt_path = os.path.join(output_dir, "_checkpoint.json")
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)

    def _save_metadata(self, output_dir: str, chunk_idx: int, total: int) -> None:
        """Write/update metadata.npz so partial results are already loadable."""
        metadata = {
            "total_positions": total,
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

    def generate_all(
        self,
        output_dir: str,
        configs: Optional[list[MaterialConfig]] = None,
    ) -> dict:
        """
        Generate the full synthetic dataset with per-config checkpointing.

        Uses a single persistent worker pool across all configs to minimize
        Stockfish startup overhead. Saves a chunk and checkpoint after each
        config so generation can be interrupted and resumed.

        Args:
            output_dir: Directory to write .npz chunks
            configs: Material configs to generate (default: DEFAULT_CONFIGS)

        Returns:
            Statistics dict with counts per config and totals.
        """
        if configs is None:
            configs = DEFAULT_CONFIGS

        os.makedirs(output_dir, exist_ok=True)
        source_id = 1

        # Load checkpoint for resume
        done_configs, stats, chunk_idx, total_positions = self._load_checkpoint(output_dir)

        remaining = [c for c in configs if c.name not in done_configs]
        print(f"Generating synthetic positions (depth={self.depth}, threads={self.threads})")
        print(f"  Total configs: {len(configs)}, remaining: {len(remaining)}, "
              f"target positions: {sum(c.num_positions for c in remaining):,}")
        print()

        if not remaining:
            print("All configs already completed.")
            return stats

        boards_buf: list = []
        values_buf: list = []
        policy_buf: list = []
        sources_buf: list = []
        weights_buf: list = []

        # Start persistent worker pool ONCE for all configs
        if self.threads > 1:
            ctx = multiprocessing.get_context("spawn")
            work_queue: multiprocessing.Queue = ctx.Queue()
            result_queue: multiprocessing.Queue = ctx.Queue()
            workers = []
            for _ in range(self.threads):
                p = ctx.Process(
                    target=_evaluate_worker,
                    args=(work_queue, result_queue, self.stockfish_path,
                          self.depth, self.hash_mb),
                )
                p.start()
                workers.append(p)
        else:
            # Single-thread: use direct evaluator
            sf_single = StockfishEvaluator(
                self.stockfish_path, depth=self.depth, hash_mb=self.hash_mb,
            )

        try:
            for ci, config in enumerate(remaining):
                # Generate positions (fast, CPU-only)
                positions = generate_positions_for_config(config)
                fens = [fen for fen, _ in positions]

                if not fens:
                    print(f"  Warning: Could not generate positions for {config.name}")
                    done_configs.add(config.name)
                    stats[config.name] = 0
                    continue

                # Evaluate with Stockfish
                if self.threads > 1:
                    batch_size = max(1, min(50, len(fens) // (self.threads * 2)))
                    batches = [fens[i:i + batch_size]
                               for i in range(0, len(fens), batch_size)]
                    for i, batch in enumerate(batches):
                        work_queue.put((i, batch))

                    results: list[tuple[str, float]] = []
                    for _ in range(len(batches)):
                        _, batch_results = result_queue.get()
                        results.extend(batch_results)
                else:
                    results = []
                    for fen in fens:
                        cp = sf_single.evaluate(fen)
                        if cp is not None:
                            results.append((fen, cp_to_value(cp)))

                stats[config.name] = len(results)

                # Encode positions into buffer
                for fen, value in results:
                    board = chess.Board(fen)
                    features = board_to_tensor(board)
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

                total_positions += len(results)
                done_configs.add(config.name)

                # Flush buffer to chunk when it's large enough
                while len(boards_buf) >= self.chunk_size:
                    _save_synthetic_chunk(
                        output_dir, chunk_idx,
                        boards_buf[:self.chunk_size],
                        values_buf[:self.chunk_size],
                        policy_buf[:self.chunk_size],
                        sources_buf[:self.chunk_size],
                        weights_buf[:self.chunk_size],
                    )
                    boards_buf = boards_buf[self.chunk_size:]
                    values_buf = values_buf[self.chunk_size:]
                    policy_buf = policy_buf[self.chunk_size:]
                    sources_buf = sources_buf[self.chunk_size:]
                    weights_buf = weights_buf[self.chunk_size:]
                    chunk_idx += 1

                # Checkpoint: flush remaining buffer so all evaluated data is on disk
                if boards_buf:
                    _save_synthetic_chunk(
                        output_dir, chunk_idx,
                        boards_buf, values_buf, policy_buf, sources_buf, weights_buf,
                    )
                    chunk_idx += 1
                    boards_buf, values_buf, policy_buf = [], [], []
                    sources_buf, weights_buf = [], []

                self._save_metadata(output_dir, chunk_idx, total_positions)
                self._save_checkpoint(output_dir, done_configs, stats,
                                      chunk_idx, total_positions)

                done_count = len(done_configs)
                total_count = len(configs)
                print(f"  [{done_count}/{total_count}] {config.name}: "
                      f"{len(results)} positions (total: {total_positions:,})")

        finally:
            # Clean up worker pool
            if self.threads > 1:
                for _ in workers:
                    work_queue.put(None)
                for p in workers:
                    p.join(timeout=10)
                    if p.is_alive():
                        p.kill()
            else:
                sf_single.close()

        # Print summary
        print(f"\nDone: {total_positions:,} positions in {chunk_idx} chunks")
        print(f"Output: {output_dir}/")
        print("\nPer-config breakdown:")
        for name, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {name:<30} {count:>6,}")

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
        "--stockfish", default=None,
        help="Path to Stockfish binary (required for generation)",
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
        "--category", default=None,
        help="Filter configs by category prefix (e.g., queen_up, rook_vs_minor)",
    )
    parser.add_argument(
        "--num", type=int, default=None,
        help="Override num_positions per config",
    )
    parser.add_argument(
        "--no-phases", action="store_true",
        help="Skip phase backgrounds (endgame only — faster, fewer configs)",
    )
    parser.add_argument(
        "--max-pawn-overlay", type=int, default=2,
        help="Max extra pawns per side in overlays (default: 2)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="Positions per chunk (default: 100000)",
    )
    parser.add_argument(
        "--hash", type=int, default=64,
        help="Stockfish hash table size in MB per instance (default: 64)",
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="Print all configs and exit (no generation)",
    )
    args = parser.parse_args()

    # Build config set with generation parameters
    all_configs = generate_default_configs(
        positions_per_config=args.num or 500,
        include_phases=not args.no_phases,
        max_pawn_overlay=args.max_pawn_overlay,
    )

    # --list-configs: print and exit
    if args.list_configs:
        categories: dict[str, list[str]] = {}
        for cfg in all_configs:
            categories.setdefault(cfg.category, []).append(cfg.name)
        print(f"Total configs: {len(all_configs)}, "
              f"target positions: {sum(c.num_positions for c in all_configs):,}\n")
        for cat in sorted(categories):
            names = categories[cat]
            print(f"  {cat} ({len(names)}):")
            for name in names:
                print(f"    {name}")
        sys.exit(0)

    # Verify Stockfish exists
    sf_path = args.stockfish
    if sf_path is None:
        parser.error("--stockfish is required for generation (use --list-configs to browse)")
    try:
        result = subprocess.run(
            [sf_path], stdout=subprocess.PIPE,
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
                print("Use --list-configs to see available configs")
                sys.exit(1)
            configs.append(cfg)
    elif args.category:
        configs = [c for c in all_configs if c.category.startswith(args.category)]
        if not configs:
            print(f"No configs match category '{args.category}'")
            print("Use --list-configs to see available categories")
            sys.exit(1)
    else:
        configs = all_configs

    # Override num_positions if specified and using named configs
    if args.num is not None and args.configs:
        for cfg in configs:
            cfg.num_positions = args.num

    generator = SyntheticGenerator(
        stockfish_path=sf_path,
        depth=args.depth,
        threads=args.threads,
        chunk_size=args.chunk_size,
        hash_mb=args.hash,
    )
    generator.generate_all(args.output, configs)


if __name__ == "__main__":
    main()
