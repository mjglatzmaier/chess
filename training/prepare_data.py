"""
Prepare training data directly from PGN files.

Plays through each game, extracting (board, move_played, game_result) triples.
This gives us both value targets (who won) and policy targets (what move was
played) -- supervised learning from strong engine games.

This is analogous to next-token prediction in LLMs: given the board state,
predict the move a strong engine would play.

Supports checkpointing — re-run the same command to resume an interrupted run.

Uses pgn-extract (C tool) for PGN parsing instead of python-chess's chess.pgn,
which suffers from unbounded memory growth on large files due to GameNode tree
construction with circular references.

Usage:
    python prepare_data.py games.pgn --output data/round_0/
    python prepare_data.py games.pgn --output data/round_0/ --max-games 250000

Requires:
    pgn-extract (apt install pgn-extract)
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess

import chess
import numpy as np
from tqdm import tqdm

from encoding import board_to_tensor, move_to_index, NUM_FEATURES, NUM_GLOBAL_FEATURES

CHECKPOINT_FILE = "checkpoint.json"

RESULT_VALUES = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]$')


def _load_checkpoint(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_checkpoint(output_dir: str, state: dict) -> None:
    path = os.path.join(output_dir, CHECKPOINT_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def _iter_uci_games(stream):
    """
    Yield (headers_dict, uci_moves_list) for each game from pgn-extract
    UCI-format output.  Parsing is plain string ops — no chess library needed.
    """
    headers = {}
    for line in stream:
        line = line.rstrip("\n\r")

        m = HEADER_RE.match(line)
        if m:
            headers[m.group(1)] = m.group(2)
            continue

        # Blank or whitespace-only lines between header block and moves
        if not line.strip():
            continue

        # Moves line: "d2d4 g8f6 ... 1-0"
        tokens = line.split()
        # Strip result token from end (1-0, 0-1, 1/2-1/2, *)
        if tokens and tokens[-1] in ("1-0", "0-1", "1/2-1/2", "*"):
            tokens.pop()

        yield headers, tokens
        headers = {}


def process_pgn(
    pgn_path: str,
    output_dir: str,
    max_games: int = 0,
    skip_moves: int = 6,
    chunk_size: int = 100_000,
    min_elo: int = 0,
    max_elo: int = 99999,
) -> None:
    """
    Process a PGN file into training data.

    For each position in each game, saves:
      - board: [65, 27] features (64 squares + 1 global context token)
      - value: float in [-1, +1] (game outcome from side-to-move perspective)
      - policy: int in [0, 4095] (index of move actually played)

    PGN parsing is delegated to pgn-extract (C tool) which streams clean
    UCI-format output.  python-chess is used only for board state tracking
    and feature encoding — no GameNode trees, no memory accumulation.

    Checkpoints after each chunk so interrupted runs can be resumed.
    """
    pgn_extract = shutil.which("pgn-extract")
    if pgn_extract is None:
        raise RuntimeError(
            "pgn-extract not found.  Install it with:  sudo apt install pgn-extract"
        )

    os.makedirs(output_dir, exist_ok=True)

    num_features = NUM_FEATURES + NUM_GLOBAL_FEATURES
    ckpt = _load_checkpoint(output_dir)
    if ckpt is not None:
        total_games = ckpt["total_games"]
        total_positions = ckpt["total_positions"]
        skipped_games = ckpt["skipped_games"]
        chunk_idx = ckpt["chunk_idx"]
        resume_games = total_games
        print(f"Resuming from checkpoint: {total_games:,} games, "
              f"{total_positions:,} positions, chunk {chunk_idx}")
    else:
        total_games = 0
        total_positions = 0
        skipped_games = 0
        chunk_idx = 0
        resume_games = 0

    # Pre-allocate chunk buffers (reused across chunks)
    boards_buf = np.zeros((chunk_size, 65, num_features), dtype=np.float32)
    values_buf = np.zeros(chunk_size, dtype=np.float32)
    policy_buf = np.zeros(chunk_size, dtype=np.int64)
    sources_buf = np.zeros(chunk_size, dtype=np.uint8)
    weights_buf = np.ones(chunk_size, dtype=np.float32)
    buf_idx = 0

    # Pipe PGN through pgn-extract for robust C-based parsing.
    # -Wuci: output moves in UCI notation (e2e4 not e4)
    # -s: silent (no per-game status on stderr)
    # -C: strip comments  -N: strip NAGs  -V: strip variations
    # -w10000: wide lines so each game's moves fit on one line
    proc = subprocess.Popen(
        [pgn_extract, "-Wuci", "-s", "-C", "-N", "-V", "-w10000", pgn_path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, errors="replace", bufsize=1,
    )

    pbar = tqdm(
        desc="Games", unit=" games", initial=total_games,
        total=max_games if max_games > 0 else None, miniters=500,
    )

    # On resume, fast-skip already-processed games.
    # pgn-extract re-parses in C so this is ~100× faster than python-chess.
    games_seen = 0

    try:
        for headers, uci_moves in _iter_uci_games(proc.stdout):
            # Fast-skip on resume
            if games_seen < resume_games:
                games_seen += 1
                continue
            games_seen += 1

            result_str = headers.get("Result", "*")
            result = RESULT_VALUES.get(result_str)
            if result is None:
                skipped_games += 1
                continue

            try:
                w_elo = int(headers.get("WhiteElo", "0") or "0")
                b_elo = int(headers.get("BlackElo", "0") or "0")
            except ValueError:
                w_elo = b_elo = 0

            if min_elo > 0 and (w_elo < min_elo or b_elo < min_elo):
                skipped_games += 1
                continue
            if max_elo < 99999 and (w_elo > max_elo or b_elo > max_elo):
                skipped_games += 1
                continue

            board = chess.Board()
            move_num = 0

            for uci_str in uci_moves:
                move_num += 1
                # pgn-extract uses uppercase for promotions (e7e8Q);
                # python-chess expects lowercase
                move = chess.Move.from_uci(uci_str.lower())

                if move_num > skip_moves and not board.is_check():
                    boards_buf[buf_idx] = board_to_tensor(board)

                    if board.turn == chess.WHITE:
                        values_buf[buf_idx] = 2.0 * result - 1.0
                    else:
                        values_buf[buf_idx] = 2.0 * (1.0 - result) - 1.0

                    policy_buf[buf_idx] = move_to_index(move)
                    buf_idx += 1
                    total_positions += 1

                board.push(move)

                if buf_idx >= chunk_size:
                    _save_chunk(output_dir, chunk_idx, boards_buf, values_buf,
                                policy_buf, sources_buf, weights_buf, buf_idx)
                    chunk_idx += 1
                    buf_idx = 0

                    _save_checkpoint(output_dir, {
                        "total_games": total_games,
                        "total_positions": total_positions,
                        "skipped_games": skipped_games,
                        "chunk_idx": chunk_idx,
                    })
                    gc.collect()

            total_games += 1
            pbar.update(1)
            pbar.set_postfix(pos=f"{total_positions:,}", chunks=chunk_idx)

            if max_games > 0 and total_games >= max_games:
                break
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()

    pbar.close()

    if buf_idx > 0:
        _save_chunk(output_dir, chunk_idx, boards_buf, values_buf,
                    policy_buf, sources_buf, weights_buf, buf_idx)
        chunk_idx += 1

    metadata = {
        "total_positions": total_positions,
        "total_games": total_games,
        "skipped_games": skipped_games,
        "num_chunks": chunk_idx,
        "chunk_size": chunk_size,
        "num_features": NUM_FEATURES,
        "skip_moves": skip_moves,
        "source": pgn_path,
        "has_policy": True,
    }
    np.savez(os.path.join(output_dir, "metadata.npz"), **metadata)

    # Clean up checkpoint on successful completion
    ckpt_path = os.path.join(output_dir, CHECKPOINT_FILE)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"\nDone: {total_games:,} games -> {total_positions:,} positions in {chunk_idx} chunks")
    print(f"Skipped: {skipped_games:,} games")
    print(f"Output: {output_dir}/")


def _save_chunk(output_dir, chunk_idx, boards, values, policies,
                sources, weights, count):
    n = count
    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        boards=boards[:n], values=values[:n], policies=policies[:n],
        sources=sources[:n], weights=weights[:n],
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare PGN data for transformer training")
    parser.add_argument("pgn_file", help="Input PGN file")
    parser.add_argument("-o", "--output", default="data/round_0/", help="Output directory")
    parser.add_argument("--max-games", type=int, default=0, help="Max games (0=all)")
    parser.add_argument("--skip-moves", type=int, default=6, help="Skip first N moves")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Positions per chunk")
    parser.add_argument("--min-elo", type=int, default=0, help="Minimum Elo filter")
    parser.add_argument("--max-elo", type=int, default=99999, help="Maximum Elo filter")
    args = parser.parse_args()

    process_pgn(
        pgn_path=args.pgn_file,
        output_dir=args.output,
        max_games=args.max_games,
        skip_moves=args.skip_moves,
        chunk_size=args.chunk_size,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
    )


if __name__ == "__main__":
    main()
