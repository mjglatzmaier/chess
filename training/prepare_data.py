"""
Prepare training data directly from PGN files.

Plays through each game, extracting (board, move_played, game_result) triples.
This gives us both value targets (who won) and policy targets (what move was
played) -- supervised learning from strong engine games.

This is analogous to next-token prediction in LLMs: given the board state,
predict the move a strong engine would play.

Usage:
    python prepare_data.py games.pgn --output data/round_0/
    python prepare_data.py games.pgn --output data/round_0/ --max-games 50000
"""

import argparse
import os

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

from encoding import board_to_tensor, move_to_index, NUM_FEATURES


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
      - board: [64, 25] features
      - value: float in [-1, +1] (game outcome from side-to-move perspective)
      - policy: int in [0, 4095] (index of move actually played)
    """
    os.makedirs(output_dir, exist_ok=True)

    boards_buf = []
    values_buf = []
    policy_buf = []

    total_games = 0
    total_positions = 0
    skipped_games = 0
    chunk_idx = 0

    pgn_file = open(pgn_path, errors="replace")
    pbar = tqdm(desc="Games", unit=" games")

    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break

        result_str = game.headers.get("Result", "*")
        if result_str == "1-0":
            result = 1.0
        elif result_str == "0-1":
            result = 0.0
        elif result_str == "1/2-1/2":
            result = 0.5
        else:
            skipped_games += 1
            continue

        try:
            w_elo = int(game.headers.get("WhiteElo", "0") or "0")
            b_elo = int(game.headers.get("BlackElo", "0") or "0")
        except ValueError:
            w_elo = b_elo = 0

        if min_elo > 0 and (w_elo < min_elo or b_elo < min_elo):
            skipped_games += 1
            continue
        if max_elo < 99999 and (w_elo > max_elo or b_elo > max_elo):
            skipped_games += 1
            continue

        board = game.board()
        move_num = 0

        for move in game.mainline_moves():
            move_num += 1

            if move_num > skip_moves and not board.is_check():
                features = board_to_tensor(board)

                if board.turn == chess.WHITE:
                    value = 2.0 * result - 1.0
                else:
                    value = 2.0 * (1.0 - result) - 1.0

                policy_idx = move_to_index(move)

                boards_buf.append(features)
                values_buf.append(value)
                policy_buf.append(policy_idx)
                total_positions += 1

            board.push(move)

            if len(boards_buf) >= chunk_size:
                _save_chunk(output_dir, chunk_idx, boards_buf, values_buf, policy_buf)
                chunk_idx += 1
                boards_buf = []
                values_buf = []
                policy_buf = []

        total_games += 1
        pbar.update(1)
        pbar.set_postfix(positions=total_positions, skipped=skipped_games)

        if max_games > 0 and total_games >= max_games:
            break

    pbar.close()
    pgn_file.close()

    if boards_buf:
        _save_chunk(output_dir, chunk_idx, boards_buf, values_buf, policy_buf)
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

    print(f"\nDone: {total_games:,} games -> {total_positions:,} positions in {chunk_idx} chunks")
    print(f"Skipped: {skipped_games:,} games")
    print(f"Output: {output_dir}/")


def _save_chunk(output_dir, chunk_idx, boards, values, policies):
    boards_array = np.stack(boards, axis=0)
    values_array = np.array(values, dtype=np.float32)
    policy_array = np.array(policies, dtype=np.int64)
    n = len(boards)
    sources_array = np.zeros(n, dtype=np.uint8)   # 0 = ccrl
    weights_array = np.ones(n, dtype=np.float32)   # uniform weight

    path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        boards=boards_array, values=values_array, policies=policy_array,
        sources=sources_array, weights=weights_array,
    )
    tqdm.write(f"  Saved chunk {chunk_idx}: {len(boards):,} positions")


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
