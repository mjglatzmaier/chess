"""
Validate a trained chess transformer model.

Runs evaluation on standard positions and prints analysis:
  - Eval scores for known positions (startpos, imbalanced positions)
  - Eval symmetry check (white/black mirror)
  - Value distribution statistics
  - Policy top-move analysis

Usage:
    python validate.py models/round_0.pt
    python validate.py models/round_0.pt --positions test_positions.epd
"""

import argparse

import chess
import numpy as np
import torch

from model import ChessTransformer
from config import ModelConfig
from encoding import board_to_tensor, legal_move_mask, index_to_move


STANDARD_POSITIONS = [
    ("Startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("Queen odds", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
    ("Rook odds", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1"),
    ("KRK endgame", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
    ("KPK endgame", "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"),
    ("Drawn KNK", "4k3/8/8/8/8/8/8/4K1N1 w - - 0 1"),
]


def validate_model(model_path: str) -> None:
    """Run validation checks on a trained model."""

    # Load model
    print(f"Loading {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    model = ChessTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model: {model.count_parameters():,} params, trained for "
          f"{checkpoint.get('epoch', '?')} epochs, loss={checkpoint.get('loss', '?'):.6f}")
    print()

    # Evaluate standard positions
    print("=== Standard Position Evaluation ===")
    print(f"{'Position':<20} {'Value':>8} {'Centipawns':>12} {'Top Move':>10}")
    print("-" * 55)

    for name, fen in STANDARD_POSITIONS:
        board = chess.Board(fen)
        features = board_to_tensor(board)
        input_tensor = torch.from_numpy(features).unsqueeze(0)  # [1, 64, 25]

        with torch.no_grad():
            value, policy = model(input_tensor)

        val = value.item()
        # Convert value [-1, +1] to approximate centipawns
        # Using inverse sigmoid: cp = -K * log10((1 - val) / (1 + val) + eps)
        eps = 1e-7
        cp = -400.0 * np.log10(max(eps, (1 - val)) / max(eps, (1 + val)))

        # Find top policy move (masked to legal moves)
        mask = torch.from_numpy(legal_move_mask(board))
        masked_policy = policy.squeeze() + (mask - 1) * 1e9  # mask illegal to -inf
        top_idx = masked_policy.argmax().item()
        top_move = index_to_move(top_idx)

        # Verify it's legal
        move_str = top_move.uci() if top_move in board.legal_moves else "???"

        print(f"{name:<20} {val:>8.4f} {cp:>12.1f} {move_str:>10}")

    # Symmetry check
    print("\n=== Symmetry Check ===")
    fen_w = "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    fen_b = "rnbqkb1r/pppp1ppp/5n2/4p3/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 2 3"

    board_w = chess.Board(fen_w)
    board_b = chess.Board(fen_b)

    feat_w = torch.from_numpy(board_to_tensor(board_w)).unsqueeze(0)
    feat_b = torch.from_numpy(board_to_tensor(board_b)).unsqueeze(0)

    with torch.no_grad():
        val_w = model.evaluate(feat_w).item()
        val_b = model.evaluate(feat_b).item()

    print(f"White to move: {val_w:+.4f}")
    print(f"Black to move (mirror): {val_b:+.4f}")
    print(f"Difference: {abs(val_w - val_b):.4f} (ideal: ~0)")

    print("\n=== Value Distribution (random positions) ===")
    values = []
    for _ in range(100):
        board = chess.Board()
        # Play random moves
        for _ in range(np.random.randint(5, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(np.random.choice(moves))

        feat = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        with torch.no_grad():
            val = model.evaluate(feat).item()
        values.append(val)

    values = np.array(values)
    print(f"Mean:   {values.mean():+.4f}")
    print(f"Std:    {values.std():.4f}")
    print(f"Min:    {values.min():+.4f}")
    print(f"Max:    {values.max():+.4f}")
    print(f"Near 0 (|v| < 0.1): {(np.abs(values) < 0.1).sum()}/100")


def main():
    parser = argparse.ArgumentParser(description="Validate trained model")
    parser.add_argument("model", help="Path to .pt checkpoint")
    args = parser.parse_args()

    validate_model(args.model)


if __name__ == "__main__":
    main()
