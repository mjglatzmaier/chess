"""
Board encoding: convert chess positions to/from tensor representations.

Encoding scheme (v2 — global context token):
  64 square tokens, each with 13 features:
    - Piece presence: 12 planes (P,N,B,R,Q,K for white, p,n,b,r,q,k for black)
    - Empty square indicator: 1 plane

  1 global context token (index 64) with 14 features:
    - Side to move: 1 plane
    - Castling rights: 4 planes (KQkq)
    - En passant file: 8 planes (one per file)
    - Halfmove clock: 1 plane (normalized to [0, 1])

  Total: 65 tokens × 27 features (zero-padded to uniform width).

  The global token approach avoids broadcasting side-to-move and castling
  to all 64 squares, letting the model learn the difference between
  per-square and global information through attention.
"""

import chess
import numpy as np


# Piece to plane index mapping
PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

NUM_FEATURES = 13  # 12 piece planes + 1 empty indicator (per-square only)
NUM_GLOBAL_FEATURES = 14  # 1 stm + 4 castling + 8 ep + 1 halfmove clock


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board as a [65, 27] float32 array.

    64 square tokens with per-square features:
      [0:12]  - piece presence (one-hot over 12 piece types)
      [12]    - empty square indicator (1 if no piece, 0 if occupied)

    1 global context token (index 64) with board-level features:
      [13]    - side to move (1 if white, 0 if black)
      [14:18] - castling rights (K, Q, k, q)
      [18:26] - en passant file (one-hot, 8 files)
      [26]    - halfmove clock (normalized: clock / 100, clamped to [0, 1])

    Total features per token: 27 (13 per-square + 14 global, zero-padded).
    Square ordering: a1=0, b1=1, ..., h8=63, global=64.
    """
    num_features = NUM_FEATURES + NUM_GLOBAL_FEATURES
    features = np.zeros((65, num_features), dtype=np.float32)

    # Piece planes (0-11) for square tokens
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            features[sq, plane] = 1.0
        else:
            features[sq, 12] = 1.0  # empty square indicator

    # Global context token (index 64)
    g = 64
    features[g, 13] = 1.0 if board.turn == chess.WHITE else 0.0
    features[g, 14] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    features[g, 15] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    features[g, 16] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    features[g, 17] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.has_legal_en_passant():
        ep_file = chess.square_file(board.ep_square)
        features[g, 18 + ep_file] = 1.0

    # Halfmove clock (normalized to [0, 1])
    features[g, 26] = min(board.halfmove_clock / 100.0, 1.0)

    return features


def fen_to_tensor(fen: str) -> np.ndarray:
    """Convert a FEN string to a [64, 25] tensor."""
    board = chess.Board(fen)
    return board_to_tensor(board)


def tensor_to_torch(features: np.ndarray):
    """Convert numpy array to PyTorch tensor."""
    import torch
    return torch.from_numpy(features)


def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess move to a policy index.

    Encoding: from_square * 64 + to_square = index in [0, 4095]
    Promotions: we encode queen promotion as the default; underpromotions
    use additional indices 4096+ (not implemented in v1 — queen promo only).
    """
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int) -> chess.Move:
    """Convert a policy index back to a chess move (queen promotion assumed)."""
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Create a binary mask over the 4096-dim policy space.
    1.0 for legal moves, 0.0 for illegal moves.
    Used to mask policy logits before softmax during inference.
    """
    mask = np.zeros(4096, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1.0
    return mask
