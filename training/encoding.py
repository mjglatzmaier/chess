"""
Board encoding: convert chess positions to/from tensor representations.

Encoding scheme:
  Per square (64 squares total), we encode:
    - Piece presence: 12 planes (P,N,B,R,Q,K for white, p,n,b,r,q,k for black)
    - Side to move: 1 plane (all 1s if white to move, all 0s if black)
    - Castling rights: 4 planes (KQkq, each all 1s or all 0s)
    - En passant: 8 planes (one per file, all 0s if no ep)
  Total: 25 binary features per square → [64, 25] tensor

  This flat-per-square encoding is natural for transformer input where
  each square becomes a token in the sequence.
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

NUM_FEATURES = 25  # 12 piece + 1 stm + 4 castling + 8 ep


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board as a [64, 25] float32 array.

    Each of the 64 squares gets a 25-dimensional feature vector:
      [0:12]  - piece presence (one-hot over 12 piece types)
      [12]    - side to move (1 if white, 0 if black)
      [13:17] - castling rights (K, Q, k, q)
      [17:25] - en passant file (one-hot, 8 files)

    Square ordering: a1=0, b1=1, ..., h8=63 (same as python-chess).
    """
    features = np.zeros((64, NUM_FEATURES), dtype=np.float32)

    # Piece planes (0-11)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            features[sq, plane] = 1.0

    # Side to move (12) — broadcast to all squares
    stm = 1.0 if board.turn == chess.WHITE else 0.0
    features[:, 12] = stm

    # Castling rights (13-16) — broadcast to all squares
    features[:, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    features[:, 14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    features[:, 15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    features[:, 16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En passant (17-24) — broadcast ep file to all squares
    if board.has_legal_en_passant():
        ep_file = chess.square_file(board.ep_square)
        features[:, 17 + ep_file] = 1.0

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
