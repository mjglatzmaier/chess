#pragma once

#include "havoc/types.hpp"

namespace havoc::zobrist {

/// Initialize zobrist hash tables. Must be called once at startup.
void init();

/// Zobrist key for a piece on a square.
[[nodiscard]] U64 piece(Square sq, Color c, Piece p);

/// Zobrist key for castling rights.
[[nodiscard]] U64 castle(Color c, U16 rights);

/// Zobrist key for en-passant file.
[[nodiscard]] U64 ep(int file);

/// Zobrist key for side to move.
[[nodiscard]] U64 stm(Color c);

/// Zobrist key for 50-move rule counter.
[[nodiscard]] U64 mv50(int move50);

/// Zobrist key for half-move counter.
[[nodiscard]] U64 hmvs(int halfmoves);

} // namespace havoc::zobrist
