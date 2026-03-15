#pragma once

#include "havoc/bitboard.hpp"
#include "havoc/types.hpp"

namespace havoc::magics {

/// Compute sliding-piece attacks for a given occupancy and square.
/// Specialized for bishop and rook only.
template <Piece p> [[nodiscard]] U64 attacks(U64 occ, Square sq);

/// Initialize magic bitboard tables. Must be called once at startup.
void init();

} // namespace havoc::magics
