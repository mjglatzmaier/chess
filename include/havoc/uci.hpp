#pragma once

/// @file uci.hpp
/// @brief UCI (Universal Chess Interface) protocol handler.

#include "havoc/search.hpp"

#include <string>
#include <string_view>

namespace havoc {

class SearchEngine;
class position;

namespace uci {

/// Start FEN for standard chess.
inline constexpr std::string_view START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Main UCI command loop. Blocks until "quit" or EOF.
void loop(SearchEngine& engine);

/// Parse and dispatch a single UCI command. Returns false on quit.
bool parse_command(const std::string& input, SearchEngine& engine, position& pos);

/// Convert a Move struct to UCI long-algebraic notation (e.g. "e2e4", "e7e8q").
std::string move_to_string(const Move& m);

/// Apply a sequence of moves from the "position ... moves ..." command.
void load_position(const std::string& pos, position& uci_pos);

} // namespace uci
} // namespace havoc
