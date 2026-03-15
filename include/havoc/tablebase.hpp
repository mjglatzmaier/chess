#pragma once

/// @file tablebase.hpp
/// @brief Syzygy tablebase probing interface (stub).

#include "havoc/position.hpp"

#include <string>

namespace havoc::tablebase {

/// Initialize tablebase access.
/// @param path Directory containing Syzygy tablebase files.
/// @return true if at least some tables were found.
bool init(const std::string& path);

/// Probe Win/Draw/Loss for the given position.
/// @return +1 (win), 0 (draw), -1 (loss), -2 (probe failed/not available)
[[nodiscard]] int probe_wdl(const position& pos);

/// Probe Distance To Zeroing (used at root for best move selection).
/// @return DTZ value, or 0 if probe failed.
[[nodiscard]] int probe_dtz(const position& pos);

/// @return true if tablebase files are loaded and available.
[[nodiscard]] bool available();

/// @return maximum number of pieces in available tablebases.
[[nodiscard]] int max_pieces();

} // namespace havoc::tablebase
