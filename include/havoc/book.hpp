#pragma once

/// @file book.hpp
/// @brief Polyglot opening book interface (stub).

#include "havoc/position.hpp"
#include "havoc/types.hpp"

#include <string>

namespace havoc::book {

/// Load a Polyglot opening book.
bool load(const std::string& path);

/// Probe the book for the current position.
/// @return A book move, or a null move if not in book.
[[nodiscard]] Move probe(const position& pos);

/// @return true if a book is loaded.
[[nodiscard]] bool is_loaded();

} // namespace havoc::book
