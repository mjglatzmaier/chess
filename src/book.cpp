#include "havoc/book.hpp"

// TODO: implement Polyglot .bin book parsing

namespace havoc::book {

bool load(const std::string& /*path*/) {
    return false;
}

Move probe(const position& /*pos*/) {
    return Move{}; // null move
}

bool is_loaded() {
    return false;
}

} // namespace havoc::book
