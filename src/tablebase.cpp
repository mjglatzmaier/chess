#include "havoc/tablebase.hpp"

// TODO: integrate Fathom library for actual Syzygy probing

namespace havoc::tablebase {

bool init(const std::string& /*path*/) {
    return false;
}

int probe_wdl(const position& /*pos*/) {
    return -2; // probe failed / not available
}

int probe_dtz(const position& /*pos*/) {
    return 0; // probe failed
}

bool available() {
    return false;
}

int max_pieces() {
    return 0;
}

} // namespace havoc::tablebase
