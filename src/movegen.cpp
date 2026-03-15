#include "havoc/movegen.hpp"

namespace havoc {

void Movegen::print() {
    for (int j = 0; j < last; ++j) {
        std::cout << kSanSquares[list[j].f] << kSanSquares[list[j].t] << " ";
    }
    std::cout << "\n";
}

void Movegen::print_legal(position& p) {
    for (int j = 0; j < last; ++j) {
        if (p.is_legal(list[j]))
            std::cout << kSanSquares[list[j].f] << kSanSquares[list[j].t] << " ";
    }
    std::cout << "\n";
}

} // namespace havoc
