# haVoc
A multi-threaded, UCI compliant C++ chess engine written in C++.

## Features
- Bitboard-based move generation
- UCI protocol support for GUI integration
- Optimized evaluation function with tunable parameters
- Multithreaded search
  
## Build Instructions

### Prerequisites
- **C++ Compiler**: Requires C++20 (GCC, Clang, or MSVC)
- **CMake**: Version 3.10+ is recommended

### Windows (MSVC or MinGW)
```sh
git clone https://github.com/mjglatzmaier/chess.git
cd chess
mkdir build && cd build
cmake ..
cmake --build .
```

# Example options
```
setoption name threads value 4
setoption name hash value 1000
```
