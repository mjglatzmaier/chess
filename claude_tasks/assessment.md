# haVoc Chess Engine — Comprehensive Assessment

## Executive Summary

haVoc is a **~9,000 LOC** UCI-compliant, multi-threaded chess engine written in C++ with
bitboard-based move generation, alpha-beta search with LMR/null-move pruning, and a
hand-crafted evaluation function. The core architecture is **solid and worth preserving** —
bitboard representation, magic move generation, and the overall search/eval split are
the right foundation for a strong engine.

**Estimated current strength**: ~2000–2200 Elo (based on feature set and tuning state)
**Target**: 3200+ Elo
**Gap**: Significant — requires NNUE or equivalent eval, deeper search optimizations,
and tablebases. The HCE (hand-crafted eval) ceiling is ~2800 Elo even with perfect tuning.

---

## Architecture Overview

```
┌─────────────┐
│   UCI Loop   │  uci.cpp — parse commands, manage game state
├─────────────┤
│   Search     │  search.h/hpp — iterative deepening, alpha-beta, quiescence
├─────────────┤
│  Evaluation  │  evaluate.cpp — HCE with material/pawn/piece/king/threats
├─────────────┤
│  Move Order  │  order.cpp — hash move, killers, history, SEE
├─────────────┤
│   Movegen    │  move.h/hpp — bitboard-based pseudo-legal generation
├─────────────┤
│  Position    │  position.cpp — make/unmake, legal move validation, SEE
├─────────────┤
│ Bitboards    │  bitboards.cpp, magics.cpp — attack tables, magic numbers
├─────────────┤
│  Hash Table  │  hashtable.cpp — transposition table with clustering
├─────────────┤
│  Threading   │  threads.cpp — thread pool with per-thread pawn/material cache
└─────────────┘
```

---

## What to KEEP (Good Foundations)

| Component | Why Keep | Notes |
|-----------|----------|-------|
| Bitboard representation | Industry standard, fast | Core data structures are sound |
| Magic bitboard movegen | O(1) sliding piece attacks | Pre-computed magics work correctly |
| Move encoding (from/to/type) | Simple, compact 3-byte moves | Could pack tighter but works fine |
| Zobrist hashing scheme | Multiple hash keys (position, pawn, material, repetition) | Well-thought-out |
| Pawn/material hash tables | Per-thread caching avoids lock contention | Good design |
| Piece-list with square_of arrays | O(1) piece iteration | Efficient for eval loops |
| UCI protocol handling | Feature-complete | Covers go/position/setoption/etc. |

---

## What to REWRITE

### 1. Evaluation Function → NNUE (Critical for 3200+ Elo)
**Current**: Hand-crafted eval (~1100 LOC) with material, mobility, pawn structure,
king safety, threats, space, and basic endgames. Many magic numbers, no Texel tuning.

**Problem**: HCE ceiling is ~2800 Elo even perfectly tuned. The eval has:
- Arbitrary bonus values (not tuned via Texel/SPSA)
- Missing important concepts (connectivity, piece coordination, pawn storms)
- Incomplete endgame knowledge (KRK, KQK, KBN vs K basics missing)
- Lazy eval that can skip critical positions

**Plan**:
- Phase 1: Keep HCE but properly Texel-tune all parameters
- Phase 2: Implement NNUE inference (HalfKP or HalfKAv2 architecture)
- Phase 3: Train NNUE net from self-play or Leela data
- Hook point: `eval::evaluate()` becomes a virtual dispatch

### 2. Threading Model → Modern SMP
**Current**: Custom `Threadpool<T>` with raw `new/delete`, `std::bind`, and global
mutable state. Search threads share `mPositions` vector without synchronization.

**Problems**:
- Memory leaks in pool destruction
- Race conditions on `searching_positions[]`, `selDepth`, `hashHits`
- No Lazy SMP (threads don't share hash table entries properly)
- `volatile` used instead of `std::atomic`

**Plan**: Implement proper Lazy SMP where each thread searches independently
with slight depth/aspiration variation, sharing only the transposition table.

### 3. Transposition Table → Thread-Safe, Correct Encoding
**Current**: 4-entry cluster, XOR-based collision detection, bit-packed entries.

**Critical Bugs**:
- `entry::age()` has operator precedence bug — always returns 0
- No atomic access — concurrent reads/writes cause data races
- Replacement policy never actually uses age (because age() is broken)

**Plan**: Rewrite with `std::atomic` or lock-free CAS, fix encoding,
add proper aging with generation counter.

### 4. Search → Add Missing Techniques
**Current**: Alpha-beta + LMR + null-move + aspiration windows (partial).

**Missing for 3200+**:
- Singular extensions (extend moves that are much better than alternatives)
- Countermove history / follow-up history
- Reverse futility pruning (static null-move pruning)
- Multi-cut pruning
- Probcut
- History-based pruning (not just ordering)
- Better time management (dynamic based on score stability)
- Contempt factor
- Syzygy tablebase probing

### 5. Build System → Modern CMake
**Current**: Single CMakeLists.txt with hardcoded compiler flags, no targets
for tests/benchmarks, platform-specific flag soup.

**Plan**:
```
cmake_minimum_required(VERSION 3.20)
project(haVoc VERSION 2.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

add_library(havoc_core STATIC src/...)
add_executable(havoc src/main.cpp)
target_link_libraries(havoc PRIVATE havoc_core)

# Tests
enable_testing()
add_executable(havoc_tests tests/...)
target_link_libraries(havoc_tests PRIVATE havoc_core GTest::gtest_main)

# Benchmarks
add_executable(havoc_bench bench/...)
```

---

## What to IMPROVE (Keep but Modernize)

### C++ Modernization Checklist

| Issue | Where | Fix |
|-------|-------|-----|
| `typedef` → `using` | types.h | `using U64 = uint64_t;` |
| Raw `new/delete` | threads.h, History | `std::unique_ptr`, RAII |
| `memset` for init | position.cpp, hashtable.cpp | Aggregate init, constructors |
| `volatile` signals | uci.h | `std::atomic<bool>` |
| `#include <stdio.h>` | multiple | `<cstdio>` |
| `#include <stdint.h>` | types.h | `<cstdint>` |
| C-style casts | throughout | `static_cast<>` |
| `atoi()` | uci.cpp | `std::stoi()` or `std::from_chars()` |
| Copy constructors | position.h | Rule of Five, `= default` |
| Global mutable state | search.hpp, uci.cpp | Encapsulate in classes |
| `const std::vector` globals | types.h | `constexpr std::array` |
| `std::bind` | threads.h | Lambdas |
| Platform #ifdefs for types | types.h | Just use `<cstdint>` |
| Enum operators | types.h | `enum class` + `constexpr` |
| `#pragma once` + include guards | everywhere | Pick one (prefer `#pragma once`) |

### Specific Code Smells

1. **`search.h` includes `search.hpp`** — Implementation in header means every
   translation unit that includes search.h recompiles the entire search. Move to .cpp.

2. **`info history[1024]`** — Fixed-size stack array. Use `std::vector` with
   reserve, or at least validate bounds.

3. **`const std::vector<Piece> Pieces{...}`** in types.h — Constructed at static
   init time in every TU that includes types.h. Use `constexpr std::array`.

4. **`SanSquares[64]`** defined in types.h as `const std::string[]` — Same
   static-init-order fiasco risk. Use `constexpr std::string_view[]`.

5. **`Move` struct** — The copy constructor and `operator=` are trivial; delete
   them and let the compiler generate them (or `= default`).

6. **`node` struct** — Allocates `History::bmHistory` on the heap in constructor;
   should use `std::unique_ptr` or embed directly.

---

## Critical Bugs Found

| # | Severity | Location | Description |
|---|----------|----------|-------------|
| 1 | 🔴 HIGH | hashtable.h:39 | `age()` operator precedence: `0x7F80000000000000 >> 55` evaluates shift before `&`. Age is always 0. |
| 2 | 🔴 HIGH | material.cpp copy ctor | Allocates array then overwrites pointer with `o.entries` — memory leak |
| 3 | 🔴 HIGH | pawns.cpp copy ctor | Same memory leak as material.cpp |
| 4 | 🟡 MED | search.hpp globals | `selDepth`, `hashHits`, `mPositions` are global — data races with SMP |
| 5 | 🟡 MED | search.hpp:19 | `volatile double elapsed` — should be `std::atomic<double>` |
| 6 | 🟡 MED | hashtable.cpp | No atomic access — concurrent TT reads/writes are UB |
| 7 | 🟡 MED | parameter.h:29 | `reinterpret_cast<unsigned long*>(value.get())` — strict aliasing violation |
| 8 | 🟡 MED | types.h:174-195 | `History::bmHistory` uses raw `new[]`/`delete[]` — leak-prone |
| 9 | 🟢 LOW | search.hpp:250 | `hashHits = 0` reset inside ID loop — not per-depth intentional? |
| 10 | 🟢 LOW | uci.cpp:58 | `atoi(cmd.c_str())` — no error handling for non-numeric input |

---

## Missing Features for 3200+ Elo

### Search (est. +400–600 Elo)
- [ ] Singular extensions
- [ ] Countermove history heuristic
- [ ] Capture history heuristic
- [ ] Reverse futility pruning / static null move pruning
- [ ] Multi-cut pruning
- [ ] Probcut
- [ ] Internal iterative reductions (IIR)
- [ ] Better aspiration window handling (exponential widening)
- [ ] Dynamic time management (score stability, best-move changes)
- [ ] Contempt factor for anti-draw play

### Evaluation (est. +600–1000 Elo)
- [ ] NNUE inference engine (HalfKP/HalfKAv2)
- [ ] Or: fully Texel-tuned HCE with SPSA optimization
- [ ] Proper tapered eval (middlegame/endgame interpolation per-term)
- [ ] Piece imbalance tables (bishop pair, rook vs 2 minors, etc.)
- [ ] Connectivity and coordination terms
- [ ] Pawn storm evaluation
- [ ] Proper king danger formula (quadratic, not linear)

### Infrastructure (est. +50–100 Elo indirect)
- [ ] Syzygy tablebase probing (6-man or 7-man)
- [ ] Opening book support (Polyglot format)
- [ ] Benchmark suite (fixed-depth search on test positions)
- [ ] Perft testing for move generation correctness
- [ ] CI/CD pipeline with regression testing
- [ ] SPRT testing framework (already have cutechess scripts)

---

## Proposed Remake Architecture

```
remake/
├── CMakeLists.txt              # Modern CMake with targets
├── include/
│   └── havoc/
│       ├── types.hpp           # Core types, enums, constexpr
│       ├── bitboard.hpp        # Bitboard utilities
│       ├── position.hpp        # Position representation
│       ├── movegen.hpp         # Move generation
│       ├── search.hpp          # Search interface
│       ├── eval/
│       │   ├── evaluator.hpp   # Abstract eval interface
│       │   ├── hce.hpp         # Hand-crafted eval
│       │   └── nnue.hpp        # NNUE eval (future)
│       ├── tt.hpp              # Transposition table
│       ├── thread_pool.hpp     # Threading
│       ├── uci.hpp             # UCI protocol
│       └── options.hpp         # Configuration
├── src/
│   ├── main.cpp
│   ├── bitboard.cpp
│   ├── position.cpp
│   ├── movegen.cpp
│   ├── search.cpp
│   ├── eval/
│   │   ├── hce.cpp
│   │   └── nnue.cpp
│   ├── tt.cpp
│   ├── thread_pool.cpp
│   ├── uci.cpp
│   └── zobrist.cpp
├── tests/
│   ├── test_bitboard.cpp
│   ├── test_movegen.cpp        # Perft tests
│   ├── test_position.cpp       # FEN parsing, make/unmake
│   ├── test_eval.cpp           # Known position evaluations
│   ├── test_search.cpp         # Mate-in-N, tactical puzzles
│   └── test_tt.cpp             # Hash table correctness
├── bench/
│   ├── bench_perft.cpp         # NPS benchmarks
│   ├── bench_search.cpp        # Fixed-depth benchmarks
│   └── bench_eval.cpp          # Eval speed benchmarks
└── .github/
    └── workflows/
        └── ci.yml              # Build + test + regression
```

### Key Design Principles for Remake
1. **Separation of concerns**: Eval is an interface, search doesn't know about eval internals
2. **No global mutable state**: Everything owned by an `Engine` class
3. **Thread safety by design**: Atomic TT, per-thread state, no shared mutables
4. **Testability**: Every component can be tested in isolation
5. **Extensibility hooks**: `IEvaluator` interface for HCE/NNUE/GPU backends
6. **Modern C++20**: concepts, ranges, constexpr, std::span, std::format

---

## Recommended Phased Plan

### Phase 1: Foundation (remake/ scaffold)
- Set up modern CMake with library + executable + test targets
- Port types, bitboards, magics (fixing bugs) with unit tests
- Perft test suite to validate move generation

### Phase 2: Core Engine
- Port position, movegen, zobrist with full test coverage
- Port search with bug fixes (threading, globals)
- Port HCE eval with Texel tuning infrastructure

### Phase 3: Strength Gains
- Add missing search features (singular extensions, countermove history, etc.)
- Implement Syzygy tablebase probing
- Texel-tune all eval parameters
- CI pipeline with SPRT regression testing

### Phase 4: NNUE
- Implement NNUE inference (HalfKAv2 architecture)
- Train initial net from Stockfish NNUE training data or self-play
- Integrate with search (lazy NNUE updates on make/unmake)

### Phase 5: Advanced
- GPU inference hooks for larger NNUE nets
- MCTS/hybrid search exploration
- Chess960 support
- Multi-PV analysis mode improvements

---

## Questions for You

1. **Target scope**: Do you want to reach 3200+ with HCE-only first, or jump straight to NNUE?
2. **Compatibility**: Should the remake maintain UCI compatibility with the original, or can we break the protocol?
3. **Tuning data**: Do you have game databases or positions for Texel tuning?
4. **Testing**: Want to use Google Test, Catch2, or something lighter?
5. **Tablebase support**: Do you have Syzygy tablebases available?
6. **GPU hooks**: Any specific GPU framework preference (CUDA, Vulkan compute, Metal)?
