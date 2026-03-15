# Task 02 — Port Core Types, Bitboards & Move Generation

## Goal
Port the foundational layer — types, bitboard utilities, magic bitboards, zobrist
hashing, and move generation — into the remake with C++20 idioms, full documentation,
and comprehensive unit tests (especially perft for movegen correctness).

---

## Steps

### 2.1 — Core Types (`include/havoc/types.hpp`)
Port from `src/types.h` with these changes:
- Replace all `typedef` with `using` declarations
- Use `<cstdint>` only — remove all MSVC-specific `__int16` etc.
- Use `enum class` for `Piece`, `Color`, `Square`, `Row`, `Col`, `Movetype`, etc.
  - Add `constexpr` conversion helpers: `to_int()`, `from_int<T>()`
  - Provide `operator++`, `operator--` as free functions with `constexpr`
- Replace `const std::vector<Piece> Pieces{...}` with `constexpr std::array`
- Replace `const std::string SanSquares[64]` with `constexpr std::array<std::string_view, 64>`
- Replace `const std::vector<char> SanPiece` with `constexpr std::array<char, 12>`
- Replace `const std::map<char, U16> CastleRights` with `constexpr` lookup function
- Move `Move` struct here, simplify:
  - Delete explicit copy ctor/operator= (let compiler generate as `= default`)
  - Add `[[nodiscard]] constexpr bool is_null() const`
  - Add `constexpr` to all operators
- Move `Score` enum values to a proper `constexpr` constants namespace
- Replace `node` struct's raw `History::bmHistory*` with `std::unique_ptr` or
  embed array directly with `std::array`
- Remove `History` struct's manual `new[]`/`delete[]` — use `std::array` or `std::vector`
- Add thorough Doxygen-style doc comments to every enum and struct

### 2.2 — Bitboard Utilities (`include/havoc/bitboard.hpp`, `src/bitboard.cpp`)
Port from `src/bits.h` and `src/bitboards.h/.cpp`:
- `namespace havoc::bitboard`
- Use C++20 `<bit>` header: `std::popcount`, `std::countr_zero` instead of
  compiler intrinsics — this handles cross-platform automatically
- If `<bit>` not available (older compilers), provide fallback with `#if __has_include`
- Make all lookup tables `constexpr` where possible (pawn attacks, knight masks, king masks)
- For tables that require runtime init (between, reductions), use a singleton
  `BitboardInit` class or `std::call_once`
- Remove all global mutable state — tables become `static const` after init
- Add `[[nodiscard]]` to all query functions
- Document each table with its purpose and indexing scheme

**Unit tests** (`tests/test_bitboard.cpp`):
- `popcount` returns correct values for known bitboards
- `lsb` / `pop_lsb` correct for edge cases (single bit, all bits, adjacent bits)
- Pawn attack masks correct for corner/edge squares
- Knight masks correct for all 64 squares (spot check corners + center)
- King masks correct
- `between[A1][H8]` returns diagonal
- `between[A1][A8]` returns file

### 2.3 — Magic Bitboards (`include/havoc/magics.hpp`, `src/magics.cpp`)
Port from `src/magics.h/.cpp` and `src/magicsrands.h`:
- Keep pre-computed magic numbers (they're correct and well-tested)
- Fix the VLA issue: `U64 stored[64][144]` → `std::array` or heap allocation
- Fix logic bug: `if (!prev && prev != atk)` → clarify intent
- Make `attacks<bishop>()` and `attacks<rook>()` return `[[nodiscard]] U64`
- Encapsulate in `namespace havoc::magics`
- Init via `magics::init()` called once at startup

**Unit tests** (`tests/test_magics.cpp`):
- Bishop attacks from center square with empty board = correct diagonal squares
- Rook attacks from center square with empty board = correct rank+file
- Bishop attacks with blockers = stops at blocker
- Rook attacks with blockers = stops at blocker
- Edge cases: corner squares, board edges

### 2.4 — Zobrist Hashing (`include/havoc/zobrist.hpp`, `src/zobrist.cpp`)
Port from `src/zobrist.h/.cpp` and `src/zobristrands.h`:
- Keep pre-computed random numbers (changing them breaks compatibility)
- Make all accessors `constexpr` or `inline`
- Encapsulate in `namespace havoc::zobrist`
- Document each hash component (piece, castling, ep, stm, move50, halfmoves)

**Unit tests** (`tests/test_zobrist.cpp`):
- Same position reached via different move orders produces same hash
- Hash changes when piece moves
- Hash changes when side to move flips
- Hash changes with castling rights

### 2.5 — Move Generation (`include/havoc/movegen.hpp`, `src/movegen.cpp`)
Port from `src/move.h` and `src/move.hpp`:
- **Critical change**: Move implementation from `.hpp` to `.cpp` — the current design
  forces recompilation of everything when movegen changes
- Keep template specialization approach (it's efficient)
- Add bounds checking on move list: `static_assert` or runtime check on `last < MAX_MOVES`
- Fix pawn capture direction documentation
- Encapsulate in `namespace havoc`

**Unit tests — PERFT** (`tests/test_perft.cpp`):
This is the most important test suite. Use standard perft positions:

| Position | Depth | Expected Nodes |
|----------|-------|----------------|
| Startpos | 1 | 20 |
| Startpos | 2 | 400 |
| Startpos | 3 | 8,902 |
| Startpos | 4 | 197,281 |
| Startpos | 5 | 4,865,609 |
| Kiwipete (`r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -`) | 1 | 48 |
| Kiwipete | 2 | 2,039 |
| Kiwipete | 3 | 97,862 |
| Kiwipete | 4 | 4,085,603 |
| Position 3 (`8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -`) | 1-5 | standard values |
| Position 4 (`r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -`) | 1-5 | |
| Position 5 (`rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -`) | 1-5 | |

Each perft test validates:
- Total node count matches expected
- All move types generated correctly (quiet, capture, ep, castling, promotion)

### 2.6 — Position Representation (`include/havoc/position.hpp`, `src/position.cpp`)
Port from `src/position.h/.cpp`:
- Replace `info history[1024]` with `std::vector<info>` (reserve 1024)
- Add bounds check on `hidx` in `do_move` / `undo_move`
- Make `piece_data` use `std::array` explicitly (it already does, just clean up)
- Fix `squares_of()` const-correctness — don't cast away const
- Add `[[nodiscard]]` to all query methods
- Document each method with pre/post conditions
- Make FEN parsing more robust (validate input)

**Unit tests** (`tests/test_position.cpp`):
- Parse startpos FEN → verify all piece positions
- Parse complex FEN → verify piece counts, castling rights, ep square
- `to_fen()` round-trips: parse FEN → to_fen → parse again → identical
- `do_move` + `undo_move` = original position (hash, pieces, all state)
- `do_null_move` + `undo_null_move` = original position
- `is_legal()` rejects known illegal moves
- `is_legal()` accepts known legal moves
- `in_check()` correct for known positions
- `pinned()` correct for known positions
- `is_draw()` detects repetition
- `is_draw()` detects 50-move rule

### 2.7 — Code Style Pass
- Run clang-format on all new files
- Run clang-tidy and fix all warnings
- Ensure consistent naming: `snake_case` for functions/variables, `PascalCase` for types
- Add file-level doc comments explaining purpose of each file
- Verify no compiler warnings on GCC, Clang, MSVC (at `/W4` / `-Wall -Wextra`)

### 2.8 — Verify
- All perft tests pass (this validates movegen + position correctness)
- All unit tests pass on local machine
- Zero compiler warnings
- clang-format clean
- Do NOT commit — leave for manual review

---

## Acceptance Criteria
- [ ] All perft positions match expected node counts through depth 4+
- [ ] All unit tests pass (bitboard, magics, zobrist, position)
- [ ] Zero compiler warnings with full warnings enabled
- [ ] No global mutable state (all state in classes or function-local)
- [ ] clang-format produces no changes
- [ ] Every public function has a doc comment
