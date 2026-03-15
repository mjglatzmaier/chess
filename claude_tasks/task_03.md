# Task 03 — Port Search, Evaluation, Hash Table & UCI

## Goal
Port the search engine, evaluation, transposition table, move ordering, threading,
and UCI protocol into the remake — fixing all known bugs, adding thread safety,
and establishing a fully playable engine.

---

## Steps

### 3.1 — Transposition Table (`include/havoc/tt.hpp`, `src/tt.cpp`)
Port from `src/hashtable.h/.cpp` with these **critical fixes**:

**Bug fixes**:
- Fix `entry::age()` operator precedence: `(dkey >> 55) & 0xFF` not `dkey & (0x7F80.. >> 55)`
- Fix `entry::depth()` similarly — verify all bit extraction masks
- Add thread-safe access: use `std::atomic<uint64_t>` for both `pkey` and `dkey`,
  or use a simple spinlock per cluster
- Verify XOR collision scheme: `pkey = key ^ dkey` on write, `(pkey ^ dkey) == key` on read

**Improvements**:
- Add generation counter (incremented each `ucinewgame`) for proper aging
- Replace `_mm_prefetch` with `__builtin_prefetch` (cross-platform) or
  wrap in a portable `havoc::prefetch()` function
- Replace `#include <xmmintrin.h>` — not available on all platforms
- Add `clear()` method that uses `std::memset` safely
- Add `hashfull()` method (UCI protocol requires this)
- Make `resize()` use `std::make_unique<hash_cluster[]>(count)` (already does, just verify)

**Unit tests** (`tests/test_tt.cpp`):
- Store and retrieve entry → get same data back
- Overwrite with deeper entry → deeper entry preserved
- Overwrite with shallower entry → depends on replacement policy
- Collision detection works (different key, same index)
- `hashfull()` returns 0 for empty table, >0 after stores
- Thread-safety test: multiple threads storing/fetching concurrently without crash

### 3.2 — Move Ordering (`include/havoc/move_order.hpp`, `src/move_order.cpp`)
Port from `src/order.h/.cpp`:
- Replace `std::shared_ptr` with `std::unique_ptr` where ownership is clear
- Replace `std::function<>` callbacks with templates (avoid allocation overhead)
- Fix `Movehistory` copy — ensure deep copy of all arrays
- Replace `std::bind` with lambdas
- Add `[[nodiscard]]` to scoring functions
- Document move ordering phases clearly

**Unit tests** (`tests/test_move_order.cpp`):
- Hash move is returned first
- Captures before quiets
- Killer moves in correct phase
- History scoring affects quiet move order

### 3.3 — Evaluation (`include/havoc/eval/evaluator.hpp`, `include/havoc/eval/hce.hpp`, `src/eval/hce.cpp`)
Port from `src/evaluate.h/.cpp`, `src/parameter.h`, `src/squares.h`, `src/endgame.h`:

**Architecture change** — introduce evaluator interface:
```cpp
namespace havoc {
    class IEvaluator {
    public:
        virtual ~IEvaluator() = default;
        virtual int evaluate(const Position& pos, int lazy_margin = -1) = 0;
        virtual std::string name() const = 0;
    };

    class HCEEvaluator : public IEvaluator {
        // ... current HCE implementation
    };
}
```

**Improvements**:
- Move `einfo` struct members into the HCE class (not needed by other evaluators)
- Replace all magic numbers with named constants in `parameters` struct
- Fix `parameter<T>` class — remove broken `reinterpret_cast` / `memcpy` aliasing
- Make `square_score<Color>()` use proper tapered eval formula
- Document each eval term with chess knowledge explanation
- Keep lazy eval but document its risks

**Port pawn/material tables**:
- Fix memory leak in pawn_table copy constructor (allocate-then-overwrite bug)
- Fix memory leak in material_table copy constructor (same bug)
- Make these per-thread (already are via Searchthread, just verify)

**Unit tests** (`tests/test_eval.cpp`):
- Startpos evaluates to ~0 (symmetric)
- Position with extra queen evaluates strongly positive for that side
- Known drawn positions (KK, KNK, KBK) evaluate to 0
- Pawn structure scoring: isolated pawn penalized, passed pawn bonused
- Evaluation is symmetric: eval(pos) == -eval(mirror(pos))

### 3.4 — Threading (`include/havoc/thread_pool.hpp`, `src/thread_pool.cpp`)
Port from `src/threads.h/.cpp`:
- Replace raw `new T(...)` with `std::make_unique<T>(...)`
- Replace `std::bind` with lambdas in `enqueue()`
- Replace `std::vector<T*>` with `std::vector<std::unique_ptr<T>>`
- Add proper destructor that joins all threads safely
- Replace `volatile` with `std::atomic`
- Each `SearchThread` owns its own `pawn_table` and `material_table` (already does)

**Unit tests** (`tests/test_thread_pool.cpp`):
- Enqueue 100 tasks → all complete
- Thread pool with 4 threads → tasks run concurrently
- Pool destruction joins all threads cleanly
- No memory leaks (run under ASan)

### 3.5 — Search (`include/havoc/search.hpp`, `src/search.cpp`)
Port from `src/search.h` and `src/search.hpp`:

**Critical changes**:
- **Move ALL code from search.hpp to search.cpp** — template instantiations
  for `search<root>`, `search<pv>`, `search<non_pv>` go in .cpp with explicit
  template instantiation
- Replace ALL global variables with members of a `SearchEngine` class:
  ```cpp
  class SearchEngine {
      std::vector<std::unique_ptr<Position>> thread_positions;
      std::atomic<int> sel_depth{0};
      std::atomic<size_t> hash_hits{0};
      std::atomic<bool> searching{false};
      // ...
  };
  ```
- Replace `volatile double elapsed` with `std::atomic<double>`
- Replace `volatile bool stop` with `std::atomic<bool>`
- Replace `searching_positions[]` global with per-SearchEngine state
- Fix aspiration window: the current code has fail-high/fail-low swapped
  (line 257-263: `failLow` triggers beta widening, `failHigh` triggers alpha widening — verify)
- Add `std::mutex` protection on `readout_pv` (already has it, verify correctness)
- Null move pruning: add zugzwang detection (skip NMP in pawn-only endgames)
- Time management: use `std::chrono::steady_clock` instead of custom timer

**Unit tests** (`tests/test_search.cpp`):
- Mate in 1: finds correct move
- Mate in 2: finds correct move within depth 4
- Mate in 3: finds correct move within depth 6
- Avoid stalemate: doesn't play stalemate when winning
- Known tactical positions: finds winning tactic
- Search returns `Score::draw` for drawn positions
- Multi-threaded search doesn't crash (run with 4 threads, ASan)

### 3.6 — UCI Protocol (`include/havoc/uci.hpp`, `src/uci.cpp`)
Port from `src/uci.h/.cpp`:
- Replace `atoi()` with `std::stoi()` or `std::from_chars()`
- Replace `memset(&lims, 0, sizeof(limits))` with aggregate init `limits lims{}`
- Add `"id name haVoc 2.0"` and `"id author M.Glatzmaier"`
- Add `hashfull` output in `info` string
- Add `nps` calculation (currently commented out)
- Proper `ucinewgame` handling (clear all caches)
- Handle `ponderhit` correctly
- Validate all input (don't crash on malformed commands)

**Unit tests** (`tests/test_uci.cpp`):
- Parse `"position startpos moves e2e4 e7e5"` → correct position
- Parse `"go depth 5"` → correct limits struct
- Parse `"setoption name Hash value 256"` → hash resized
- `move_to_string()` produces correct UCI notation for all move types

### 3.7 — Options (`include/havoc/options.hpp`, `src/options.cpp`)
Port from `src/options.h`:
- Simplify: just use `std::unordered_map<std::string, std::string>` with mutex
- Remove broken `parameter<T>` template class (not used meaningfully)
- Add proper defaults in constructor
- Remove fragile string matching in `set_engine_params()`

### 3.8 — Integration Test
- Build full engine: `cmake --build build`
- Run against cutechess-cli or similar:
  ```
  echo -e "uci\nisready\nposition startpos\ngo depth 10\nquit" | ./havoc
  ```
- Verify UCI output is correct
- Run `bench` command if implemented (fixed depth search on set of positions)

### 3.9 — Code Quality Pass
- clang-format all files
- clang-tidy all files, fix warnings
- Verify zero compiler warnings on GCC/Clang/MSVC
- Every public method documented
- Do NOT commit — leave for manual review

---

## Acceptance Criteria
- [ ] Engine plays legal chess via UCI protocol
- [ ] All unit tests pass (TT, move order, eval, search, UCI)
- [ ] Mate-in-N tests pass reliably
- [ ] No global mutable state outside of `SearchEngine` / `UCI` classes
- [ ] Thread-safe TT (no crashes under concurrent access)
- [ ] Zero compiler warnings, clang-format clean
- [ ] ASan clean (no memory leaks, no UB)
