# Task 06 — Advanced Features & Extensibility

## Goal
Add tablebases, Chess960 support, and extensibility hooks for future work
(GPU inference, MCTS, distributed search). Polish for release quality.

---

## Steps

### 6.1 — Syzygy Tablebase Probing

#### 6.1.1 — Integration
- Use Fathom library (https://github.com/jdart1/Fathom) as a git submodule
  or vendored dependency
- Add to CMake: `add_subdirectory(extern/fathom)`
- UCI option: `setoption name SyzygyPath value /path/to/tablebases`
- Probe at root for DTZ (Distance To Zeroing) to determine best move
- Probe in search for WDL (Win/Draw/Loss) to adjust eval

#### 6.1.2 — Search Integration
- At root: if position has ≤ N pieces (N = tablebase size), use TB result
- In search: if probe returns WDL, adjust score:
  - Win → set score to `mate_in(max_ply)` equivalent
  - Loss → set score to `mated_in(max_ply)` equivalent  
  - Draw → set score to `draw`
- Only probe when piece count ≤ TB piece limit

#### 6.1.3 — Testing
- Test known endgame positions against tablebase results
- KRK: verify engine finds mate
- KQK: verify engine finds mate
- KBNK: verify engine finds mate (tricky endgame)

### 6.2 — Chess960 / Fischer Random Support
- Castling rules differ: king and rook can start on any squares
- Modify castling logic in `Position::do_move()` / `undo_move()`
- Modify movegen castling generation
- UCI option: `setoption name UCI_Chess960 value true`
- FEN parsing: handle Shredder-FEN notation for castling (e.g., `KQkq` → `HAha`)

### 6.3 — Extensibility Hooks

#### 6.3.1 — GPU Inference Interface
```cpp
namespace havoc {
    class IGPUBackend {
    public:
        virtual ~IGPUBackend() = default;
        virtual bool init(const std::string& model_path) = 0;
        virtual std::vector<float> evaluate_batch(
            const std::vector<InputFeatures>& batch) = 0;
        virtual void shutdown() = 0;
    };
}
```
- Design for batch evaluation (GPU is efficient with batches)
- Support CUDA, Metal, and Vulkan Compute via compile-time selection
- Placeholder implementation that falls back to CPU

#### 6.3.2 — MCTS Interface
```cpp
namespace havoc {
    class ISearchStrategy {
    public:
        virtual ~ISearchStrategy() = default;
        virtual Move search(Position& pos, const SearchLimits& limits) = 0;
        virtual void stop() = 0;
    };

    class AlphaBetaSearch : public ISearchStrategy { /* current search */ };
    class MCTSSearch : public ISearchStrategy { /* future MCTS */ };
}
```
- Abstract search strategy allows swapping alpha-beta for MCTS
- MCTS would use neural net eval for node expansion (AlphaZero-style)
- Leave as interface + stub for now

#### 6.3.3 — Plugin Architecture (Optional)
- Dynamic loading of evaluation backends via `dlopen` / `LoadLibrary`
- Allows third-party NNUE nets or custom evaluators without recompilation
- Lower priority — implement only if time permits

### 6.4 — Multi-PV Improvements
- Current multi-PV is basic — improve to properly track N best lines
- Each PV line gets its own score and depth
- UCI output: `multipv 1 ... pv e2e4 ...`, `multipv 2 ... pv d2d4 ...`

### 6.5 — Analysis Mode Features
- `go infinite` works correctly (search until `stop`)
- Proper `ponderhit` handling
- `go mate N` — search for mate in N moves
- `go nodes N` — search exactly N nodes
- `go movetime N` — search for exactly N milliseconds

### 6.6 — Documentation & README
- Comprehensive README.md for the remake:
  - Build instructions (Linux/macOS/Windows)
  - UCI options reference
  - Architecture overview
  - Contributing guidelines
  - Acknowledgments
- Doxygen configuration for API docs
- Architecture diagram (ASCII or Mermaid)

### 6.7 — Release Packaging
- CMake install target: `cmake --install build --prefix /usr/local`
- Create release builds for Linux (x86_64), macOS (arm64 + x86_64), Windows (x64)
- GitHub Release workflow: build on tag push, upload binaries
- Version string from CMake project version

### 6.8 — Final Quality Pass
- Full clang-tidy analysis
- Valgrind / ASan / UBSan / TSan clean
- Profile with `perf` / Instruments — identify and fix hotspots
- Memory usage audit (TT should be configurable, no leaks)
- Verify cross-platform: builds and tests pass on Linux, macOS, Windows

---

## Acceptance Criteria
- [ ] Syzygy tablebase probing works for 5/6-man positions
- [ ] Chess960 mode passes validation suite
- [ ] GPU and MCTS interfaces defined (implementation optional)
- [ ] Multi-PV works correctly for N ≤ 4
- [ ] Analysis mode features all work
- [ ] README comprehensive and accurate
- [ ] Release binaries build for 3 platforms
- [ ] All sanitizers pass clean
- [ ] Engine reaches target Elo (3200+ with NNUE, 2700+ with tuned HCE)
