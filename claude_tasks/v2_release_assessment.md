# haVoc v2.0.0 — Release Assessment

## Summary

haVoc v2.0.0 is a complete C++20 rewrite of the original haVoc chess engine (v1.0.0, circa 2010).
The engine is a fully functional, UCI-compliant, multi-threaded chess engine with a hand-crafted
evaluation function, modern search techniques, and a tuning infrastructure.

## Metrics

| Metric | v1.0.0 (legacy) | v2.0.0 |
|--------|-----------------|--------|
| Language standard | C++11/14 mix | C++20 |
| Lines of code | ~9,000 | ~9,650 |
| Source files | 40 | 49 |
| Unit tests | 0 | 46 |
| Perft tests | 0 | 16 (5 standard positions) |
| Compiler warnings | Many | 0 |
| Global mutable state | Everywhere | Encapsulated in SearchEngine |
| Elo (self-play estimate) | ~2100 | ~2270 (+170) |
| Perft NPS | ~10.5 Mnps | ~12.8 Mnps (+17%) |
| Bench NPS | N/A | 623K nps |
| CI pipeline | None | GitHub Actions (Linux/macOS/Windows) |

## Completed Work

### Task 01 — Project Scaffold
- Modern CMake 3.20 with library/executable/test targets
- Google Test via FetchContent
- clang-format + clang-tidy configuration
- GitHub Actions CI (GCC, Clang, AppleClang, MSVC)
- ASan/UBSan support via cmake option

### Task 02 — Core Engine Port
- C++20 types: `using` aliases, `constexpr` arrays, `std::string_view`
- Bitboard utilities using `<bit>` header (`std::popcount`, `std::countr_zero`)
- Magic bitboard move generation
- Zobrist hashing with pre-computed random numbers
- Position representation with `std::vector<info>` history
- 16 perft tests across 5 standard positions — all passing

### Task 03 — Search, Eval, UCI
- `IEvaluator` interface for HCE/NNUE/Transformer extensibility
- `HCEEvaluator` with material, PST, mobility, king safety, threats, passed pawns
- `SearchEngine` class encapsulating all state (no globals)
- Transposition table with bug fixes (age() operator precedence)
- Pawn/material table memory leak fixes
- Async search (go infinite/stop works correctly)
- Per-thread evaluators with own pawn/material caches

### Task 04 — Search Improvements
- Reverse futility pruning
- Internal iterative reductions
- Singular extensions with multi-cut
- Countermove history
- History-based pruning
- Improved LMR with history-aware reductions
- Bench command for regression testing
- +104 Elo vs v1.0.0 in 100-game SPRT match

### Task 04b — Eval Overhaul
- KRK, KQK, KBNK endgame evaluators
- Endgame scaling (opposite-color bishops, pawnless positions)
- Quadratic king safety formula
- Parameter serialization (load/save)
- Tablebase and opening book interfaces (stubs)
- Category-level eval scale factors

### Task 04c — Tuning Pipeline
- `havoc_datagen`: parallel self-play training data generator with checkpointing
- `havoc_texel`: batch gradient descent tuner with momentum, LR decay, clamping
- `havoc_pgn2epd`: PGN-to-EPD converter for external game databases
- `bake_params.py`: bake tuned parameters into source code
- `tune.sh`: end-to-end pipeline script
- Staged tuning (category → shape → fine)

## Known Limitations

- HCE parameter tuning showed marginal gains — the eval architecture itself is the ceiling
- Mobility tables have implicit monotonicity constraints the tuner can't enforce
- No tablebase or opening book implementations (interfaces only)
- No NNUE or neural network evaluation

## Architecture

```
┌─────────────────────────────────────────────┐
│              UCI Protocol                    │
├─────────────────────────────────────────────┤
│           SearchEngine                       │
│  Alpha-Beta + LMR + Singular + NMP          │
│  Quiescence + SEE + Delta pruning           │
├─────────────────────────────────────────────┤
│  IEvaluator ──► HCEEvaluator                │
│                 (future: TransformerEval)    │
├─────────────────────────────────────────────┤
│  Position │ Movegen │ Bitboards │ TT        │
└─────────────────────────────────────────────┘
```

## Next Phase: Transformer Evaluation (v3.0)

The v3.0 development will replace the HCE with a self-attention transformer
trained via self-play, combining alpha-beta search with learned evaluation.
See task_07 through task_10 for the detailed plan.
