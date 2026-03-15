# Task 04 — Search Improvements & HCE Tuning Pipeline

## Goal
Add missing search techniques that account for ~400–600 Elo, implement proper
time management, set up the PBIL/Texel tuning pipeline for the HCE, and establish
SPRT regression testing to validate every change.

---

## Steps

### 4.1 — Search Enhancements (Priority Order)

Each enhancement should be implemented, tested, then validated via fixed-depth
bench (nodes searched should change but not crash). Implement in this order
(roughly by Elo gain potential):

#### 4.1.1 — Reverse Futility Pruning (Static Null Move Pruning)
- At shallow depths (≤6), if `static_eval - margin >= beta`, return `static_eval`
- Margin increases with depth: `margin = depth * 80`
- Skip if in check, PV node, or zugzwang-prone position
- **Expected gain**: +30–50 Elo

#### 4.1.2 — Internal Iterative Reductions (IIR)
- If no hash move available at PV node with `depth >= 4`, reduce depth by 1
- Cheaper than IID (internal iterative deepening) and nearly as effective
- **Expected gain**: +10–20 Elo

#### 4.1.3 — Countermove History
- 2D array indexed by `[previous_move.to][previous_move.piece]` → counter move
- Use as additional move ordering signal (after killers, before regular quiets)
- **Expected gain**: +20–40 Elo

#### 4.1.4 — Capture History
- Track history statistics for captures separately from quiets
- Index by `[piece][to_square][captured_piece]`
- Use for ordering captures (supplement SEE)
- **Expected gain**: +10–20 Elo

#### 4.1.5 — Singular Extensions
- At non-root nodes with hash move and sufficient depth:
  - Search all other moves with reduced window `(ttvalue - depth * 2, ttvalue - depth * 2 + 1)`
  - If no other move beats this bound, the hash move is "singular" — extend by 1
- **Expected gain**: +30–60 Elo

#### 4.1.6 — Probcut
- At moderate depths, if a shallow search with raised beta finds a cutoff,
  prune the full-depth search
- `beta_cut = beta + 200; if shallow_search >= beta_cut: return beta_cut`
- **Expected gain**: +10–20 Elo

#### 4.1.7 — Multi-Cut
- If many moves at a cut-node exceed beta in a shallow search, prune
- Less commonly used in modern engines, but can help at lower depths
- **Expected gain**: +5–15 Elo

#### 4.1.8 — History-Based Pruning
- Prune quiet moves with very negative history scores at shallow depths
- `if (!in_check && depth <= 4 && history_score < -4000 * depth) continue;`
- **Expected gain**: +10–20 Elo

### 4.2 — Improved LMR Formula
- Current formula uses `log(depth) * log(move_count)`
- Modern engines also adjust by:
  - History score (reduce less for moves with good history)
  - PV node (reduce less)
  - Improving (reduce less if static eval is improving)
  - Cut node (reduce more)
  - Tactical moves (reduce less for checks, threats)
- Retune the LMR table after all other changes

### 4.3 — Time Management Overhaul
Replace the simple `remainder / moves_to_go` with:
- **Stability-based allocation**: If best move changes between iterations, allocate more time
- **Score-based allocation**: If score is dropping, allocate more time
- **Move difficulty**: Allocate less time for forced moves (only 1 legal move)
- **Ponder time recovery**: When ponderhit occurs, transfer saved time
- Use `std::chrono::steady_clock` throughout
- Implement `check_time()` called every N nodes instead of separate timer thread
  (simpler and more portable)

### 4.4 — Contempt Factor
- Add configurable contempt (default: 10 centipawns)
- Draws score as `contempt * (1 - phase/24)` instead of 0
- Helps engine avoid draws when stronger, accept draws when weaker
- UCI option: `setoption name Contempt value 10`

### 4.5 — Texel Tuning Pipeline
Build tooling to automatically tune HCE parameters:

#### 4.5.1 — Position Extraction
- Parse PGN databases → extract quiet positions (not in check, no captures in last 2 plies)
- Store as list of (FEN, game_result) pairs where result ∈ {1.0, 0.5, 0.0}
- Target: 1–5 million positions from high-quality games

#### 4.5.2 — Texel Tuning Core
- Objective function: minimize `Σ (result - sigmoid(eval))²` over all positions
- `sigmoid(eval) = 1 / (1 + 10^(-eval / 400))`
- Optimization: gradient descent or SPSA (Simultaneous Perturbation Stochastic Approximation)
- Parameters to tune: all values in `parameters` struct (~50–100 parameters)
- Framework: standalone binary `havoc_tune` that links `havoc_core`

#### 4.5.3 — PBIL Integration
- Port existing `tuning/` PBIL code to work with new parameter structure
- PBIL operates on bit representations of parameters
- Alternative to Texel — can tune non-differentiable parameters
- Keep both options available

### 4.6 — Regression Testing with SPRT
Set up automated strength testing:

#### 4.6.1 — Bench Command
- `bench` UCI command: run fixed-depth search on 30–50 standard positions
- Output total nodes searched as a "bench signature"
- If signature changes, search behavior changed (intentional or bug)
- Store expected signature in `tests/bench.h` or similar

#### 4.6.2 — SPRT Testing Script
- Use cutechess-cli for engine-vs-engine matches
- SPRT (Sequential Probability Ratio Test) to determine if a change is:
  - Positive (Elo gain with 95% confidence)
  - Negative (Elo loss with 95% confidence)
  - Neutral (no significant change)
- Script: `scripts/sprt.sh <base_binary> <test_binary> <num_games>`
- Parameters: Elo0=0, Elo1=5, alpha=0.05, beta=0.05
- Time control: 10+0.1 (10 seconds + 0.1 second increment)

#### 4.6.3 — CI Integration
- Add SPRT job to GitHub Actions (optional, can be slow)
- At minimum: bench signature check in CI
- Run `havoc bench` and compare to expected node count

### 4.7 — Verify All Improvements
For each search enhancement:
1. Implement the change
2. Run bench → signature should change
3. Run unit tests → all pass
4. Run short SPRT test (100–500 games) against previous version
5. If positive or neutral: keep. If negative: revert.

---

## Acceptance Criteria
- [ ] All search enhancements implemented and individually tested
- [ ] Time management handles sudden death, increment, and fixed-time correctly
- [ ] Texel tuning pipeline produces optimized parameters
- [ ] Bench command works and signature is tracked
- [ ] SPRT script runs successfully
- [ ] Engine strength measurably improved (target: +200–400 Elo over Task 03)
- [ ] Zero regressions in unit tests
- [ ] No new compiler warnings
