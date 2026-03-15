# Task 05 — NNUE Evaluation

## Goal
Implement an NNUE (Efficiently Updatable Neural Network) evaluation function
as a drop-in replacement for the HCE, enabling 3200+ Elo strength.

---

## Steps

### 5.1 — NNUE Architecture Design

#### Network Architecture (HalfKAv2 recommended)
```
Input: HalfKAv2 features
  - 2 perspectives × (64 king squares × 640 piece features) = 81,920 inputs
  - Each perspective = king_square * (piece_type * 2 * 64) + piece_sq

Layer 1: 81,920 → 1024 (ClippedReLU activation, int16 weights)
Layer 2: 2048 → 8 (two perspectives concatenated, int8 weights)  
Layer 3: 8 → 1 (int8 weights, output = evaluation in centipawns)
```

Alternative: Start simpler with HalfKP:
```
Input: 2 × (64 × 640) = 81,920
Layer 1: 81,920 → 256 (ClippedReLU)
Layer 2: 512 → 32 (ClippedReLU)
Layer 3: 32 → 1
```

### 5.2 — NNUE Inference Engine (`include/havoc/eval/nnue.hpp`, `src/eval/nnue.cpp`)

#### 5.2.1 — Data Types
```cpp
namespace havoc::nnue {
    using Weight = int16_t;     // Layer 1 weights
    using Bias = int16_t;       // Layer 1 biases
    using L2Weight = int8_t;    // Layer 2+ weights
    using Accumulator = int16_t; // Accumulated values
}
```

#### 5.2.2 — Accumulator
The key NNUE innovation — incremental updates instead of full recomputation:
```cpp
struct NNUEAccumulator {
    alignas(64) int16_t values[2][L1_SIZE];  // [perspective][neuron]
    bool needs_refresh[2] = {true, true};
};
```
- On `do_move()`: update accumulator by adding/removing feature deltas
- On king move: full refresh (king square changes input mapping)
- On `undo_move()`: restore from stack (same as position history)

#### 5.2.3 — SIMD Acceleration
- Use `#include <immintrin.h>` for x86 (AVX2/AVX-512)
- Use `#include <arm_neon.h>` for ARM (Apple Silicon)
- Provide scalar fallback for portability
- Use compile-time detection: `#if defined(__AVX2__)`, `#if defined(__ARM_NEON)`
- Key operations to vectorize:
  - Dot product (int16 × int16 → int32 accumulate)
  - ClippedReLU: `max(0, min(x, 127))`
  - Feature add/subtract for accumulator updates

#### 5.2.4 — Network File Format
- Binary format compatible with Stockfish NNUE (for initial testing with SF nets)
- Or custom format: header + layer weights in row-major order
- File loaded at engine startup via `NNUEEvaluator::load(path)`
- UCI option: `setoption name EvalFile value nn.bin`

#### 5.2.5 — Evaluator Implementation
```cpp
class NNUEEvaluator : public IEvaluator {
public:
    bool load(const std::string& path);
    int evaluate(const Position& pos, int lazy_margin = -1) override;
    std::string name() const override { return "NNUE"; }

private:
    Network network_;
    void refresh_accumulator(const Position& pos, Color perspective);
    void update_accumulator(const Position& pos, Move m, Color perspective);
    int forward(const Position& pos);
};
```

### 5.3 — Accumulator Stack Integration
- Add `NNUEAccumulator` to position's move history stack
- On `do_move()`: copy current accumulator, apply delta
- On `undo_move()`: restore from stack (automatic via history)
- On king move: mark accumulator as needing refresh
- This is the performance-critical path — must be fast

### 5.4 — Training Pipeline (Separate Tool)

#### 5.4.1 — Training Data Generation
Two approaches:
1. **From Stockfish data**: Download existing training data from Stockfish repos
2. **Self-play**: Engine plays games against itself, recording positions and outcomes

Data format per sample:
```
FEN, evaluation (from search), game result (1/0.5/0)
```

#### 5.4.2 — Trainer
- Recommend using existing NNUE trainers:
  - `nnue-pytorch` (Stockfish's trainer)
  - Or write custom PyTorch trainer
- Loss function: `MSE(sigmoid(predicted_eval), lambda * result + (1-lambda) * sigmoid(target_eval))`
  where `lambda ≈ 0.75`
- Output: binary weight file

#### 5.4.3 — Quantization
- Train in float32, quantize to int16/int8 for inference
- Verify quantized net produces similar eval to float net

### 5.5 — Hybrid Eval Mode
- UCI option: `setoption name EvalMode value [hce|nnue|hybrid]`
- Hybrid mode: use NNUE for most positions, fall back to HCE for edge cases
  (e.g., positions the NNUE wasn't trained on)
- Or: NNUE eval + HCE endgame corrections

### 5.6 — Search Adjustments for NNUE
NNUE eval has different characteristics than HCE:
- Eval is more accurate → can trust it more for pruning
- Adjust futility margins, razoring margins, null move conditions
- NNUE eval is more expensive than HCE → lazy eval more important
- Contempt factor may need retuning

### 5.7 — Testing & Validation

**Unit tests** (`tests/test_nnue.cpp`):
- Load network file → no crash
- Evaluate startpos → returns reasonable value (~0)
- Accumulator update matches full refresh (consistency check)
- SIMD path matches scalar path (verify vectorization correctness)
- Symmetric positions → symmetric evals

**Strength tests**:
- SPRT test NNUE vs HCE: expect significant Elo gain (+200–500)
- Run on standard test suites (STS, WAC, ECM)
- Profile: NNUE eval should be <2x slower than HCE per call, but
  total NPS may drop 30–50% (offset by better eval = fewer nodes needed)

---

## Acceptance Criteria
- [ ] NNUE inference engine loads and evaluates positions correctly
- [ ] Accumulator incremental updates match full refresh
- [ ] SIMD acceleration works on x86 (AVX2) and ARM (NEON)
- [ ] Scalar fallback works on all platforms
- [ ] UCI option to switch between HCE and NNUE
- [ ] SPRT confirms Elo gain over HCE
- [ ] Engine reaches 3000+ Elo in testing (target 3200+)
- [ ] No memory leaks, no UB (ASan clean)
