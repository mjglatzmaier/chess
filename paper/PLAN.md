# Paper Plan: Latency-Constrained Transformer Evaluation in an Alpha-Beta Chess Engine

## Working Title
**Latency-Constrained Transformer Evaluation in an Alpha-Beta Chess Engine**

## Thesis (one sentence)
Compact transformer evaluation can be integrated into alpha-beta search on commodity CPUs if the model and inference path are explicitly designed for latency, and this yields measurable strength gains over a classical handcrafted evaluator under realistic time controls.

## Three Contributions
1. **Architecture**: Compact chess transformer evaluator designed for low-latency CPU inference inside alpha-beta
2. **Systems**: Custom inference path using quantization and SIMD-aware kernels for UCI deployment
3. **Empirical**: Benchmark study of Elo, node throughput, and tactical performance across evaluator types and model scales

## Research Questions
- RQ1: Can a compact transformer evaluator improve playing strength over HCE inside alpha-beta under fixed CPU time controls?
- RQ2: How does playing strength vary with model size and evaluator latency?
- RQ3: Does adding a policy head improve move ordering enough to offset extra inference cost?
- RQ4: How close can a compact transformer-eval alpha-beta engine get to an NNUE baseline under equal hardware constraints?

## The One Experiment That Matters
Under the same CPU and time control, does the transformer-eval alpha-beta engine beat the classical handcrafted evaluator by enough Elo to justify the inference cost?

## Claims to AVOID
- First transformer + alpha-beta chess engine
- State of the art
- Competitive with Stockfish
- Interpretable attention proves understanding

## Claims that ARE defensible
- Compact transformers can work inside alpha-beta on CPU
- Best strength/latency tradeoff in small-model regime
- Policy head can improve move ordering to offset NPS loss
- Systems engineering (quantization, SIMD) is necessary, not optional

## Section Outline
1. Introduction: Problem, gap, contributions
2. Background and Related Work (alpha-beta/NNUE, ChessBench, Chessformer, gap)
3. Engine Architecture (haVoc search stack)
4. Transformer Evaluator (encoding, architecture, targets, sizes)
5. CPU Inference Implementation (quantization, SIMD, memory layout, latency)
6. Experimental Setup (datasets, match protocol, hardware, metrics)
7. Results (strength vs latency curves, ablations)
8. Analysis (move ordering effects, failure modes, examples)
9. Limitations (slower than NNUE, not incrementally updatable)
10. Conclusion (viable only when treated as systems problem)

## Experiment Phases
- Phase A: Feasibility (tiny model in engine, measure NPS/latency/tactical)
- Phase B: Model-Size Sweep (tiny/small/medium Elo vs latency - HEART OF PAPER)
- Phase C: Training Target Ablations (outcome vs distillation vs policy)
- Phase D: Search-Role Ablations (value-only vs value+policy ordering)
- Phase E: Quantization/Backend (fp32 vs fp16 vs int8)

## Engineering Roadmap
1. Track 1: Match harness, Elo/SPRT, tactical benchmarks, profiling
2. Track 2: Minimal C++ inference (tiny model, leaf eval only)
3. Track 3: Throughput optimization (quantization, SIMD, memory layout)
4. Track 4: Search-aware features (policy move ordering, selective search)

## Key Metrics
Elo (SPRT), NPS, evals/sec, avg depth, inference latency, tactical suite, memory footprint
