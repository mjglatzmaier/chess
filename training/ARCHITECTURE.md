# Chess Transformer — Architecture Decisions & Rationale

## Influences

This implementation draws from several sources but follows none exactly:

- **AlphaZero** (Silver et al., 2018) — the dual value/policy head design and self-play training loop
- **Chessformer** (Shao et al., 2024, arXiv:2409.12272) — demonstrated transformers can match CNNs for chess with less compute
- **Vision Transformer (ViT)** (Dosovitskiy et al., 2020) — treating a 2D grid as a sequence of patch tokens
- **GPT/BERT architecture patterns** — pre-norm, GELU activation, learned positional encoding
- **Leela Chess Zero (Lc0)** — practical engineering of neural network chess engines

No existing chess transformer implementation was directly copied. The design reflects
tradeoffs specific to our use case: alpha-beta search integration (not MCTS), CPU
inference requirement, and training from supervised game data (not pure self-play).

---

## Decision 1: Per-Square Tokens (not patch-based, not flat)

### Options Considered
1. **Flat input**: Concatenate all 768 features (12 planes × 64 squares) into one vector
2. **Patch-based**: Group squares into 2×2 or 4×4 patches (like ViT)
3. **Per-square tokens**: Each of 64 squares is a token with 25 features ← chosen

### Decision
Per-square tokens. Each square becomes a token in a 64-token sequence.

### Rationale
- Chess pieces interact at the **individual square level** — a knight on f3 attacks
  specific squares, not a 4×4 region. Per-square granularity preserves this.
- Self-attention between 64 tokens is O(64² = 4096), which is trivially small
  compared to NLP sequences of 512-4096 tokens. No efficiency concern.
- Flat input loses spatial structure — the model can't easily learn "these features
  are on the same square." Per-square grouping provides this for free.
- Patch-based (ViT-style) would merge squares that have different semantic roles.
  A 2×2 patch containing e1 (king) and d1 (queen) loses the distinction between
  pieces that have very different movement and strategic roles.

### Tradeoff
Per-square means 64 tokens even for sparse boards (endgames with 4 pieces).
A variable-length "piece-list" encoding could be more efficient but would
require dynamic sequence lengths and special handling of empty squares.

---

## Decision 2: 25-Dimensional Feature Vector Per Square

### Encoding
```
[0:12]  — Piece presence (one-hot): P,N,B,R,Q,K,p,n,b,r,q,k
[12]    — Side to move (1=white, 0=black), broadcast to all squares
[13:16] — Castling rights (K,Q,k,q), broadcast to all squares
[17:24] — En passant file (one-hot, 8 files), broadcast to all squares
```

### Why not 12 planes only?
The 12 piece planes describe WHAT is on the board. But the position also includes
WHO moves next, castling rights, and en passant — these change the eval significantly.
A position with white to move vs black to move can differ by 20-50cp.

### Why broadcast global features?
Side-to-move, castling, and en passant are global (not per-square) properties. Two
options: (a) add a special [CLS]-like token with global features, or (b) broadcast
them to every square's feature vector.

Broadcasting was chosen because it lets every square's attention computation directly
incorporate global state without routing through a special token. This is simpler and
empirically works well in ViT-style architectures where global conditioning is needed.

### Why not HalfKP features (like NNUE)?
HalfKP encodes features relative to the king square (piece × square × king_square).
This creates 2 × 64 × 640 = 81,920 input features — very sparse, designed for
efficiently updatable feedforward nets. Transformers don't benefit from this encoding
because self-attention already captures piece-king relationships. The simpler 25-dim
per-square encoding is sufficient and more interpretable.

---

## Decision 3: Mean Pooling (not CLS token)

### Options Considered
1. **CLS token**: Prepend a learnable [CLS] token, use its final representation
2. **Mean pooling**: Average all 64 token representations ← chosen
3. **Max pooling**: Take element-wise maximum across tokens
4. **Attention pooling**: Learn a weighted combination

### Decision
Mean pooling over all 64 square tokens.

### Rationale
- CLS tokens work well in NLP where the first token aggregates sequential information.
  In chess, there's no sequential ordering — a1 is not "first" in any meaningful way.
  CLS would need to learn to attend to all squares equally, which mean pooling does
  explicitly.
- Mean pooling treats every square as equally important for the aggregate
  representation. This is correct — the value of a position depends on ALL pieces
  and ALL squares, not just a subset.
- In experiments with ViT for image classification, mean pooling slightly outperforms
  CLS tokens when the spatial structure is regular (which an 8×8 grid is).
- Simpler to implement, one fewer parameter, no special token handling.

### Tradeoff
Mean pooling can dilute signal from rare but important features (e.g., a single
pawn about to promote). Attention pooling could learn to upweight these, but adds
complexity and parameters. For our initial model, simplicity wins.

---

## Decision 4: Policy Head as 64×64 From-To Matrix

### Options Considered
1. **Move embedding**: Learn an embedding per legal move, score each ← complex
2. **From-To matrix**: Each source square predicts 64 target squares ← chosen
3. **Action-value (Q-values)**: Predict value for each possible next state ← expensive

### Decision
Policy head: `Linear(embed_dim, 64)` applied per-token → reshape to [64, 64] = 4096 logits.

### Rationale
- Chess moves are naturally (from, to) pairs. A 64×64 matrix captures all possible
  from-to combinations with a single linear layer per token.
- The from-square's token representation already encodes "what piece is here and what's
  around it" from self-attention. Projecting to 64 target logits asks "where should
  this piece go?" — a natural question.
- 4096 outputs cover all moves except underpromotions (rare, <0.1% of moves).
  Queen promotion is assumed as default — a practical simplification.
- This is the same encoding used by AlphaZero and Lc0.

### Tradeoff
- Can't distinguish promotion types (queen vs rook vs bishop vs knight promotion).
  Could be extended to 64 × 73 (64 targets + 9 underpromotion options) if needed.
- Wastes capacity on impossible moves (e.g., pawn from-square predicting queen-like
  target squares). But the cross-entropy loss naturally pushes these to zero probability.

---

## Decision 5: Pre-Norm Transformer (not Post-Norm)

### Decision
`norm_first=True` in TransformerEncoderLayer — LayerNorm before attention and FFN,
not after.

### Rationale
Pre-norm transformers train more stably, especially for small-to-medium models.
Post-norm (the original "Attention Is All You Need" design) can have gradient
issues in deeper networks without careful learning rate warmup. Pre-norm eliminates
this and allows higher learning rates out of the box.

This is now standard practice — GPT-2+, LLaMA, and most modern transformers use
pre-norm.

---

## Decision 6: GELU Activation (not ReLU)

### Decision
GELU activation in the feed-forward network, not ReLU.

### Rationale
GELU (Gaussian Error Linear Unit) provides a smooth approximation to ReLU that
empirically trains better in transformers. It's used by BERT, GPT-2+, and nearly
all modern transformers. The smooth gradient near zero helps with optimization.

For the value head's hidden layer, GELU is also used. The final value output uses
tanh to bound the output to [-1, +1].

---

## Decision 7: Separate Value and Policy Heads (Shared Encoder)

### Options Considered
1. **Separate networks**: One transformer for value, another for policy
2. **Shared encoder, separate heads**: Single transformer, two output heads ← chosen
3. **Single head**: Predict value and policy from the same output

### Decision
Shared encoder backbone with separate value and policy heads.

### Rationale
- **Shared encoder** forces the transformer to learn representations useful for BOTH
  tasks. Understanding "who's winning" (value) and "what's the best move" (policy)
  requires similar features — piece activity, king safety, tactical threats. Sharing
  the encoder is more parameter-efficient and provides implicit regularization.
- **Separate heads** allow each task its own projection. The value head pools to a
  scalar; the policy head preserves per-square structure. These are fundamentally
  different output shapes requiring different final layers.
- This is the same design as AlphaZero. It's well-validated and the standard approach
  for dual-objective game-playing networks.

### Tradeoff
Value and policy objectives can sometimes conflict (a position might be "good" overall
but have a single critical move that's hard to find). The shared encoder must balance
both objectives. The `value_loss_weight` and `policy_loss_weight` hyperparameters
control this balance.

---

## Decision 8: Xavier Init with Small Gain + Zero-Init Value Head

### Decision
- All weights: Xavier uniform with gain=0.1
- Value head final layer: zero-initialized
- Biases: zero-initialized

### Rationale
Small initialization prevents the randomly initialized model from producing extreme
values at the start of training. A model that outputs +0.9 or -0.8 for random
positions creates large gradients that can destabilize early training.

Zero-initializing the value head output layer means the untrained model predicts 0
(draw) for every position — a reasonable starting point. The model then learns to
deviate from this as it sees data.

This follows the "fixup initialization" principle used in residual networks.

---

## Decision 9: Cosine Annealing LR Schedule

### Decision
Cosine annealing from `lr` to `lr * 0.01` over the total training steps.

### Rationale
Cosine annealing provides a smooth learning rate decay that works well without
hyperparameter tuning. It starts high (exploring broadly), gradually decreases
(refining), and ends very low (final convergence). It's simpler than step decay
(which requires choosing step points) and more robust than linear decay.

No warmup is used for the initial implementation. For larger models or larger
datasets, a linear warmup of 1000-5000 steps would improve stability.

---

## Future Considerations

1. **Rotary positional encoding (RoPE)**: Instead of learned positional embeddings,
   RoPE could provide better generalization to unseen square relationships. Used in
   LLaMA and most modern transformers.

2. **Flash Attention**: For larger models, Flash Attention 2 would reduce memory
   usage and increase speed. Not needed at our scale (64 tokens) but relevant if
   we add more input tokens.

3. **Mixture of Experts (MoE)**: Different positions (openings vs endgames) might
   benefit from specialized sub-networks. MoE could route positions to specialized
   experts while sharing the attention backbone.

4. **Distillation from Stockfish**: Instead of training on game outcomes only,
   train on Stockfish's per-position evaluations as additional targets. This provides
   denser supervision than binary game outcomes.
