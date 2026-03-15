"""
Hyperparameter configurations for the chess transformer.

Three model sizes for experimentation:
- Tiny:   rapid iteration, CPU-feasible inference (~200K params)
- Small:  competitive play, CPU inference with SIMD (~1.5M params)
- Medium: research/analysis, may need GPU inference (~10M params)
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Transformer model hyperparameters."""
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 512       # feed-forward hidden dimension
    dropout: float = 0.1
    num_squares: int = 64
    num_input_features: int = 25  # 12 piece planes + 1 stm + 4 castling + 8 ep
    num_policy_moves: int = 4096  # 64 * 64 from-to pairs

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 10
    warmup_steps: int = 1000
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    save_every: int = 1       # save checkpoint every N epochs
    eval_every: int = 1       # evaluate every N epochs
    num_workers: int = 4      # data loader workers


# Pre-defined model sizes
TINY = ModelConfig(
    embed_dim=64,
    num_layers=2,
    num_heads=4,
    ff_dim=256,
    dropout=0.1,
)

SMALL = ModelConfig(
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    ff_dim=512,
    dropout=0.1,
)

MEDIUM = ModelConfig(
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    ff_dim=1024,
    dropout=0.1,
)

MODEL_CONFIGS = {
    "tiny": TINY,
    "small": SMALL,
    "medium": MEDIUM,
}
