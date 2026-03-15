"""
Chess Transformer — neural network architecture for board evaluation.

Architecture:
    Input:    [batch, 64, 25] — 64 square tokens, 25 features each
    Embed:    Linear(25, embed_dim) + learned positional encoding
    Encoder:  N × TransformerEncoderLayer (self-attention + FFN)
    Value:    mean-pool → Linear(embed_dim, 1) → tanh
    Policy:   per-token → Linear(embed_dim, 64) → reshape to [64, 64]

The value head predicts game outcome: +1 white wins, -1 black wins.
The policy head predicts move probabilities as a 64×64 from-to matrix.

Design rationale:
    - Self-attention lets every square attend to every other square in one layer,
      enabling the model to learn long-range relationships (pins, skewers, batteries)
      that CNNs need many layers to capture.
    - Mean pooling (not CLS token) gives a position-aware aggregate that considers
      all squares equally — important since chess has no "most important" square.
    - Separate value/policy heads share the encoder backbone, which is efficient
      and forces the encoder to learn features useful for both evaluation and
      move selection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, SMALL


class ChessTransformer(nn.Module):
    """Transformer model for chess evaluation and move prediction."""

    def __init__(self, config: ModelConfig = SMALL):
        super().__init__()
        self.config = config

        # Square feature embedding: project 25-dim input to embed_dim
        self.input_projection = nn.Linear(config.num_input_features, config.embed_dim)

        # Learned positional encoding for 64 squares
        # Unlike NLP where position is sequential, chess squares have 2D spatial
        # meaning. Learned embeddings let the model discover this structure.
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.num_squares, config.embed_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Layer norm before heads
        self.output_norm = nn.LayerNorm(config.embed_dim)

        # Value head: scalar evaluation
        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 1),
            nn.Tanh(),  # output in [-1, +1]
        )

        # Policy head: per-square logits for target square
        # Each source square produces 64 logits (one per target square)
        self.policy_head = nn.Linear(config.embed_dim, 64)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Zero-init the value head output layer for stable start
        nn.init.zeros_(self.value_head[-2].weight)
        nn.init.zeros_(self.value_head[-2].bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode board positions through the transformer.

        Args:
            x: [batch, 64, 25] input features

        Returns:
            [batch, 64, embed_dim] encoded representations
        """
        # Project input features to embedding dimension
        x = self.input_projection(x)  # [batch, 64, embed_dim]

        # Add positional encoding
        x = x + self.pos_encoding

        # Pass through transformer encoder
        x = self.encoder(x)  # [batch, 64, embed_dim]

        # Final layer norm
        x = self.output_norm(x)

        return x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → value + policy.

        Args:
            x: [batch, 64, 25] input features

        Returns:
            value:  [batch, 1] — game outcome prediction in [-1, +1]
            policy: [batch, 4096] — move logits (from_sq * 64 + to_sq)
        """
        encoded = self.encode(x)  # [batch, 64, embed_dim]

        # Value head: mean pool over all squares, then predict scalar
        pooled = encoded.mean(dim=1)  # [batch, embed_dim]
        value = self.value_head(pooled)  # [batch, 1]

        # Policy head: each square produces 64 target-square logits
        policy_per_square = self.policy_head(encoded)  # [batch, 64, 64]
        policy = policy_per_square.reshape(-1, 4096)  # [batch, 4096]

        return value, policy

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Value-only forward pass (faster, no policy computation).

        Args:
            x: [batch, 64, 25] input features

        Returns:
            [batch, 1] — game outcome prediction in [-1, +1]
        """
        encoded = self.encode(x)
        pooled = encoded.mean(dim=1)
        return self.value_head(pooled)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(size: str = "small") -> ChessTransformer:
    """Create a model by size name."""
    from config import MODEL_CONFIGS
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Choose from: {list(MODEL_CONFIGS)}")
    config = MODEL_CONFIGS[size]
    model = ChessTransformer(config)
    print(f"Created {size} model: {model.count_parameters():,} parameters")
    print(f"  embed_dim={config.embed_dim}, layers={config.num_layers}, "
          f"heads={config.num_heads}, ff_dim={config.ff_dim}")
    return model


if __name__ == "__main__":
    # Quick test: create each model size and run a forward pass
    for size in ["tiny", "small", "medium"]:
        model = create_model(size)
        dummy_input = torch.randn(2, 64, 25)
        value, policy = model(dummy_input)
        print(f"  value shape: {value.shape}, policy shape: {policy.shape}")
        print(f"  value range: [{value.min():.3f}, {value.max():.3f}]")
        print()
