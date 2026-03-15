# haVoc Transformer Training Pipeline

A self-play training system for learning chess evaluation from scratch using a
transformer neural network. This is the training side of haVoc v3.0 — the trained
model plugs into the C++ engine as an `IEvaluator` replacement for the hand-crafted
evaluation (HCE).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Pipeline                          │
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  Self-Play   │    │   PyTorch     │    │    Export       │  │
│  │  Data Gen    │───▶│   Training    │───▶│   ONNX/Binary  │  │
│  │  (C++ engine)│    │   (GPU)       │    │                │  │
│  └─────────────┘    └──────────────┘    └────────┬───────┘  │
│                                                    │          │
└────────────────────────────────────────────────────┼──────────┘
                                                     │
┌────────────────────────────────────────────────────▼──────────┐
│                    C++ Engine (haVoc)                          │
│                                                               │
│  SearchEngine + IEvaluator ◀── TransformerEvaluator           │
│                                (loads exported model)         │
└───────────────────────────────────────────────────────────────┘
```

## Why a Transformer?

Traditional chess neural networks use either:
- **Feedforward nets (NNUE)**: Fast but can only learn local piece interactions
- **CNNs (Lc0/AlphaZero)**: Good at spatial patterns but limited receptive field

Transformers with self-attention can learn **global board relationships** — a bishop
on a1 directly attends to a pawn on h8 in a single layer. This is ideal for chess
concepts like pins, skewers, discovered attacks, and long-range piece coordination.

**This combination (alpha-beta search + transformer eval) is novel** — most engines
use either NNUE+alpha-beta or CNN+MCTS, but not transformer+alpha-beta.

## Network Architecture

```
Input Encoding (per square):
  12 piece planes (P,N,B,R,Q,K × white,black)  →  one-hot
  + side-to-move (1 bit, broadcast)
  + castling rights (4 bits, broadcast)
  + en passant file (8 bits, broadcast)
  Total: 25 features per square → projected to embed_dim

Transformer Encoder:
  [64 square tokens] → Linear(25, embed_dim) + positional encoding
  → N × TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
  → pooled output (mean of all tokens)

Value Head:
  pooled → Linear(embed_dim, 1) → tanh → [-1, +1]
  Represents win probability: +1 = white wins, -1 = black wins

Policy Head:
  per-token → Linear(embed_dim, 64) → [64, 64] move logits
  Represents from-square × to-square move probabilities
```

## Training Loop (AlphaZero-style)

```
Round 0: Generate training data using HCE engine
         Train transformer to mimic HCE (distillation)

Round N: Generate data using transformer-powered engine
         Train on game outcomes (self-play improvement)
         Each round produces a stronger evaluator
```

## Quick Start

```bash
# Install dependencies
pip install -r training/requirements.txt

# Round 0: Generate initial training data from HCE engine
cd /path/to/chess
./build/havoc_datagen --games 5000 --depth 8 --threads 8 --output training/data/round_0.epd

# Prepare training data
python training/prepare_data.py training/data/round_0.epd --output training/data/round_0/

# Train the model
python training/train.py --data training/data/round_0/ --output training/models/round_0.pt

# Export to ONNX
python training/export_onnx.py training/models/round_0.pt --output training/models/round_0.onnx

# Validate
python training/validate.py training/models/round_0.pt
```

## File Structure

```
training/
├── README.md               # This file
├── requirements.txt         # Python dependencies
├── model.py                 # Transformer architecture
├── encoding.py              # Board ↔ tensor conversion
├── prepare_data.py          # EPD → training tensors
├── train.py                 # Training loop
├── export_onnx.py           # Export to ONNX format
├── validate.py              # Model validation & analysis
├── config.py                # Hyperparameter configs
├── data/                    # Training data (generated, not committed)
│   └── round_0/
└── models/                  # Trained models (not committed)
    └── round_0.pt
```
