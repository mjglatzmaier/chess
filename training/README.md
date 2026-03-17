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

# Prepare training data (PGN or EPD → .npz chunks)
python training/prepare_data.py training/data/round_0.epd --output training/data/round_0/

# Generate synthetic positions for material imbalance coverage (see below)
python training/synthetic_gen.py --stockfish stockfish --output training/data/synthetic/

# Mix data sources with configurable ratios
python training/data_mixer.py --ccrl training/data/round_0/ 0.7 \
    --synthetic training/data/synthetic/ 0.3 --output training/data/mixed/

# Train the model
python training/train.py --data training/data/mixed/ --output training/models/round_0.pt

# Export to ONNX
python training/export_onnx.py training/models/round_0.pt --output training/models/round_0.onnx

# Validate
python training/validate.py training/models/round_0.pt

# Inspect data quality
python training/data_stats.py training/data/mixed/ --detailed
```

## Synthetic Position Generator (`synthetic_gen.py`)

Generates random legal positions with controlled material configurations and
evaluates each with Stockfish. This fills gaps in the CCRL data — material
imbalances, rare endgames, and edge cases that engine games rarely produce.

### How it works

1. **Config generation** — Systematically builds material configurations from
   3 dimensions: base imbalances (Q vs RR, R vs BN, etc.), pawn overlays
   (0–2 extra pawns per side), and phase backgrounds (extra pieces on both
   sides to simulate opening/middlegame/endgame density).

2. **Position generation** — For each config, places pieces on random legal
   squares (kings not adjacent, pawns not on ranks 1/8, no stalemate/checkmate).

3. **Stockfish evaluation** — Persistent worker processes each run a long-lived
   Stockfish instance. Positions are distributed via work queues for parallel
   evaluation. Hash tables are preserved across positions for speed.

4. **Checkpointing** — After each config completes, results are saved to disk
   and a checkpoint file is written. Generation can be interrupted (Ctrl+C) and
   resumed by re-running the same command.

### Usage

```bash
# List all material configs without generating (no Stockfish needed)
python training/synthetic_gen.py --list-configs

# Full generation (recommended for 12-thread machine)
python training/synthetic_gen.py \
    --stockfish stockfish \
    --output training/data/synthetic/ \
    --depth 20 \
    --threads 12 \
    --hash 128

# Quick test run — 3 specific configs, 10 positions each, shallow depth
python training/synthetic_gen.py \
    --stockfish stockfish \
    --output training/data/synthetic_test/ \
    --configs KQvK KRvK KBvK \
    --num 10 \
    --depth 6

# Generate only queen imbalance positions
python training/synthetic_gen.py \
    --stockfish stockfish \
    --output training/data/synthetic/ \
    --category queen_vs_pieces

# Endgame-only (skip phase backgrounds — ~1/4 the configs, much faster)
python training/synthetic_gen.py \
    --stockfish stockfish \
    --output training/data/synthetic/ \
    --no-phases

# Resume an interrupted run (just re-run the same command)
python training/synthetic_gen.py \
    --stockfish stockfish \
    --output training/data/synthetic/ \
    --depth 20 --threads 12
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stockfish` | *(required)* | Path to Stockfish binary |
| `--output` | `data/synthetic/` | Output directory for .npz chunks |
| `--depth` | `20` | Stockfish search depth (higher = slower but more accurate) |
| `--threads` | `4` | Number of parallel Stockfish worker processes |
| `--hash` | `64` | Stockfish hash table size in MB per worker instance |
| `--configs` | all | Specific named configs (e.g., `KQvK KRvK`) |
| `--category` | all | Filter by category prefix (e.g., `queen_up`, `rook_vs_minor`) |
| `--num` | `500` | Positions per config |
| `--no-phases` | off | Skip phase backgrounds (endgame-only, ~4× fewer configs) |
| `--max-pawn-overlay` | `2` | Max extra pawns per side in overlay combinations |
| `--chunk-size` | `100000` | Positions per .npz output chunk |
| `--list-configs` | off | Print all configs and exit |

### Performance tips

- **Threads**: Use your CPU core count. Each worker runs its own Stockfish
  instance with 1 thread. 12 workers × depth 20 ≈ ~50 positions/sec.
- **Hash**: Increase `--hash` if you have spare RAM. Workers evaluate similar
  positions sequentially, so transposition hits are common. 128–256MB per
  worker is a good target. Total RAM ≈ `threads × hash`.
- **Depth**: Depth 20 is a good balance. Depth 12 is ~5× faster but less
  accurate. Depth 6 is useful only for smoke tests.
- **Resume**: The generator checkpoints after every config. If interrupted,
  re-run the same command — it skips completed configs automatically.

### Material config categories

| Category | Description | Example configs |
|----------|-------------|-----------------|
| `queen_up` | Queen advantage over single piece | KQvK, KQvKR, KQvKN |
| `queen_vs_pieces` | Queen vs multiple pieces | KQvKRR, KQvKBBN, KQvKRBN |
| `rook_up` | Rook advantage | KRvK, KRRvKR |
| `rook_vs_minor` | Rook vs minor piece | KRvKB, KRvKN |
| `rook_vs_pieces` | Rook(s) vs piece combinations | KRvKBN, KRRvKBN |
| `minor_up` | Minor piece advantage | KBvK, KNvK, KBNvKB |
| `minor_pair_up` | Two minors up | KBNvK, KBBvK |
| `piece_vs_pawns` | Piece vs pawn compensation | KQvKPPPP, KRvKPPP |

Each category is crossed with pawn overlays (0–2 extra pawns per side) and
phase backgrounds (endgame / late-middle / middlegame / early-middle).

## Data Mixer (`data_mixer.py`)

Blends multiple data source directories with configurable ratios, producing
either a mixed DataLoader (for online sampling) or pre-merged .npz chunks.

```bash
# Merge sources into pre-shuffled chunks (for training)
python training/data_mixer.py \
    --ccrl training/data/round_0/ 0.6 \
    --synthetic training/data/synthetic/ 0.2 \
    --output training/data/mixed/
```

| Argument | Description |
|----------|-------------|
| `--ccrl DIR RATIO` | CCRL game data directory and sampling ratio |
| `--synthetic DIR RATIO` | Synthetic position data directory and ratio |
| `--endgame DIR RATIO` | Endgame tablebase data directory and ratio |
| `--opening DIR RATIO` | Opening database data directory and ratio |
| `--output` | Output directory for merged chunks |
| `--chunk-size` | Positions per output chunk (default: 100000) |

## Data Statistics (`data_stats.py`)

Analyze training data quality: value distributions, source breakdown, material
counts, game phase estimates, and anomaly detection.

```bash
# Basic stats
python training/data_stats.py training/data/mixed/

# Detailed analysis with material and phase breakdown
python training/data_stats.py training/data/mixed/ --detailed

# Compare two datasets side by side
python training/data_stats.py training/data/round_0/ training/data/synthetic/ --compare
```

## File Structure

```
training/
├── README.md                # This file
├── ARCHITECTURE.md          # Detailed design rationale
├── requirements.txt         # Python dependencies
├── config.py                # Hyperparameter configs (model sizes, training)
├── encoding.py              # Board ↔ tensor conversion (25-dim per square)
├── model.py                 # Transformer architecture (value + policy heads)
├── prepare_data.py          # PGN/EPD → .npz training chunks
├── synthetic_gen.py         # Synthetic position generator with Stockfish eval
├── data_mixer.py            # Multi-source dataset mixer with ratio sampling
├── data_stats.py            # Data validation, statistics, anomaly detection
├── train.py                 # Training loop (lazy loading, mixed precision)
├── export_onnx.py           # Export to ONNX format for C++ engine
├── validate.py              # Model validation & analysis
├── data/                    # Training data (generated, not committed)
│   ├── round_0/             #   CCRL/self-play chunks
│   ├── synthetic/           #   Stockfish-evaluated synthetic positions
│   └── mixed/               #   Blended multi-source chunks
└── models/                  # Trained models (not committed)
    └── round_0.pt
```
