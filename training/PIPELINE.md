# Training Data Pipeline — Quick Reference

End-to-end command sequence for regenerating all training data and running
a training round. Run all commands from the repository root.

> **When to regenerate:** Any change to `training/encoding.py` (feature count,
> board representation) invalidates all `.npz` chunks. Delete stale data and
> re-run this pipeline.

## Prerequisites

```bash
pip install -r training/requirements.txt

# Stockfish must be on PATH (or pass full path to --stockfish)
which stockfish
```

## Step 0 — Clean Stale Data

Remove old `.npz` chunks built with a previous encoding:

```bash
rm -rf training/data/round_0 training/data/round_0_large \
       training/data/round_0_subset training/data/synthetic \
       training/data/mixed
```

## Step 1 — Generate CCRL Training Data

Convert the CCRL PGN (~2.3M games) into `.npz` chunks using the current encoding.
The full file produces ~279M positions — use `--max-games` to cap it.

```bash
python training/prepare_data.py \
    "CCRL-4040.[2299856].pgn" \
    --output training/data/round_0/ \
    --max-games 80000 \
    --skip-moves 6 \
    --chunk-size 100000
```

Typical output: ~9.7M positions, ~97 chunks (~80K games).

## Step 2 — Generate Synthetic Positions (⏱ ~12 hrs)

Fills coverage gaps (rare endgames, material imbalances) with Stockfish evals.
Resumable — re-run the same command to pick up where it left off.

```bash
python training/synthetic_gen.py \
    --stockfish stockfish \
    --output training/data/synthetic/ \
    --depth 20 \
    --threads 12 \
    --hash 128
```

Typical output: ~550K positions, ~1100 chunks.

### Faster alternatives

```bash
# Endgame-only (~4x fewer configs, ~3 hrs)
python training/synthetic_gen.py \
    --stockfish stockfish \
    --output training/data/synthetic/ \
    --depth 20 --threads 12 --hash 128 \
    --no-phases

# Smoke test (seconds)
python training/synthetic_gen.py \
    --stockfish stockfish \
    --output training/data/synthetic_test/ \
    --configs KQvK KRvK --num 10 --depth 6
```

## Step 3 — Mix Data Sources

Blend CCRL and synthetic data with configurable ratios:

```bash
python training/data_mixer.py \
    --ccrl training/data/round_0/ 0.7 \
    --synthetic training/data/synthetic/ 0.3 \
    --output training/data/mixed/
```

## Step 4 — Inspect Data Quality

Sanity-check the mixed dataset before training:

```bash
python training/data_stats.py training/data/mixed/ --detailed
```

## Step 5 — Train

```bash
python training/train.py \
    --data training/data/mixed/ \
    --output training/models/round_0.pt \
    --model-size small \
    --epochs 10 \
    --batch-size 256 \
    --lr 1e-4

# Resume from checkpoint
python training/train.py \
    --data training/data/mixed/ \
    --checkpoint training/models/round_0.pt \
    --output training/models/round_1.pt \
    --epochs 10
```

Monitor with TensorBoard:

```bash
tensorboard --logdir runs/
```

## Step 6 — Export & Validate

```bash
python training/export_onnx.py training/models/round_0.pt \
    --output training/models/round_0.onnx

python training/validate.py training/models/round_0.pt
```

## Full One-Liner (Steps 1–5)

For an unattended overnight run after cleaning stale data:

```bash
python training/prepare_data.py "CCRL-4040.[2299856].pgn" -o training/data/round_0/ --max-games 80000 && \
python training/synthetic_gen.py --stockfish stockfish -o training/data/synthetic/ --depth 20 --threads 12 --hash 128 && \
python training/data_mixer.py --ccrl training/data/round_0/ 0.7 --synthetic training/data/synthetic/ 0.3 -o training/data/mixed/ && \
python training/data_stats.py training/data/mixed/ --detailed && \
python training/train.py --data training/data/mixed/ --output training/models/round_0.pt --model-size small --epochs 10
```
