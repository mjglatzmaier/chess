# Task: Training Pipeline Overhaul

## Motivation

The current data generation → training pipeline has accumulated several reliability
and correctness issues that have caused repeated crashes (OOM system freezes) and
data quality problems. This task tracks a comprehensive overhaul.

### Issues Found in Audit (2026-03-22)

**Critical (crash-causing):**
1. `train.py` ignores `--epoch-size` for single-source `--data` runs, attempting to
   load all 271M positions (~476 GB uint8 → ~1.9 TB float32 peak) and freezing the
   system. **Fixed in current working tree.**
2. Peak memory during uint8→float32→torch conversion was ~9× raw data size due to
   intermediate numpy copy. **Fixed to ~5× in current working tree.**

**Data quality (synthetic data):**
3. Halfmove clock always 0 across all 552K synthetic positions — global context token
   plane 26 is dead. **Fixed in synthetic_gen.py (working tree), but existing
   `data/synthetic/` and `synthetic.h5` still have the old data.**
4. Black queen (plane 10) never appears — the generator always gives white the stronger
   material. **Fixed in synthetic_gen.py (working tree), but existing data still has
   the old encoding.**
5. 52.8% of synthetic positions have an ALL-ZERO global context token (black to move +
   no castling + no en passant + halfmove=0).

**Latent bugs:**
6. `board_dtype` in `PreloadedDataset` is overwritten per source in the loop — if one
   source is uint8 and another float32, the halfmove denormalization is applied to
   the wrong data.
7. The original HDF5 subsampling used per-index fancy indexing on gzip-compressed data,
   which is extremely slow (decompresses each chunk per index). **Fixed to chunk-aligned
   reads in working tree.**

---

## Tasks

### 1. Patch existing synthetic data on disk
**Priority: Immediate**

Write a `fix_synthetic_data.py` script that patches the existing NPZ chunks and
regenerates `synthetic.h5` without re-running the 12-hour Stockfish evaluation:

- Add random halfmove clocks (0–40) to the global context token
- Mirror ~50% of positions (swap piece planes 0–5 ↔ 6–11, flip squares vertically,
  negate side-to-move) so black gets queens/stronger material
- Adjust policy indices for mirrored positions (mirror from_sq and to_sq)
- Regenerate `synthetic.h5` via `convert_to_hdf5.py`
- Verify round-trip accuracy

This avoids re-running Stockfish while fixing the data quality issues.

### 2. Add memory guardrails to train.py
**Priority: High**

Prevent future OOM crashes:

- Estimate memory requirements before loading and warn/abort if insufficient
  (`total_positions × 65 × 27 × 5` bytes for peak)
- Add a `--max-memory` flag (default: 80% of available RAM)
- Print clear memory estimates during loading: "Loading X positions, estimated
  Y GB peak memory (Z GB available)"

### 3. Fix board_dtype multi-source bug
**Priority: High**

The `board_dtype` variable is overwritten per source in the loop. If sources have
mixed dtypes (e.g., one uint8 HDF5 and one float32 NPZ), the halfmove denormalization
is applied to all data based on the LAST source's dtype.

Fix: track dtype per source and apply halfmove fix per source before concatenation.

### 4. Unify data format as HDF5
**Priority: Medium**

Complete the migration started in `train_tasks_01.md`:
- Deprecate NPZ chunk format for training
- Update `prepare_data.py` to write HDF5 directly (or at least make the NPZ→HDF5
  conversion automatic)
- Update `synthetic_gen.py` to write HDF5 directly
- Remove legacy NPZ loading code paths from `train.py`

### 5. Add data validation to the training pipeline
**Priority: Medium**

Add automatic pre-training validation that catches issues like the ones found:

- Per-plane activation statistics (detect dead planes)
- Global context token coverage (ensure castling/ep/halfmove are represented)
- Value distribution by source (detect extreme skew)
- Policy range and uniqueness checks
- Source balance verification (actual vs configured ratios)

This should run as a fast pre-flight check before training starts, not as a
separate script the user has to remember to run.

### 6. Add synthetic data color/feature diversity
**Priority: Medium**

Beyond the immediate fixes:
- Add positions with castling rights for middlegame/early-middle phase backgrounds
  (requires constraining king/rook placement to starting squares)
- Generate both-sides-have-queens configs (e.g., KQRvKQB)
- Add opening-like positions (full starting material minus a few pieces)
- Track piece plane coverage in checkpoint stats so imbalances are visible

### 7. Streaming/chunked loading for large datasets
**Priority: Low (after epoch-size fix)**

The current preload-everything approach works when epoch-size limits memory, but
for maximum flexibility:
- Implement a streaming HDF5 dataset that reads chunks on demand during training
- Use a prefetch buffer (2-3 chunks ahead) to hide I/O latency
- Eliminate the need for epoch-size as a memory management tool

---

## Current State (2026-03-22)

| Item | Status |
|------|--------|
| epoch-size bug fix | ✅ Fixed in working tree (not yet committed) |
| Peak memory reduction | ✅ Fixed in working tree (not yet committed) |
| HDF5 chunk-aligned reads | ✅ Fixed in working tree (not yet committed) |
| synthetic_gen.py halfmove | ✅ Fixed in working tree (not yet committed) |
| synthetic_gen.py color swap | ✅ Fixed in working tree (not yet committed) |
| Patch existing synthetic data | ❌ Not started |
| Memory guardrails | ❌ Not started |
| board_dtype multi-source fix | ❌ Not started |
| HDF5 unification | Partially done (convert_to_hdf5.py exists) |
| Pre-training validation | ❌ Not started |
| Synthetic diversity | ❌ Not started |
| Streaming HDF5 loader | ❌ Not started |
