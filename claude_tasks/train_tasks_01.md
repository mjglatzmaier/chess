# Task: Training Data Pipeline Redesign — HDF5 + uint8 Boards

## Motivation

The current compressed `.npz` chunk format causes memory issues during training:

- **Boards stored as float32** but values are binary (0/1) — 4x wasted storage
- **Each `np.load()` on compressed `.npz` decompresses the entire chunk** (~702 MB
  for CCRL) into temporary buffers. glibc's `malloc` doesn't reliably return these
  to the OS, causing RSS to grow ~10 GB/epoch → OOM crash after 6-7 epochs
- **LRU chunk cache** with 100K-position chunks means each cache entry is ~704 MB,
  limiting practical cache depth
- No support for partial reads — must decompress full chunk to access one position

### Current Data Profile

| Source    | Positions     | Chunks | Chunk Size | Disk    |
|-----------|---------------|--------|------------|---------|
| CCRL      | 271,326,839   | 2,714  | 100,000    | ~21 GB  |
| Synthetic | 552,500       | 1,105  | 500        | ~39 MB  |

### Current Array Dtypes (per chunk)

| Array      | Dtype     | Shape             | Memory/chunk (CCRL) | Notes                          |
|------------|-----------|-------------------|---------------------|--------------------------------|
| `boards`   | float32   | `[100K, 65, 27]`  | 702 MB              | Binary 0/1 (except halfmove)   |
| `values`   | float32   | `[100K]`          | 0.4 MB              | Range [-1, +1]                 |
| `policies` | int64     | `[100K]`          | 0.8 MB              | Range [0, 4095] — fits uint16  |
| `sources`  | uint8     | `[100K]`          | 0.1 MB              | Range [0, 3]                   |
| `weights`  | float32   | `[100K]`          | 0.4 MB              | Typically 1.0                  |

---

## Design: HDF5 + Strict uint8 Boards

### New Array Dtypes

| Array      | Dtype     | Shape             | Memory/chunk (CCRL) | Savings |
|------------|-----------|-------------------|---------------------|---------|
| `boards`   | **uint8** | `[100K, 65, 27]`  | **175 MB**          | **4x**  |
| `values`   | float32   | `[100K]`          | 0.4 MB              | —       |
| `policies` | **uint16**| `[100K]`          | **0.2 MB**          | **4x**  |
| `sources`  | uint8     | `[100K]`          | 0.1 MB              | —       |
| `weights`  | float32   | `[100K]`          | 0.4 MB              | —       |

**Per-chunk CCRL: 702 MB → 176 MB (4x reduction)**

### Halfmove Clock Handling

The halfmove clock (global token index 64, feature index 26) is the only non-binary
board feature. Currently stored as `min(halfmove_clock / 100.0, 1.0)` (float in [0, 1]).

**Approach:** Quantize to uint8: `min(halfmove_clock, 255)`. Reconstruct the [0, 1]
float at load time: `value / 255.0`. Precision loss is negligible (1/255 ≈ 0.004).

### HDF5 File Structure

One `.h5` file per source directory (replaces the directory of `.npz` chunks):

```
round_0.h5
├── boards    [N, 65, 27]  uint8    chunked=(10000, 65, 27), compression=lz4
├── values    [N]           float32  chunked=(10000,)
├── policies  [N]           uint16   chunked=(10000,)
├── sources   [N]           uint8    chunked=(10000,)
├── weights   [N]           float32  chunked=(10000,)
└── attrs:
    ├── total_positions: int
    ├── source: str
    ├── has_policy: bool
    ├── num_features: int
    ├── encoding_version: int  (for future-proofing)
    └── halfmove_scale: float  (255.0, for reconstructing halfmove clock)
```

**Key HDF5 properties:**
- **Chunked storage** with chunk shape `(10000, 65, 27)` for boards — enables
  efficient partial reads without loading the full dataset
- **LZ4 compression** — fast decompression (~2 GB/s), low CPU overhead, good ratio
  on sparse binary data. Requires `hdf5plugin` or `h5py` with LZ4 filter.
  Alternative: gzip level 1 (no extra dependency, slightly slower)
- **Single file per source** — no more 2,714 chunk files cluttering the filesystem
- **Random access** — `h5py` supports NumPy-style indexing directly into the file

### Why HDF5 Over Parquet/Arrow

- **Native NumPy integration** — `h5py` datasets support direct slicing (`ds[1000:2000]`)
- **Chunked partial reads** — reads only the HDF5 chunks covering the requested range
- **Compression per-chunk** — each HDF5 chunk is independently compressed/decompressed
- **Mature ecosystem** — well-tested with large scientific datasets
- Parquet is column-oriented (great for tabular), but board tensors are not tabular

---

## Implementation Plan

### Task 1: Conversion Script (`convert_to_hdf5.py`)

New script that reads existing `.npz` chunk directories and writes `.h5` files.

```
python convert_to_hdf5.py \
    --input training/data/round_0/ \
    --output training/data/round_0.h5 \
    --compression gzip   # or lz4 if hdf5plugin available
```

**Steps:**
1. Read `metadata.npz` from input directory
2. Create HDF5 file with pre-sized datasets (total_positions known)
3. Iterate over `chunk_*.npz` files in order:
   - Load each chunk
   - Convert boards: `float32 → uint8` (multiply halfmove_clock by 255, cast rest)
   - Convert policies: `int64 → uint16`
   - Write to HDF5 at the correct offset
4. Set HDF5 attributes from metadata
5. Verify: spot-check random positions match original

**Validation:** The script should verify round-trip accuracy:
- `boards_uint8.astype(float32)` should match original boards for all binary features
- Halfmove clock: `abs(original - uint8/255.0) < 0.005`

### Task 2: Update `encoding.py`

- Add `board_to_tensor_uint8()` that returns `np.uint8` directly
- Or modify `board_to_tensor()` to accept a `dtype` parameter
- Halfmove clock: store as `min(halfmove_clock, 255)` in uint8 mode
- Update `NUM_FEATURES` or add constants for the new encoding

### Task 3: New HDF5 Dataset Class (`hdf5_dataset.py`)

Replace `MixedChessDataset` for HDF5 sources.

```python
class HDF5ChessDataset(Dataset):
    def __init__(self, sources: dict[str, tuple[str, float]]):
        """
        sources: {"ccrl": ("data/round_0.h5", 0.8), "synthetic": ("data/synthetic.h5", 0.2)}
        """
        self.files = {}  # name → h5py.File (kept open for the process lifetime)
        # ... build index of source ranges

    def __getitem__(self, idx):
        # Direct HDF5 read — no chunk cache needed
        source, file, local_idx = self._find_source(idx)
        board = self.files[source]["boards"][local_idx]  # uint8, shape [65, 27]
        board_float = torch.from_numpy(board).float()    # uint8 → float32
        # Reconstruct halfmove clock if needed:
        # board_float[64, 26] /= 255.0
        value = torch.tensor(self.files[source]["values"][local_idx])
        policy = torch.tensor(int(self.files[source]["policies"][local_idx]),
                              dtype=torch.long)
        return board_float, value, policy
```

**Key design points:**
- **No LRU cache needed** — HDF5 handles caching internally via its chunk cache
- **No decompression of full chunks** — reads only the HDF5 chunks that overlap
  the requested indices
- **h5py.File stays open** for the process lifetime (one FD per source file)
- **uint8 → float32 conversion** happens per-item in `__getitem__` (tiny: 1.7 KB)
- **Configure h5py chunk cache** via `h5py.File(..., rdcc_nbytes=...)` to tune
  memory usage explicitly

### Task 4: Update `ProportionalSampler`

The sampler should work with the new dataset. The current chunk-aware sampling
strategy can be simplified since HDF5 doesn't need chunk-aligned access:

- Keep source-proportional sampling (80/20 etc.)
- Generate random indices per source per epoch (no chunk grouping needed)
- The HDF5 chunk cache handles locality automatically

However, **sequential access patterns are still beneficial** for HDF5 read
performance. Consider:
- Sort sampled indices per source before yielding (improves read locality)
- Or keep the current chunk-group approach, mapped to HDF5 chunk boundaries

### Task 5: Update `prepare_data.py`

Option A: Write HDF5 directly during PGN processing (preferred)
Option B: Keep `.npz` output, convert separately with `convert_to_hdf5.py`

For Option A:
- Open HDF5 file with resizable datasets (use `maxshape=(None, 65, 27)`)
- Write positions as they're processed
- Or pre-count positions (two-pass), then write

### Task 6: Update `data_mixer.py`

Update `merge_to_chunks()` to output HDF5 instead of `.npz` chunks.

### Task 7: Update `train.py`

- Detect `.h5` vs directory input and use appropriate dataset class
- Remove `ChessDataset` lazy mode (no longer needed)
- Remove `malloc_trim()` workaround (no longer needed)
- Remove chunk cache diagnostics (no longer needed)

### Task 8: Update `validate.py` and `export_onnx.py`

- `validate.py` uses `board_to_tensor()` directly — update if encoding changes
- `export_onnx.py` has a **pre-existing bug**: dummy input shape is `(1, 64, 25)`
  but model expects `(1, 65, 27)` — fix this while updating

---

## Migration Path

1. **Build conversion script** (Task 1) — converts existing data in-place
2. **Build HDF5 dataset** (Task 3) — new loading code
3. **Wire into training** (Task 7) — support both formats during transition
4. **Update preparation** (Task 5) — new data goes directly to HDF5
5. **Remove old code** — drop `.npz` support after validating

### Backward Compatibility

During migration, `train.py` should support both formats:
```bash
# Old format (directory of .npz chunks):
python train.py --ccrl training/data/round_0/ 0.8

# New format (.h5 file):
python train.py --ccrl training/data/round_0.h5 0.8
```

Detect by checking if the path is a file (`.h5`) or directory (`.npz` chunks).

---

## Expected Impact

| Metric                  | Before (npz)    | After (HDF5)     |
|-------------------------|-----------------|-------------------|
| Board memory per chunk  | 702 MB          | 176 MB            |
| Full CCRL on disk       | ~21 GB          | ~6 GB (+ compression) |
| Cache memory (training) | ~31 GB          | ~2-4 GB (h5py managed) |
| RSS growth per epoch    | ~10 GB          | ~0 (no malloc churn) |
| Chunk load time         | Decompress full | Partial read      |
| File count (CCRL)       | 2,714 files     | 1 file            |

---

## Dependencies

Add to `requirements.txt`:
```
h5py>=3.8.0
```

Optional (for LZ4 compression):
```
hdf5plugin>=4.0.0
```

---

## Open Questions

1. **HDF5 chunk size:** `10000` positions per HDF5 chunk is a reasonable default
   (~1.8 MB per chunk for uint8 boards). Larger chunks = better compression,
   smaller chunks = more granular reads. Worth benchmarking 5K vs 10K vs 50K.

2. **num_workers > 0:** With HDF5, multi-worker DataLoading requires care —
   each worker needs its own `h5py.File` handle (HDF5 is not fork-safe by default).
   Use `h5py.File(..., swmr=True)` or open files in `worker_init_fn`.

3. **Should `data_mixer.py` merge into a single HDF5?** Or keep separate per-source
   `.h5` files and mix at sampling time (current approach)? Keeping separate files
   is simpler and more flexible.
