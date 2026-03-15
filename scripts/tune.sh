#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
GAMES=${1:-500}
DEPTH=${2:-4}
ITERATIONS=${3:-3}
BUILD_DIR="./build"
DATAGEN="$BUILD_DIR/havoc_datagen"
TUNER="$BUILD_DIR/havoc_texel"
ENGINE="$BUILD_DIR/havoc"
DATA_FILE="training_data.epd"
PARAMS_FILE="tuned_params.txt"

echo "=== haVoc Parameter Tuning Pipeline ==="
echo "Games: $GAMES, Depth: $DEPTH, Iterations: $ITERATIONS"
echo ""

# ── Step 1: Build ─────────────────────────────────────────────────────────────
echo "--- Building ---"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DHAVOC_BUILD_TOOLS=ON
cmake --build "$BUILD_DIR" --parallel
echo ""

# ── Step 2: Generate training data ───────────────────────────────────────────
echo "--- Generating training data ($GAMES games at depth $DEPTH) ---"
"$DATAGEN" --games "$GAMES" --depth "$DEPTH" --output "$DATA_FILE"
echo ""

# ── Step 3: Run Texel tuning ─────────────────────────────────────────────────
echo "--- Running Texel tuning ($ITERATIONS iterations) ---"
"$TUNER" --data "$DATA_FILE" --output "$PARAMS_FILE" --iterations "$ITERATIONS"
echo ""

# ── Step 4: Bake into source ─────────────────────────────────────────────────
echo "--- Baking tuned params into source ---"
python3 scripts/bake_params.py "$PARAMS_FILE"
echo ""

# ── Step 5: Rebuild with tuned defaults ──────────────────────────────────────
echo "--- Rebuilding with new defaults ---"
cmake --build "$BUILD_DIR" --parallel
echo ""

# ── Step 6: Verify with bench ────────────────────────────────────────────────
echo "--- Running bench with baked-in params ---"
printf "bench 8\nquit\n" | "$ENGINE"
echo ""

echo "=== Done ==="
echo "Tuned params baked into include/havoc/parameters.hpp"
echo "Rebuild complete — new defaults are active."
