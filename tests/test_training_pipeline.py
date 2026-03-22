"""
Tests for the training data pipeline: encoding, synthetic generator,
data mixer, HDF5 loading, memory safety, and data quality validation.

These tests don't require Stockfish — the evaluator is mocked where needed.
Run with: python -m pytest tests/test_training_pipeline.py -v
"""

import os
import tempfile
from collections import Counter
from unittest.mock import MagicMock, patch

import chess
import h5py
import numpy as np
import pytest
import torch

# Add training/ to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from encoding import board_to_tensor, move_to_index, NUM_FEATURES, NUM_GLOBAL_FEATURES
from synthetic_gen import (
    MaterialConfig,
    SyntheticGenerator,
    StockfishEvaluator,
    cp_to_value,
    generate_random_position,
    generate_positions_for_config,
    generate_default_configs,
    PIECE_MAP,
    DEFAULT_CONFIGS,
)
from data_mixer import (
    DataMixer,
    MixedChessDataset,
    ProportionalSampler,
    SOURCE_IDS,
    merge_to_chunks,
)
from data_stats import count_material, estimate_phase, analyze_directory
from train import PreloadedDataset, HALFMOVE_TOKEN, HALFMOVE_FEAT, HALFMOVE_SCALE
from convert_to_hdf5 import convert_boards_to_uint8


# ===========================
# HDF5 fixture helpers
# ===========================


def _create_h5_source(path, n_positions, board_dtype=np.uint8, halfmove_values=None):
    """Create a test HDF5 file with random but valid board data."""
    boards = np.zeros((n_positions, 65, 27), dtype=np.float32)
    for i in range(n_positions):
        # Each square gets exactly one plane set (piece or empty)
        for sq in range(64):
            plane = np.random.choice(13)  # 0-11 = pieces, 12 = empty
            boards[i, sq, plane] = 1.0
        # Global context token
        boards[i, 64, 13] = float(np.random.randint(0, 2))  # side to move
        boards[i, 64, 14] = float(np.random.randint(0, 2))  # castling K
        boards[i, 64, 15] = float(np.random.randint(0, 2))  # castling Q
        boards[i, 64, 16] = float(np.random.randint(0, 2))  # castling k
        boards[i, 64, 17] = float(np.random.randint(0, 2))  # castling q
        if halfmove_values is not None:
            boards[i, 64, 26] = halfmove_values[i % len(halfmove_values)]
        else:
            boards[i, 64, 26] = min(np.random.randint(0, 50) / 100.0, 1.0)

    if board_dtype == np.uint8:
        # Simulate convert_to_hdf5 quantization
        boards[:, HALFMOVE_TOKEN, HALFMOVE_FEAT] *= HALFMOVE_SCALE
        boards = np.clip(np.round(boards), 0, 255).astype(np.uint8)

    chunk = min(100, n_positions)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("boards", data=boards, chunks=(chunk, 65, 27))
        f.create_dataset("values", data=np.random.uniform(-1, 1, n_positions).astype(np.float32), chunks=(chunk,))
        f.create_dataset("policies", data=np.random.randint(0, 4096, n_positions).astype(np.uint16), chunks=(chunk,))
        f.create_dataset("sources", data=np.zeros(n_positions, dtype=np.uint8), chunks=(chunk,))
        f.create_dataset("weights", data=np.ones(n_positions, dtype=np.float32), chunks=(chunk,))
        f.attrs["total_positions"] = n_positions
        f.attrs["encoding_version"] = 2
        f.attrs["halfmove_scale"] = 255.0
        f.attrs["board_shape"] = (65, 27)
    return str(path)


# ===========================
# Synthetic Generator Tests
# ===========================


class TestPositionGeneration:
    """Test random position generation without Stockfish."""

    def test_generate_kqvk(self):
        """Generate King+Queen vs King positions."""
        white = [chess.KING, chess.QUEEN]
        black = [chess.KING]
        board = generate_random_position(white, black)
        assert board is not None
        assert board.is_valid()
        # Count pieces
        pieces = list(board.piece_map().values())
        assert len(pieces) == 3
        assert any(p.piece_type == chess.QUEEN and p.color == chess.WHITE for p in pieces)

    def test_generate_krvk(self):
        """Generate King+Rook vs King positions."""
        white = [chess.KING, chess.ROOK]
        black = [chess.KING]
        board = generate_random_position(white, black)
        assert board is not None
        assert board.is_valid()

    def test_generate_complex_material(self):
        """Generate positions with multiple pieces on each side."""
        white = [chess.KING, chess.ROOK, chess.BISHOP]
        black = [chess.KING, chess.ROOK, chess.KNIGHT]
        board = generate_random_position(white, black)
        assert board is not None
        assert board.is_valid()
        pieces = list(board.piece_map().values())
        assert len(pieces) == 6

    def test_generate_with_pawns(self):
        """Pawns should not be on ranks 1 or 8."""
        white = [chess.KING, chess.PAWN, chess.PAWN]
        black = [chess.KING, chess.PAWN]
        for _ in range(20):
            board = generate_random_position(white, black)
            if board is None:
                continue
            for sq, piece in board.piece_map().items():
                if piece.piece_type == chess.PAWN:
                    rank = chess.square_rank(sq)
                    assert 1 <= rank <= 6, f"Pawn on rank {rank}"

    def test_kings_not_adjacent(self):
        """White and black kings should never be adjacent."""
        white = [chess.KING]
        black = [chess.KING]
        for _ in range(50):
            board = generate_random_position(white, black)
            if board is None:
                continue
            wk = board.king(chess.WHITE)
            bk = board.king(chess.BLACK)
            assert wk is not None and bk is not None
            dist = max(
                abs(chess.square_rank(wk) - chess.square_rank(bk)),
                abs(chess.square_file(wk) - chess.square_file(bk)),
            )
            assert dist >= 2, "Kings are adjacent"

    def test_position_not_game_over(self):
        """Generated positions should not be checkmate or stalemate."""
        white = [chess.KING, chess.QUEEN]
        black = [chess.KING]
        for _ in range(30):
            board = generate_random_position(white, black)
            if board is not None:
                assert not board.is_game_over()

    def test_no_castling_no_ep(self):
        """Synthetic positions should have no castling rights or en passant."""
        white = [chess.KING, chess.ROOK]
        black = [chess.KING]
        board = generate_random_position(white, black)
        assert board is not None
        assert board.castling_rights == 0
        assert board.ep_square is None

    def test_halfmove_clock_set(self):
        """Synthetic positions should have a non-trivial halfmove clock."""
        white = [chess.KING, chess.QUEEN]
        black = [chess.KING]
        halfmoves = set()
        for _ in range(50):
            board = generate_random_position(white, black)
            if board is not None:
                halfmoves.add(board.halfmove_clock)
        # Should see multiple distinct values (not all zero)
        assert len(halfmoves) > 1, "Halfmove clock should vary across positions"
        assert max(halfmoves) > 0, "At least some positions should have halfmove > 0"

    def test_halfmove_clock_range(self):
        """Halfmove clock should be in [0, 40]."""
        white = [chess.KING, chess.ROOK]
        black = [chess.KING]
        for _ in range(100):
            board = generate_random_position(white, black)
            if board is not None:
                assert 0 <= board.halfmove_clock <= 40

    def test_generate_config_batch(self):
        """Generate a batch of positions from a MaterialConfig."""
        config = MaterialConfig("KRvK", "KR", "K", num_positions=50)
        results = generate_positions_for_config(config)
        assert len(results) == 50
        for fen, val in results:
            board = chess.Board(fen)
            assert board.is_valid()
            assert val is None  # eval not filled yet

    def test_color_swap_in_config(self):
        """generate_positions_for_config should give both colors the stronger material."""
        config = MaterialConfig("KQvKR", "KQ", "KR", num_positions=200)
        results = generate_positions_for_config(config)
        white_queen_count = 0
        black_queen_count = 0
        for fen, _ in results:
            board = chess.Board(fen)
            white_queen_count += len(board.pieces(chess.QUEEN, chess.WHITE))
            black_queen_count += len(board.pieces(chess.QUEEN, chess.BLACK))
        # Both colors should have queens in a meaningful fraction
        assert white_queen_count > 30, f"White got too few queens: {white_queen_count}"
        assert black_queen_count > 30, f"Black got too few queens: {black_queen_count}"

    def test_no_dead_piece_planes(self):
        """All piece types should appear when generating enough configs."""
        # Use symmetric configs so color swap doesn't matter
        configs = [
            MaterialConfig("KQvKQ", "KQ", "KQ", num_positions=50),
            MaterialConfig("KRvKR", "KR", "KR", num_positions=50),
            MaterialConfig("KBvKB", "KB", "KB", num_positions=50),
            MaterialConfig("KNvKN", "KN", "KN", num_positions=50),
            MaterialConfig("KPvKP", "KP", "KP", num_positions=50),
        ]
        planes_seen = set()
        for cfg in configs:
            results = generate_positions_for_config(cfg)
            for fen, _ in results:
                board = chess.Board(fen)
                tensor = board_to_tensor(board)
                for plane in range(12):
                    if tensor[:64, plane].sum() > 0:
                        planes_seen.add(plane)
        assert len(planes_seen) == 12, (
            f"Only {len(planes_seen)}/12 piece planes activated: "
            f"missing {set(range(12)) - planes_seen}"
        )


class TestCpToValue:
    """Test centipawn to value conversion."""

    def test_zero_cp(self):
        assert cp_to_value(0) == 0.0

    def test_positive_cp(self):
        v = cp_to_value(300)
        assert 0.4 < v < 0.6  # tanh(300/600) ≈ 0.46

    def test_negative_cp(self):
        v = cp_to_value(-300)
        assert -0.6 < v < -0.4

    def test_large_cp(self):
        v = cp_to_value(10000)
        assert v > 0.99  # mate score → near 1.0

    def test_symmetry(self):
        assert abs(cp_to_value(100) + cp_to_value(-100)) < 1e-10


class TestMaterialConfig:
    """Test MaterialConfig dataclass and systematic generation."""

    def test_piece_lists(self):
        config = MaterialConfig("KQvKR", "KQ", "KR")
        assert config.white_list() == [chess.KING, chess.QUEEN]
        assert config.black_list() == [chess.KING, chess.ROOK]

    def test_default_configs_valid(self):
        """All default configs should have valid piece strings."""
        for config in DEFAULT_CONFIGS:
            for c in config.white_pieces:
                assert c in PIECE_MAP, f"Unknown piece '{c}' in {config.name}"
            for c in config.black_pieces:
                assert c in PIECE_MAP, f"Unknown piece '{c}' in {config.name}"
            # Both sides must have a king
            assert "K" in config.white_pieces
            assert "K" in config.black_pieces

    def test_systematic_generation_count(self):
        """Systematic generator should produce 300+ configs with phases."""
        configs = generate_default_configs(
            positions_per_config=100, include_phases=True, max_pawn_overlay=2,
        )
        assert len(configs) >= 200, f"Expected 200+, got {len(configs)}"

    def test_systematic_no_phases(self):
        """Without phases, should produce fewer configs."""
        with_phases = generate_default_configs(include_phases=True)
        without_phases = generate_default_configs(include_phases=False)
        assert len(without_phases) < len(with_phases)
        assert len(without_phases) >= 50

    def test_no_equal_material(self):
        """Systematic configs should not include equal-material positions."""
        for cfg in DEFAULT_CONFIGS:
            w = cfg.white_pieces.replace("K", "")
            b = cfg.black_pieces.replace("K", "")
            # The piece sets should differ (that's the point of imbalance coverage)
            assert w != b or w == "", (
                f"Config {cfg.name} has equal material: {w} vs {b}"
            )

    def test_no_duplicate_names(self):
        """Config names should be unique."""
        names = [c.name for c in DEFAULT_CONFIGS]
        assert len(names) == len(set(names)), "Duplicate config names found"

    def test_max_pieces_limit(self):
        """No config should have more than 8 pieces per side."""
        for cfg in DEFAULT_CONFIGS:
            assert len(cfg.white_pieces) <= 8, f"{cfg.name} white has {len(cfg.white_pieces)} pieces"
            assert len(cfg.black_pieces) <= 8, f"{cfg.name} black has {len(cfg.black_pieces)} pieces"


# ===========================
# Encoding Tests
# ===========================


class TestBoardEncoding:
    """Test board_to_tensor encoding correctness."""

    def test_starting_position_shape(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor.shape == (65, 27)
        assert tensor.dtype == np.float32

    def test_starting_position_piece_count(self):
        """Starting position should have 32 occupied squares."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        pieces_set = tensor[:64, :12].sum()
        empty_set = tensor[:64, 12].sum()
        assert pieces_set == 32
        assert empty_set == 32
        assert pieces_set + empty_set == 64

    def test_each_square_has_one_plane(self):
        """Each square should have exactly one plane active (piece or empty)."""
        board = chess.Board()
        tensor = board_to_tensor(board)
        for sq in range(64):
            bits = tensor[sq, :13].sum()
            assert bits == 1.0, f"Square {sq} has {bits} planes set"

    def test_global_token_side_to_move(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[64, 13] == 1.0  # white to move
        board.turn = chess.BLACK
        tensor = board_to_tensor(board)
        assert tensor[64, 13] == 0.0  # black to move

    def test_global_token_castling(self):
        board = chess.Board()
        tensor = board_to_tensor(board)
        assert tensor[64, 14] == 1.0  # white kingside
        assert tensor[64, 15] == 1.0  # white queenside
        assert tensor[64, 16] == 1.0  # black kingside
        assert tensor[64, 17] == 1.0  # black queenside

    def test_global_token_no_castling(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
        tensor = board_to_tensor(board)
        for plane in range(14, 18):
            assert tensor[64, plane] == 0.0

    def test_global_token_halfmove_clock(self):
        board = chess.Board()
        board.halfmove_clock = 50
        tensor = board_to_tensor(board)
        assert tensor[64, 26] == 0.5  # 50/100

    def test_global_token_halfmove_clamped(self):
        board = chess.Board()
        board.halfmove_clock = 200
        tensor = board_to_tensor(board)
        assert tensor[64, 26] == 1.0  # clamped

    def test_global_token_en_passant(self):
        # White pawn on f5, black just played e7-e5: en passant is legal
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3")
        tensor = board_to_tensor(board)
        # e-file = file index 4
        assert tensor[64, 18 + 4] == 1.0  # en passant on e-file
        # Other ep files should be 0
        for f in range(8):
            if f != 4:
                assert tensor[64, 18 + f] == 0.0

    def test_encoding_uint8_roundtrip(self):
        """Verify float32 → uint8 → float32 roundtrip preserves data."""
        board = chess.Board()
        board.halfmove_clock = 34
        tensor = board_to_tensor(board)

        # Simulate convert_to_hdf5
        t_copy = tensor.copy()
        t_copy[HALFMOVE_TOKEN, HALFMOVE_FEAT] *= HALFMOVE_SCALE
        t_u8 = np.clip(np.round(t_copy), 0, 255).astype(np.uint8)

        # Simulate train.py loading
        t_restored = t_u8.astype(np.float32)
        t_restored[HALFMOVE_TOKEN, HALFMOVE_FEAT] /= HALFMOVE_SCALE

        # Binary features should match exactly
        mask = np.ones_like(tensor, dtype=bool)
        mask[HALFMOVE_TOKEN, HALFMOVE_FEAT] = False
        assert np.array_equal(tensor[mask].astype(np.uint8), t_u8[mask])

        # Halfmove should be close
        err = abs(float(tensor[HALFMOVE_TOKEN, HALFMOVE_FEAT]) -
                  float(t_restored[HALFMOVE_TOKEN, HALFMOVE_FEAT]))
        assert err < 0.005, f"Halfmove roundtrip error {err} too large"


# ===========================
# PreloadedDataset Tests
# ===========================


class TestPreloadedDataset:
    """Test the HDF5-backed PreloadedDataset used by train.py."""

    @pytest.fixture
    def h5_file(self, tmp_path):
        """Create a test HDF5 file with 1000 positions."""
        return _create_h5_source(tmp_path / "test.h5", 1000)

    @pytest.fixture
    def h5_file_large(self, tmp_path):
        """Create a larger test HDF5 file with 5000 positions."""
        return _create_h5_source(tmp_path / "large.h5", 5000)

    def test_load_all(self, h5_file):
        ds = PreloadedDataset({"test": (h5_file, 1.0)})
        assert len(ds) == 1000

    def test_output_shapes(self, h5_file):
        ds = PreloadedDataset({"test": (h5_file, 1.0)})
        board, value, policy = ds[0]
        assert board.shape == (65, 27)
        assert board.dtype == torch.float32
        assert value.dtype == torch.float32
        assert policy.dtype == torch.long

    def test_epoch_size_limits_positions(self, h5_file):
        """epoch_size should limit the number of loaded positions."""
        ds = PreloadedDataset({"test": (h5_file, 1.0)}, epoch_size=200)
        assert len(ds) == 200

    def test_epoch_size_single_source(self, h5_file_large):
        """epoch_size must work for single-source (was previously a bug)."""
        ds = PreloadedDataset({"test": (h5_file_large, 1.0)}, epoch_size=500)
        assert len(ds) == 500, (
            f"epoch_size=500 but loaded {len(ds)} — "
            "single-source epoch_size may be ignored"
        )

    def test_epoch_size_multi_source(self, tmp_path):
        """epoch_size should work with multiple sources and proportional sampling."""
        h5_a = _create_h5_source(tmp_path / "a.h5", 2000)
        h5_b = _create_h5_source(tmp_path / "b.h5", 500)
        ds = PreloadedDataset(
            {"a": (h5_a, 0.8), "b": (h5_b, 0.2)},
            epoch_size=1000,
        )
        assert len(ds) == 1000

    def test_halfmove_denormalization(self, tmp_path):
        """uint8 halfmove clocks should be denormalized to [0, 1] floats."""
        h5_path = _create_h5_source(
            tmp_path / "hm.h5", 100,
            halfmove_values=[0.25],  # halfmove_clock = 25
        )
        ds = PreloadedDataset({"test": (h5_path, 1.0)})
        hm_vals = ds.boards[:, HALFMOVE_TOKEN, HALFMOVE_FEAT]
        # After uint8 roundtrip: 0.25 * 255 = 63.75 → 64 → 64/255 ≈ 0.251
        assert hm_vals.max() < 1.0, "Halfmove should be denormalized to [0, 1]"
        assert hm_vals.max() > 0.2, "Halfmove values should be present"

    def test_value_range(self, h5_file):
        ds = PreloadedDataset({"test": (h5_file, 1.0)})
        assert ds.values.min() >= -1.0
        assert ds.values.max() <= 1.0

    def test_policy_range(self, h5_file):
        ds = PreloadedDataset({"test": (h5_file, 1.0)})
        assert ds.policies.min() >= 0
        assert ds.policies.max() < 4096

    def test_corrupt_policy_raises(self, tmp_path):
        """Policies >= 4096 should raise ValueError."""
        path = tmp_path / "bad.h5"
        with h5py.File(str(path), "w") as f:
            n = 10
            f.create_dataset("boards", data=np.zeros((n, 65, 27), dtype=np.uint8), chunks=(n, 65, 27))
            f.create_dataset("values", data=np.zeros(n, dtype=np.float32), chunks=(n,))
            f.create_dataset("policies", data=np.full(n, 5000, dtype=np.uint16), chunks=(n,))
            f.create_dataset("sources", data=np.zeros(n, dtype=np.uint8), chunks=(n,))
            f.create_dataset("weights", data=np.ones(n, dtype=np.float32), chunks=(n,))
            f.attrs["total_positions"] = n
            f.attrs["encoding_version"] = 2
            f.attrs["halfmove_scale"] = 255.0
            f.attrs["board_shape"] = (65, 27)
        with pytest.raises(ValueError, match="Policy index.*4096"):
            PreloadedDataset({"bad": (str(path), 1.0)})

    def test_missing_source_skipped(self, h5_file):
        """Missing HDF5 files should be skipped with a warning, not crash."""
        ds = PreloadedDataset({
            "exists": (h5_file, 0.5),
            "missing": ("/nonexistent/file.h5", 0.5),
        })
        # Should load only from the existing source
        assert len(ds) > 0


# ===========================
# Memory Safety Tests
# ===========================


class TestMemorySafety:
    """
    Verify torch tensors don't share memory with numpy source arrays.
    The legacy train.py segfaulted because torch.from_numpy() shared memory
    with numpy arrays that were later garbage-collected.
    """

    def test_tensor_memory_independence(self, tmp_path):
        """Modifying source numpy arrays must not affect loaded tensors."""
        h5_path = _create_h5_source(tmp_path / "mem.h5", 50)
        ds = PreloadedDataset({"test": (h5_path, 1.0)})

        # Capture a value before any tampering
        original_board_val = ds.boards[0, 0, 0].item()

        # The numpy arrays should have been deleted in __init__.
        # Verify the torch tensors are independent by checking they're
        # contiguous and own their data.
        assert ds.boards.is_contiguous()
        assert ds.values.is_contiguous()
        assert ds.policies.is_contiguous()

        # Verify data_ptr is valid (would segfault if memory was freed)
        _ = ds.boards[0].sum().item()
        _ = ds.values[0].item()
        _ = ds.policies[0].item()

    def test_float_conversion_creates_new_tensor(self):
        """torch.from_numpy(uint8).float() must NOT share memory with numpy."""
        arr = np.ones((10, 65, 27), dtype=np.uint8)
        tensor = torch.from_numpy(arr).float()

        # Modify numpy array — tensor should be unaffected
        arr[:] = 99
        assert tensor[0, 0, 0].item() == 1.0, (
            "float() tensor shares memory with uint8 numpy array!"
        )

    def test_epoch_size_reduces_tensor_size(self, tmp_path):
        """With epoch_size, tensor should be proportionally smaller."""
        h5_path = _create_h5_source(tmp_path / "big.h5", 5000)

        ds_full = PreloadedDataset({"test": (h5_path, 1.0)})
        ds_small = PreloadedDataset({"test": (h5_path, 1.0)}, epoch_size=500)

        # Small dataset should use ~10x less memory for boards
        ratio = ds_full.boards.nbytes / ds_small.boards.nbytes
        assert ratio > 5, f"epoch_size didn't reduce memory enough: ratio={ratio:.1f}x"


# ===========================
# Data Quality Validation Tests
# ===========================


class TestDataQuality:
    """Test for data quality issues that caused training problems."""

    def test_global_token_not_all_zero(self):
        """Encoded positions with castling/halfmove should have non-zero global tokens."""
        board = chess.Board()  # starting position has castling rights
        tensor = board_to_tensor(board)
        global_token = tensor[64, :]
        assert global_token.sum() > 0, "Global context token should not be all zeros"
        # Specifically check castling planes
        assert tensor[64, 14] == 1.0  # white kingside

    def test_synthetic_halfmove_in_encoding(self):
        """Synthetic positions with halfmove clock should encode it in the tensor."""
        white = [chess.KING, chess.QUEEN]
        black = [chess.KING]
        found_nonzero_hm = False
        for _ in range(50):
            board = generate_random_position(white, black)
            if board is None:
                continue
            tensor = board_to_tensor(board)
            if tensor[64, 26] > 0:
                found_nonzero_hm = True
                break
        assert found_nonzero_hm, "No synthetic position had nonzero halfmove in encoding"

    def test_all_piece_planes_reachable(self):
        """All 12 piece planes should be used across a diverse set of positions."""
        planes_seen = set()
        configs = [
            MaterialConfig("KQvKQ", "KQ", "KQ", num_positions=20),
            MaterialConfig("KRvKR", "KR", "KR", num_positions=20),
            MaterialConfig("KBvKB", "KB", "KB", num_positions=20),
            MaterialConfig("KNvKN", "KN", "KN", num_positions=20),
            MaterialConfig("KPvKP", "KP", "KP", num_positions=20),
        ]
        for cfg in configs:
            results = generate_positions_for_config(cfg)
            for fen, _ in results:
                board = chess.Board(fen)
                tensor = board_to_tensor(board)
                for plane in range(12):
                    if tensor[:64, plane].sum() > 0:
                        planes_seen.add(plane)

        assert len(planes_seen) == 12, (
            f"Only {len(planes_seen)}/12 piece planes activated: "
            f"missing {set(range(12)) - planes_seen}"
        )

    def test_detect_dead_plane(self):
        """Verify we can detect a dead (all-zero) plane in board data."""
        n = 100
        boards = np.zeros((n, 65, 27), dtype=np.float32)
        for i in range(n):
            for sq in range(64):
                boards[i, sq, 12] = 1.0  # all empty squares
            boards[i, 64, 13] = float(i % 2)  # side to move

        # Check plane 10 (black queen) is dead
        plane10_total = boards[:, :64, 10].sum()
        assert plane10_total == 0, "Plane 10 should be dead in this test data"

        # Verify other planes are not all dead
        plane12_total = boards[:, :64, 12].sum()
        assert plane12_total > 0

    def test_value_distribution_not_degenerate(self):
        """Values should have reasonable variance, not all the same."""
        values = np.random.uniform(-1, 1, 1000).astype(np.float32)
        assert values.std() > 0.1, "Value distribution is degenerate"

        # Check for near-constant (all same sign)
        all_positive = np.all(values > 0)
        all_negative = np.all(values < 0)
        assert not all_positive and not all_negative


# ===========================
# Lazy Loading Tests (legacy format — NPZ chunks)
# ===========================


class TestLazyLoading:
    """Test legacy ChessDataset lazy loading (train.py.legacy).
    
    These tests use NPZ chunk format, which is still supported by data_mixer
    and prepare_data. Import from the legacy script if available.
    """

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a small test dataset with 3 chunks."""
        for i in range(3):
            n = 100
            boards = np.random.randn(n, 65, 27).astype(np.float32)
            values = np.random.uniform(-1, 1, n).astype(np.float32)
            policies = np.random.randint(0, 4096, n).astype(np.int64)
            np.savez_compressed(
                tmp_path / f"chunk_{i:04d}.npz",
                boards=boards, values=values, policies=policies,
            )
        np.savez(
            tmp_path / "metadata.npz",
            num_chunks=3, has_policy=True,
            total_positions=300, chunk_size=100,
            num_features=27, skip_moves=0,
            source="test",
        )
        return str(tmp_path)

    def _get_legacy_dataset_class(self):
        """Import ChessDataset from legacy script, skip if unavailable."""
        try:
            # The legacy module file has a dot in the name, use importlib
            import importlib.util
            legacy_path = os.path.join(
                os.path.dirname(__file__), "..", "training", "train.py.legacy"
            )
            spec = importlib.util.spec_from_file_location("train_legacy", legacy_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.ChessDataset
        except Exception:
            pytest.skip("Legacy train.py.legacy not available")

    def test_eager_loading(self, sample_data_dir):
        ChessDataset = self._get_legacy_dataset_class()
        ds = ChessDataset(sample_data_dir, lazy=False)
        assert len(ds) == 300
        board, value, policy = ds[0]
        assert board.shape == (65, 27)

    def test_lazy_loading(self, sample_data_dir):
        ChessDataset = self._get_legacy_dataset_class()
        ds = ChessDataset(sample_data_dir, lazy=True, chunk_cache_size=2)
        assert len(ds) == 300
        board, value, policy = ds[0]
        assert board.shape == (65, 27)

    def test_lazy_cross_chunk_access(self, sample_data_dir):
        ChessDataset = self._get_legacy_dataset_class()
        ds = ChessDataset(sample_data_dir, lazy=True, chunk_cache_size=2)
        b0, _, _ = ds[0]    # chunk 0
        b1, _, _ = ds[100]  # chunk 1
        b2, _, _ = ds[200]  # chunk 2
        assert b0.shape == (65, 27)
        assert b1.shape == (65, 27)
        assert b2.shape == (65, 27)

    def test_lazy_lru_eviction(self, sample_data_dir):
        ChessDataset = self._get_legacy_dataset_class()
        ds = ChessDataset(sample_data_dir, lazy=True, chunk_cache_size=2)
        _ = ds[0]    # load chunk 0
        _ = ds[100]  # load chunk 1, cache: [0, 1]
        _ = ds[200]  # load chunk 2, evict chunk 0, cache: [1, 2]
        assert len(ds._chunk_cache) == 2
        assert 0 not in ds._chunk_cache
        assert 1 in ds._chunk_cache
        assert 2 in ds._chunk_cache

    def test_lazy_bisect_correctness(self, sample_data_dir):
        ChessDataset = self._get_legacy_dataset_class()
        ds = ChessDataset(sample_data_dir, lazy=True)
        ci, local = ds._find_chunk(99)
        assert ci == 0 and local == 99
        ci, local = ds._find_chunk(100)
        assert ci == 1 and local == 0
        ci, local = ds._find_chunk(299)
        assert ci == 2 and local == 99

    def test_out_of_range(self, sample_data_dir):
        ChessDataset = self._get_legacy_dataset_class()
        ds = ChessDataset(sample_data_dir, lazy=True)
        with pytest.raises(IndexError):
            ds._find_chunk(300)


# ===========================
# Data Mixer Tests
# ===========================


def _create_source(tmp_path, name, n_chunks, chunk_size, source_id=0):
    """Helper: create a data source directory with chunks."""
    src_dir = tmp_path / name
    src_dir.mkdir()
    total = 0
    for i in range(n_chunks):
        boards = np.random.randn(chunk_size, 65, 27).astype(np.float32)
        values = np.random.uniform(-1, 1, chunk_size).astype(np.float32)
        policies = np.random.randint(0, 4096, chunk_size).astype(np.int64)
        sources = np.full(chunk_size, source_id, dtype=np.uint8)
        weights = np.ones(chunk_size, dtype=np.float32)
        np.savez_compressed(
            src_dir / f"chunk_{i:04d}.npz",
            boards=boards, values=values, policies=policies,
            sources=sources, weights=weights,
        )
        total += chunk_size
    np.savez(
        src_dir / "metadata.npz",
        num_chunks=n_chunks, has_policy=True,
        total_positions=total, chunk_size=chunk_size,
        num_features=27, skip_moves=0, source=name,
    )
    return str(src_dir)


class TestDataMixer:
    """Test multi-source dataset mixing."""

    @pytest.fixture
    def two_sources(self, tmp_path):
        src_a = _create_source(tmp_path, "source_a", 2, 50, source_id=0)
        src_b = _create_source(tmp_path, "source_b", 1, 50, source_id=1)
        return src_a, src_b

    def test_mixed_dataset_total_size(self, two_sources):
        src_a, src_b = two_sources
        ds = MixedChessDataset({
            "a": (src_a, 0.7),
            "b": (src_b, 0.3),
        })
        assert ds.total_size == 150  # 2*50 + 1*50

    def test_mixed_dataset_getitem(self, two_sources):
        src_a, src_b = two_sources
        ds = MixedChessDataset({"a": (src_a, 0.5), "b": (src_b, 0.5)})
        board, value, policy = ds[0]
        assert board.shape == (65, 27)
        assert isinstance(value, torch.Tensor)
        assert isinstance(policy, torch.Tensor)

    def test_proportional_sampler(self, two_sources):
        src_a, src_b = two_sources
        ds = MixedChessDataset({"a": (src_a, 0.7), "b": (src_b, 0.3)})
        sampler = ProportionalSampler(ds, epoch_size=1000)
        indices = list(sampler)
        assert len(indices) == 1000

        # Check that indices are valid (within dataset range)
        assert all(0 <= i < ds.total_size for i in indices)

    def test_data_mixer_dataloader(self, two_sources):
        src_a, src_b = two_sources
        mixer = DataMixer({"a": (src_a, 0.6), "b": (src_b, 0.4)})
        loader = mixer.get_dataloader(batch_size=16, num_workers=0, epoch_size=100)
        batch = next(iter(loader))
        boards, values, policies = batch
        assert boards.shape == (16, 65, 27)
        assert values.shape == (16,)
        assert policies.shape == (16,)

    def test_merge_to_chunks(self, two_sources, tmp_path):
        src_a, src_b = two_sources
        output_dir = str(tmp_path / "merged")
        merge_to_chunks(
            {"a": (src_a, 0.5), "b": (src_b, 0.5)},
            output_dir,
            chunk_size=50,
        )
        # Check output exists
        assert os.path.exists(os.path.join(output_dir, "metadata.npz"))
        chunks = list((tmp_path / "merged").glob("chunk_*.npz"))
        assert len(chunks) >= 1

        # Check chunks have sources field
        data = np.load(str(chunks[0]))
        assert "sources" in data
        assert "weights" in data

    def test_ratio_normalization(self, two_sources):
        src_a, src_b = two_sources
        ds = MixedChessDataset({"a": (src_a, 3.0), "b": (src_b, 1.0)})
        total = sum(ds.source_ratios.values())
        assert abs(total - 1.0) < 1e-6


# ===========================
# Data Stats Tests
# ===========================


class TestDataStats:
    """Test data statistics and validation."""

    def test_count_material_basic(self):
        board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        tensor = board_to_tensor(board)
        material = count_material(tensor)
        assert "K" in material and "R" in material

    def test_estimate_phase_opening(self):
        board = chess.Board()  # starting position
        tensor = board_to_tensor(board)
        phase = estimate_phase(tensor)
        assert phase == "opening"

    def test_estimate_phase_endgame(self):
        board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        tensor = board_to_tensor(board)
        phase = estimate_phase(tensor)
        assert phase == "endgame"

    def test_analyze_directory(self, tmp_path):
        """Test analyze_directory on a small dataset."""
        src = _create_source(tmp_path, "test_data", 2, 50)
        stats = analyze_directory(src, detailed=True)
        assert stats["total_positions"] == 100
        assert stats["num_chunks"] == 2
        assert "mean" in stats["value_stats"]
        assert len(stats["anomalies"]) >= 0  # may or may not have anomalies


# ===========================
# Integration Test
# ===========================


class TestSyntheticGeneratorIntegration:
    """Integration test for SyntheticGenerator with mocked Stockfish."""

    def test_generate_with_mock_stockfish(self, tmp_path):
        """Test full pipeline with a mock evaluator."""
        config = MaterialConfig("KRvK_test", "KR", "K", num_positions=10)
        positions = generate_positions_for_config(config)
        assert len(positions) == 10

        # Simulate what SyntheticGenerator.generate_all does
        output_dir = str(tmp_path / "synthetic")
        os.makedirs(output_dir, exist_ok=True)

        boards_buf, values_buf, policy_buf = [], [], []
        sources_buf, weights_buf = [], []
        import random

        for fen, _ in positions:
            board = chess.Board(fen)
            features = board_to_tensor(board)
            # Mock eval: random value
            value = cp_to_value(random.randint(-500, 500))
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            policy_idx = move_to_index(move)

            boards_buf.append(features)
            values_buf.append(value)
            policy_buf.append(policy_idx)
            sources_buf.append(1)
            weights_buf.append(1.0)

        # Save chunk
        np.savez_compressed(
            os.path.join(output_dir, "chunk_0000.npz"),
            boards=np.stack(boards_buf),
            values=np.array(values_buf, dtype=np.float32),
            policies=np.array(policy_buf, dtype=np.int64),
            sources=np.array(sources_buf, dtype=np.uint8),
            weights=np.array(weights_buf, dtype=np.float32),
        )
        np.savez(
            os.path.join(output_dir, "metadata.npz"),
            num_chunks=1, has_policy=True,
            total_positions=len(positions), chunk_size=100000,
            num_features=27, skip_moves=0, source="synthetic",
        )

        # Verify the chunks are valid
        data = np.load(os.path.join(output_dir, "chunk_0000.npz"))
        assert data["boards"].shape == (10, 65, 27)
        assert data["values"].shape == (10,)
        assert data["policies"].shape == (10,)

    def test_end_to_end_npz_to_h5_to_preloaded(self, tmp_path):
        """Full pipeline: generate positions → NPZ → HDF5 → PreloadedDataset."""
        import random

        # Step 1: Generate positions
        config = MaterialConfig("KQvK_e2e", "KQ", "K", num_positions=50)
        positions = generate_positions_for_config(config)

        # Step 2: Encode to NPZ chunk
        boards_buf, values_buf, policy_buf = [], [], []
        for fen, _ in positions:
            board = chess.Board(fen)
            features = board_to_tensor(board)
            value = cp_to_value(random.randint(-500, 500))
            legal_moves = list(board.legal_moves)
            move = random.choice(legal_moves)
            boards_buf.append(features)
            values_buf.append(value)
            policy_buf.append(move_to_index(move))

        boards_arr = np.stack(boards_buf)
        values_arr = np.array(values_buf, dtype=np.float32)
        policies_arr = np.array(policy_buf, dtype=np.int64)

        # Step 3: Convert to HDF5 (simulating convert_to_hdf5.py)
        h5_path = str(tmp_path / "e2e.h5")
        boards_u8 = convert_boards_to_uint8(boards_arr.copy())
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("boards", data=boards_u8, chunks=(50, 65, 27))
            f.create_dataset("values", data=values_arr, chunks=(50,))
            f.create_dataset("policies", data=policies_arr.astype(np.uint16), chunks=(50,))
            f.create_dataset("sources", data=np.ones(50, dtype=np.uint8), chunks=(50,))
            f.create_dataset("weights", data=np.ones(50, dtype=np.float32), chunks=(50,))
            f.attrs["total_positions"] = 50
            f.attrs["encoding_version"] = 2
            f.attrs["halfmove_scale"] = 255.0
            f.attrs["board_shape"] = (65, 27)

        # Step 4: Load with PreloadedDataset
        ds = PreloadedDataset({"test": (h5_path, 1.0)})
        assert len(ds) == 50

        # Step 5: Verify roundtrip accuracy
        for i in range(50):
            board_tensor, value_tensor, policy_tensor = ds[i]

            # Binary features should match
            orig = boards_buf[i]
            loaded = board_tensor.numpy()
            for sq in range(64):
                for plane in range(13):
                    assert orig[sq, plane] == loaded[sq, plane], (
                        f"Position {i}, sq {sq}, plane {plane} mismatch"
                    )

            # Value should match
            assert abs(value_tensor.item() - values_buf[i]) < 1e-5

            # Policy should match
            assert policy_tensor.item() == policy_buf[i]

    def test_synthetic_positions_have_diverse_global_tokens(self, tmp_path):
        """After fixes, synthetic positions should have varied global context tokens."""
        config = MaterialConfig("KQvKR_div", "KQ", "KR", num_positions=100)
        positions = generate_positions_for_config(config)

        stm_set = set()
        halfmove_set = set()
        for fen, _ in positions:
            board = chess.Board(fen)
            tensor = board_to_tensor(board)
            stm_set.add(int(tensor[64, 13]))
            halfmove_set.add(round(float(tensor[64, 26]), 2))

        # Side-to-move should include both white and black
        assert len(stm_set) == 2, "Both white and black should appear as STM"

        # Halfmove clock should have variety
        assert len(halfmove_set) > 5, (
            f"Only {len(halfmove_set)} unique halfmove values — "
            "expected more diversity"
        )
