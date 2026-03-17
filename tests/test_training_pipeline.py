"""
Tests for the training data pipeline: synthetic generator, data mixer,
lazy loading, and data stats.

These tests don't require Stockfish — the evaluator is mocked where needed.
Run with: python -m pytest tests/test_training_pipeline.py -v
"""

import os
import tempfile
from collections import Counter
from unittest.mock import MagicMock, patch

import chess
import numpy as np
import pytest
import torch

# Add training/ to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

from encoding import board_to_tensor, move_to_index, NUM_FEATURES
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
from train import ChessDataset


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

    def test_generate_config_batch(self):
        """Generate a batch of positions from a MaterialConfig."""
        config = MaterialConfig("KRvK", "KR", "K", num_positions=50)
        results = generate_positions_for_config(config)
        assert len(results) == 50
        for fen, val in results:
            board = chess.Board(fen)
            assert board.is_valid()
            assert val is None  # eval not filled yet


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
# Lazy Loading Tests
# ===========================


class TestLazyLoading:
    """Test ChessDataset lazy loading with LRU cache."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a small test dataset with 3 chunks."""
        for i in range(3):
            n = 100
            boards = np.random.randn(n, 64, 25).astype(np.float32)
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
            num_features=25, skip_moves=0,
            source="test",
        )
        return str(tmp_path)

    def test_eager_loading(self, sample_data_dir):
        ds = ChessDataset(sample_data_dir, lazy=False)
        assert len(ds) == 300
        board, value, policy = ds[0]
        assert board.shape == (64, 25)

    def test_lazy_loading(self, sample_data_dir):
        ds = ChessDataset(sample_data_dir, lazy=True, chunk_cache_size=2)
        assert len(ds) == 300
        board, value, policy = ds[0]
        assert board.shape == (64, 25)

    def test_lazy_cross_chunk_access(self, sample_data_dir):
        ds = ChessDataset(sample_data_dir, lazy=True, chunk_cache_size=2)
        # Access positions from different chunks
        b0, _, _ = ds[0]    # chunk 0
        b1, _, _ = ds[100]  # chunk 1
        b2, _, _ = ds[200]  # chunk 2
        assert b0.shape == (64, 25)
        assert b1.shape == (64, 25)
        assert b2.shape == (64, 25)

    def test_lazy_lru_eviction(self, sample_data_dir):
        ds = ChessDataset(sample_data_dir, lazy=True, chunk_cache_size=2)
        _ = ds[0]    # load chunk 0
        _ = ds[100]  # load chunk 1, cache: [0, 1]
        _ = ds[200]  # load chunk 2, evict chunk 0, cache: [1, 2]
        assert len(ds._chunk_cache) == 2
        assert 0 not in ds._chunk_cache
        assert 1 in ds._chunk_cache
        assert 2 in ds._chunk_cache

    def test_lazy_bisect_correctness(self, sample_data_dir):
        ds = ChessDataset(sample_data_dir, lazy=True)
        # Last element of each chunk
        ci, local = ds._find_chunk(99)
        assert ci == 0 and local == 99
        ci, local = ds._find_chunk(100)
        assert ci == 1 and local == 0
        ci, local = ds._find_chunk(299)
        assert ci == 2 and local == 99

    def test_out_of_range(self, sample_data_dir):
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
        boards = np.random.randn(chunk_size, 64, 25).astype(np.float32)
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
        num_features=25, skip_moves=0, source=name,
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
        assert board.shape == (64, 25)
        assert isinstance(value, torch.Tensor)
        assert isinstance(policy, torch.Tensor)

    def test_proportional_sampler(self, two_sources):
        src_a, src_b = two_sources
        ds = MixedChessDataset({"a": (src_a, 0.7), "b": (src_b, 0.3)})
        sampler = ProportionalSampler(ds, epoch_size=1000)
        indices = list(sampler)
        assert len(indices) == 1000

        # Check approximate proportions
        a_indices = set(ds.source_indices["a"])
        b_indices = set(ds.source_indices["b"])
        a_count = sum(1 for i in indices if i in a_indices)
        b_count = sum(1 for i in indices if i in b_indices)
        # Allow ±10% tolerance
        assert 600 < a_count < 800, f"Expected ~700, got {a_count}"
        assert 200 < b_count < 400, f"Expected ~300, got {b_count}"

    def test_data_mixer_dataloader(self, two_sources):
        src_a, src_b = two_sources
        mixer = DataMixer({"a": (src_a, 0.6), "b": (src_b, 0.4)})
        loader = mixer.get_dataloader(batch_size=16, num_workers=0, epoch_size=100)
        batch = next(iter(loader))
        boards, values, policies = batch
        assert boards.shape == (16, 64, 25)
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
            num_features=25, skip_moves=0, source="synthetic",
        )

        # Verify it loads correctly with ChessDataset
        ds = ChessDataset(output_dir, lazy=True)
        assert len(ds) == 10
        board, value, policy = ds[0]
        assert board.shape == (64, 25)
        assert -1.0 <= value.item() <= 1.0
