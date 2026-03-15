#include "havoc/bitboard.hpp"
#include "havoc/eval/hce.hpp"
#include "havoc/magics.hpp"
#include "havoc/material_table.hpp"
#include "havoc/parameters.hpp"
#include "havoc/pawn_table.hpp"
#include "havoc/position.hpp"
#include "havoc/tt.hpp"
#include "havoc/zobrist.hpp"

#include <cmath>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace {

/// Helper to build a position from a FEN string.
havoc::position make_pos(const std::string& fen) {
    std::istringstream iss(fen);
    return havoc::position(iss);
}

/// Fixture that initializes tables once for all tests.
class EvalTest : public ::testing::Test {
  protected:
    static void SetUpTestSuite() {
        havoc::bitboards::init();
        havoc::magics::init();
        havoc::zobrist::init();
    }
};

// ─── Startpos eval ≈ 0 ─────────────────────────────────────────────────────

TEST_F(EvalTest, StartposIsApproximatelyZero) {
    havoc::parameters params;
    havoc::pawn_table pt(params);
    havoc::material_table mt;
    havoc::HCEEvaluator eval(pt, mt, params);

    auto pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    int score = eval.evaluate(pos);
    EXPECT_GE(score, -50) << "Startpos eval too low: " << score;
    EXPECT_LE(score, 50) << "Startpos eval too high: " << score;
}

// ─── Extra queen → large advantage ─────────────────────────────────────────

TEST_F(EvalTest, ExtraQueenForWhite) {
    havoc::parameters params;
    havoc::pawn_table pt(params);
    havoc::material_table mt;
    havoc::HCEEvaluator eval(pt, mt, params);

    // White has a queen, black doesn't (removed from d8)
    auto pos = make_pos("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    int score = eval.evaluate(pos);
    EXPECT_GT(score, 800) << "Missing black queen eval too low: " << score;
}

// ─── KNK (king+knight vs king) is drawn ─────────────────────────────────────

TEST_F(EvalTest, KingKnightVsKingIsDraw) {
    havoc::parameters params;
    havoc::pawn_table pt(params);
    havoc::material_table mt;
    havoc::HCEEvaluator eval(pt, mt, params);

    auto pos = make_pos("8/8/8/8/4k3/8/8/K1N5 w - - 0 1");
    int score = eval.evaluate(pos);
    EXPECT_EQ(score, 0) << "KNK should be drawn, got: " << score;
}

// ─── Eval symmetry ──────────────────────────────────────────────────────────

TEST_F(EvalTest, EvalIsSymmetric) {
    havoc::parameters params;
    havoc::pawn_table pt(params);
    havoc::material_table mt;
    havoc::HCEEvaluator eval(pt, mt, params);

    // Symmetric position with white to move
    auto pos_w = make_pos("r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    int score_w = eval.evaluate(pos_w);

    // Mirrored position with black to move
    auto pos_b = make_pos("rnbqkb1r/pppp1ppp/5n2/4p3/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 2 3");
    int score_b = eval.evaluate(pos_b);

    // Both should be similar magnitude (from side-to-move perspective)
    EXPECT_NEAR(score_w, score_b, 15)
        << "White eval: " << score_w << ", Black mirrored eval: " << score_b;
}

// ─── TT basic operations ───────────────────────────────────────────────────

TEST_F(EvalTest, TTStoreAndFetch) {
    havoc::hash_table tt;
    tt.resize(1); // 1 MB

    havoc::Move m(havoc::E2, havoc::E4, havoc::quiet);
    tt.save(0x123456789ABCDEF0ULL, 10, havoc::bound_exact, 1, m, 150, true);

    havoc::hash_data hd;
    bool found = tt.fetch(0x123456789ABCDEF0ULL, hd);
    EXPECT_TRUE(found);
    EXPECT_EQ(hd.depth, 10);
    EXPECT_EQ(hd.bound, havoc::bound_exact);
    EXPECT_EQ(hd.score, 150);
    EXPECT_EQ(hd.move.f, havoc::E2);
    EXPECT_EQ(hd.move.t, havoc::E4);
}

TEST_F(EvalTest, TTNegativeScore) {
    havoc::hash_table tt;
    tt.resize(1);

    havoc::Move m(havoc::D7, havoc::D5, havoc::quiet);
    tt.save(0xFEDCBA9876543210ULL, 5, havoc::bound_low, 2, m, -300, false);

    havoc::hash_data hd;
    bool found = tt.fetch(0xFEDCBA9876543210ULL, hd);
    EXPECT_TRUE(found);
    EXPECT_EQ(hd.score, -300);
}

TEST_F(EvalTest, TTHashfull) {
    havoc::hash_table tt;
    tt.resize(1);
    tt.clear();
    EXPECT_EQ(tt.hashfull(), 0);
}

} // namespace
