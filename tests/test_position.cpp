#include "havoc/bitboard.hpp"
#include "havoc/magics.hpp"
#include "havoc/movegen.hpp"
#include "havoc/position.hpp"
#include "havoc/zobrist.hpp"

#include <sstream>

#include <gtest/gtest.h>

namespace havoc {

class PositionTest : public ::testing::Test {
  protected:
    void SetUp() override {
        bitboards::init();
        magics::init();
        zobrist::init();
    }
};

// ── FEN round-trip ──────────────────────────────────────────────────────────

TEST_F(PositionTest, FenRoundTrip_Startpos) {
    const std::string fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    std::istringstream iss1(fen_str);
    position pos1(iss1);
    std::string out = pos1.to_fen();

    std::istringstream iss2(out);
    position pos2(iss2);
    EXPECT_EQ(pos1.key(), pos2.key());
    EXPECT_EQ(pos1.to_fen(), pos2.to_fen());
}

TEST_F(PositionTest, FenRoundTrip_Kiwipete) {
    const std::string fen_str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    std::istringstream iss1(fen_str);
    position pos1(iss1);
    std::string out = pos1.to_fen();

    std::istringstream iss2(out);
    position pos2(iss2);
    EXPECT_EQ(pos1.key(), pos2.key());
    EXPECT_EQ(pos1.to_fen(), pos2.to_fen());
}

// ── do_move / undo_move preserves state ─────────────────────────────────────

TEST_F(PositionTest, DoUndo_PreservesKey) {
    std::istringstream fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    position pos(fen);
    U64 key_before = pos.repkey();

    Movegen mvs(pos);
    mvs.generate<pseudo_legal, pieces>();

    for (int i = 0; i < mvs.size(); ++i) {
        if (!pos.is_legal(mvs[i]))
            continue;
        pos.do_move(mvs[i]);
        pos.undo_move(mvs[i]);
        EXPECT_EQ(pos.repkey(), key_before) << "repkey mismatch after do/undo move " << i;
    }
}

TEST_F(PositionTest, DoUndo_PreservesAllPieces) {
    std::istringstream fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    position pos(fen);
    U64 pieces_before = pos.all_pieces();

    Movegen mvs(pos);
    mvs.generate<pseudo_legal, pieces>();

    for (int i = 0; i < mvs.size(); ++i) {
        if (!pos.is_legal(mvs[i]))
            continue;
        pos.do_move(mvs[i]);
        pos.undo_move(mvs[i]);
        EXPECT_EQ(pos.all_pieces(), pieces_before) << "pieces mismatch after do/undo move " << i;
    }
}

// ── null move do/undo ───────────────────────────────────────────────────────

TEST_F(PositionTest, NullMove_PreservesState) {
    std::istringstream fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    position pos(fen);
    U64 key_before = pos.repkey();
    Color stm_before = pos.to_move();

    pos.do_null_move();
    EXPECT_NE(pos.to_move(), stm_before);
    pos.undo_null_move();

    EXPECT_EQ(pos.to_move(), stm_before);
    EXPECT_EQ(pos.repkey(), key_before);
}

// ── is_draw: repetition ─────────────────────────────────────────────────────

TEST_F(PositionTest, IsDraw_Repetition) {
    std::istringstream fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    position pos(fen);

    // Ng1-f3
    Move m1(G1, F3, quiet);
    // Ng8-f6
    Move m2(G8, F6, quiet);
    // Nf3-g1
    Move m3(F3, G1, quiet);
    // Nf6-g8
    Move m4(F6, G8, quiet);

    EXPECT_FALSE(pos.is_draw());

    pos.do_move(m1);
    pos.do_move(m2);
    pos.do_move(m3);
    pos.do_move(m4);

    // Position repeats — should be draw
    EXPECT_TRUE(pos.is_draw());
}

// ── is_draw: 50-move rule ───────────────────────────────────────────────────

TEST_F(PositionTest, IsDraw_50MoveRule) {
    // FEN with move50 = 99
    std::istringstream fen("8/8/8/4k3/8/8/8/4K3 w - - 99 100");
    position pos(fen);
    EXPECT_FALSE(pos.is_draw());

    // One more quiet move -> move50 becomes 100
    Move m(E1, D1, quiet);
    pos.do_move(m);
    EXPECT_TRUE(pos.is_draw());
}

// ── in_check ────────────────────────────────────────────────────────────────

TEST_F(PositionTest, InCheck_No) {
    std::istringstream fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    position pos(fen);
    EXPECT_FALSE(pos.in_check());
}

TEST_F(PositionTest, InCheck_Yes) {
    std::istringstream fen("rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2");
    position pos(fen);
    EXPECT_TRUE(pos.in_check());
}

// ── piece counts ────────────────────────────────────────────────────────────

TEST_F(PositionTest, PieceCounts_Startpos) {
    std::istringstream fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    position pos(fen);

    EXPECT_EQ(pos.number_of(white, pawn), 8u);
    EXPECT_EQ(pos.number_of(black, pawn), 8u);
    EXPECT_EQ(pos.number_of(white, knight), 2u);
    EXPECT_EQ(pos.number_of(black, knight), 2u);
    EXPECT_EQ(pos.number_of(white, bishop), 2u);
    EXPECT_EQ(pos.number_of(black, bishop), 2u);
    EXPECT_EQ(pos.number_of(white, rook), 2u);
    EXPECT_EQ(pos.number_of(black, rook), 2u);
    EXPECT_EQ(pos.number_of(white, queen), 1u);
    EXPECT_EQ(pos.number_of(black, queen), 1u);
    EXPECT_EQ(pos.number_of(white, king), 1u);
    EXPECT_EQ(pos.number_of(black, king), 1u);
}

TEST_F(PositionTest, KingSquare_Startpos) {
    std::istringstream fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    position pos(fen);
    EXPECT_EQ(pos.king_square(white), E1);
    EXPECT_EQ(pos.king_square(black), E8);
}

} // namespace havoc
