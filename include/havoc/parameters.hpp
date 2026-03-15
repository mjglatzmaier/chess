#pragma once

/// @file parameters.hpp
/// @brief Evaluation parameters / tuning constants for the HCE evaluator.

#include <array>
#include <vector>

namespace havoc {

struct parameters {
    float tempo = 1.0f;

    // Piece-square score scaling (indexed by Piece enum)
    std::vector<int> sq_score_scaling{1, 1, 1, 1, 1};

    // Mobility scaling (indexed by Piece enum)
    std::vector<int> mobility_scaling{1, 1, 1, 1, 1};

    // Square attack bonuses (pawn, knight, bishop, rook, queen)
    std::vector<int> square_attks{7, 4, 3, 2, 1};

    // Piece attack scaling
    std::vector<int> attack_scaling{1, 1, 1, 1, 1};

    // King attack tables per piece type
    static constexpr int knight_attks[6] = {1, 3, 4, 9, 16, 25};
    static constexpr int bishop_attks[6] = {1, 3, 4, 9, 16, 25};
    static constexpr int rook_attks[6] = {0, 2, 5, 5, 7, 15};
    static constexpr int queen_attks[6] = {0, 1, 2, 3, 4, 9};

    std::vector<int> trapped_rook_penalty{1, 2}; // mg, eg

    std::vector<int> attk_queen_bonus{2, 1, 1, 1, 0};

    // Pinned piece scaling
    std::vector<int> pinned_scaling{1, 1, 2, 3, 4};

    // Minor piece bonuses
    std::vector<int> knight_outpost_bonus{0, 1, 2, 3, 3, 2, 1, 0};
    std::vector<int> bishop_outpost_bonus{0, 0, 1, 2, 2, 1, 0, 0};
    std::vector<int> center_influence_bonus{0, 1, 1, 1, 1, 0};

    // King harassment tables
    static constexpr int pawn_king[3] = {1, 2, 3};
    static constexpr int knight_king[3] = {1, 2, 3};
    static constexpr int bishop_king[3] = {1, 2, 3};
    static constexpr int rook_king[5] = {1, 2, 3, 3, 4};
    static constexpr int queen_king[7] = {1, 3, 3, 4, 4, 5, 6};

    static constexpr int attack_combos[5][5] = {
        {0, 0, 0, 4, 10},     // pawn
        {0, 4, 4, 4, 15},     // knight
        {0, 4, 4, 4, 12},     // bishop
        {0, 4, 4, 10, 15},    // rook
        {10, 15, 12, 15, 20}, // queen
    };

    std::vector<int> attacker_weight{1, 4, 8, 16, 32};
    std::vector<int> king_shelter{-3, -2, 2, 3}; // 0,1,2,3 pawns
    std::vector<int> king_safe_sqs{-4, -2, -1, 0, 0, 1, 2, 4};

    int uncastled_penalty = 5;
    static constexpr int connected_rook_bonus = 1;
    static constexpr int doubled_bishop_bonus = 4;
    static constexpr int open_file_bonus = 1;
    static constexpr int bishop_open_center_bonus = 1;
    static constexpr int bishop_color_complex_penalty = 1;
    static constexpr int bishop_penalty_pawns_same_color = 1;
    static constexpr int rook_7th_bonus = 2;

    // Pawn structure
    static constexpr int doubled_pawn_penalty = 4;
    static constexpr int backward_pawn_penalty = 1;
    static constexpr int isolated_pawn_penalty = 4;
    static constexpr int passed_pawn_bonus = 2;
    static constexpr int semi_open_pawn_penalty = 1;

    // Move ordering
    static constexpr int counter_move_bonus = 5;
    static constexpr int threat_evasion_bonus = 2;

    // Search
    int fixed_depth = -1;

    static constexpr int pawn_lever_score[64] = {
        1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3,
        2, 1, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4,
        4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4, 4, 3, 2, 1,
    };
};

} // namespace havoc
