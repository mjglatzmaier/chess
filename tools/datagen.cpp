/// @file datagen.cpp
/// @brief Training data generator: self-play games → quiet position EPD file.

#include "havoc/bitboard.hpp"
#include "havoc/magics.hpp"
#include "havoc/movegen.hpp"
#include "havoc/position.hpp"
#include "havoc/search.hpp"
#include "havoc/uci.hpp"
#include "havoc/zobrist.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace havoc;

struct DatagenPosition {
    std::string fen;
    double result; // 1.0 = white wins, 0.5 = draw, 0.0 = black wins
};

/// Collect legal moves for the current position.
static std::vector<Move> legal_moves(position& pos) {
    Movegen mvs(pos);
    mvs.generate<pseudo_legal, pieces>();
    std::vector<Move> legals;
    for (int i = 0; i < mvs.size(); ++i) {
        if (pos.is_legal(mvs[i]))
            legals.push_back(mvs[i]);
    }
    return legals;
}

/// Play one self-play game and collect quiet positions.
/// @param random_plies Number of random opening plies for diversity.
static double play_game(SearchEngine& engine, int depth, int random_plies,
                        std::vector<DatagenPosition>& positions, std::mt19937& rng) {
    std::string start_fen(uci::START_FEN);
    std::istringstream fen_stream(start_fen);
    position pos(fen_stream);

    int ply = 0;

    while (ply < 500) {
        if (pos.is_draw())
            return 0.5;

        auto legals = legal_moves(pos);
        if (legals.empty())
            return pos.in_check() ? (pos.to_move() == white ? 0.0 : 1.0) : 0.5;

        Move best{};
        int score = 0;

        if (ply < random_plies) {
            // Play a random legal move for opening diversity
            std::uniform_int_distribution<int> dist(0, static_cast<int>(legals.size()) - 1);
            best = legals[dist(rng)];
        } else {
            // Search for the best move
            SearchLimits lims{};
            lims.depth = static_cast<unsigned>(depth);
            engine.start(pos, lims, true);
            engine.wait();

            if (pos.root_moves.empty())
                return 0.5;

            best = pos.root_moves[0].pv[0];
            score = pos.root_moves[0].score;

            // Resign if losing badly
            if (std::abs(score) > 5000) {
                return score > 0 ? (pos.to_move() == white ? 1.0 : 0.0)
                                 : (pos.to_move() == white ? 0.0 : 1.0);
            }

            // Collect quiet positions: after opening, not in check, quiet move
            bool is_quiet_move = (best.type == static_cast<U8>(quiet));
            if (ply >= 16 && !pos.in_check() && is_quiet_move && std::abs(score) < 3000) {
                positions.push_back({pos.to_fen(), 0.0});
            }
        }

        pos.do_move(best);
        ++ply;
        engine.clear();
    }

    return 0.5; // too long = draw
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --games N        Number of self-play games (default: 100)\n"
              << "  --depth N        Search depth per move (default: 4)\n"
              << "  --random-plies N Random opening plies (default: 4)\n"
              << "  --output FILE    Output EPD file (default: training_data.epd)\n"
              << "  --seed N         Random seed (default: time-based)\n";
}

int main(int argc, char* argv[]) {
    int num_games = 100;
    int depth = 4;
    int random_plies = 4;
    std::string output = "training_data.epd";
    unsigned seed =
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count());

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if ((key == "--games" || key == "-g") && i + 1 < argc)
            num_games = std::stoi(argv[++i]);
        else if ((key == "--depth" || key == "-d") && i + 1 < argc)
            depth = std::stoi(argv[++i]);
        else if (key == "--random-plies" && i + 1 < argc)
            random_plies = std::stoi(argv[++i]);
        else if ((key == "--output" || key == "-o") && i + 1 < argc)
            output = argv[++i];
        else if (key == "--seed" && i + 1 < argc)
            seed = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    bitboards::init();
    magics::init();
    zobrist::init();

    std::mt19937 rng(seed);
    std::vector<DatagenPosition> all_positions;
    SearchEngine engine;

    auto t0 = std::chrono::steady_clock::now();

    for (int g = 0; g < num_games; ++g) {
        std::vector<DatagenPosition> game_positions;
        double result = play_game(engine, depth, random_plies, game_positions, rng);

        for (auto& p : game_positions) {
            p.result = result;
            all_positions.push_back(p);
        }

        if ((g + 1) % 10 == 0 || g + 1 == num_games) {
            auto elapsed = std::chrono::steady_clock::now() - t0;
            double secs = std::chrono::duration<double>(elapsed).count();
            std::cout << "Game " << (g + 1) << "/" << num_games << "  result=" << result
                      << "  positions=" << game_positions.size()
                      << "  total=" << all_positions.size()
                      << "  elapsed=" << static_cast<int>(secs) << "s" << std::endl;
        }
    }

    // Write EPD file
    std::ofstream out(output);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open " << output << " for writing" << std::endl;
        return 1;
    }

    for (const auto& p : all_positions) {
        out << p.fen << " c9 \"" << p.result << "\";" << std::endl;
    }
    out.close();

    auto total_time = std::chrono::steady_clock::now() - t0;
    double total_secs = std::chrono::duration<double>(total_time).count();
    std::cout << "\nWrote " << all_positions.size() << " positions to " << output << " in "
              << static_cast<int>(total_secs) << "s" << std::endl;

    return 0;
}
