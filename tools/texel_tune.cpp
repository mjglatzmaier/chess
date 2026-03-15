/// @file texel_tune.cpp
/// @brief Texel tuner: optimize evaluation parameters via coordinate descent.

#include "havoc/bitboard.hpp"
#include "havoc/eval/hce.hpp"
#include "havoc/magics.hpp"
#include "havoc/material_table.hpp"
#include "havoc/parameters.hpp"
#include "havoc/pawn_table.hpp"
#include "havoc/position.hpp"
#include "havoc/zobrist.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace havoc;

struct TuningPosition {
    std::string fen;
    double result; // 1.0, 0.5, 0.0
};

/// Sigmoid: maps centipawn eval to expected score in [0, 1].
static double sigmoid(double eval, double K = 400.0) {
    return 1.0 / (1.0 + std::pow(10.0, -eval / K));
}

class TexelTuner {
  public:
    std::vector<TuningPosition> positions;
    parameters params;

    bool load_data(const std::string& filename) {
        std::ifstream in(filename);
        if (!in.is_open())
            return false;

        std::string line;
        while (std::getline(in, line)) {
            // Format: FEN c9 "result";
            auto c9_pos = line.find(" c9 ");
            if (c9_pos == std::string::npos)
                continue;

            std::string fen = line.substr(0, c9_pos);

            auto q1 = line.find('"', c9_pos);
            auto q2 = line.find('"', q1 + 1);
            if (q1 == std::string::npos || q2 == std::string::npos)
                continue;

            double result = std::stod(line.substr(q1 + 1, q2 - q1 - 1));
            positions.push_back({fen, result});
        }

        std::cout << "Loaded " << positions.size() << " positions" << std::endl;
        return !positions.empty();
    }

    /// Compute mean squared error over all positions.
    double compute_error(double K = 400.0) {
        pawn_table pt(params);
        material_table mt;
        HCEEvaluator eval(pt, mt, params);

        double total_error = 0.0;
        for (const auto& tp : positions) {
            std::istringstream fen(tp.fen);
            position pos(fen);

            int score = eval.evaluate(pos, -1); // full eval, no lazy margin
            double predicted = sigmoid(static_cast<double>(score), K);
            double diff = tp.result - predicted;
            total_error += diff * diff;
        }

        return total_error / static_cast<double>(positions.size());
    }

    /// Find optimal K (scaling constant) via binary search.
    double find_optimal_K() {
        std::cout << "Finding optimal K..." << std::endl;
        double lo = 50.0, hi = 800.0;
        for (int i = 0; i < 20; ++i) {
            double m1 = lo + (hi - lo) / 3.0;
            double m2 = hi - (hi - lo) / 3.0;
            double e1 = compute_error(m1);
            double e2 = compute_error(m2);
            if (e1 < e2)
                hi = m2;
            else
                lo = m1;
        }
        double K = (lo + hi) / 2.0;
        std::cout << "Optimal K = " << K << " (error = " << compute_error(K) << ")" << std::endl;
        return K;
    }

    /// Coordinate descent: tune one parameter at a time.
    void optimize(int iterations = 5) {
        double K = find_optimal_K();

        auto tunable = params.all_params();
        double best_error = compute_error(K);
        std::cout << "\nInitial error: " << best_error << std::endl;
        std::cout << "Tunable parameters: " << tunable.size() << std::endl;

        for (int iter = 0; iter < iterations; ++iter) {
            auto iter_start = std::chrono::steady_clock::now();
            std::cout << "\n=== Iteration " << (iter + 1) << " ===" << std::endl;
            int improved = 0;

            for (auto& [name, ptr] : tunable) {
                int original = *ptr;
                double best_param_error = best_error;
                int best_value = original;

                // Try delta values: ±1, ±2, ±5, ±10
                for (int delta : {-10, -5, -2, -1, 1, 2, 5, 10}) {
                    *ptr = original + delta;
                    double err = compute_error(K);

                    if (err < best_param_error) {
                        best_param_error = err;
                        best_value = original + delta;
                    }
                }

                if (best_value != original) {
                    *ptr = best_value;
                    best_error = best_param_error;
                    ++improved;
                    std::cout << "  " << name << ": " << original << " -> " << best_value
                              << " (error: " << best_error << ")" << std::endl;
                } else {
                    *ptr = original;
                }
            }

            auto iter_elapsed = std::chrono::steady_clock::now() - iter_start;
            double iter_secs = std::chrono::duration<double>(iter_elapsed).count();

            std::cout << "Improved " << improved << " / " << tunable.size()
                      << " params. Error: " << best_error << "  (" << static_cast<int>(iter_secs)
                      << "s)" << std::endl;

            if (improved == 0) {
                std::cout << "Converged!" << std::endl;
                break;
            }
        }
    }
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "  --data FILE       Input EPD training data (default: training_data.epd)\n"
              << "  --params FILE     Initial parameters file (optional)\n"
              << "  --output FILE     Output tuned parameters (default: tuned_params.txt)\n"
              << "  --iterations N    Tuning iterations (default: 5)\n";
}

int main(int argc, char* argv[]) {
    std::string data_file = "training_data.epd";
    std::string params_file;
    std::string output_file = "tuned_params.txt";
    int iterations = 5;

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if ((key == "--data" || key == "-d") && i + 1 < argc)
            data_file = argv[++i];
        else if ((key == "--params" || key == "-p") && i + 1 < argc)
            params_file = argv[++i];
        else if ((key == "--output" || key == "-o") && i + 1 < argc)
            output_file = argv[++i];
        else if ((key == "--iterations" || key == "-i") && i + 1 < argc)
            iterations = std::stoi(argv[++i]);
        else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    bitboards::init();
    magics::init();
    zobrist::init();

    TexelTuner tuner;

    if (!params_file.empty()) {
        if (tuner.params.load(params_file))
            std::cout << "Loaded initial params from " << params_file << std::endl;
        else
            std::cerr << "Warning: failed to load params from " << params_file << std::endl;
    }

    if (!tuner.load_data(data_file)) {
        std::cerr << "Failed to load training data from " << data_file << std::endl;
        return 1;
    }

    tuner.optimize(iterations);

    if (tuner.params.save(output_file))
        std::cout << "\nSaved tuned parameters to " << output_file << std::endl;
    else
        std::cerr << "\nFailed to save parameters to " << output_file << std::endl;

    return 0;
}
