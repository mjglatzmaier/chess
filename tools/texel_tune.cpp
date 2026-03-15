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

    /// Batch gradient descent with momentum, learning rate decay, and parameter clamping.
    void optimize(int iterations, TuneStage stage) {
        double K = find_optimal_K();

        auto tunable = params.all_params(stage);
        const size_t N = tunable.size();

        // Stage-appropriate hyperparameters
        double learning_rate;
        int perturbation;
        double lr_decay;      // multiply LR by this each iteration
        double momentum_beta; // exponential moving average factor for gradient
        switch (stage) {
        case TuneStage::category:
            learning_rate = 8.0;
            perturbation = 2;
            lr_decay = 0.7;    // decay faster — few params, converges quick
            momentum_beta = 0.5;
            break;
        case TuneStage::shape:
            learning_rate = 3.0;
            perturbation = 1;
            lr_decay = 0.85;
            momentum_beta = 0.7;
            break;
        case TuneStage::fine:
            learning_rate = 1.5;
            perturbation = 1;
            lr_decay = 0.9;
            momentum_beta = 0.8;
            break;
        }

        // Parameter bounds: {name_prefix, min, max}
        // Category scales should stay in [10, 200] to prevent zeroing out
        // King danger divisor should stay positive
        struct ParamBounds { int lo; int hi; };
        auto get_bounds = [&](const std::string& name) -> ParamBounds {
            if (name.find("category_scale") != std::string::npos) return {10, 200};
            if (name.find("mobility_scale") != std::string::npos) return {10, 200};
            if (name == "king_danger_divisor") return {64, 1024};
            if (name.find("material_value") != std::string::npos) return {10, 30000};
            if (name.find("_scale") != std::string::npos) return {1, 256};
            return {-500, 500}; // general default
        };

        // Momentum: running average of gradients
        std::vector<double> velocity(N, 0.0);

        double current_error = compute_error(K);
        std::cout << "\nStage " << static_cast<int>(stage) + 1
                  << " | Initial error: " << current_error << std::endl;
        std::cout << "Tunable parameters: " << N
                  << " | LR: " << learning_rate
                  << " | LR decay: " << lr_decay
                  << " | Momentum: " << momentum_beta
                  << " | Perturbation: ±" << perturbation << std::endl;

        for (int iter = 0; iter < iterations; ++iter) {
            auto iter_start = std::chrono::steady_clock::now();
            std::cout << "\n=== Iteration " << (iter + 1)
                      << " (lr=" << learning_rate << ") ===" << std::endl;

            // Phase 1: Compute numerical gradient for every parameter
            std::vector<double> gradient(N, 0.0);
            std::vector<int> originals(N);

            for (size_t i = 0; i < N; ++i)
                originals[i] = *tunable[i].second;

            for (size_t i = 0; i < N; ++i) {
                int original = originals[i];

                *tunable[i].second = original + perturbation;
                double err_plus = compute_error(K);

                *tunable[i].second = original - perturbation;
                double err_minus = compute_error(K);

                *tunable[i].second = original;

                gradient[i] = (err_plus - err_minus) / (2.0 * perturbation);
            }

            // Phase 2: Update velocity (momentum) and apply batch update
            int updated = 0;
            for (size_t i = 0; i < N; ++i) {
                // Momentum: smooth the gradient over iterations
                velocity[i] = momentum_beta * velocity[i] + (1.0 - momentum_beta) * gradient[i];

                int delta = static_cast<int>(-learning_rate * velocity[i] * 1e6);

                // Clamp step size
                int max_step = (stage == TuneStage::category) ? 20 : 8;
                delta = std::max(-max_step, std::min(max_step, delta));

                if (delta != 0) {
                    int new_val = originals[i] + delta;

                    // Clamp to parameter bounds
                    auto bounds = get_bounds(tunable[i].first);
                    new_val = std::max(bounds.lo, std::min(bounds.hi, new_val));
                    delta = new_val - originals[i];

                    if (delta != 0) {
                        *tunable[i].second = new_val;
                        ++updated;
                        std::cout << "  " << tunable[i].first << ": " << originals[i]
                                  << " -> " << new_val
                                  << " (grad: " << gradient[i]
                                  << ", vel: " << velocity[i] << ")" << std::endl;
                    }
                }
            }

            double new_error = compute_error(K);

            // If error got worse, revert and halve learning rate
            if (new_error >= current_error) {
                std::cout << "  Error increased (" << current_error << " -> " << new_error
                          << "), reverting and halving LR" << std::endl;
                for (size_t i = 0; i < N; ++i)
                    *tunable[i].second = originals[i];
                learning_rate *= 0.5;
                // Also dampen velocity to avoid overshooting again
                for (auto& v : velocity) v *= 0.5;
                new_error = current_error;
            } else {
                current_error = new_error;
                // Decay learning rate
                learning_rate *= lr_decay;
            }

            auto iter_elapsed = std::chrono::steady_clock::now() - iter_start;
            double secs = std::chrono::duration<double>(iter_elapsed).count();
            std::cout << "Updated " << updated << " / " << N
                      << " params. Error: " << current_error
                      << "  (" << static_cast<int>(secs) << "s)" << std::endl;

            if (updated == 0) {
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
              << "  --iterations N    Tuning iterations (default: 5)\n"
              << "  --stage N         Tuning stage: 1=category, 2=shape, 3=fine (default: 2)\n";
}

int main(int argc, char* argv[]) {
    std::string data_file = "training_data.epd";
    std::string params_file;
    std::string output_file = "tuned_params.txt";
    int iterations = 5;
    int stage_num = 2;

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
        else if ((key == "--stage" || key == "-s") && i + 1 < argc)
            stage_num = std::stoi(argv[++i]);
        else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    TuneStage stage = (stage_num == 1   ? TuneStage::category
                       : stage_num == 3 ? TuneStage::fine
                                        : TuneStage::shape);

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

    tuner.optimize(iterations, stage);

    if (tuner.params.save(output_file))
        std::cout << "\nSaved tuned parameters to " << output_file << std::endl;
    else
        std::cerr << "\nFailed to save parameters to " << output_file << std::endl;

    return 0;
}
