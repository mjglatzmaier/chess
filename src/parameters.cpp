#include "havoc/parameters.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace havoc {

bool parameters::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open())
        return false;

    std::string line;
    auto params = all_params();

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        auto eq = line.find('=');
        if (eq == std::string::npos)
            continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        // Trim whitespace
        auto trim = [](std::string& s) {
            while (!s.empty() && s.front() == ' ')
                s.erase(s.begin());
            while (!s.empty() && s.back() == ' ')
                s.pop_back();
        };
        trim(key);
        trim(val);

        for (auto& [name, ptr] : params) {
            if (name == key) {
                *ptr = std::stoi(val);
                break;
            }
        }
    }
    return true;
}

bool parameters::save(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open())
        return false;

    // const_cast to call all_params() which needs non-const this
    auto params = const_cast<parameters*>(this)->all_params();
    for (const auto& [name, ptr] : params) {
        out << name << " = " << *ptr << "\n";
    }
    return true;
}

std::vector<std::pair<std::string, int*>> parameters::all_params() {
    std::vector<std::pair<std::string, int*>> result;

    // Tempo (stored as int via cast)
    result.emplace_back("tempo", reinterpret_cast<int*>(&tempo));

    // Material values
    for (size_t i = 0; i < material_value.size(); ++i)
        result.emplace_back("material_value_" + std::to_string(i), &material_value[i]);

    // Mobility tables
    for (size_t i = 0; i < knight_mobility_table.size(); ++i)
        result.emplace_back("knight_mobility_" + std::to_string(i), &knight_mobility_table[i]);
    for (size_t i = 0; i < bishop_mobility_table.size(); ++i)
        result.emplace_back("bishop_mobility_" + std::to_string(i), &bishop_mobility_table[i]);
    for (size_t i = 0; i < rook_mobility_table.size(); ++i)
        result.emplace_back("rook_mobility_" + std::to_string(i), &rook_mobility_table[i]);

    // Endgame scaling
    result.emplace_back("opposite_bishop_scale", &opposite_bishop_scale);
    result.emplace_back("no_pawn_scale", &no_pawn_scale);
    result.emplace_back("minor_advantage_no_pawn_scale", &minor_advantage_no_pawn_scale);

    // Passed pawn rank bonuses
    for (size_t i = 0; i < passed_pawn_rank_bonus.size(); ++i)
        result.emplace_back("passed_pawn_rank_bonus_" + std::to_string(i),
                            &passed_pawn_rank_bonus[i]);

    // Attacker weights
    for (size_t i = 0; i < attacker_weight.size(); ++i)
        result.emplace_back("attacker_weight_" + std::to_string(i), &attacker_weight[i]);

    // King shelter
    for (size_t i = 0; i < king_shelter.size(); ++i)
        result.emplace_back("king_shelter_" + std::to_string(i), &king_shelter[i]);

    // King safe squares
    for (size_t i = 0; i < king_safe_sqs.size(); ++i)
        result.emplace_back("king_safe_sqs_" + std::to_string(i), &king_safe_sqs[i]);

    result.emplace_back("uncastled_penalty", &uncastled_penalty);

    return result;
}

} // namespace havoc
