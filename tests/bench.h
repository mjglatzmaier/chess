
#pragma once

#ifndef BENCH_H
#define BENCH_H

#include <string>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cassert>
#include <thread>
#include <fstream>

#include "position.h"
#include "types.h"
#include "move.h"
#include "utils.h"
#include "search.h"
#include "evaluate.h"
#include "pbil.h"
#include "epd.hpp"
#include "options.h"
#include "parameter.h"

std::mutex mtx;

struct scores {
	size_t correct;
	size_t total;
	double acc_score;
	double mnps_score;
	std::vector<double> times_ms;
	std::vector<U64> nodes;
	std::vector<U64> qnodes;
};


namespace pbil_score {

	size_t iteration = 0;
	double best_score = std::numeric_limits<double>::max();
	parameters engine_params;
	std::vector<float> tuneable_params;
	std::unique_ptr<epd> E;
};

class Perft {
	std::vector<double> do_mv_times;
	std::vector<double> undo_mv_times;
	std::vector<double> gen_times;
	std::vector<double> legal_times;

	util::clock dom_timer;
	util::clock tot_timer;
	util::clock gen_timer;
	util::clock legal_timer;

public:
	Perft() {};
	Perft(const Perft& p) = delete;
	Perft(const Perft&& p) = delete;
	Perft& operator=(const Perft& p) = delete;
	Perft& operator=(const Perft&& p) = delete;

	inline void go(const int& depth);
	inline U64 search(position& p, const int& depth);
	inline void divide(position& p, int d);
	inline void gen(position& p, U64& times);
	inline double pbil_search(position& p, const int& depth, scores& S, bool silent);
	inline void auto_tune();
	inline void bench(const int& depth, bool silent);
};


inline void Perft::go(const int& depth) {

	std::string positions[5] =
	{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
		"rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6"
	};

	long int results[5][7] = {
					{ 20,  400,  8902,  197281,   4865609, 119060324, 3195901860 },
					{ 48, 2039, 97862, 4085603, 193690690,         0,          0 },
					{ 14,  191,  2812,   43238,    674624,  11030083,  178633661 },
					{ 6,   264,  9467,  422333,  15833292, 706045033,          0 },
					{ 42, 1352, 53392,       0,         0,         0,          0 } };

	if (depth > 7) {
		std::cout << "Abort, max depth = 7" << std::endl;
		return;
	}

	U64 nb = 0ULL;
	for (int i = 0; i < 5; ++i) {
		std::istringstream fen(positions[i]);
		position board(fen);

		std::cout << "Position : " << positions[i] << std::endl;
		std::cout << "" << std::endl;
		for (int d = 0; d < depth; d++) {
			tot_timer.start();
			nb = search(board, d + 1);
			tot_timer.stop();
			std::cout << "depth "
				<< (d + 1) << "\t"
				<< std::right << std::setw(14)
				<< results[i][d]
				<< "\t" << "perft " << std::setw(14)
				<< nb << "\t " << std::setw(15)
				<< tot_timer.ms() << " ms " << std::endl;
		}
		std::cout << "" << std::endl;
		std::cout << "" << std::endl;
	}
}

inline void Perft::gen(position& p, U64& times) {

	tot_timer.start();
	int count = 0;
	for (U64 i = 0; i < times; ++i) {
		Movegen mvs(p);
		mvs.generate<pseudo_legal, pieces>();
		count = 0;
		for (int j = 0; j < mvs.size(); ++j) {
			if (!p.is_legal(mvs[j])) continue;

			p.do_move(mvs[j]);
			p.undo_move(mvs[j]);

			++count;
		}
	}
	tot_timer.stop();
	std::cout << "---------------------------------" << std::endl;
	std::cout << count << " legal mvs" << std::endl;
	std::cout << "time: " << dom_timer.ms() << " ms " << std::endl;
}

void Perft::divide(position& p, int d) {
	tot_timer.start();

	U64 total = 0;

	Movegen mvs(p);
	gen_timer.start();
	mvs.generate<pseudo_legal, pieces>();
	gen_timer.stop();
	gen_times.push_back(gen_timer.ms() * 1000);

	for (int i = 0; i < mvs.size(); ++i) {

		legal_timer.start();
		if (!p.is_legal(mvs[i])) {
			legal_timer.stop();
			legal_times.push_back(legal_timer.ms() * 1000);
			continue;
		}
		legal_timer.stop();
		legal_times.push_back(legal_timer.ms() * 1000);

		dom_timer.start();
		p.do_move(mvs[i]);
		dom_timer.stop();
		do_mv_times.push_back(1000 * dom_timer.ms());

		int n = d > 1 ? search(p, d - 1) : 1;
		total += n;

		dom_timer.start();
		p.undo_move(mvs[i]);
		dom_timer.stop();
		undo_mv_times.push_back(1000 * dom_timer.ms());

		std::cout << SanSquares[mvs[i].f]
			<< SanSquares[mvs[i].t]
			<< '\t' << n << std::endl;
	}
	tot_timer.stop();

	double dm_avg = 0;
	for (auto& t : do_mv_times) dm_avg += t;
	dm_avg /= do_mv_times.size();

	double udm_avg = 0;
	for (auto& t : undo_mv_times) udm_avg += t;
	udm_avg /= undo_mv_times.size();

	double gen_avg = 0;
	for (auto& t : gen_times) gen_avg += t;
	gen_avg /= gen_times.size();

	double legal_avg = 0;
	for (auto& t : legal_times) legal_avg += t;
	legal_avg /= legal_times.size();

	std::cout << "---------------------------------" << std::endl;
	std::cout << "total " << '\t' << total << std::endl;
	std::cout << "time " << tot_timer.ms() << " ms " << std::endl;
	std::cout << "do-mv time: " << dm_avg << " ns " << std::endl;
	std::cout << "undo-mv time: " << udm_avg << " ns " << std::endl;
	std::cout << "gen-avg time: " << gen_avg << " ns " << std::endl;
	std::cout << "legal-avg time: " << legal_avg << " ns " << std::endl;
}

inline U64 Perft::search(position& p, const int& depth) {
	if (depth == 1) {

		Movegen mvs(p);
		mvs.generate<pseudo_legal, pieces>();

		U64 sz = 0;
		for (int i = 0; i < mvs.size(); ++i) {

			if (!p.is_legal(mvs[i])) {
				continue;
			}
			++sz;
		}
		return sz;
	}

	U64 cnt = 0;
	Movegen mvs(p);
	mvs.generate<pseudo_legal, pieces>();

	for (int i = 0; i < mvs.size(); ++i) {

		if (!p.is_legal(mvs[i])) {
			continue;
		}

		p.do_move(mvs[i]);

		cnt += search(p, depth - 1);

		p.undo_move(mvs[i]);

	}

	return cnt;
}


inline void update_options_file(const position& p) {
	using namespace eval;
	std::cout << "saving options file.." << std::endl;
	opts->set<float>("tempo", p.params.tempo);
	opts->set<float>("pawn ss", p.params.sq_score_scaling[pawn]);
	opts->set<float>("knight ss", p.params.sq_score_scaling[knight]);
	opts->set<float>("bishop ss", p.params.sq_score_scaling[bishop]);
	opts->set<float>("rook ss", p.params.sq_score_scaling[rook]);
	opts->set<float>("queen ss", p.params.sq_score_scaling[queen]);
	opts->set<float>("king ss", p.params.sq_score_scaling[king]);
	opts->set<float>("pawn  ms", p.params.mobility_scaling[pawn]);
	opts->set<float>("knight ms", p.params.mobility_scaling[knight]);
	opts->set<float>("bishop ms", p.params.mobility_scaling[bishop]);
	opts->set<float>("rook ms", p.params.mobility_scaling[rook]);
	opts->set<float>("queen ms", p.params.mobility_scaling[queen]);
	opts->set<float>("pawn as", p.params.attack_scaling[pawn]);
	opts->set<float>("knight as", p.params.attack_scaling[knight]);
	opts->set<float>("bishop as", p.params.attack_scaling[bishop]);
	opts->set<float>("rook as", p.params.attack_scaling[rook]);
	opts->set<float>("queen as", p.params.attack_scaling[queen]);
	opts->set<float>("castle pen", p.params.uncastled_penalty);
	opts->set<float>("knight ak", p.params.attacker_weight[knight]);
	opts->set<float>("bishop ak", p.params.attacker_weight[bishop]);
	opts->set<float>("rook ak", p.params.attacker_weight[rook]);
	opts->set<float>("queen ak", p.params.attacker_weight[queen]);
	opts->set<float>("bishop pin", p.params.pinned_scaling[bishop]);
	opts->set<float>("rook pin", p.params.pinned_scaling[rook]);
	opts->set<float>("queen pin", p.params.pinned_scaling[queen]);
	opts->set<float>("king s1", p.params.king_safe_sqs[0]);
	opts->set<float>("king s2", p.params.king_safe_sqs[1]);
	opts->set<float>("king s3", p.params.king_safe_sqs[2]);
	opts->set<float>("king s4", p.params.king_safe_sqs[3]);
	opts->set<float>("king s5", p.params.king_safe_sqs[4]);
	opts->set<float>("king s6", p.params.king_safe_sqs[5]);
	opts->set<float>("king s7", p.params.king_safe_sqs[6]);
	opts->set<float>("king s8", p.params.king_safe_sqs[7]);
	opts->set<int>("fixed depth", p.params.fixed_depth);

	//opts->save_param_file(std::string(""));
}

inline double pbil_residual(const std::vector<int>& new_bits) {

	++pbil_score::iteration;

	position p;

	{
		std::unique_lock<std::mutex> lock(mtx);

		p.params = pbil_score::engine_params;
		size_t num_floats = pbil_score::tuneable_params.size();
		size_t param_floats = new_bits.size() / 32;
		assert(num_floats == param_floats);
	}


	std::vector<float> new_params;

	for (size_t i = 0, idx = 0; idx < pbil_score::tuneable_params.size(); ++idx) {

		std::bitset<sizeof(float)* CHAR_BIT> b;
		int bidx = 0;

		std::for_each(new_bits.begin() + i, new_bits.begin() + i + 32,
			[&](int val) { b[bidx++] = val; ++i; });


		const auto val = b.to_ulong();
		float new_value = 0;
		memcpy(&new_value, &val, sizeof(float));

		new_params.push_back(new_value);
	}

	// update tuneable params in position class
	p.params.tempo = new_params[0];
	p.params.sq_score_scaling[pawn] = new_params[1];
	p.params.sq_score_scaling[knight] = new_params[2];
	p.params.sq_score_scaling[bishop] = new_params[3];
	p.params.sq_score_scaling[rook] = new_params[4];
	p.params.sq_score_scaling[queen] = new_params[5];
	p.params.sq_score_scaling[king] = new_params[6];
	p.params.mobility_scaling[knight] = new_params[7];
	p.params.mobility_scaling[bishop] = new_params[8];
	p.params.mobility_scaling[rook] = new_params[9];
	p.params.mobility_scaling[queen] = new_params[10];
	p.params.attack_scaling[knight] = new_params[11];
	p.params.attack_scaling[bishop] = new_params[12];
	p.params.attack_scaling[rook] = new_params[13];
	p.params.attack_scaling[queen] = new_params[14];
	p.params.uncastled_penalty = new_params[15];
	p.params.attacker_weight[knight] = new_params[16];
	p.params.attacker_weight[bishop] = new_params[17];
	p.params.attacker_weight[rook] = new_params[18];
	p.params.attacker_weight[queen] = new_params[19];
	p.params.pinned_scaling[bishop] = new_params[20];
	p.params.pinned_scaling[rook] = new_params[21];
	p.params.pinned_scaling[queen] = new_params[22];
	p.params.king_safe_sqs[0] = new_params[23];
	p.params.king_safe_sqs[1] = new_params[24];
	p.params.king_safe_sqs[2] = new_params[25];
	p.params.king_safe_sqs[3] = new_params[26];
	p.params.king_safe_sqs[4] = new_params[27];
	p.params.king_safe_sqs[5] = new_params[28];
	p.params.king_safe_sqs[6] = new_params[29];
	p.params.king_safe_sqs[7] = new_params[30];

	ttable.clear();
	//mtable.clear();
	//ptable.clear();

	scores S;
	Perft perft;
	unsigned depth = 8;
	double minimized_score = perft.pbil_search(p, depth, S, true);


	// log best param set thus far (under lock)
	if (minimized_score < pbil_score::best_score) {
		std::unique_lock<std::mutex> lock(mtx);

		pbil_score::best_score = minimized_score;

		update_options_file(p);
	}

	return std::abs(minimized_score);

}


inline void Perft::auto_tune() {

	pbil_score::engine_params = eval::Parameters;
	pbil_score::tuneable_params =
	{
		eval::Parameters.tempo,
		eval::Parameters.sq_score_scaling[pawn],
		eval::Parameters.sq_score_scaling[knight],
		eval::Parameters.sq_score_scaling[bishop],
		eval::Parameters.sq_score_scaling[rook],
		eval::Parameters.sq_score_scaling[queen],
		eval::Parameters.sq_score_scaling[king],
		eval::Parameters.mobility_scaling[knight],
		eval::Parameters.mobility_scaling[bishop],
		eval::Parameters.mobility_scaling[rook],
		eval::Parameters.mobility_scaling[queen],
		eval::Parameters.attack_scaling[knight],
		eval::Parameters.attack_scaling[bishop],
		eval::Parameters.attack_scaling[rook],
		eval::Parameters.attack_scaling[queen],
		eval::Parameters.uncastled_penalty,
		eval::Parameters.attacker_weight[knight],
		eval::Parameters.attacker_weight[bishop],
		eval::Parameters.attacker_weight[rook],
		eval::Parameters.attacker_weight[queen],
		eval::Parameters.pinned_scaling[bishop],
		eval::Parameters.pinned_scaling[rook],
		eval::Parameters.pinned_scaling[queen],
		eval::Parameters.king_safe_sqs[0],
		eval::Parameters.king_safe_sqs[1],
		eval::Parameters.king_safe_sqs[2],
		eval::Parameters.king_safe_sqs[3],
		eval::Parameters.king_safe_sqs[4],
		eval::Parameters.king_safe_sqs[5],
		eval::Parameters.king_safe_sqs[6],
		eval::Parameters.king_safe_sqs[7]
	};


	pbil_score::E = util::make_unique<epd>("A:\\code\\chess\\sbchess\\tuning\\epd\\mini-test.txt");



	size_t length = pbil_score::tuneable_params.size() * 32;
	pbil_score::iteration = 0;
	pbil_score::best_score = std::numeric_limits<double>::max();

	// was 300
	pbil p(30, length, 0.7, 0.15, 0.3, 0.05, 1e-6);

	std::vector<int> i0;
	for (auto& p : pbil_score::tuneable_params) {
		std::bitset<sizeof(float)* CHAR_BIT> bits = *reinterpret_cast<unsigned long*>(&p);
		for (size_t i = 0; i < bits.size(); ++i) {
			i0.push_back(bits[i]);
		}
	}

	p.set_initial_guess(i0);
	p.optimize(pbil_residual);
}


inline double Perft::pbil_search(position& p, const int& depth, scores& S, bool silent) {

	std::vector<epd_entry> positions = pbil_score::E->get_positions();


	limits lims;
	memset(&lims, 0, sizeof(limits));
	lims.depth = depth;

	S.correct = 0;
	S.total = positions.size();
	size_t counter = 1;
	for (const epd_entry& e : positions) {

		std::cout << "iteration: " << pbil_score::iteration << " pos: " << counter << "/" << positions.size() << " best score: " << pbil_score::best_score << "\r" << std::flush;
		std::istringstream fen(e.pos);
		++counter;

		p.setup(fen);

		Search::start(p, lims, silent);

		S.correct += (p.bestmove == e.bestmove);
		S.times_ms.push_back(p.elapsed_ms);
		S.nodes.push_back(p.nodes());
		S.qnodes.push_back(p.qnodes());
	}

	// avg nps
	double avg_nodes = 0;
	double avg_time_ms = 0;
	double avg_nps = 0;

	for (auto& n : S.nodes) { avg_nodes += n; }
	for (auto& qn : S.qnodes) { avg_nodes += qn; }
	avg_nodes /= (S.nodes.size());
	for (auto& t : S.times_ms) { avg_time_ms += t; }
	avg_time_ms /= S.times_ms.size();
	avg_nps = avg_nodes / avg_time_ms * 1000.0f;

	// convert to Mega nodes / sec
	avg_nps /= 1e6;

	// assume max MNPS ~ 500 (perfect score for speed = 500 MNPS)
	S.mnps_score = avg_nps / 500.0f;
	S.acc_score = ((float)(S.correct / (float)S.total));
	double minimized_score = (1.0f - (0.90 * S.acc_score + 0.10 * S.mnps_score));


	return minimized_score;
}

inline void Perft::bench(const int& depth, bool silent) {

	pbil_score::E = util::make_unique<epd>("A:\\code\\chess\\sbchess\\tuning\\epd\\mini-test.txt");
	std::vector<epd_entry> positions = pbil_score::E->get_positions();
	std::vector<std::string> csv_data;

	limits lims;
	memset(&lims, 0, sizeof(limits));
	lims.depth = depth;

	position p;
	scores S;

	S.correct = 0;
	S.total = positions.size();
	size_t counter = 1;

	for (const epd_entry& e : positions) {

		std::cout << "test pos: " << counter << "/" << positions.size() << " correct: " << S.correct << "/" << S.total << "\r" << std::flush;
		std::istringstream fen(e.pos);
		++counter;

		ttable.clear();
		//mtable.clear();
		//ptable.clear();

		p.setup(fen);

		Search::start(p, lims, silent);


		S.correct += (p.bestmove == e.bestmove);
		S.times_ms.push_back(p.elapsed_ms);
		S.nodes.push_back(p.nodes());
		S.qnodes.push_back(p.qnodes());
	}

	// avg nps
	double avg_nodes = 0;
	double avg_qnodes = 0;
	double avg_time_ms = 0;
	double avg_nps = 0;

	for (auto& n : S.nodes) { avg_nodes += n; }
	for (auto& qn : S.qnodes) { avg_qnodes += qn; }
	avg_nodes /= (S.nodes.size());
	avg_qnodes /= (S.qnodes.size());

	for (auto& t : S.times_ms) { avg_time_ms += t; }
	avg_time_ms /= S.times_ms.size();
	avg_nps = (avg_nodes + avg_qnodes) / avg_time_ms * 1000.0f;

	// convert to Mega nodes / sec
	avg_nps /= 1e6;
	S.acc_score = ((float)(S.correct / (float)S.total));


	std::ofstream result_csv("mini-test-result.csv", std::ios_base::app);

	result_csv << "\n";
	result_csv << "depth, acc, corr, total, avg ms, avg nodes, avg qnodes, avg MNPS\n";
	result_csv <<
		depth << "," <<
		S.acc_score << "," <<
		S.correct << "," <<
		S.total << "," <<
		avg_time_ms << "," <<
		avg_nodes << "," <<
		avg_qnodes << "," <<
		avg_nps << "\n";

	result_csv.close();
}

#endif
