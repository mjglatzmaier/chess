
#pragma once

#ifndef PBIL_H
#define PBIL_H


// Population based incremental learning

#include <vector>
#include <random>
#include <functional>

#include "threads.h"


class pbil {

private:
	double mutate_prob, mutate_shift;
	double best_err, learn_rate, neg_learn_rate, etol;
	std::vector<double> probabilities;
	std::vector<std::vector<int>> samples;
	std::vector<int> best_sample, initial_guess;
	std::mt19937 rng;

	void educate();
	void educate(float min, float max);

	void initialize_probabilities();

	void update_probabilities(const std::vector<int>& min_gene,
		const std::vector<int>& max_gene);

	void mutate();
	void init();

public:

	pbil(const size_t& popsz,
		const size_t& nbits,
		const double& mutate_p,
		const double& mutate_s,
		const double& lr,
		const double& nlr,
		const double& tol) :
		mutate_prob(mutate_p), mutate_shift(mutate_s), best_err(1e10),
		learn_rate(lr), neg_learn_rate(nlr), etol(tol)
	{
		samples =
			std::vector<std::vector<int>>(popsz, std::vector<int>(nbits));

		probabilities = std::vector<double>(nbits);

		rng.seed(std::random_device{}());
	};

	pbil& operator=(const pbil& o) = delete;
	pbil& operator=(const pbil&& o) = delete;
	pbil(const pbil& o) = delete;
	~pbil() { }


	template<class T, typename... Args>
	void optimize(T&& residual, Args&&... args);

	void set_initial_guess(const std::vector<int>& guess) {
		for (auto& g : guess) initial_guess.push_back(g);
	}

};

#include "pbil.hpp"

#endif
