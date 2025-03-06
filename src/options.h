
#pragma once

#ifndef OPTIONS_H
#define OPTIONS_H

#include <string>
#include <sstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <fstream>

#include "utils.h"
#include "evaluate.h"

class options {

private:
	std::mutex m;
	std::map<std::string, std::string> opts;
	void load_args(int argc, char* argv[]);

public:
	options() = delete;
	options(const options&) = delete;
	options(const options&&) = delete;
	options& operator=(const options&) = delete;
	options& operator=(const options&&) = delete;

	options(int argc, char* argv[]) { load_args(argc, argv); }
	~options() {}

	template<typename T>
	inline T value(const char* s) {
		std::unique_lock<std::mutex> lock(m);
		std::istringstream ss{ opts[std::string{s}] };
		T v;
		ss >> v;
		return v;
	}

	template<typename T>
	inline void set(const std::string key, const T value) {
		std::unique_lock<std::mutex> lock(m);
		std::string vs = std::to_string(value);
		if (opts.find(key) == opts.end()) {
			opts.emplace(key, vs);
		}
		else opts[key] = vs;
	}

	bool read_param_file(std::string& filename);
	bool save_param_file(std::string& filename);
	void set_engine_params();
};

inline void options::load_args(int argc, char* argv[]) {

	//read_param_file(std::string("engine.conf"));

	auto matches = [](std::string& s1, const char* s2) { return strcmp(s1.c_str(), s2) == 0; };
	auto set = [this](const std::string& k, const std::string& v) { this->opts.emplace(k.substr(1), v); };

	for (int j = 1; j < argc - 1; j += 2) {
		std::string key = argv[j]; std::string val = argv[j + 1];
		if (matches(key, "-threads")) set(key, val);
		else if (matches(key, "-book")) set(key, val);
		else if (matches(key, "-hashsize")) set(key, val);
		else if (matches(key, "-tune")) set(key, val);
		else if (matches(key, "-bench")) set(key, val);
		else if (matches(key, "-param"))
		{
			std::string paramKey = argv[j + 1];
			std::string paramVal = argv[j + 2]; 
			std::cout << " .. setting new parameter argument: " << paramKey << "=" << paramVal << std::endl;
			this->opts[paramKey] = paramVal;
			set_engine_params();
		}
	}

	// load defaults
	if (opts.find("threads") == opts.end())
		set(std::string("-threads"), std::string("1"));

	if (opts.find("multipv") == opts.end())
		set(std::string("-multipv"), std::string("1"));

	if (opts.find("hashsize") == opts.end())
		set(std::string("-hashsize"), std::string("1000"));
}


inline bool options::read_param_file(std::string& filename) {
	std::string line("");

	if (filename == "") {
		filename = value<std::string>("param");
		if (filename == "") {
			filename = "engine.conf";
		}
	}
	else std::cout << "..reading param file " << filename << std::endl;

	std::ifstream param_file(filename);
	auto set = [this](std::string& k, std::string& v) { this->opts.emplace(k, v); };

	while (std::getline(param_file, line)) {

		// assumed format "param-tag:param-value"
		std::vector<std::string> tokens = util::split(line, ':');

		if (tokens.size() != 2) {
			std::cout << "skipping invalid line" << line << std::endl;
			continue;
		}

		set(tokens[0], tokens[1]);
		//std::cout << "stored param: " << tokens[0] << " value: " << tokens[1] << std::endl;
	}

	set_engine_params();
	return true;
}

inline bool options::save_param_file(std::string& filename) {

	if (filename == "") {
		filename = value<std::string>("param");
		if (filename == "") {
			filename = "engine.conf"; // default
		}
	}

	std::ofstream param_file(filename, std::ofstream::out);

	for (const auto& p : opts) {
		std::string line = p.first + ":" + p.second + "\n";
		param_file << line;
		//std::cout << "saved " << line << " into engine.conf " << std::endl;
	}
	param_file.close();
	return true;
}


inline void options::set_engine_params() {
	auto matches = [](const std::string& s1, const char* s2) { return strcmp(s1.c_str(), s2) == 0; };
	using namespace eval;

	for (const auto& p : opts) {
		std::cout << "engine param: " << p.first << " = " << p.second << std::endl;
		if (matches(p.first, "tempo")) Parameters.tempo = value<float>("tempo");
		else if (matches(p.first, "pawn ss")) Parameters.sq_score_scaling[pawn] = value<float>("pawn ss");
		else if (matches(p.first, "knight ss")) Parameters.sq_score_scaling[knight] = value<float>("knight ss");
		else if (matches(p.first, "bishop ss")) Parameters.sq_score_scaling[bishop] = value<float>("bishop ss");
		else if (matches(p.first, "rook ss")) Parameters.sq_score_scaling[rook] = value<float>("rook ss");
		else if (matches(p.first, "queen ss")) Parameters.sq_score_scaling[queen] = value<float>("queen ss");
		else if (matches(p.first, "king ss")) Parameters.sq_score_scaling[king] = value<float>("king ss");
		else if (matches(p.first, "pawn ms")) Parameters.mobility_scaling[pawn] = value<float>("pawn ms");
		else if (matches(p.first, "knight ms")) Parameters.mobility_scaling[knight] = value<float>("knight ms");
		else if (matches(p.first, "bishop ms")) Parameters.mobility_scaling[bishop] = value<float>("bishop ms");
		else if (matches(p.first, "rook ms")) Parameters.mobility_scaling[rook] = value<float>("rook ms");
		else if (matches(p.first, "queen ms")) Parameters.mobility_scaling[queen] = value<float>("queen ms");
		else if (matches(p.first, "pawn as")) Parameters.attack_scaling[pawn] = value<float>("pawn as");
		else if (matches(p.first, "knight as")) Parameters.attack_scaling[knight] = value<float>("knight as");
		else if (matches(p.first, "bishop as")) Parameters.attack_scaling[bishop] = value<float>("bishop as");
		else if (matches(p.first, "rook as")) Parameters.attack_scaling[rook] = value<float>("rook as");
		else if (matches(p.first, "queen as")) Parameters.attack_scaling[queen] = value<float>("queen as");
		else if (matches(p.first, "castle pen")) Parameters.uncastled_penalty = value<float>("castle pen");
		else if (matches(p.first, "pawn ak")) Parameters.attacker_weight[pawn] = value<float>("pawn ak");
		else if (matches(p.first, "knight ak")) Parameters.attacker_weight[knight] = value<float>("knight ak");
		else if (matches(p.first, "bishop ak")) Parameters.attacker_weight[bishop] = value<float>("bishop ak");
		else if (matches(p.first, "rook ak")) Parameters.attacker_weight[rook] = value<float>("rook ak");
		else if (matches(p.first, "queen ak")) Parameters.attacker_weight[queen] = value<float>("queen ak");
		else if (matches(p.first, "bishop pin")) Parameters.pinned_scaling[bishop] = value<float>("bishop pin");
		else if (matches(p.first, "rook pin")) Parameters.pinned_scaling[rook] = value<float>("rook pin");
		else if (matches(p.first, "queen pin")) Parameters.pinned_scaling[queen] = value<float>("queen pin");
		else if (matches(p.first, "king s1")) Parameters.king_safe_sqs[0] = value<float>("king s1");
		else if (matches(p.first, "king s2")) Parameters.king_safe_sqs[1] = value<float>("king s2");
		else if (matches(p.first, "king s3")) Parameters.king_safe_sqs[2] = value<float>("king s3");
		else if (matches(p.first, "king s4")) Parameters.king_safe_sqs[3] = value<float>("king s4");
		else if (matches(p.first, "king s5")) Parameters.king_safe_sqs[4] = value<float>("king s5");
		else if (matches(p.first, "king s6")) Parameters.king_safe_sqs[5] = value<float>("king s6");
		else if (matches(p.first, "king s7")) Parameters.king_safe_sqs[6] = value<float>("king s7");
		else if (matches(p.first, "king s8")) Parameters.king_safe_sqs[7] = value<float>("king s8");
		else if (matches(p.first, "fixed_depth")) Parameters.fixed_depth = value<int>("fixed_depth");
	}
}

extern std::unique_ptr<options> opts;

#endif
