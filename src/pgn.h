
#ifndef PGN_H
#define PGN_H

#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>
#include <cmath>

#include "move.h"
#include "position.h"
#include "types.h"
#include "utils.h"
#include "uci.h"
enum Result { pgn_draw, pgn_wwin, pgn_bwin, pgn_none };

//const std::string START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct game {
	Result result;
	unsigned white_elo, black_elo;
	std::vector<Move> moves;

	game() { clear(); }
	bool finished() { return result != Result::pgn_none; }
	void clear() { moves.clear(); result = Result::pgn_none; white_elo = 0; black_elo = 0; }
	inline unsigned rating_diff() const { return fabs(white_elo - black_elo); }
};


class pgn {
private:
	std::vector<game> games;
	std::vector<std::string> pgn_files;

	bool parse_files();
	bool parse_moves(position& p, const std::string& line, game& g);

	inline bool is_header(const std::string& line);
	inline bool is_elo(const std::string& line);
	inline bool is_empty(const std::string& line);
	inline void parse_elo(game& g, const std::string& line);
	inline void strip(std::string& token);
	inline Square get_square(const std::string& s, int start);

	template<Piece piece>
	bool find_move(position& p, const Square& to, Move& m, int row = -1, int col = -1);

public:
	pgn() {}
	pgn(const std::vector<std::string>& files);
	~pgn() { }
	pgn(const pgn& o) = delete;
	pgn(const pgn&& o) = delete;
	pgn& operator=(const pgn& o) = delete;
	pgn& operator=(const pgn&& o) = delete;


	std::vector<game>& parsed_games() { return games; }
	bool move_from_san(position& p, std::string& s, Move& m);

};

inline bool pgn::is_elo(const std::string& line) {
	return (line.find("[WhiteElo") != std::string::npos ||
		line.find("[BlackElo") != std::string::npos);
}


inline void pgn::parse_elo(game& g, const std::string& line) {
	// assume this is a valid elo-tag from a pgn file (!!)
	bool white = line.find("[WhiteElo") != std::string::npos;

	std::string segment;
	std::stringstream tmp(line);
	std::getline(tmp, segment, ' ');
	std::getline(tmp, segment, ' ');
	strip(segment);

	if (white) { g.white_elo = std::stoi(segment); }
	else { g.black_elo = std::stoi(segment); }
}


inline bool pgn::is_header(const std::string& line) {
	return (line.find("[") != std::string::npos ||
		line.find("]") != std::string::npos);
}


inline bool pgn::is_empty(const std::string& line) {
	return (line.size() <= 0 || line == "\n");
}

inline void pgn::strip(std::string& token) {
	std::string skip = "!?+#[]\"{}";
	std::string result = "";
	for (const auto& c : token) {
		if (skip.find(c) != std::string::npos) { continue; }
		else result += c;
	}
	token = result;
}


template<Piece piece>
bool pgn::find_move(position& p, const Square& to, Move& m, int row, int col) {

	Movegen mvs(p);
	mvs.generate<piece>();
	bool promotion = (m.type != Movetype::no_type);


	for (int j = 0; j < mvs.size(); ++j) {

		if (!p.is_legal(mvs[j])) continue;

		if (promotion && m.type == mvs[j].type && mvs[j].t == to) {

			if (row >= 0 && row == util::row(mvs[j].f)) { m = mvs[j]; return true; }
			if (col >= 0 && col == util::col(mvs[j].f)) { m = mvs[j]; return true; }
			if (col < 0 && row < 0) { m = mvs[j]; return true; }
		}

		else if (!promotion && mvs[j].t == to) {
			if (row >= 0 && row == util::row(mvs[j].f)) { m = mvs[j]; return true; }
			if (col >= 0 && col == util::col(mvs[j].f)) { m = mvs[j]; return true; }
			if (col < 0 && row < 0) { m = mvs[j]; return true; }
		}

	}
	return false;
}


inline Square pgn::get_square(const std::string& s, int start) {

	std::string str = s.substr(start - 1, 2);

	int i = 0;
	for (const auto& sq : SanSquares) {
		if (sq == str) break;
		++i;
	}
	return Square(i);
}

#endif
