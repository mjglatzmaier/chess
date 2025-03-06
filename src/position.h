
#pragma once

#ifndef POSITION_H
#define POSITION_H

#include <thread>
#include <algorithm>
#include <vector>
#include <iostream>
#include <array>
#include <string>
#include <sstream>
#include <cstring>
#include <memory>

#include "types.h"
#include "utils.h"
#include "bitboards.h"
#include "magics.h"
#include "zobrist.h"
#include "order.h"
#include "parameter.h" // just for parameter reference (todo: refactor)
#include "pawns.h"
#include "material.h"

struct Move;


struct info {
	U64 checkers;
	U64 pinned[2];
	U64 key;
	U64 mkey;
	U64 pawnkey;
	U64 repkey;
	U16 hmvs;
	U16 cmask;
	U8 move50;
	Color stm;
	Square eps;
	Square ks[2];
	bool has_castled[2];
	Piece captured;
	bool incheck;
};

/// <summary>
/// Rootmove idea taken from Stockfish
/// Allows easy mgmt of pv/multi-pv/time mgmt
/// </summary>
struct Rootmove {
	Rootmove(const Move& m) : pv(1, m) {}
	std::vector<Move> pv;
	int selDepth = 0;
	Score score = Score::ninf; // score for pv[0] == root
	Score prevScore = Score::ninf; // last score for pv[0] == root
	bool operator==(const Move& m) const {
		return pv[0] == m;
	}
	bool operator<(const Rootmove& m) const {
		return m.score != score ? m.score < score : m.prevScore < prevScore;
	}
};
typedef std::vector<Rootmove> Rootmoves;


struct piece_data {

	std::array<U64, 2> bycolor;
	std::array<Square, 2> king_sq;
	std::array<Color, squares> color_on;
	std::array<Piece, squares> piece_on;
	std::array<std::array<int, pieces>, 2> number_of;
	std::array<std::array<U64, squares>, colors> bitmap;
	std::array<std::array<std::array<int, squares>, pieces>, 2> piece_idx;
	std::array<std::array<std::array<Square, 11>, pieces>, 2> square_of;

	piece_data() { };
	piece_data(const piece_data& pd);
	piece_data& operator=(const piece_data& pd);

	// utility methods for moving pieces
	void clear();

	void set(const Color& c, const Piece& p, const Square& s, info& ifo);

	inline void do_quiet(const Color& c, const Piece& p, const Square& f, const Square& t, info& ifo);

	template<Color c>
	inline void do_castle(const bool& kingside);

	template<Color c>
	inline void undo_castle(const bool& kingside);

	inline void do_cap(const Color& c, const Piece& p, const Square& f, const Square& t, info& ifo);

	inline void do_ep(const Color& c, const Square& f, const Square& t, info& ifo);

	inline void do_promotion(const Color& c, const Piece& p,
		const Square& f, const Square& t, info& ifo);

	inline void do_promotion_cap(const Color& c,
		const Piece& p, const Square& f, const Square& t, info& ifo);

	inline void do_castle_ks(const Color& c, const Square& f, const Square& t, info& ifo);

	inline void do_castle_qs(const Color& c, const Square& f, const Square& t, info& ifo);

	inline void remove_piece(const Color& c, const Piece& p, const Square& s, info& ifo);

	inline void add_piece(const Color& c, const Piece& p, const Square& s, info& ifo);
};


class position {
	U16 thread_id;
	info history[1024];
	haVoc::Movehistory stats;
	info ifo;
	piece_data pcs;
	U64 hidx;
	U64 nodes_searched;
	U64 qnodes_searched;

public:
	position() {}
	position(std::istringstream& s);
	position(const std::string& fen);
	position(const position& p, const std::thread& t);
	position(const position& p);
	position(const position&& p);
	position& operator=(const position& p);
	position& operator=(const position&&);
	~position() { }

	double elapsed_ms;
	std::string bestmove;
	parameters params; // reference to our tuneable parameters
	bool debug_search = false;
	Rootmoves root_moves;

	// setup/clear a position
	void setup(std::istringstream& fen);
	std::string to_fen() const;
	void clear();
	void set_piece(const char& p, const Square& s);
	void print() const;
	void do_move(const Move& m);
	void undo_move(const Move& m);
	void do_null_move();
	void undo_null_move();
	int see_move(const Move& m) const;
	int see(const Move& m) const;

	inline void stats_update(const Move& m,
		const Move& previous,
		const int16& depth,
		const Score& score,
		const std::vector<Move>& quiets,
		Move* killers) {
		stats.update(*this, m, previous, depth, score, quiets, killers);
	}
	const haVoc::Movehistory* history_stats() const { return &stats; }

		/// <summary>
	/// Returns true if square 's' owned by 'us' is attacked by 'them'.
	/// </summary>
	/// <param name="s"></param>
	/// <param name="bb"></param>
	/// <returns></returns>
	bool is_attacked(const Square& s, const Color& us, const Color& them, U64 m = 0ULL) const;
	U64 attackers_of2(const Square& s, const Color& c) const;
	U64 attackers_of(const Square& s, const U64& bb) const;
	U64 checkers() const { return ifo.checkers; }
	bool in_check() const;

	/// <summary>
	/// This method assumes the side to move is in check.  Dangerous
	/// checks are contact checks with the king or double-checks
	/// </summary>
	/// <returns></returns>
	bool in_dangerous_check();

	/// <summary>
	/// Returns true if move checks opposing king
	/// </summary>
	bool gives_check(const Move& m);

	/// <summary>
	/// Returns true if a quiet move is a discovered check or a contact check with the enemy king
	/// </summary>
	/// <returns></returns>
	bool quiet_gives_dangerous_check(const Move& m);

	/// <summary>
	/// Returns true if the move can be legally played in the current position
	/// Useful for checking moves from the hash table which are often illegal
	/// </summary>
	/// <param name="m"></param>
	/// <returns></returns>
	bool is_legal(const Move& m);
	U64 pinned(const Color us);
	bool is_draw();

	inline bool can_castle_ks() const {
		return ((ifo.cmask & (ifo.stm == white ? wks : bks))) == (ifo.stm == white ? wks : bks);
	}

	inline bool can_castle_qs() const {
		return (((ifo.cmask & (ifo.stm == white ? wqs : bqs))) == (ifo.stm == white ? wqs : bqs));
	}

	template<Color c>
	inline bool can_castle_ks() const {
		return (((ifo.cmask & (c == white ? wks : bks))) == (c == white ? wks : bks));
	}

	template<Color c>
	inline bool can_castle_qs() const {
		return (((ifo.cmask & (c == white ? wqs : bqs))) == (c == white ? wqs : bqs));
	}

	template<Color c>
	inline bool can_castle() const {
		return can_castle_ks<c>() || can_castle_qs<c>();
	}

	template<Color c>
	inline bool has_castled() const { return ifo.has_castled[c]; }

	template<Color c>
	inline U64 pinned() const { return ifo.pinned[c]; }

	// returns true if there are pawns on the 7th 
	// rank for either side (hint to search not to aggressively reduce search depth)
	inline bool pawns_near_promotion() const {
		return
			(
			((get_pieces<white, pawn>() & bitboards::row[Row::r7]) != 0ULL) ||
			((get_pieces<black, pawn>() & bitboards::row[Row::r2]) != 0ULL)
			);
	}

	// returns true if there are pawns on the 7th rank
	// for the side to move
	inline bool pawns_on_7th() const
	{
		return ifo.stm == white ? 
			((get_pieces<white, pawn>() & bitboards::row[Row::r7]) != 0ULL) :
			((get_pieces<black, pawn>() & bitboards::row[Row::r2]) != 0ULL);
	}

	template<Color c>
	inline bool non_pawn_material() const {
		return ((get_pieces<c, knight>() |
			get_pieces<c, bishop>() |
			get_pieces<c, rook>() |
			get_pieces<c, queen>()) != 0ULL);
	}


	// position info access wrappers
	inline Square eps() const { return ifo.eps; }
	inline Color to_move() const { return ifo.stm; }
	inline U64 key() { return ifo.key; }
	inline U64 repkey() { return ifo.repkey; }
	inline U64 pawnkey() const { return ifo.pawnkey; }
	inline U64 material_key() const { return ifo.mkey; }
	// piece access wrappers
	inline U64 all_pieces() const { return pcs.bycolor[white] | pcs.bycolor[black]; }

	inline unsigned number_of(const Color& c, const Piece& p) const { return pcs.number_of[c][p]; }

	inline Piece piece_on(const Square& s) const { return pcs.piece_on[s]; }

	inline Square king_square(const Color& c) const { return ifo.ks[c]; }

	inline Square king_square() const { return ifo.ks[ifo.stm]; }

	inline Color color_on(const Square& s) const { return pcs.color_on[s]; }

	inline U16 id() { return thread_id; }

	inline void set_id(U16 id) { thread_id = id; }
	inline void set_nodes_searched(U64 n) { nodes_searched = n; }

	inline void set_qnodes_searched(U64 qn) { qnodes_searched = qn; }

	bool is_cap_promotion(const Movetype& mt);
	bool is_promotion(const U8& mt);

	inline bool is_master() { return thread_id == 0; }

	inline U64 nodes() const { return nodes_searched; }
	inline U64 qnodes() const { return qnodes_searched; }
	inline void adjust_nodes(const U64& dn) { nodes_searched += dn; }
	inline void adjust_qnodes(const U64& dn) { qnodes_searched += dn; }

	template<Color c, Piece p>
	inline U64 get_pieces() const { return pcs.bitmap[c][p]; }

	template<Color c>
	inline U64 get_pieces() const { return pcs.bycolor[c]; }

	template<Color c, Piece p>
	inline Square* squares_of() const {
		return const_cast<Square*>(pcs.square_of[c][p].data() + 1);
	}

};


inline void piece_data::clear() {
	std::fill(bycolor.begin(), bycolor.end(), 0);
	std::fill(king_sq.begin(), king_sq.end(), Square::no_square);
	std::fill(color_on.begin(), color_on.end(), Color::no_color);
	std::fill(piece_on.begin(), piece_on.end(), Piece::no_piece);

	for (auto& v : number_of) std::fill(v.begin(), v.end(), 0);
	for (auto& v : bitmap) std::fill(v.begin(), v.end(), 0ULL);
	for (auto& v : piece_idx) { for (auto& w : v) { std::fill(w.begin(), w.end(), 0); } }
	for (auto& v : square_of) { for (auto& w : v) { std::fill(w.begin(), w.end(), Square::no_square); } }
}

inline void piece_data::do_quiet(const Color& c, const Piece& p,
	const Square& f, const Square& t, info& ifo) {

	// bitmaps
	U64 fto = bitboards::squares[f] | bitboards::squares[t];

	int idx = piece_idx[c][p][f];
	piece_idx[c][p][f] = 0;
	piece_idx[c][p][t] = idx;

	bycolor[c] ^= fto;
	bitmap[c][p] ^= fto;

	square_of[c][p][idx] = t;
	color_on[f] = no_color;
	color_on[t] = c;

	piece_on[t] = p;
	piece_on[f] = no_piece;

	ifo.key = ifo.key ^ zobrist::piece(f, c, p);
	ifo.key = ifo.key ^ zobrist::piece(t, c, p);

	ifo.repkey = ifo.repkey ^ zobrist::piece(f, c, p);
	ifo.repkey = ifo.repkey ^ zobrist::piece(t, c, p);

	if (p == Piece::pawn) {
		ifo.pawnkey = ifo.pawnkey ^ zobrist::piece(f, c, p);
		ifo.pawnkey = ifo.pawnkey ^ zobrist::piece(t, c, p);
	}
}

inline void piece_data::do_cap(const Color& c, const Piece& p,
	const Square& f, const Square& t, info& ifo) {
	Color them = Color(c ^ 1);
	Piece cap = piece_on[t];
	remove_piece(them, cap, t, ifo);
	do_quiet(c, p, f, t, ifo);
}

inline void piece_data::do_promotion(const Color& c, const Piece& p,
	const Square& f, const Square& t, info& ifo) {
	remove_piece(c, Piece::pawn, f, ifo);
	add_piece(c, p, t, ifo);
}

inline void piece_data::do_ep(const Color& c, const Square& f, const Square& t, info& ifo) {
	Color them = Color(c ^ 1);
	Square cs = Square(them == white ? t + 8 : t - 8);
	remove_piece(them, Piece::pawn, cs, ifo);
	do_quiet(c, Piece::pawn, f, t, ifo);
}

inline void piece_data::do_promotion_cap(const Color& c, const Piece& p,
	const Square& f, const Square& t, info& ifo) {
	Color them = Color(c ^ 1);
	Piece cap = piece_on[t];
	remove_piece(them, cap, t, ifo);
	remove_piece(c, Piece::pawn, f, ifo);
	add_piece(c, p, t, ifo);
}

inline void piece_data::do_castle_ks(const Color& c, const Square& f, const Square& t, info& ifo) {
	Square rf = (c == white ? H1 : H8);
	Square rt = (c == white ? F1 : F8);

	do_quiet(c, king, f, t, ifo);
	do_quiet(c, rook, rf, rt, ifo);
}

inline void piece_data::do_castle_qs(const Color& c, const Square& f, const Square& t, info& ifo) {
	Square rf = (c == white ? A1 : A8);
	Square rt = (c == white ? D1 : D8);

	do_quiet(c, king, f, t, ifo);
	do_quiet(c, rook, rf, rt, ifo);
}

inline void piece_data::remove_piece(const Color& c, const Piece& p, const Square& s, info& ifo) {
	U64 sq = bitboards::squares[s];
	bycolor[c] ^= sq;
	bitmap[c][p] ^= sq;

	// carefully remove this piece so when we add it back in undo, we
	// do not overwrite an existing piece index
	int tmp_idx = piece_idx[c][p][s];
	int max_idx = number_of[c][p];
	Square tmp_sq = square_of[c][p][max_idx];
	square_of[c][p][tmp_idx] = square_of[c][p][max_idx];
	square_of[c][p][max_idx] = no_square;
	piece_idx[c][p][tmp_sq] = tmp_idx;
	number_of[c][p] -= 1;
	piece_idx[c][p][s] = 0;
	color_on[s] = no_color;
	piece_on[s] = no_piece;
	ifo.key ^= zobrist::piece(s, c, p);
	ifo.mkey ^= zobrist::piece(s, c, p);
	ifo.repkey ^= zobrist::piece(s, c, p);
	if (p == Piece::pawn) ifo.pawnkey ^= zobrist::piece(s, c, p);
}

inline void piece_data::add_piece(const Color& c, const Piece& p, const Square& s, info& ifo) {
	U64 sq = bitboards::squares[s];
	bycolor[c] |= sq;
	bitmap[c][p] |= sq;

	number_of[c][p] += 1;
	square_of[c][p][number_of[c][p]] = s;
	piece_on[s] = p;
	piece_idx[c][p][s] = number_of[c][p];
	color_on[s] = c;
	ifo.key ^= zobrist::piece(s, c, p);
	ifo.mkey ^= zobrist::piece(s, c, p);
	ifo.repkey ^= zobrist::piece(s, c, p);
	if (p == Piece::pawn) ifo.pawnkey ^= zobrist::piece(s, c, p);
}

inline void piece_data::set(const Color& c, const Piece& p, const Square& s, info& ifo) {
	bitmap[c][p] |= bitboards::squares[s];
	bycolor[c] |= bitboards::squares[s];
	color_on[s] = c;
	number_of[c][p] += 1;
	piece_idx[c][p][s] = number_of[c][p];
	square_of[c][p][number_of[c][p]] = s;
	piece_on[s] = p;
	if (p == Piece::king) king_sq[c] = s;

	ifo.key ^= zobrist::piece(s, c, p);
	ifo.mkey ^= zobrist::piece(s, c, p);
	ifo.repkey ^= zobrist::piece(s, c, p);
	if (p == Piece::pawn) ifo.pawnkey ^= zobrist::piece(s, c, p);
}

#endif
