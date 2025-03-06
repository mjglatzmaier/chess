
#include <vector>

#include "pawns.h"
#include "types.h"
#include "utils.h"
#include "bitboards.h"
#include "squares.h"
#include "evaluate.h"
#include "position.h"




template<Color c>
int16 evaluate(const position& p, pawn_entry& e);

inline size_t pow2(size_t x) {
	return x <= 2 ? x : pow2(x >> 1) << 1;
}

pawn_table::pawn_table() : sz_mb(0), count(0) {
	init();
}

pawn_table::pawn_table(const pawn_table& o) {
	entries = std::shared_ptr<pawn_entry[]>(new pawn_entry[count]());
	std::memcpy(entries.get(), o.entries.get(), count * sizeof(pawn_entry));
	sz_mb = o.sz_mb;
	count = o.count;
}

pawn_table& pawn_table::operator=(const pawn_table& o) {
	entries = std::shared_ptr<pawn_entry[]>(new pawn_entry[count]());
	std::memcpy(entries.get(), o.entries.get(), count * sizeof(pawn_entry));
	sz_mb = o.sz_mb;
	count = o.count;
	return *this;
}

void pawn_table::init() {
	sz_mb = 10 * 1024; // todo : input mb parameter
	count = 1024 * sz_mb / sizeof(pawn_entry);
	count = pow2(count);
	count = (count < 1024 ? 1024 : count);
	entries = std::shared_ptr<pawn_entry[]>(new pawn_entry[count]());
	clear();
}



void pawn_table::clear() {
	memset(entries.get(), 0, count * sizeof(pawn_entry));
}



pawn_entry* pawn_table::fetch(const position& p) const {
	U64 k = p.pawnkey();
	unsigned idx = k & (count - 1);
	if (entries[idx].key == k) {
		return &entries[idx];
	}
	else {
		std::memset(&entries[idx], 0, sizeof(pawn_entry));
		entries[idx].key = k;
		entries[idx].score = evaluate<white>(p, entries[idx]) - evaluate<black>(p, entries[idx]);
		return &entries[idx];
	}
}


template<Color c>
inline bool backward_pawn(const int& row, const int& col, const U64& pawns) {
	int left = col - 1 < Col::A ? -1 : col - 1;
	int right = col + 1 > Col::H ? -1 : col + 1;
	bool left_greater = false;
	bool right_greater = false;

	if (c == white) {
		if (left != -1) {
			int sq = -1;
			U64 left_pawns = bitboards::col[left] & pawns;
			while (left_pawns) {
				int tmp = bits::pop_lsb(left_pawns);
				if (tmp > sq) sq = tmp;
			}
			left_greater = sq > 0 && util::row(sq) > row;
		}

		if (right != -1) {
			int sq = -1;
			U64 right_pawns = bitboards::col[right] & pawns;
			bool no_right_pawns = (right_pawns == 0ULL);
			while (right_pawns) {
				int tmp = bits::pop_lsb(right_pawns);
				if (tmp > sq) sq = tmp;
			}
			right_greater = sq > 0 && util::row(sq) > row || no_right_pawns;
		}
	}
	else {
		if (left != -1) {
			int sq = 100;
			U64 left_pawns = bitboards::col[left] & pawns;
			bool no_left_pawns = (left_pawns == 0ULL);
			while (left_pawns) {
				int tmp = bits::pop_lsb(left_pawns);
				if (tmp < sq) sq = tmp;
			}
			left_greater = sq < 100 && util::row(sq) < row || no_left_pawns;
		}
		if (right != -1) {
			int sq = 100;
			U64 right_pawns = bitboards::col[right] & pawns;
			bool no_right_pawns = (right_pawns == 0ULL);
			while (right_pawns) {
				int tmp = bits::pop_lsb(right_pawns);
				if (tmp < sq) sq = tmp;
			}
			right_greater = sq < 100 && util::row(sq) < row || no_right_pawns;
		}
	}

	return (left == -1 && right_greater) ||
		(right == -1 && left_greater) ||
		(left_greater && right_greater);
}

// sq score scale factors by column
std::vector<float> pawn_scaling{ 0.86f, 0.90f, 0.95f, 1.00f, 1.00f, 0.95f, 0.90f, 0.86f };
std::vector<float> material_vals{ 100.0f, 300.0f, 315.0f, 480.0f, 910.0f };

template<Color c>
int16 evaluate(const position& p, pawn_entry& e) {

	Color them = Color(c ^ 1);

	U64 pawns = p.get_pieces<c, pawn>();
	U64 epawns = them == white ?
		p.get_pieces<white, pawn>() :
		p.get_pieces<black, pawn>();

	Square* sqs = p.squares_of<c, pawn>();

	Square ksq = p.king_square(c);

	int16 score = 0;
	U64 locked_bb = 0ULL;

	for (Square s = *sqs; s != no_square; s = *++sqs) {

		U64 fbb = bitboards::squares[s];
		U64 front = (c == white ? bitboards::squares[s + 8] : bitboards::squares[s - 8]);
		int row = util::row(s);
		int col = util::col(s);

		score += p.params.sq_score_scaling[pawn] * square_score<c>(pawn, Square(s));
		score += pawn_scaling[col] * material_vals[pawn];


		// Pawn attacks
		e.attacks[c] |= bitboards::pattks[c][s];

		// Track undefended pawns
		auto defend_mask = (c == white ? bitboards::pattks[black][s] : bitboards::pattks[white][s]);
		auto defenders = pawns & defend_mask;
		if (defenders == 0ULL) {
			score -= 1;
			e.undefended[c] |= fbb;
		}

		// King shelter
		if ((bitboards::kmask[ksq] & fbb)) {
			e.king[c] |= fbb;
		}

		// Track passed pawns
		U64 mask = bitboards::passpawn_mask[c][s] & epawns;
		if (mask == 0ULL) {
			e.passed[c] |= fbb;
			e.score += p.params.passed_pawn_bonus;
			e.weak_squares[c] |= front;
		}

		// Track isolated pawns
		U64 neighbors_bb = bitboards::neighbor_cols[col] & pawns;
		if (neighbors_bb == 0ULL) {
			e.isolated[c] |= fbb;
			score -= p.params.isolated_pawn_penalty;
			e.weak_squares[c] |= front;
		}

		// Track backward pawns
		if (backward_pawn<c>(row, col, pawns)) {
			e.backward[c] |= fbb;
			score -= p.params.backward_pawn_penalty;
			e.weak_squares[c] |= front;
		}

		// Sort pawns by square color
		U64 wsq = bitboards::colored_sqs[white] & fbb;
		if (wsq) e.light[c] |= fbb;
		U64 bsq = bitboards::colored_sqs[black] & fbb;
		if (bsq) e.dark[c] |= fbb;

		// Track doubled pawns
		U64 doubled = bitboards::col[col] & pawns;
		if (bits::more_than_one(doubled)) {
			e.doubled[c] |= doubled;
			if (e.isolated[c] & doubled)
				score -= 2 * p.params.doubled_pawn_penalty;
			else score -= p.params.doubled_pawn_penalty;
		}

		// Track pawns on semi-open files
		U64 column = bitboards::col[col];
		if ((column & epawns) == 0ULL) {
			e.semiopen[c] |= fbb;
			if ((fbb & e.backward[c])) 
				score -= 2 * p.params.backward_pawn_penalty;
			if ((fbb & e.isolated[c]))
				score -= 2 * p.params.isolated_pawn_penalty;
			if ((fbb && e.doubled[c]))
				score -= 2 * p.params.doubled_pawn_penalty;
			e.weak_squares[c] |= front;
		}

		// Track king/queen side pawn configurations
		if (util::col(s) <= Col::D) 
			e.qsidepawns[c] |= fbb;
		else 
			e.ksidepawns[c] |= fbb;

		// .. locked center pawns
		// count nb of center pawns while computing this too
		// e.g. french advanced, caro-kahn advanced, 4-pawns attack in KID etc.
		// favors flank attacks, knights, and small penalties for bishop
		if ((bitboards::squares[s] & bitboards::small_center_mask) != 0ULL) {
			Square front_sq = Square(c == white ? s + 8 : s - 8);
			if (util::on_board(front_sq)) {
				U64 fbb = bitboards::squares[front_sq];
				e.center_pawn_count++;
				if ((epawns & fbb)) {
					locked_bb |= fbb;
				}
			}
		}
		
		// pawn islands
		// pawn chain tips
		// pawn chain bases
		// pawn majorities
	}

	// note : evaluated 2x's when we only need to evaluate once
	// since it is the same per side - but the performance hit should be small
	if (bits::count(locked_bb) >= 2) {
		e.locked_center = true;
	}

	return score;
}
