
#include <vector>

#include "material.h"
#include "types.h"
#include "utils.h"
#include "bitboards.h"
#include "position.h"


//material_table mtable;


int16 evaluate(const position& p, material_entry& e);


inline size_t pow2(size_t x) {
	return x <= 2 ? x : pow2(x >> 1) >> 1;
}


material_table::material_table() : sz_mb(0), count(0) {
	init();
}

material_table::material_table(const material_table& o) {
	entries = std::shared_ptr<material_entry[]>(new material_entry[count]());
	std::memcpy(entries.get(), o.entries.get(), count * sizeof(material_entry));
	entries = o.entries;
	sz_mb = o.sz_mb;
	count = o.count;
}


material_table& material_table::operator=(const material_table& o) {
	entries = std::shared_ptr<material_entry[]>(new material_entry[count]());
	std::memcpy(entries.get(), o.entries.get(), count * sizeof(material_entry));
	sz_mb = o.sz_mb;
	count = o.count;
	return *this;
}


void material_table::init() {
	sz_mb = 50 * 1024;
	count = 1024 * sz_mb / sizeof(material_entry);
	if (count < 1024) count = 1024;
	entries = std::shared_ptr<material_entry[]>(new material_entry[count]());
}

void material_table::clear() {
	memset(entries.get(), 0, count * sizeof(material_entry));
}


material_entry* material_table::fetch(const position& p) const {
	U64 k = p.material_key();
	unsigned idx = k & (count - 1);
	if (entries[idx].key == k) {
		return &entries[idx];
	}
	else {
		entries[idx] = {};
		entries[idx].key = k;
		entries[idx].score = evaluate(p, entries[idx]);
		return &entries[idx];
	}
}



int16 evaluate(const position& p, material_entry& e) {

	std::vector<int> sign{ 1, -1 };
	std::vector<float> material_vals{ 0.0f, 300.0f, 315.0f, 480.0f, 910.0f };
	const std::vector<Piece> pieces{ knight, bishop, rook, queen }; // pawns handled in pawns.cpp

	// pawn count adjustments for the rook and knight
	// 1. the knight becomes less valuable as pawns dissapear
	// 2. the bishop/rook/queen become more valuable as pawns dissapear
	// 3. adjustment ~2 pts / pawn so that 16*2 = 32 max adjustment
	//{
	//	const float pawn_adjustment = 2.0;
	//	U64 wpawns = p.get_pieces<white, pawn>();
	//	U64 bpawns = p.get_pieces<black, pawn>();
	//	auto pawnCount = bits::count(wpawns) + bits::count(bpawns);
	//	int minor_pawn_adjust = pawn_adjustment * (16 - pawnCount);
	//	material_vals[knight] -= minor_pawn_adjust;
	//	//material_vals[bishop] += minor_pawn_adjust;
	//	material_vals[rook] += minor_pawn_adjust;
	//	//material_vals[queen] += minor_pawn_adjust;
	//}

	int16 score = 0;
	e.endgame = EndgameType::none;
	auto eg_pieces = std::vector< std::vector<int> >{
		{ 0, 0, 0, 0, 0 }, // p, n, b, r, q
		{ 0, 0, 0, 0, 0 } // p, n, b, r, q
	};
	auto total = std::vector<int>{ 0, 0 };

	for (Color c = white; c <= black; ++c) {
		for (const auto& piece : pieces) {
			int n = p.number_of(c, piece);
			e.number[piece] += n;
			score += sign[c] * n * material_vals[piece];
			total[c] += n;
			eg_pieces[c][piece] += n;
		}
	}
	auto total_eg = (total[white] + total[black]);
	// encoding endgame type if applicable
	// see types.h for enumeration of different endgame types
	if (total_eg == 0)
		e.endgame = EndgameType::KpK;

	else if (total_eg == 2 && eg_pieces[white][rook] == 1 && eg_pieces[black][rook] == 1)
		e.endgame = EndgameType::KrrK;

	else if (total_eg == 2 && eg_pieces[white][bishop] == 1 && eg_pieces[black][knight] == 1)
		e.endgame = EndgameType::KbnK;

	else if (total_eg == 2 && eg_pieces[white][knight] == 1 && eg_pieces[black][bishop] == 1)
		e.endgame = EndgameType::KnbK;

	else if (total_eg == 2 && eg_pieces[white][bishop] == 1 && eg_pieces[black][bishop] == 1)
		e.endgame = EndgameType::KbbK;

	else if (total_eg == 2 && eg_pieces[white][knight] == 1 && eg_pieces[black][knight] == 1)
		e.endgame = EndgameType::KnnK;

	else if (total_eg == 1 && (eg_pieces[white][bishop] == 1 || eg_pieces[black][bishop] == 1))
		e.endgame = EndgameType::KbK;

	else if (total_eg == 1 && (eg_pieces[white][knight] == 1 || eg_pieces[black][knight] == 1))
		e.endgame = EndgameType::KnK;

	else if (total_eg <= 2)
		e.endgame = EndgameType::Unknown;


	// endgame linear interpolation coefficient
	// computed from piece count - excludes pawns
	// endgame_coeff of 0 --> piece count = 14
	// endgame_coeff of 1 --> piece count = 2
	// coeff = -1/12*total + 7/6
	//e.endgame_coeff = std::min(-0.083333f * total + 1.16667f, 1.0f);

	return score;
}
