
#pragma once

#ifndef EVALUATE_H
#define EVALUATE_H

#include "types.h"
#include "utils.h"
#include "pawns.h"
#include "material.h"
#include "parameter.h"
#include "utils.h"
#include "threads.h"

struct endgame_info {
	bool evaluated_fence;
	bool is_fence;
};

struct einfo {
	pawn_entry* pe;
	material_entry* me;
	endgame_info endgame;
	U64 pawn_holes[2];
	U64 all_pieces;
	U64 pieces[2];
	U64 weak_pawns[2];
	U64 empty;
	U64 kmask[2];
	U64 kattk_points[2][5];
	U64 piece_attacks[2][5];
	bool bishop_colors[2][2];
	U64 central_pawns[2];
	U64 queen_sqs[2];
	U64 white_pawns[2];
	U64 black_pawns[2];

	bool closed_center;
	unsigned kattackers[2][5];
};


namespace eval {

	float evaluate(const position& p, const Searchthread& t, const float& lazy_margin);

}

namespace eval {
	extern parameters Parameters;
}

#endif
