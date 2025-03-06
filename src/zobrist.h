
#pragma once

#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "types.h"
#include "utils.h"

namespace zobrist {
	bool load();
	U64 piece(const Square& s, const Color& c, const Piece& p);
	U64 castle(const Color& c, const U16& bit);
	U64 ep(const U8& column);
	U64 stm(const Color& c);
	U64 mv50(const U16& count);
	U64 hmvs(const U16& count);
	U64 gen(const unsigned int& bits, util::rand<unsigned int>& r);

	extern U64 piece_rands[Square::squares][2][pieces];
	extern U64 castle_rands[2][16];
	extern U64 ep_rands[8];
	extern U64 stm_rands[2];
	extern U64 move50_rands[512];
	extern U64 hmv_rands[512];
}

#endif
