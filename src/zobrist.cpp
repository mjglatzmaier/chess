
#include "zobrist.h"
#include "bits.h"
#include "zobristrands.h"

namespace zobrist {
	U64 piece_rands[Square::squares][2][pieces];
	U64 castle_rands[2][16];
	U64 ep_rands[8];
	U64 stm_rands[2];
	U64 move50_rands[512];
	U64 hmv_rands[512];
}


U64 zobrist::gen(const unsigned int& bits, util::rand<unsigned int>& r) {
	U64 res = 0ULL;
	for (unsigned int i = 0; i < bits; ++i) res |= (1ULL << (r.next() & 63));
	return res;
}

const unsigned _bits = 25;

bool zobrist::load() {

	const unsigned int N = 1835;
	std::vector<U64> rands;
	util::rand<unsigned int> R;

	//bool ok = true;
	//unsigned count = 0;
	//while (rands.size() < N) {

	//  U64 rand = R.next(); // gen(bits, R);
	//  
	//  // filter zobrist keys to be unique
	//  // and favor smaller values (bits distributed near lower half)
	//  if (bits::count(rand) > 4) continue;
	//  
	//  U64 filter_low = (rand >> 20); // 0x000000FFFFFFFF;
	//  if (bits::count(filter_low) >= 3) continue;

	//  //U64 filter_high = (rand << 20); // 0xFFFFFFFF;
	//  //if (bits::count(filter_high) <= 3) continue;

	//  for (unsigned i = 0; i < rands.size(); ++i) {
	//    ok = (rand != rands[i]);
	//    if (!ok) break;
	//  }
	//  
	//  if (ok) {
	//    count = ((count + 1) % 6 == 0 ? 0 : count + 1);
	//    std::string eol((count != 0 ? "), " : "),\n"));
	//    std::cout << "ULL(" << rand << eol;
	//    rands.emplace_back(rand);
	//  }

	//}

	// pieces
	unsigned int idx = 0;
	for (Square sq = A1; sq <= H8; ++sq) {
		for (Color c = white; c <= black; ++c) {
			for (Piece p = pawn; p <= king; ++p, ++idx) {
				piece_rands[sq][c][p] = zobrist_rands[idx]; // gen(bits, R);//gen.next();
			}
		}
	}


	// castle rights
	for (Color c = white; c <= black; ++c) {
		for (int bit = 0; bit < 16; ++bit, ++idx) {
			castle_rands[c][bit] = zobrist_rands[idx]; // gen(bits, R);// .next();
		}
	}


	// ep
	for (int col = 0; col < 8; ++col, ++idx) {
		ep_rands[col] = zobrist_rands[idx]; // gen(bits, R); // gen.next();
	}

	// stm
	stm_rands[white] = zobrist_rands[idx++]; // gen(bits, R); // .next();
	stm_rands[black] = zobrist_rands[idx++]; // gen(bits, R); // .next();


	for (int m = 0; m < 512; ++m, idx += 2) {
		move50_rands[m] = zobrist_rands[idx]; // gen(bits, R); // .next();
		hmv_rands[m] = zobrist_rands[idx + 1]; // gen(bits, R);// .next();
	}

	return true;
}


U64 zobrist::piece(const Square& s, const Color& c, const Piece& p) {
	return piece_rands[s][c][p];
}

U64 zobrist::castle(const Color& c, const U16& bit) {
	return castle_rands[c][bit];
}

U64 zobrist::ep(const U8& column) {
	return ep_rands[column];
}

U64 zobrist::stm(const Color& c) {
	return stm_rands[c];
}

U64 zobrist::mv50(const U16& count) {
	return move50_rands[count];
}

U64 zobrist::hmvs(const U16& count) {
	return hmv_rands[count];
}
