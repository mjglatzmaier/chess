
#include "magics.h"
#include "magicsrands.h"
#include "types.h"

namespace magics {
	namespace detail {

		struct table {
			U64 mask;
			U64 magic;
			U16 offset;
			U8 shift;

			inline unsigned entry(const U64& occ_) { return unsigned(magic * (mask & occ_) >> shift); }
		};

		std::vector<std::vector<U8>> ridx(64, std::vector<U8>(4096));
		std::vector<std::vector<U8>> bidx(64, std::vector<U8>(512));
		std::vector<U64> battks;
		std::vector<U64> rattks;

		table rtable[64];
		table btable[64];

	} // end namespace detail
}

U64 magics::next_magic(const unsigned int& bits, util::rand<unsigned int>& r) {
	U64 res = 0ULL;
	for (unsigned int i = 0; i < bits; ++i) 
		res |= (1ULL << (r.next() & 63));
	return res;
}

template<Piece p>
U64 magics::attacks(const Square& s, const U64& block) {
	std::vector< std::vector<int> > steps =
	{ {}, // pawn
		{}, // knight
		{-7, 7, -9, 9}, // bishop
		{-1, 1, -8, 8}  // rook
	};
	U64 bm = 0ULL;
	for (auto& step : steps[p]) {
		U64 tmp = 0ULL;
		int sqs = 1;

		while ((tmp & block) == 0ULL && sqs < 8) {
			int to = s + (sqs++) * step;
			if (util::on_board(to) &&
				util::col_dist(s, to) <= 7 &&
				util::row_dist(s, to) <= 7 &&
				((p == Piece::bishop && util::on_diagonal(s, to)) ||
					(p == Piece::rook && (util::same_row(s, to) || util::same_col(s, to))))) {
				tmp |= (1ULL << to);
			}
		}
		bm |= tmp;
	}
	return bm;
}

bool magics::load() {

	std::vector<U64> occupancy, atks, used;
	occupancy.reserve(4096); atks.reserve(4096); used.reserve(4096);
	occupancy.resize(4096); atks.resize(4096); used.resize(4096);
	util::rand<unsigned int> r;

	// 4096 is computed from counting the number of possible blockers for a rook/bishop at a given square.
	// E.g. the rook@A1 has 12 squares which can be blocked (A8,H1 have been removed)
	// in mathematica : sum[12!/(n!*(12-n)!),{n,1,12}] = 4095 .. similar computation for bishop@E4.
	detail::battks.reserve(1428); detail::battks.resize(1428);
	detail::rattks.reserve(4900); detail::rattks.resize(4900);

	// structs to provide shorthand indexing
	struct _attks {
		std::vector<U64>& operator[](const int& idx) { return (idx == 2 ? detail::battks : detail::rattks); }
	};
	struct _idx {
		std::vector<std::vector<U8>>& operator[](const int& idx) { return (idx == 2 ? detail::bidx : detail::ridx); }
	};
	_attks attack_arr;
	_idx indices;

	// these offsets tally the number of unique attack arrays
	// that exist for bishop/rook at the given square. E.g. all values
	// for the rook can be computed by 
	// value = empty_row1_sqs * empty_row2_sqs * empty_col1_sqs * empty_col2_sqs
	// rook@A1 has 49 unique attack patterns (but 4096 occupancies at A1)
	std::vector<std::vector<int>> offsets =
	{
		{}, // pawn
		{}, // knight
		{
7, 6, 10, 12, 12, 10, 6, 7,
6, 6, 10, 12, 12, 10, 6, 6,
10, 10, 40, 48, 48, 40, 10, 10,
12, 12, 48, 108, 108, 48, 12, 12,
12, 12, 48, 108, 108, 48, 12, 12,
10, 10, 40, 48, 48, 40, 10, 10,
6, 6, 10, 12, 12, 10, 6, 6,
7, 6, 10, 12, 12, 10, 6, 7
		},
		{
49, 42, 70, 84, 84, 70, 42, 49,
42, 36, 60, 72, 72, 60, 36, 42,
70, 60, 100, 120, 120, 100, 60, 70,
84, 72, 120, 144, 144, 120, 72, 84,
84, 72, 120, 144, 144, 120, 72, 84,
70, 60, 100, 120, 120, 100, 60, 70,
42, 36, 60, 72, 72, 60, 36, 42,
49, 42, 70, 84, 84, 70, 42, 49
		}
	};

	for (Piece p = Piece::bishop; p <= Piece::rook; ++p) {

		size_t bits = 6;

		for (Square s = Square::A1; s <= Square::H8; ++s) {

			U64 mask = (p == Piece::rook ? bitboards::rmask[s] : bitboards::bmask[s]);
			U8 shift = 64 - bits::count(mask);
			U64 b = 0ULL;
			int occ_size = 0;
			U64 magic = 0ULL;
			U64 filter = 0ULL;

			// enumerate all occupancy combinations of the bishop/rook mask
			do {
				occupancy[occ_size] = b;
				atks[occ_size++] = (p == Piece::bishop ? attacks<Piece::bishop>(s, b) :
					attacks<Piece::rook>(s, b));
				b = (b - mask) & mask;
			} while (b);

			int count = 0;
			//do {

			//  do {
			//    magic = next_magic(bits, r);
			//    filter = (magic * mask) & 0xFFFF000000000000;
			//  } while (bits::count(filter) < 7);

			//  used.assign(occ_size, 0);

			//  for (count = 0; count < occ_size; ++count) {
			//    unsigned int idx = unsigned(magic * (mask & occupancy[count]) >> shift);

			//    if ((used[idx] == 0ULL &&
			//      (used[idx] != occupancy[count] ||
			//        occupancy[count] == 0ULL))) used[idx] = occupancy[count];
			//    else break;
			//  }
			//} while (count != occ_size);

			// remove redundant occupancies 
			U16 offset = 0;
			magic = (p == bishop ? bishop_magics[s] : rook_magics[s]);

			for (int i = 0; i < s; ++i) offset += offsets[p][i];

			U64 stored[64][144] = { {} };

			// filter those occupancies which are redundant, the index : r_occ[s][idx] + offset
			for (int i = 0; i < occ_size; ++i) {
				unsigned int idx = unsigned(magic * (mask & occupancy[i]) >> shift);
				U64 atk = atks[i];

				int k = 0;
				while (k < 144) {
					U64 prev = stored[s][k];
					if (!prev && prev != atk) {
						stored[s][k] = atk;
						break;
					}
					else if (prev == atk) break;
					++k;
				}
				indices[p][s][idx] = k;

				int o = indices[p][s][idx] + offset; // total offset
				attack_arr[p][o] = atks[i];
				detail::table* tab = (p == Piece::bishop ? detail::btable : detail::rtable);
				tab[s].magic = magic;
				tab[s].mask = mask;
				tab[s].shift = shift;
				tab[s].offset = offset;

			}
			//std::cout << "s " << s <<  "occ size: " << occ_size << std::endl;
			//std::string eol = (s % 6 != 0 ? "), " : "),\n");
			//std::cout << "U64(" << magic << eol;
			//if (s == 63) std::cout << "==============================" << std::endl;
		}
	}
	return true;
}

namespace magics {
	template<> U64 attacks<Piece::rook>(const U64& occ, const Square& s) {
		using namespace detail;
		return rattks[ridx[s][rtable[s].entry(occ)] + rtable[s].offset];
	}

	template<> U64 attacks<Piece::bishop>(const U64& occ, const Square& s) {
		using namespace detail;
		return battks[bidx[s][btable[s].entry(occ)] + btable[s].offset];
	}
}
