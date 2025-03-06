
#include "bitboards.h"
#include "bits.h"

namespace bitboards {
	U64 row[8];
	U64 col[8];
	U64 bmask[64]; // bishop mask (outer board edges are trimmed)
	U64 pawnmask[2]; // 2nd - 6th rank mask for pawns (to exclude promotion candidates)
	U64 pattks[2][64];
	U64 pawnmaskleft[2];
	U64 pawnmaskright[2];
	U64 nmask[64]; // step attacks for the knight
	U64 kmask[64]; // step attacks for the king
	U64 kchecks[5][64]; // check mask for king
	U64 kzone[64]; // large zone around king (for eval)
	U64 kflanks[8]; // 3 rows of squares (including king square) for pawn cover detection
	U64 rmask[64]; // rook mask (outer board edges are trimmed)
	U64 kpawnstorm[2][2]; // to detect enemy pawn storms against our king
	U64 battks[64];
	U64 rattks[64];
	U64 small_center_mask;
	U64 big_center_mask;
	U64 squares[64];
	U64 diagonals[64];
	U64 between[64][64];
	U64 passpawn_mask[2][64];
	U64 neighbor_cols[8];
	U64 colored_sqs[2];
	U64 front_region[2][64];
	U64 pawn_majority_masks[3]; // 1 for each flank
	unsigned reductions[2][2][64][64]; // max-depth = 64
	U64 edges;
	U64 corners;
}

void bitboards::load() {

	std::vector<std::vector<int>> steps =
	{
		{ }, // pawn 
		{ 10, -6, -10, 6, 17, 15, -15, -17}, // knight
		{ -7, -9, 7, 9 }, // bishop
		{ -1, 1, 8, -8 }, // rook
		{ }, // queen
		{ -1, 1, 8, -8, -9, -7, 9, 7 }  // king
	};

	// king zone steps
	std::vector<int> zsteps = {
		-1, 1, 8, -8, -9, -7, 9, 7, // normal kmask
		-2, 2, -2 + 8, -2 - 8, 2 + 8, 2 - 8, -2 - 16, -2 + 16, 2 - 16, 2 + 16,
		-16, 16, -16 - 1, -16 + 1, 16 + 1, 16 - 1
	};

	for (Square s = A1; s <= H8; ++s) squares[s] = (1ULL << s);

	// row/col masks
	for (Row r = r1; r <= r8; ++r) {
		U64 rw = 0ULL; U64 cl = 0ULL;
		for (Col c = A; c <= H; ++c) {
			cl |= squares[c * 8 + r];
			rw |= squares[r * 8 + c];
		}
		row[r] = rw;
		col[r] = cl;
	}


	// pawn majority masks
	pawn_majority_masks[0] = col[Col::A] | col[Col::B] | col[Col::C];
	pawn_majority_masks[1] = col[Col::D] | col[Col::E];
	pawn_majority_masks[2] = col[Col::F] | col[Col::G] | col[Col::H];

	// helpful definitions for board corners/edges
	edges = row[r1] | col[A] | row[r8] | col[H];
	corners = squares[A1] | squares[H1] | squares[H8] | squares[A8];

	// pawn masks for captures/promotions
	pawnmask[white] = row[r2] | row[r3] | row[r4] | row[r5] | row[r6];
	pawnmask[black] = row[r3] | row[r4] | row[r5] | row[r6] | row[r7];
	for (Color color = white; color <= black; ++color) {
		pawnmaskleft[color] = 0ULL;
		pawnmaskright[color] = 0ULL;
		for (int r = (color == white ? 1 : 2); r <= (color == white ? 5 : 6); ++r) {
			for (int c = 0; c <= 6; ++c) pawnmaskleft[color] |= squares[r * 8 + c];
			for (int c = 1; c <= 7; ++c) pawnmaskright[color] |= squares[r * 8 + c];
		}
	}

	// central control masks
	big_center_mask = squares[C3] | squares[D3] | squares[E3] | squares[F3] |
		squares[C4] | squares[D4] | squares[E4] | squares[F4] |
		squares[C5] | squares[D5] | squares[E5] | squares[F5] |
		squares[C6] | squares[D6] | squares[E6] | squares[F6];

	small_center_mask = squares[C4] | squares[C5] | squares[D4] | squares[D5] | squares[E4] | squares[E5];

	// king flank masks
	U64 roi = ~(row[0] | row[7]);
	for (Col c = Col::A; c <= Col::H; ++c) {
		kflanks[c] = 0ULL;

		int lidx = (c - 1 < 0 ? 0 : c - 1);
		int ridx = (c + 1 > Col::H ? Col::H : c + 1);

		U64 mask = c <= Col::C || c >= Col::F ?
			(col[lidx] | col[c] | col[ridx])
			: (col[lidx - 1] | col[lidx] | col[c] | col[ridx] | col[ridx + 1]);

		kflanks[c] |= roi & mask;

	}

	for (Square s = A1; s <= H8; ++s) {

		if ((util::row(s) % 2 == 0) && (s % 2 == 0)) 
			colored_sqs[black] |= squares[s];
		else if ((util::row(s) % 2) != 0 && (s % 2 != 0)) 
			colored_sqs[black] |= squares[s];
		else 
			colored_sqs[white] |= squares[s];


		// search reduction array
		// index assignment [pv_node][improving][depth][move count]
		for (int sd = 0; sd < 64; ++sd) {
			for (int mc = 0; mc < 64; ++mc) {
				// pv nodes
				double small_r = log(double(sd + 1)) * log(double(mc + 1)) / 2.0;
				double big_r = 0.25 + log(double(sd + 1)) * log(double(mc + 1)) / 1.5;

				// pv-nodes
				reductions[1][0][sd][mc] = int(big_r >= 1.0 ? big_r + 0.5 : 0);
				reductions[1][1][sd][mc] = int(small_r >= 1.0 ? small_r + 0.5 : 0);

				// non-pv nodes
				reductions[0][0][sd][mc] = reductions[1][0][sd][mc] + 1;
				reductions[0][1][sd][mc] = reductions[1][1][sd][mc] + 1;
			}
		}

		// knight step attacks
		U64 bm = 0ULL;
		for (auto& step : steps[Piece::knight]) {
			int to = s + step;
			if (util::on_board(to) &&
				util::col_dist(s, to) <= 2 &&
				util::row_dist(s, to) <= 2) bm |= squares[to];
		}
		nmask[s] = bm;

		// king step attacks
		bm = 0ULL;
		for (auto& step : steps[Piece::king]) {
			int to = s + step;
			if (util::on_board(to) &&
				util::col_dist(s, to) <= 1 &&
				util::row_dist(s, to) <= 1) bm |= squares[to];
		}
		kmask[s] = bm;

		// king zone bitboard (for eval)
		bm = 0ULL;
		for (auto& step : zsteps) {
			int to = s + step;
			if (util::on_board(to) &&
				util::col_dist(s, to) <= 2 &&
				util::row_dist(s, to) <= 2) bm |= squares[to];
		}
		kzone[s] = bm;

		// pawn attack masks for each color
		int pawn_steps[2][2] = { {9, 7}, {-7, -9} };
		for (Color c = white; c <= black; ++c) {
			pattks[c][s] = 0ULL;
			for (auto& step : pawn_steps[c]) {
				int to = s + step;
				if (util::on_board(to) &&
					util::row_dist(s, to) < 2 &&
					util::col_dist(s, to) < 2) {
					pattks[c][s] |= squares[to];
				}
			}
		}

		// front region for each square
		for (Color c = white; c <= black; ++c) {
			front_region[c][s] = 0ULL;
			if (c == white)
				for (auto r = util::row(s) + 1; r <= Row::r8; ++r) 
					front_region[c][s] |= bitboards::row[r];
			else 
				for (auto r = util::row(s) - 1; r >= Row::r1; --r)
					front_region[c][s] |= bitboards::row[r];
		}

		// between bitboard
		for (Square s2 = A1; s2 <= H8; ++s2) {
			if (s != s2) {
				U64 btwn = 0ULL;
				int delta = 0;

				if (util::col_dist(s, s2) == 0) delta = (s < s2 ? 8 : -8);
				else if (util::row_dist(s, s2) == 0) delta = (s < s2 ? 1 : -1);
				else if (util::on_diagonal(s, s2)) {
					if (s < s2 && util::col(s) < util::col(s2)) delta = 9;
					else if (s < s2 && util::col(s) > util::col(s2)) delta = 7;
					else if (s > s2 && util::col(s) < util::col(s2)) delta = -7;
					else delta = -9;
				}

				if (delta != 0) {
					int iter = 0;
					int sq = 0;
					do {
						sq = s + iter * delta;
						btwn |= squares[sq];
						iter++;
					} while (sq != s2);
				}
				between[s][s2] = btwn;
			}
		}


		// passed pawn masks
		U64 roi = ~(row[0] | row[7]);
		if (squares[s] & roi) {

			neighbor_cols[util::col(s)] = 0ULL;

			for (Color c = white; c <= black; ++c) {
				passpawn_mask[c][s] = 0ULL;

				U64 neighbors =
					(kmask[s] & row[util::row(s)]) | squares[s];

				while (neighbors) {
					int sq = bits::pop_lsb(neighbors);
					passpawn_mask[c][s] |=
						util::squares_infront(col[util::col(sq)], c, sq);
					if (s != sq)
						neighbor_cols[util::col(s)] |= col[util::col(sq)];
				}
			}
		}

		// bishop diagonal masks (outer bits trimmed)
		bm = 0ULL;
		U64 trim = 0ULL;
		for (auto& step : steps[Piece::bishop]) {
			int j = 0;
			while (true) {
				int to = s + (j++) * step;
				if (util::on_board(to) && util::on_diagonal(s, to)) bm |= squares[to];
				else break;
			}
		}
		trim = squares[s] | (bm & edges);
		battks[s] = bm;
		bm ^= trim;
		bmask[s] = bm;

		// rook masks (outer-bits trimmed)
		bm = (row[util::row(s)] | col[util::col(s)]);
		rattks[s] = bm;
		trim = squares[s] | squares[8 * util::row(s)] | squares[8 * util::row(s) + 7] |
			squares[util::col(s)] | squares[util::col(s) + 56];
		bm ^= trim;
		rmask[s] = bm;


		// king check mask
		kchecks[knight][s] = nmask[s];
		kchecks[bishop][s] = battks[s];
		kchecks[rook][s] = rattks[s];
		kchecks[queen][s] = battks[s] | rattks[s];
	}

	// King pawn storm detection
	kpawnstorm[white][0] = between[Square::F2][Square::F5] |
		between[Square::G2][Square::G5] |
		between[Square::H2][Square::H5];
	kpawnstorm[white][1] = 
		between[Square::A2][Square::A5] |
		between[Square::B2][Square::B5] |
		between[Square::C2][Square::C5];

	kpawnstorm[black][0] = between[Square::F7][Square::F4] |
		between[Square::G7][Square::G4] |
		between[Square::H7][Square::H4];
	kpawnstorm[black][1] =
		between[Square::A7][Square::A4] |
		between[Square::B7][Square::B4] |
		between[Square::C7][Square::C4];
}

