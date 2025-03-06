
#pragma once

#include <vector>
#include <algorithm>

#include "position.h"
#include "evaluate.h"
#include "bits.h"
#include "types.h"
#include "bitboards.h"

namespace eval {

	template<Color c>
	inline bool has_opposition(const position& p, einfo& ei) {
		Square wks = p.king_square(white);
		Square bks = p.king_square(black);
		Color tmv = p.to_move();

		int cols = util::col_dist(wks, bks) - 1;
		int rows = util::row_dist(wks, bks) - 1;
		bool odd_rows = ((rows & 1) == 1);
		bool odd_cols = ((cols & 1) == 1);

		// distant opposition
		if (cols > 0 && rows > 0) {
			return (tmv != c && odd_rows && odd_cols);
		}
		// direct opposition
		else return (tmv != c && (odd_rows || odd_cols));
	}


	template<Color c>
	inline void get_pawn_majorities(const position& p, einfo& ei, std::vector<U64>& majorities) {
		U64 our_pawns = p.get_pieces<c, pawn>();
		U64 their_pawns = (c == white ? p.get_pieces<black, pawn>() :
			p.get_pieces<white, pawn>());

		majorities = { 0ULL, 0ULL, 0ULL, };

		// queenside majority
		U64 our_qs = our_pawns & bitboards::pawn_majority_masks[0];
		U64 their_qs = their_pawns & bitboards::pawn_majority_masks[0];
		if (our_qs != 0ULL && their_qs != 0) {
			if (bits::count(our_qs) > bits::count(their_qs)) {
				majorities[0] = our_qs;
			}
		}

		// central majority
		U64 our_c = our_pawns & bitboards::pawn_majority_masks[1];
		U64 their_c = their_pawns & bitboards::pawn_majority_masks[1];
		if (our_c != 0ULL && their_c != 0) {
			if (bits::count(our_c) > bits::count(their_c)) {
				majorities[1] = our_c;
			}
		}

		// kingside majority
		U64 our_ks = our_pawns & bitboards::pawn_majority_masks[2];
		U64 their_ks = their_pawns & bitboards::pawn_majority_masks[2];
		if (our_ks != 0ULL && their_ks != 0) {
			if (bits::count(our_ks) > bits::count(their_ks)) {
				majorities[2] = our_ks;
			}
		}
	}


	template<Color c>
	inline float eval_passed_kpk(const position& p, einfo& ei, const Square& f, const bool& has_opposition) {
		float score = 0;

		const float advanced_passed_pawn_bonus = 15;
		const float good_king_bonus = 5;

		Color them = Color(c ^ 1);

		Square ks = p.king_square(c);
		int row_ks = util::row(ks);
		int col_ks = util::col(ks);

		Square eks = p.king_square(them);
		int row_eks = util::row(eks);
		int col_eks = util::col(eks);

		int row = util::row(f);
		int col = util::col(f);

		Square in_front = Square(f + (c == white ? 8 : -8));

		U64 eks_bb = (bitboards::kmask[f] & bitboards::squares[eks]);
		U64 fks_bb = (bitboards::kmask[f] & bitboards::squares[ks]);

		bool e_control_next = eks_bb != 0ULL;
		bool f_control_next = fks_bb != 0ULL;

		bool f_king_infront = (c == white ? row_ks >= row : row_ks <= row);
		bool e_king_infront = (c == white ? row_eks > row : row_eks < row);

		// edge column draw
		if (col == Col::A || col == Col::H) {
			if (e_control_next) 
				return 0;
			if (col_eks == col && e_king_infront) 
				return 0;
		}

		// case 1. bad king position (enemy king blocking pawn)
		if (e_king_infront && !f_king_infront && e_control_next && !has_opposition)
			return 0;


		// case 2. we control front square and have opposition (winning)
		if (f_control_next && has_opposition)
			score += good_king_bonus;

		int dist = (c == white ? 7 - row : row);

		// case 3. we are behind the pawn
		//int f_min_dist = std::min(row_ks, col_ks);
		//int e_min_dist = std::min(row_eks, col_eks);
		bool inside_pawn_box = util::col_dist(eks, f) <= dist;

		int fk_dist = std::max(util::col_dist(ks, f), util::row_dist(ks, f));
		int ek_dist = std::max(util::col_dist(eks, f), util::row_dist(eks, f));
		bool too_far = fk_dist >= ek_dist;
		if (too_far && !f_king_infront && inside_pawn_box) 
			return 0;

		// bonus for being closer to queening
		switch (dist) {
		case 0: score += 7 * advanced_passed_pawn_bonus; break;
		case 1: score += 6 * advanced_passed_pawn_bonus; break;
		case 2: score += 5 * advanced_passed_pawn_bonus; break;
		case 3: score += 4 * advanced_passed_pawn_bonus; break;
		case 4: score += 3 * advanced_passed_pawn_bonus; break;
		case 5: score += 2 * advanced_passed_pawn_bonus; break;
		case 6: score += 1 * advanced_passed_pawn_bonus; break;
		}

		return score;
	}

	template<Color c>
	inline float eval_passed_krrk(const position& p, einfo& ei, const Square& f, const bool& has_opposition) {
		float score = 0;

		const float advanced_passed_pawn_bonus = 15;
		const float rook_behind_pawn_bonus = 8;
		const float good_king_bonus = 5;
		const float enemy_rook_behind_pawn = 4;
		const float lucena_pattern_bonus = 30;

		Color them = Color(c ^ 1);

		Square ks = p.king_square(c);
		int row_ks = util::row(ks);
		int col_ks = util::col(ks);

		Square eks = p.king_square(them);
		int row_eks = util::row(eks);
		int col_eks = util::col(eks);

		int row = util::row(f);
		int col = util::col(f);
		auto pawnOn7th = (c == white ? row == Row::r7 : row == Row::r2);
		auto kingOn5th = (c == white ? row_ks == Row::r5 : row_ks == Row::r4);
		auto kingOn6th = (c == white ? row_ks == Row::r6 : row_ks == Row::r3);
		auto kingOn7th = (c == white ? row_ks == Row::r7 : row_ks == Row::r2);
		auto kingOn8th = (c == white ? row_ks == Row::r8 : row_ks == Row::r1);

		Square frs = p.squares_of<c, rook>()[0];
		int col_fr = util::col(frs);
		int row_fr = util::row(frs);

		Square ers = (c == white ?
			p.squares_of<black, rook>()[0] :
			p.squares_of<white, rook>()[0]);
		int col_er = util::col(ers);
		int row_er = util::row(ers);


		Square in_front = Square(f + (c == white ? 8 : -8));

		U64 eks_bb = (bitboards::kmask[f] & bitboards::squares[eks]);
		U64 fks_bb = (bitboards::kmask[f] & bitboards::squares[ks]);

		bool e_control_next = eks_bb != 0ULL;
		bool f_control_next = fks_bb != 0ULL;

		bool f_king_infront = (c == white ? row_ks >= row : row_ks <= row);
		bool e_king_infront = (c == white ? row_eks > row : row_eks < row);


		// 1. Does our king control the next square
		//if (fks_bb != 0ULL && has_opposition)
		//	score += good_king_bonus;

		// 2. Is our rook behind the passed pawn
		bool fr_behind = (c == white ? row_fr < row : row_fr > row);
		if (fr_behind) {
			score += rook_behind_pawn_bonus;
			if (col_fr == col)
				score += rook_behind_pawn_bonus;
		}

		// 3. Inactive enemy king (farther from passer is better)
		auto kingPawnDist = std::max(util::row_dist(eks, f), util::col_dist(eks, f));
		score += kingPawnDist;

		// 4. Penalty for enemy rook behind our passer
		//if (col_er == col) {
		//	if (c == white && row_er < row)
		//		score -= enemy_rook_behind_pawn;
		//	if (c == black && row_er > row)
		//		score -= enemy_rook_behind_pawn;
		//}
		
		// 5. Lucena pattern
		//if (pawnOn7th && 
		//	(kingOn8th || kingOn7th || kingOn6th || kingOn5th) && 
		//	cutoff_enemy_king) {
		//	if ((c == white && row_fr == Row::r4) || (c == black && row_fr == Row::r5)) {
		//		score += lucena_pattern_bonus;
		//		if (kingOn7th) score += 5;
		//		if (kingOn6th) score += 35;
		//		if (kingOn5th) {
		//			score += 50;
		//			if (col == col_ks)
		//				score += 50;
		//		}
		//	}
		//}

		// 5. Philidor pattern

		// Bonus for being closer to queening
		int dist = (c == white ? 7 - row : row);
		switch (dist) {
		case 1: score += 6 * advanced_passed_pawn_bonus; break;
		case 2: score += 5 * advanced_passed_pawn_bonus; break;
		case 3: score += 4 * advanced_passed_pawn_bonus; break;
		case 4: score += 3 * advanced_passed_pawn_bonus; break;
		case 5: score += 2 * advanced_passed_pawn_bonus; break;
		case 6: score += 1 * advanced_passed_pawn_bonus; break;
		}

		return score;
	}


	template<Color c>
	inline float eval_passed_knbk(const position& p, einfo& ei, const Square& f, const bool& has_opposition) {
		float score = 0;

		const float advanced_passed_pawn_bonus = 2;
		const float good_king_bonus = 5;
		const float controls_front_square_bonus = 4;
		const float same_bishop_as_queen_sq_bonus = 2;
		const float blockade_penalty = 2;

		Color them = Color(c ^ 1);

		Square ks = p.king_square(c);
		int row_ks = util::row(ks);
		int col_ks = util::col(ks);

		Square eks = p.king_square(them);
		int row_eks = util::row(eks);
		int col_eks = util::col(eks);

		int row = util::row(f);
		int col = util::col(f);
		Square frontSquare = Square(c == white ? f + 8 : f - 8);
		Square bishopSquare = (c == white ?
			p.squares_of<white, bishop>()[0] :
			p.squares_of<black, bishop>()[0]);
		auto hasBishop = bishopSquare != Square::no_square;

		// 1. Do our minors control the next square
		//auto control = p.attackers_of2(frontSquare, c);
		//score += bits::count(control);

		// 2. Inactive enemy king (farther from passer is better)
		auto kingPawnDist = std::max(util::row_dist(eks, f), util::col_dist(eks, f));
		score += kingPawnDist;
		//std::cout << " .. PASSED PAWN EVAL .. " << std::endl;
		//bits::print(bitboards::squares[f]);
		//std::cout << "DBG: kingPawnDist = " << kingPawnDist << std::endl;

		// 3. If we have the bishop, can we attack the queening square?
		if (hasBishop) {
			auto bishopSquareBB = bitboards::squares[bishopSquare];
			auto lightSqBishop = ((bishopSquareBB & bitboards::colored_sqs[white]) != 0ULL);
			auto queenSquare = bitboards::squares[(c == white ? col + 56 : col)];
			auto lightQueenSq = (bitboards::colored_sqs[white] & queenSquare) != 0ULL;
			if ((lightQueenSq && lightSqBishop) || (!lightSqBishop && !lightQueenSq)) {
				//std::cout << "DBG: queens on bishop color bonus" << std::endl;
				score += same_bishop_as_queen_sq_bonus;
			}

			// 4. If knight blockades pawn and we cannot attack the knight
			auto frontSquareLight = ((bitboards::squares[frontSquare] & bitboards::colored_sqs[white]) != 0ULL);
			Square knightSquare = (c == white ? p.squares_of<black, knight>()[0] : p.squares_of<white, knight>()[0]);
			if (knightSquare == frontSquare) {
				//std::cout << "DBG: knight blockade penalty 1" << std::endl;
				score -= blockade_penalty;
				if ((frontSquareLight && !lightSqBishop) || (!frontSquareLight && lightSqBishop)) {
					score -= blockade_penalty;
					//std::cout << "DBG: knight blockade penalty 2" << std::endl;
					if (kingPawnDist <= 2) {
						score -= blockade_penalty; // very drawish
						//std::cout << "DBG: knight blockade penalty 3" << std::endl;
					}
				}
			}

			// 5. Inactive enemy knight
			auto knightPawnDist = std::max(util::row_dist(knightSquare, f), util::col_dist(knightSquare, f));
			score += knightPawnDist;
			//std::cout << "DBG: knight inactivity bonus " << (0.5f * knightPawnDist) << std::endl;
		}

		// Bonus for being closer to queening
		int dist = (c == white ? 7 - row : row);
		switch (dist) {
		case 1: score += 6 * advanced_passed_pawn_bonus; break;
		case 2: score += 5 * advanced_passed_pawn_bonus; break;
		case 3: score += 4 * advanced_passed_pawn_bonus; break;
		case 4: score += 3 * advanced_passed_pawn_bonus; break;
		case 5: score += 2 * advanced_passed_pawn_bonus; break;
		case 6: score += 1 * advanced_passed_pawn_bonus; break;
		}

		return score;
	}

	inline bool is_fence(const position& p, einfo& ei) {

		if (ei.pe->semiopen[black] != 0ULL) {
			//bits::print(ei.pe->semiopen[black]);
			//std::cout << "..dbg semi-open file in pawn-entry for black, not a fence pos" << std::endl;
			return false;
		}

		U64 enemies = p.get_pieces<black, pawn>();
		if (enemies == 0ULL) return false;

		U64 attacks = ei.pe->attacks[black];
		U64 friends = p.get_pieces<white, pawn>();
		U64 wking = p.get_pieces<white, king>();
		U64 bking = p.get_pieces<black, king>();

		std::vector<int> delta{ -1, 1 };
		std::vector<Square> blocked;

		U64 dbg_bb = 0ULL;

		while (friends) {
			Square start = Square(bits::pop_lsb(friends));

			Square occ = Square(start + 8);

			if (!(bitboards::squares[occ] & enemies) &&
				!(bitboards::squares[occ] & bking)) {
				//std::cout << "..dbg detected pawn-chain is not \"locked\", not a fence position" << std::endl;
				return false;
			}

			blocked.push_back(start);
			dbg_bb |= bitboards::squares[start];

			for (const auto& d : delta) {
				Square n = Square(start + d);

				if (!util::on_board(n)) continue;

				if (bitboards::squares[n] & attacks) {
					blocked.push_back(n);
					dbg_bb |= bitboards::squares[n];
				}
			}
		}

		if (blocked.size() <= 0) return false;

		Col start_col = Col(util::col(blocked[0]));
		bool connected = (start_col == Col::A) || (start_col == Col::B);

		if (!connected) return false;

		U64 side = 0ULL;

		std::sort(std::begin(blocked), std::end(blocked),
			[](const Square& s1, const Square& s2) -> bool { return util::col(s1) < util::col(s2);  }
		);

		for (int i = 1; i < blocked.size(); ++i) {
			Square curr = blocked[i];
			Square prev = blocked[i - 1];

			connected = (
				(abs(curr - prev) == 1) ||
				(abs(curr - prev) == 8) ||
				(abs(curr - prev) == 7) ||
				(abs(curr - prev) == 9));


			if (!connected) {
				//std::cout << "dbg: breaking @ " << curr << " " << prev << std::endl;
				break;
			}

			side |= util::squares_behind(bitboards::col[util::col(prev)], white, prev);
			side |= util::squares_behind(bitboards::col[util::col(curr)], white, curr);
		}

		bool wk_in = (side & wking);
		bool bk_in = (side & bking);

		if (!(wk_in && !bk_in)) return false;

		/*
		std::cout << "fence dbg: " << std::endl;
		for (const auto& b : blocked) {
			std::cout << SanSquares[b] << " ";
		}
		std::cout << std::endl;

		bits::print(dbg_bb);
		bits::print(side);
		std::cout << "white king inside: " << wk_in << std::endl;
		std::cout << "black king inside: " << bk_in << std::endl;
		*/

		return connected && wk_in && !bk_in;
	}

}
