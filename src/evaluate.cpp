
#include <mutex>
#include <thread>

#include "evaluate.h"
#include "squares.h"
#include "magics.h"
#include "endgame.h"
#include "position.h"

namespace eval {
	std::mutex mtx;
	parameters Parameters;
}

namespace {


	float do_eval(const position& p);

	template<Color c> float eval_pawns(const position& p, einfo& ei);
	template<Color c> float eval_knights(const position& p, einfo& ei);
	template<Color c> float eval_bishops(const position& p, einfo& ei);
	template<Color c> float eval_rooks(const position& p, einfo& ei);
	template<Color c> float eval_queens(const position& p, einfo& ei);
	template<Color c> float eval_king(const position& p, einfo& ei);
	template<Color c> float eval_space(const position& p, einfo& ei);
	template<Color c> float eval_center(const position& p, einfo& ei);
	template<Color c> float eval_color(const position& p, einfo& ei);
	template<Color c> float eval_threats(const position& p, einfo& ei);
	template<Color c> float eval_pawn_levers(const position& p, einfo& ei);
	template<Color c> float eval_passed_pawns(const position& p, einfo& ei);
	template<Color c> float eval_flank_attack(const position& p, einfo& ei);
	template<Color c> float eval_weak_squares(const position& p, einfo& ei);
	template<Color c> float eval_kpk(const position& p, einfo& ei);
	template<Color c> float eval_krrk(const position& p, einfo& ei);
	template<Color c> float eval_knbk(const position& p, einfo& ei);
	/*
	template<Color c> float eval_kbrk(const position& p, info& ei);
	template<Color c> float eval_knnk(const position& p, info& ei);
	template<Color c> float eval_kbbk(const position& p, info& ei);
	template<Color c> float eval_kqqk(const position& p, info& ei);
	template<Color c> float eval_knqk(const position& p, info& ei);
	template<Color c> float eval_kbqk(const position& p, info& ei);
	template<Color c> float eval_krnk(const position& p, info& ei);
	template<Color c> float eval_krqk(const position& p, info& ei);
	template<Color c> float eval_kings(const position& p, info& ei);
	template<Color c> float eval_passers(const position& p, info& ei);
	*/

	/*evaluation helpers*/
	template<Color c> bool trapped_rook(const position& p, einfo& ei, const Square& rs);


	/*TODO: Re-evaluate these..*/
	inline float knight_mobility(const unsigned& n) {
		return -50.0f * exp(-float(n) * 0.5f) + 20.0f;
	}

	inline float bishop_mobility(const unsigned& n) {
		return -50.0f * exp(-float(n) * 0.5f) + 20.0f;
	}

	inline float rook_mobility(const unsigned& n) {
		return 0.0f + 1.11f * log(n + 1); // max of ~20
	}

	inline float queen_mobility(const unsigned& n) {
		return 0.0f + 2.22f * log(n + 1); // max of ~20
	}

	float do_eval(const position& p, const Searchthread& t, const float& lazy_margin) {


		float score = 0;
		einfo ei = {};
		memset(&ei, 0, sizeof(einfo));

		ei.pe = t.pawnTable.fetch(p);
		ei.me = t.materialTable.fetch(p);

		ei.all_pieces = p.all_pieces();
		ei.empty = ~p.all_pieces();
		ei.pieces[white] = p.get_pieces<white>();
		ei.pieces[black] = p.get_pieces<black>();
		ei.weak_pawns[white] = ei.pe->doubled[white] | ei.pe->isolated[white] | ei.pe->backward[white] | ei.pe->undefended[white];
		ei.weak_pawns[black] = ei.pe->doubled[black] | ei.pe->isolated[black] | ei.pe->backward[black] | ei.pe->undefended[black];
		ei.kmask[white] = bitboards::kmask[p.king_square(white)];
		ei.kmask[black] = bitboards::kmask[p.king_square(black)];
		ei.central_pawns[white] = p.get_pieces<white, pawn>() & bitboards::big_center_mask;
		ei.central_pawns[black] = p.get_pieces<black, pawn>() & bitboards::big_center_mask;
		ei.queen_sqs[white] = p.get_pieces<white, queen>();
		ei.queen_sqs[black] = p.get_pieces<black, queen>();
		ei.pawn_holes[white] = (ei.pe->backward[white] != 0ULL ? ei.pe->backward[white] << 8 : 0ULL);
		ei.pawn_holes[black] = (ei.pe->backward[black] != 0ULL ? ei.pe->backward[black] >> 8 : 0ULL);

		score += ei.pe->score;
		score += ei.me->score;

		// Return early if eval is above the lazy margin
		if (lazy_margin > 0 && !ei.me->is_endgame() && abs(score) >= lazy_margin)
			return (p.to_move() == white ? score : -score) + p.params.tempo;

		// Specialized handling for endgame types
		if (ei.me->is_endgame()) {
			EndgameType t = ei.me->endgame;
			auto noPawns = (p.get_pieces<white, pawn>() | p.get_pieces<black, pawn>()) == 0ULL;
			switch (t) {
			case KpK:
				if (noPawns)
					return Score::draw;
				score += (eval_kpk<white>(p, ei) - eval_kpk<black>(p, ei));
				break;
			case KrrK:
				// Not necessarily drawn if no pawns...
				//score += (eval_krrk<white>(p, ei) - eval_krrk<black>(p, ei));
				break;
			case KnbK:
			case KbnK:
			case KnnK:
			case KbbK:
			case KnK:
			case KbK:
				if (noPawns)
					return Score::draw;
				break;
			}
		}

		score += (eval_pawns<white>(p, ei) - eval_pawns<black>(p, ei));
		//std::cout << "Score.pawnEval=" << score << std::endl;
		score += (eval_knights<white>(p, ei) - eval_knights<black>(p, ei));
		//std::cout << "Score.knightEval=" << score << std::endl;
		score += (eval_bishops<white>(p, ei) - eval_bishops<black>(p, ei));
		//std::cout << "Score.bishopEval=" << score << std::endl;
		score += (eval_rooks<white>(p, ei) - eval_rooks<black>(p, ei));
		//std::cout << "Score.rookEval=" << score << std::endl;
		score += (eval_queens<white>(p, ei) - eval_queens<black>(p, ei));
		//std::cout << "Score.queenEval=" << score << std::endl;
		score += (eval_king<white>(p, ei) - eval_king<black>(p, ei));
		//std::cout << "Score.kingEval=" << score << std::endl;
		score += (eval_passed_pawns<white>(p, ei) - eval_passed_pawns<black>(p, ei));
		//std::cout << "Score.passedPawnEval=" << score << std::endl;

		if (lazy_margin > 0 && !ei.me->is_endgame() && abs(score) >= lazy_margin)
			return (p.to_move() == white ? score : -score) + p.params.tempo;

		//score += (eval_weak_squares<white>(p, ei) - eval_weak_squares<black>(p, ei));
		//std::cout << "Score.weakSquareEval=" << score << std::endl;
		score += (eval_threats<white>(p, ei) - eval_threats<black>(p, ei));
		//std::cout << "Score.threatEval=" << score << std::endl;
		score += (eval_space<white>(p, ei) - eval_space<black>(p, ei));
		//std::cout << "Score.spaceEval=" << score << std::endl;

		return (p.to_move() == white ? score : -score) + p.params.tempo;
	}


	template<Color c> float eval_weak_squares(const position& p, einfo& ei) {
		float score = 0;
		Color them = Color(c ^ 1);
		auto weakSquares = ei.pe->weak_squares[them];
		auto undefended = weakSquares & (~ei.pe->attacks[them]);

		//std::cout << " \n======Weak squares for c" << c << "========" << std::endl;
		//bits::print(weakSquares);
		//bits::print(undefended);

		// Bonus for attacking weak squares with pawns
		auto pawnAttks = ei.pe->attacks[c];
		auto attacked = weakSquares & pawnAttks;
		if (attacked)
			score += p.params.square_attks[pawn];

		// Knights
		auto kOcc = (c == black ? p.get_pieces<black, knight>() : p.get_pieces<white, knight>()) & undefended;
		auto knightAttks = ei.piece_attacks[c][knight] & weakSquares;
		if (knightAttks)
			score += p.params.square_attks[knight];
		if (kOcc)
			score += 2;

		// Bishops
		auto bOcc = (c == black ? p.get_pieces<black, bishop>() : p.get_pieces<white, bishop>()) & undefended;
		auto bAttks = ei.piece_attacks[c][bishop] & weakSquares;
		if (bAttks)
			score += p.params.square_attks[bishop];
		if (bOcc)
			score += 2;

		// Rook attacks
		auto rAttks = ei.piece_attacks[c][rook] & weakSquares;
		if (rAttks)
			score += p.params.square_attks[rook];

		// Queen attacks
		auto qAttks = ei.piece_attacks[c][queen] & weakSquares;
		if (qAttks)
			score += p.params.square_attks[queen];

		//while (weakSquares) {
		//	auto s = bits::pop_lsb(weakSquares);
		//	auto attks = p.attackers_of2(Square(s), c);
		//	if (attks)
		//		score += 1; //0.5f * bits::count(attks);
		//}

		//std::cout << "Score.WeakSquares=" << score << std::endl;
		return score;
	}

	template<Color c> float eval_pawns(const position& p, einfo& ei) {


		//std::cout << " \n======eval_pawns for c" << c << "========" << std::endl;
		float score = 0;
		// pawn harassment of enemy king
		auto pawnAttacks = ei.pe->attacks[c];
		U64 kattks = pawnAttacks & ei.kmask[c^1];
		if (kattks) {
			ei.kattackers[c][pawn]++; // kattackers of "other" king
			ei.kattk_points[c][pawn] |= kattks; // attack points of "other" king
			score += p.params.pawn_king[std::min(2, bits::count(kattks))];
		}

		// pawn chain bases/undefended pawns
		auto undefended = ei.pe->undefended[c ^ 1];
		auto baseAttks = pawnAttacks & undefended;
		if (baseAttks)
			score += 0.5 * bits::count(baseAttks);


		//std::cout << "Score.PawnEval=" << score << std::endl;

		return score;
	}


	template<Color c> float eval_knights(const position& p, einfo& ei) {
		float score = 0;
		Square* knights = p.squares_of<c, knight>();
		Color them = Color(c ^ 1);
		U64 enemies = ei.pieces[them];
		U64 pawn_targets = (c == white ? p.get_pieces<black, pawn>() : p.get_pieces<white, pawn>());
		U64 equeen_sq = ei.queen_sqs[them];
		int ks = p.king_square(c);

		for (Square s = *knights; s != no_square; s = *++knights) {
			score += p.params.sq_score_scaling[knight] * square_score<c>(knight, s);

			// Mobility
			U64 mvs = bitboards::nmask[s];
			ei.piece_attacks[c][knight] |= mvs;
			if (!(bitboards::squares[s] & p.pinned<c>())) {
				U64 mobility = (mvs & ei.empty) & (~ei.pe->attacks[them]);
				score += p.params.mobility_scaling[knight] * knight_mobility(bits::count(mobility));
			}

			// Outpost (pawn-hole occupation)
			if ((bitboards::squares[s] & ei.pawn_holes[them])) {
				score += p.params.knight_outpost_bonus[util::col(s)];
			}

			// Outer rim penalty
			if (bitboards::edges & bitboards::squares[s])
				score -= 12;

			// Closed center bonus
			if (ei.pe->locked_center || ei.pe->center_pawn_count >= 4)
				score += p.params.bishop_open_center_bonus;

			// Bonus for attacking the center
			U64 center_influence = mvs & bitboards::big_center_mask;
			if (center_influence != 0ULL) {
				score += bits::count(center_influence) * p.params.center_influence_bonus[knight];
			}

			// Bonus for queen attacks
			U64 qattks = mvs & equeen_sq;
			if (qattks)
				score += p.params.attk_queen_bonus[knight];

			// King distance computation
			int dist = std::max(util::row_dist(s, ks), util::col_dist(s, ks));
			score -= dist;

			// Minor behind pawn
			auto fsq = (c == white ? s + 8 : s - 8);
			if (util::on_board(fsq)) {
				auto bbs = bitboards::squares[fsq];
				auto pawninfront = (c == white ? p.get_pieces<white, pawn>() : p.get_pieces<black, pawn>()) & bbs;
				if (pawninfront && util::row(s) != Row::r1 && util::row(s) != Row::r8)
					score += 12;
			}

			// king harassment
			U64 kattks = mvs & ei.kmask[them];
			if (kattks) {
				ei.kattackers[c][knight]++; // kattackers of "other" king
				ei.kattk_points[c][knight] |= kattks; // attack points of "other" king
				score += p.params.knight_king[std::min(2, bits::count(kattks))];
			}

			// protected
			U64 support = p.attackers_of2(s, c);
			if (support != 0ULL)
				score += bits::count(support);
		}
		return score;
	}


	template<Color c> float eval_bishops(const position& p, einfo& ei) {
		float score = 0;
		Square* bishops = p.squares_of<c, bishop>();
		Color them = Color(c ^ 1);
		U64 enemies = ei.pieces[them];
		U64 pawn_targets = (c == white ? p.get_pieces<black, pawn>() : p.get_pieces<white, pawn>());
		bool dark_sq = false;
		bool light_sq = false;
		U64 elight_sq_pawns = ei.white_pawns[them];
		U64 flight_sq_pawns = ei.white_pawns[c];
		U64 edark_sq_pawns = ei.black_pawns[them];
		U64 fdark_sq_pawns = ei.black_pawns[c];
		U64 equeen_sq = ei.queen_sqs[them];
		U64 center_pawns = ei.central_pawns[c];
		U64 valuable_enemies = (c == white ?
			p.get_pieces<black, queen>() | p.get_pieces<black, rook>() | p.get_pieces<black, king>() :
			p.get_pieces<white, queen>() | p.get_pieces<white, rook>() | p.get_pieces<white, king>());
		int ks = p.king_square(c);

		for (Square s = *bishops; s != no_square; s = *++bishops) {
			score += p.params.sq_score_scaling[bishop] * square_score<c>(bishop, s);

			if (bitboards::squares[s] & bitboards::colored_sqs[white]) {
				light_sq = true;
				ei.bishop_colors[c][white] = true;
			}
			if (bitboards::squares[s] & bitboards::colored_sqs[black]) {
				dark_sq = true;
				ei.bishop_colors[c][black] = true;
			}


			// Xray bonus
			U64 xray = bitboards::battks[s] & valuable_enemies;
			if (xray) {
				score += bits::count(xray);
			}

			// Mobility
			U64 mvs = magics::attacks<bishop>(ei.all_pieces, s);
			ei.piece_attacks[c][bishop] |= mvs;
			U64 mobility = (mvs & ei.empty) & (~ei.pe->attacks[them]);
			float mscore = p.params.mobility_scaling[bishop] * bishop_mobility(bits::count(mobility));

			if ((bitboards::squares[s] & p.pinned<c>()) && mscore > 0) 
				mscore /= p.params.pinned_scaling[bishop];

			score += mscore;


			// King distance computation
			int dist = std::max(util::row_dist(s, ks), util::col_dist(s, ks));
			score -= dist;

			// Closed center penalty
			if (ei.pe->locked_center || ei.pe->center_pawn_count >= 4)
				score -= p.params.bishop_open_center_bonus;

			// Bonus for attacking the center
			U64 center_influence = mvs & bitboards::big_center_mask;
			if (center_influence != 0ULL) {
				score += bits::count(center_influence) * p.params.center_influence_bonus[bishop];
			}

			// Long-diagonal bonus
			auto on_long_diagonal = (light_sq ? bitboards::squares[s] & bitboards::battks[Square::D5] :
				bitboards::squares[s] & bitboards::battks[Square::E5]);
			if (on_long_diagonal != 0ULL)
				score += p.params.bishop_open_center_bonus;

			// outpost bonus
			if ((bitboards::squares[s] & ei.pawn_holes[them]))
				score += p.params.bishop_outpost_bonus[util::col(s)];

			// Minor behind pawn
			auto fsq = (c == white ? s + 8 : s - 8);
			if (util::on_board(fsq)) {
				auto bbs = bitboards::squares[fsq];
				auto pawninfront = (c == white ? p.get_pieces<white, pawn>() : p.get_pieces<black, pawn>()) & bbs;
				if (pawninfront && util::row(s) != Row::r1 && util::row(s) != Row::r8)
					score += 12;
			}

			// Bonus/penalty for same color pawns as bishop
			const float same_color_penalty = (ei.me->is_endgame() ? 1.5f : 0.25f);
			auto fcolored_pawns = (light_sq ? flight_sq_pawns : fdark_sq_pawns);
			if (fcolored_pawns != 0ULL)
				score -= same_color_penalty * bits::count(flight_sq_pawns);

			// bonus for queen attacks
			U64 qattks = mvs & equeen_sq;
			if (qattks)
				score += p.params.attk_queen_bonus[bishop];

			// king harassment
			U64 kattks = mvs & ei.kmask[them];
			if (kattks) {
				ei.kattackers[c][bishop]++;
				ei.kattk_points[c][bishop] |= kattks;
				score += p.params.bishop_king[std::min(2, bits::count(kattks))];
			}

			// protected
			U64 support = p.attackers_of2(s, c);
			if (support != 0ULL) 
				score += bits::count(support);
		}

		// double bishop bonus
		if (light_sq && dark_sq) 
			score += p.params.doubled_bishop_bonus;

		return score;
	}

	Square rookSquares[10] = { 
		Square::no_square, Square::no_square, Square::no_square, 
		Square::no_square, Square::no_square, Square::no_square, 
		Square::no_square, Square::no_square, Square::no_square, Square::no_square };
	template<Color c> float eval_rooks(const position& p, einfo& ei) {
		float score = 0;
		int rookIdx = 0;
		Square* rooks = p.squares_of<c, rook>();
		Color them = Color(c ^ 1);
		U64 enemies = ei.pieces[them];
		U64 pawn_targets = (c == white ? p.get_pieces<black, pawn>() : p.get_pieces<white, pawn>());
		U64 equeen_sq = ei.queen_sqs[them];
		U64 valuable_enemies = (c == white ?
			p.get_pieces<black, queen>() | p.get_pieces<black, king>() :
			p.get_pieces<white, queen>() | p.get_pieces<white, king>());

		for (Square s = *rooks; s != no_square; s = *++rooks) {
			score += p.params.sq_score_scaling[rook] * square_score<c>(rook, s);

			rookSquares[rookIdx++] = s;


			// xray bonus
			U64 xray = bitboards::rattks[s] & valuable_enemies;
			if (xray) {
				score += bits::count(xray);
			}

			// mobility
			U64 mvs = magics::attacks<rook>(ei.all_pieces, s);
			ei.piece_attacks[c][rook] |= mvs;

			U64 mobility = (mvs & ei.empty) & (~ei.pe->attacks[them]);
			int free_sqs = bits::count(mobility);
			float mscore = p.params.mobility_scaling[rook] * rook_mobility(free_sqs);
			if ((bitboards::squares[s] & p.pinned<c>()))
				mscore /= p.params.pinned_scaling[rook];
			score += mscore;

			// malus for king "trapping" rook(s) in corner
			if (trapped_rook<c>(p, ei, s)) {
				score -= p.params.trapped_rook_penalty[ei.me->is_endgame()];
				if (!ei.me->is_endgame() && !p.has_castled<c>()) {
					score -= 2.0f;
				}
			}

			// bonus for attacking the center
			U64 center_influence = mvs & bitboards::big_center_mask;
			if (center_influence != 0ULL) {
				score += bits::count(center_influence) * p.params.center_influence_bonus[rook];
			}

			// bonus for queen attacks
			U64 qattks = mvs & equeen_sq;
			if (qattks) 
				score += p.params.attk_queen_bonus[rook];


			// open file bonus
			U64 column = bitboards::col[util::col(s)] & (p.get_pieces<white, pawn>() | p.get_pieces<black, pawn>());
			if (column == 0ULL) 
				score += p.params.open_file_bonus;

			// 7th rank bonus
			if (bitboards::squares[s] &
				(c == white ? bitboards::row[Row::r7] :
					bitboards::row[Row::r2])) {
				score += p.params.rook_7th_bonus;
			}

			// king harassment
			U64 kattks = mvs & ei.kmask[them];
			if (kattks) {
				ei.kattackers[c][rook]++;
				ei.kattk_points[c][rook] |= kattks;
				score += p.params.rook_king[std::min(4, bits::count(kattks))];
			}

			// protected
			U64 support = p.attackers_of2(s, c);
			if (support != 0ULL) {
				score += bits::count(support);
			}
		}

		// connected rooks
		if (rookIdx >= 2) {
			int row0 = util::row(rookSquares[0]);
			int row1 = util::row(rookSquares[1]);
			int col0 = util::col(rookSquares[0]);
			int col1 = util::col(rookSquares[1]);

			if ((row0 == row1) || (col0 == col1)) {
				U64 between_bb = bitboards::between[rookSquares[0]][rookSquares[1]];
				U64 sq_bb = bitboards::squares[rookSquares[0]] | bitboards::squares[rookSquares[1]];
				U64 blockers = (between_bb ^ sq_bb) & ei.all_pieces;

				if (blockers == 0ULL) {
					score += p.params.connected_rook_bonus;
				}
			}
		}

		return score;
	}


	template<Color c> float eval_queens(const position& p, einfo& ei) {
		float score = 0;
		Square* queens = p.squares_of<c, queen>();
		Color them = Color(c ^ 1);
		U64 enemies = ei.pieces[them];
		auto pawn_targets = (c == white ? p.get_pieces<black, pawn>() : p.get_pieces<white, pawn>());
		auto weakEnemies = (c == black ?
			p.get_pieces<white, pawn>() | p.get_pieces<white, knight>() | p.get_pieces<white, bishop>() | p.get_pieces<white, rook>() :
			p.get_pieces<black, pawn>() | p.get_pieces<black, knight>() | p.get_pieces<black, bishop>() | p.get_pieces<black, rook>());

			for (Square s = *queens; s != no_square; s = *++queens) {
				score += p.params.sq_score_scaling[queen] * square_score<c>(queen, s);

			// mobility
			U64 mvs = (magics::attacks<bishop>(ei.all_pieces, s) |
				magics::attacks<rook>(ei.all_pieces, s));
			ei.piece_attacks[c][queen] |= mvs;
			//U64 mobility = (mvs & ei.empty) & (~ei.pe->attacks[them]);
			//float mscore = p.params.mobility_scaling[queen] * queen_mobility(bits::count(mobility));
			//if ((bitboards::squares[s] & p.pinned<c>()))
			//	mscore = -20;
			//score += mscore;

			// Weak queen 
			auto attackers = p.attackers_of2(s, Color(c ^ 1)) & weakEnemies;
			if (attackers != 0ULL)
				score -= bits::count(weakEnemies);

			// Bonus for attacking the center
			U64 center_influence = mvs & bitboards::big_center_mask;
			if (center_influence != 0ULL) {
				score += bits::count(center_influence) * p.params.center_influence_bonus[queen];
			}


			// King harassment
			U64 kattks = mvs & ei.kmask[them];
			if (kattks) {
				ei.kattackers[c][queen]++;
				ei.kattk_points[c][queen] |= kattks;
				score += p.params.queen_king[std::min(6, bits::count(kattks))];
			}
		}

		return score;
	}


	template<Color c> float eval_king(const position& p, einfo& ei) {
		float score = 0;
		Square* kings = p.squares_of<c, king>();
		Color them = Color(c ^ 1);
		auto enemyPawns = (c == white ? p.get_pieces<black, pawn>() : p.get_pieces<white, pawn>());

		for (Square s = *kings; s != no_square; s = *++kings) {

			if (!ei.me->is_endgame()) {
				score += p.params.sq_score_scaling[king] * square_score<c>(king, s);
			}

			// Mobility 
			U64 mvs = ei.kmask[c] & ei.empty;

			// King safety based on number of attackers
			U64 unsafe_bb = 0ULL;
			for(Piece pc = pawn; pc <= queen; ++pc)
				unsafe_bb |= ei.kattk_points[them][pc];  // their attack points to our king
			
			if (unsafe_bb != 0ULL) {
				mvs &= (~unsafe_bb);
				unsigned num_attackers = 0;

				for (int j = 1; j < 5; ++j) 
					num_attackers += ei.kattackers[them][j];
				score -= 2 * p.params.attacker_weight[std::min((int)num_attackers, 4)];

				score += p.params.king_safe_sqs[std::min(7, bits::count(mvs))];
			
				// Attack combinations against our king
				for (Piece p1 = knight; p1 <= queen; ++p1) {
					for (Piece p2 = pawn; p2 < p1; ++p2) {
						auto twiceAttacked = ei.kattk_points[them][p1] & ei.kattk_points[them][p2];
						if (twiceAttacked != 0ULL) {
							score -= p.params.attack_combos[p1][p2];
							// Penalty for no defense of square attacked 2x's
							auto defenders = p.attackers_of2(Square(bits::pop_lsb(twiceAttacked)), c) ^ bitboards::squares[s];
							if (defenders == 0ULL)
								score -= 3 * p.params.attack_combos[p1][p2];
						}
					}
				}
			}

			// Enemy queen and bishop pair without challenging bishop
			// Piece mobility score


			// Pawns around king bonus
			if (!ei.me->is_endgame()) {
				U64 pawn_shelter = ei.pe->king[c] & ei.kmask[c];
				int n = 0;
				if (pawn_shelter) 
					n = std::min(3, bits::count(pawn_shelter));
				score += 0.5 * p.params.king_shelter[n];

				// Penalty for having pawnless flank in middle game
				U64 kflank = bitboards::kflanks[util::col(s)] & p.get_pieces<c, pawn>();
				if (kflank == 0ULL) 
					score -= 2;

				// Penalty for back rank threats
			}

			// Bonus for castling in middlegame
			auto didCastle = (c == white ? p.has_castled<white>() : p.has_castled<black>());
			if (!ei.me->is_endgame() && didCastle)
				score += 16;

			// Enemy pawn storm
			if (!ei.me->is_endgame()) {
				auto pawnStormMask = bitboards::kpawnstorm[c][!(util::col(s) >= Col::E)];
				auto pawnStorm = pawnStormMask & enemyPawns;
				auto numAttackers = bits::count(pawnStorm);
				if (numAttackers >= 2) {
					score -= 2;
					if (numAttackers >= 3) {
						score -= 2;
					}
				}
			}
		}
		return score;
	}


	U64 rowmask = (bitboards::row[Row::r3] | bitboards::row[Row::r4] | bitboards::row[Row::r5] | bitboards::row[Row::r6]);
	U64 colmask = (bitboards::col[Col::C] | bitboards::col[Col::D] | bitboards::col[Col::E]);
	U64 spacemask = rowmask | colmask;
	template<Color c> float eval_space(const position& p, einfo& ei) {
		float score = 0;

		if (ei.me->is_endgame())
			return score;

		U64 pawns = p.get_pieces<c, pawn>();
		U64 doubled = ei.pe->doubled[c];
		U64 isolated = ei.pe->isolated[c];
		pawns &= ~(doubled | isolated);
		pawns &= spacemask;

		U64 space = 0ULL;
		while (pawns) {
			int s = bits::pop_lsb(pawns);
			space |= util::squares_behind(bitboards::col[util::col(s)], c, s);
		}
		score += /*0.25 **/ bits::count(space);
		return score;
	}


	template<Color c> float eval_color(const position& p, einfo& ei) {
		float score = 0;
		U64 pawns = p.get_pieces<c, pawn>();
		ei.white_pawns[c] = pawns & bitboards::colored_sqs[white];
		ei.black_pawns[c] = pawns & bitboards::colored_sqs[black];
		return score;
	}


	template<Color c> float eval_center(const position& p, einfo& ei) {
		float score = 0;
		if (ei.me->is_endgame())
			return score;

		U64 center_pawns = ei.central_pawns[c] & bitboards::small_center_mask;
		score += bits::count(center_pawns);

		auto centerTargets = (c == white ? 
			bitboards::squares[C5] | bitboards::squares[D5] | bitboards::squares[E5] /*| bitboards::squares[F5]*/ :
			bitboards::squares[C4] | bitboards::squares[D4] | bitboards::squares[E4] /*| bitboards::squares[F4]*/);

		while (centerTargets) {
			auto s = bits::pop_lsb(centerTargets);
			score += bits::count(p.attackers_of2(Square(s), c));
		}

		return score;
	}


	template<Color c> float eval_threats(const position& p, einfo& ei) {

		float score = 0;
		auto pawnAttacks = ei.pe->attacks[c];
		auto enemyPawnAttacks = ei.pe->attacks[c ^ 1];
		auto enemyPawns = (c == white ? p.get_pieces<black, pawn>() : p.get_pieces<white, pawn>());
		auto ourPawns = (c == white ? p.get_pieces<white, pawn>() : p.get_pieces<black, pawn>());
		auto enemies = ei.pieces[c ^ 1];
		enemies = (enemies ^ enemyPawns);
		auto ourPieces = ei.pieces[c];
		auto ourPieceAttacks = (ei.piece_attacks[c][knight] | ei.piece_attacks[c][bishop] | ei.piece_attacks[c][rook] | ei.piece_attacks[c][queen]);
		auto enemyPieceAttacks = (ei.piece_attacks[c ^ 1][knight] | ei.piece_attacks[c ^ 1][bishop] | ei.piece_attacks[c ^ 1][rook] | ei.piece_attacks[c ^ 1][queen]);
		
		// 1. Pieces under attack by pawns
		auto attackedByPawns = enemies & pawnAttacks;
		if (attackedByPawns != 0ULL)
			score += 1;

		// 2. Hanging pieces under attack
		auto defendendEnemies = enemies & (enemyPawnAttacks | enemyPieceAttacks);
		auto undefendendEnemies = enemies ^ defendendEnemies;
		while (undefendendEnemies) {
			auto to = Square(bits::pop_lsb(undefendendEnemies));
			auto victim = p.piece_on(to);
			auto sqbb = bitboards::squares[to];
			auto byKnight = sqbb & ei.piece_attacks[c][knight];
			if (byKnight)
				score += 2 * p.params.attack_scaling[knight] * p.params.knight_attks[victim];
			auto byBishop = sqbb & ei.piece_attacks[c][bishop];
			if (byBishop)
				score += 2 * p.params.attack_scaling[bishop] * p.params.bishop_attks[victim];
			auto byRook = sqbb & ei.piece_attacks[c][rook];
			if (byRook)
				score += 2 * p.params.attack_scaling[rook] * p.params.rook_attks[victim];
			auto byQueen = sqbb & ei.piece_attacks[c][queen];
			if (byQueen)
				score += 2 * p.params.attack_scaling[queen] * p.params.queen_attks[victim];
		}

		// 3. Hanging weak pawns under attack
		auto weakPawns = ei.weak_pawns[c ^ 1];
		auto defendendWeakPawns = weakPawns & (enemyPawnAttacks | enemyPieceAttacks);
		auto undefendendWeakPawns = weakPawns ^ defendendWeakPawns;
		if (undefendendWeakPawns) {
			auto byKnight = undefendendWeakPawns & ei.piece_attacks[c][knight];
			if (byKnight)
				score += bits::count(byKnight);
			auto byBishop = undefendendWeakPawns & ei.piece_attacks[c][bishop];
			if (byBishop)
				score += bits::count(byBishop);
			auto byRook = undefendendWeakPawns & ei.piece_attacks[c][rook];
			if (byRook)
				score += bits::count(byRook);
			auto byQueen = undefendendWeakPawns & ei.piece_attacks[c][queen];
			if (byQueen)
				score += bits::count(byQueen);
		}


		// 4. Pieces pinned to queen
		auto enemyQueens = (c == white ? p.get_pieces<black, queen>() : p.get_pieces<white, queen>());
		auto ourRooks = (c == white ? p.get_pieces<white, rook>() : p.get_pieces<black, rook>());
		auto ourBishops = (c == white ? p.get_pieces<white, bishop>() : p.get_pieces<black, bishop>());
		while (enemyQueens) {
			auto queenSq = bits::pop_lsb(enemyQueens);
			auto rookPinners = bitboards::rattks[queenSq] & ourRooks;
			auto bishopPinners = bitboards::battks[queenSq] & ourBishops;
			while (rookPinners) {
				auto rookSq = bits::pop_lsb(rookPinners);
				auto betweenMask = bitboards::between[rookSq][queenSq];
				auto pinnedByRook = (enemies & betweenMask) ^ bitboards::squares[queenSq];
				if (pinnedByRook && bits::count(pinnedByRook) == 1) {
					auto pp = p.piece_on(Square(bits::pop_lsb(pinnedByRook)));
					if (pp == bishop || pp == knight)
						score += 6;
				}
			}
			while (bishopPinners) {
				auto bishopSq = bits::pop_lsb(bishopPinners);
				auto betweenMask = bitboards::between[bishopSq][queenSq];
				auto pinnedByBishop = (enemies & betweenMask) ^ bitboards::squares[queenSq];
				if (pinnedByBishop && bits::count(pinnedByBishop) == 1) {
					auto pp = p.piece_on(Square(bits::pop_lsb(pinnedByBishop)));
					if (pp == knight)
						score += 6;
					if (pp == rook)
						score += 18;
				}
			}
		}


		// 5. Latent (discovered) checks
		auto hasDiscovery = false;
		auto ourQueens = (c == white ? p.get_pieces<white, queen>() : p.get_pieces<black, queen>());
		auto enemyKing = p.king_square(Color(c ^ 1));
		auto rookCheckers = bitboards::rattks[enemyKing] & ourRooks;
		auto bishopCheckers = bitboards::battks[enemyKing] & ourBishops; 
		auto ourKnights = (c == white ? p.get_pieces<white, knight>() : p.get_pieces<black, knight>());
		auto queenCheckers = (bitboards::rattks[enemyKing] | bitboards::battks[enemyKing]) & ourQueens;
		while (bishopCheckers && !hasDiscovery) {
			auto bishopSq = bits::pop_lsb(bishopCheckers);
			auto between = (bitboards::between[bishopSq][enemyKing] & (ourRooks | ourKnights));
			if (between && bits::count(between) == 1) {
				hasDiscovery = true;
				score += 10;
			}
		}
		while (rookCheckers && !hasDiscovery) {
			auto rookSq = bits::pop_lsb(rookCheckers);
			auto between = (bitboards::between[rookSq][enemyKing] & (ourBishops | ourKnights));
			if (between && bits::count(between) == 1) {
				hasDiscovery = true;
				score += 10; 
			}
		}
		while (queenCheckers && !hasDiscovery) {
			auto queenSq = bits::pop_lsb(queenCheckers);
			auto between = (bitboards::between[queenSq][enemyKing] & (ourKnights));
			if (between && bits::count(between) == 1) {
				hasDiscovery = true;
				score += 10; 
			}
		}


		// 6. Restriction
		auto ourAttacks = (pawnAttacks | ourPieceAttacks);
		auto theirAttacks = (enemyPawnAttacks | enemyPieceAttacks);
		score += bits::count(ourAttacks) - bits::count(theirAttacks);


		// 7. Skewer detection
		bishopCheckers = bitboards::battks[enemyKing] & ourBishops;
		auto enemyKnights = (c == white ? p.get_pieces<black, knight>() : p.get_pieces<white, knight>());
		auto enemyBishops = (c == white ? p.get_pieces<black, bishop>() : p.get_pieces<white, bishop>());
		auto enemyRooks = (c == white ? p.get_pieces<black, rook>() : p.get_pieces<white, rook>());
		auto enemyPieces = (enemyKnights | enemyBishops | enemyRooks | enemyQueens);
		auto enemyKingSq = bitboards::squares[enemyKing];
		while (bishopCheckers) {
			auto bishopSq = bits::pop_lsb(bishopCheckers);
			while (enemyPieces) {
				auto enemy = bits::pop_lsb(enemyPieces);
				auto between = bitboards::between[bishopSq][enemy] & (enemyKingSq);
				if (between && bits::count(between) == 1)
				{
					auto pp = p.piece_on(Square(enemy));
					if (pp == bishop || pp == knight)
						score += 4;
					if (pp == rook)
						score += 6;
					if (pp == queen)
						score += 8;
				}
			}
		}

		return score;
	}


	template<Color c> float eval_pawn_levers(const position& p, einfo& ei)
	{
		float score = 0;
		U64 their_pawns = (c == white ? p.get_pieces<black, pawn>() : p.get_pieces<white, pawn>());

		U64 pawn_lever_attacks = ei.pe->attacks[c] & their_pawns;

		while (pawn_lever_attacks) {
			int to = bits::pop_lsb(pawn_lever_attacks);
			if (c == p.to_move()) 
				score += p.params.pawn_lever_score[to];
		}
		return score;
	}


	bool opposite_side_castling(const position& p, einfo& ei) {
		int kcw = util::col(p.king_square(white));
		int kcb = util::col(p.king_square(black));

		return !ei.me->is_endgame() &&
			p.has_castled<white>() &&
			p.has_castled<black>() &&
			((kcw < Col::E) != (kcb < Col::E));
	}

	template<Color c> float eval_flank_attack(const position& p, einfo& ei)
	{
		float score = 0;
		if (!ei.pe->locked_center) return score;

		if (!opposite_side_castling(p, ei)) return score;

		U64 flank_bb = bitboards::col[A] | bitboards::col[B] | bitboards::col[C] |
			bitboards::col[F] | bitboards::col[G] | bitboards::col[H];
		U64 their_pawns = (c == white ? p.get_pieces<black, pawn>() : p.get_pieces<white, pawn>());

		U64 flank_attacks = ei.pe->attacks[c] & their_pawns & flank_bb;

		// double-counts the pawn lever attack when the center is closed
		while (flank_attacks) {
			int to = bits::pop_lsb(flank_attacks);
			if (c == p.to_move()) score += p.params.pawn_lever_score[to];
		}

		return score;
	}


	// TODO: 
	// - king within queening sq?
	template<Color c> float eval_passed_pawns(const position& p, einfo& ei)
	{
		float score = 0;
		U64 passers = ei.pe->passed[c];
		if (passers == 0ULL) {
			return score;
		}

		//std::cout << " \n======Weak squares for c" << c << "========" << std::endl;
		//bits::print(passers);

		while (passers) {
			Square f = Square(bits::pop_lsb(passers));
			int row_dist = (c == white ? 7 - util::row(f) : util::row(f));

			if (row_dist > 3 || row_dist <= 0) {
				score += p.params.passed_pawn_bonus;
				continue;
			}


			Square front = (c == white ? Square(f + 8) : Square(f - 8));

			// 1. is next square blocked?
			if (p.piece_on(front) == Piece::no_piece) {
				score += 1;
			}

			// 2. is next square attacked - note: this needs to be dispersed throughout the piece eval (!)
			U64 our_attackers = 0ULL;
			U64 their_attackers = 0ULL;
			auto crudeControl = 0;

			if (util::on_board(front))
			{
				our_attackers = p.attackers_of2(front, c);
				their_attackers = p.attackers_of2(front, Color(c ^ 1));
				
			}

			if (our_attackers != 0ULL) {
				auto count = bits::count(our_attackers);
				crudeControl += count;
				score += 3 * bits::count(our_attackers);
			}

			if (their_attackers != 0ULL) {
				auto count = bits::count(their_attackers);
				crudeControl -= count;
				score -= 3 * bits::count(their_attackers);
			}

			// 3. rooks behind passed pawns
			auto rooks = (c == white ? p.get_pieces<white, rook>() : p.get_pieces<black, rook>());
			if (rooks)
			{
				while (rooks)
				{


					auto rf = Square(bits::pop_lsb(rooks));
					if (util::col(rf) == util::col(f))
					{
						auto rowDiff = util::row(rf) - util::row(f);
						auto isBehind = (c == white ? rowDiff < 0 : rowDiff > 0);
						auto supports = ((bitboards::between[rf][f] & p.all_pieces()) ^ (bitboards::squares[rf] | bitboards::squares[f])) == 0ULL;
						if (isBehind)
							score += 1;
						if (isBehind && supports)
						{
							crudeControl += 1;
							score += 30;
						}
					}
				}
			}


			// 4. bonus for connected passers
			auto connectedPassed = (bitboards::neighbor_cols[util::col(f)] & ei.pe->passed[c]) != 0ULL;
			if (connectedPassed)
				score += 30; // this is counted twice - so sums to 60 if exists.

			// 5. bonus for closer to promotion
			score += (
				row_dist == 3 ? 45 :
				row_dist == 2 ? 90 :
				row_dist == 1 ? 180 : 0);

			if (crudeControl < 0)
				score -= (
					row_dist == 3 ? 30 :
					row_dist == 2 ? 55 :
					row_dist == 1 ? 120 : 0);

			//std::cout << "   Score for passed pawn=" << f << "=" << score << " control=" << crudeControl << std::endl;
		}

		return score;

	}

	////////////////////////////////////////////////////////////////////////////////
	// helper functions for evaluation
	////////////////////////////////////////////////////////////////////////////////

	template<Color c> bool trapped_rook(const position& p, einfo& ei, const Square& rs) {

		int ks = p.king_square(c);
		int kcol = util::col(ks);
		int krow = util::row(ks);
		int rcol = util::col(rs);
		int rrow = util::row(rs);

		if (krow != rrow)
			return false;

		if (krow != (c == white ? Row::r1 : Row::r8))
			return false;

		if ((kcol < Col::E) != (rcol < kcol))
			return false;

		return true;
	}



	////////////////////////////////////////////////////////////////////////////////
	// endgame evaluations
	////////////////////////////////////////////////////////////////////////////////
	template<Color c> float eval_kpk(const position& p, einfo& ei) {
		float score = 0;
		const float pawn_majority_bonus = 16;
		const float opposition_bonus = 4;
		const float advanced_passed_pawn_bonus = 10;
		const float king_proximity_bonus = 2;
		const float pawn_spread_bonus = 2;

		// only evaluate the fence once
		//if (!ei.endgame.evaluated_fence) {
		//	ei.endgame.is_fence = eval::is_fence(p, ei);
		//	ei.endgame.evaluated_fence = true;

		//	if (ei.endgame.is_fence) {
		//		//std::cout << "kpk is a fence position" << std::endl;
		//		return Score::draw;
		//	}
		//	//else std::cout << "kpk is not a fence position" << std::endl;
		//}


		// 1. King opposition
		bool has_opposition = eval::has_opposition<c>(p, ei);

		// 2. Pawns on both sides of the board
		auto queensidePawns = (ei.pe->queenside[c] & (bitboards::col[Col::A] | bitboards::col[Col::B] | bitboards::col[Col::C])) != 0ULL;
		auto kingsidePawns = (ei.pe->kingside[c] & (bitboards::col[Col::H] | bitboards::col[Col::G] | bitboards::col[Col::F])) != 0ULL;
		if (queensidePawns && kingsidePawns)
			score += pawn_spread_bonus;

		// 3. Passed pawns
		U64 passed_pawns = ei.pe->passed[c];
		if (passed_pawns != 0ULL) {
			while (passed_pawns) {
				int f = bits::pop_lsb(passed_pawns);
				score += eval::eval_passed_kpk<c>(p, ei, Square(f), has_opposition);
			}
		}

		// 4. Opposition (always good)
		if (has_opposition) 
			score += opposition_bonus;

		return score;
	}


	template<Color c> float eval_krrk(const position& p, einfo& ei) {
		float score = 0;

		const float opposition_bonus = 2;
		const float pawn_spread_bonus = 4;

		// 1. King opposition
		bool has_opposition = eval::has_opposition<c>(p, ei);
		if (has_opposition)
			score += opposition_bonus;

		// 2. Pawns on both sides of the board
		auto queensidePawns = (ei.pe->queenside[c] & (bitboards::col[Col::A] | bitboards::col[Col::B] | bitboards::col[Col::C])) != 0ULL;
		auto kingsidePawns = (ei.pe->kingside[c] & (bitboards::col[Col::H] | bitboards::col[Col::G] | bitboards::col[Col::F])) != 0ULL;
		if (queensidePawns && kingsidePawns)
			score += pawn_spread_bonus;

		// 3. Passed pawn eval
		U64 passed_pawns = ei.pe->passed[c];
		if (passed_pawns != 0ULL) {
			while (passed_pawns) {
				int f = bits::pop_lsb(passed_pawns);
				score += eval::eval_passed_krrk<c>(p, ei, Square(f), has_opposition);
			}
		}

		return score;
	}


	template<Color c> float eval_knbk(const position& p, einfo& ei) {
		float score = 0;

		//std::cout << " \n\n.. KNBK EVAL DEBUG .." << std::endl;
		//std::cout << " color = " << c << std::endl;
		//std::cout << " toMv = " << p.to_move() << std::endl;
		const float opposition_bonus = 2;
		const float pawn_spread_bonus = 4;

		// 1. King opposition
		bool has_opposition = eval::has_opposition<c>(p, ei);
		if (has_opposition)
			score += opposition_bonus;

		// 2. Pawns on both sides of the board with bishop
		auto bishops = p.get_pieces<c, bishop>();
		auto hasBishop = (bishops != 0ULL);
		if (hasBishop) {
			auto queensidePawns = (ei.pe->queenside[c] & (bitboards::col[Col::A] | bitboards::col[Col::B] | bitboards::col[Col::C])) != 0ULL;
			auto kingsidePawns = (ei.pe->kingside[c] & (bitboards::col[Col::H] | bitboards::col[Col::G] | bitboards::col[Col::F])) != 0ULL;
			if (queensidePawns && kingsidePawns) {
				score += pawn_spread_bonus;
				//std::cout << "DBG: pawn spread bonus" << std::endl;
			}
		
			// 4. If we have the bishop, reward pawns on opposite color as bishop
			auto lightSqBishop = 
				(bitboards::squares[bits::pop_lsb(bishops)] & bitboards::colored_sqs[white]) != 0ULL;
			if (lightSqBishop) {
				score += bits::count(ei.pe->dark[c]); // count pawns on dark squares
			}
			else {
				score += bits::count(ei.pe->light[c]); // count pawns on light squares
			}
		}

		// 5. Passed pawn eval
		U64 passed_pawns = ei.pe->passed[c];
		if (passed_pawns != 0ULL) {
			while (passed_pawns) {
				int f = bits::pop_lsb(passed_pawns);
				score += eval::eval_passed_knbk<c>(p, ei, Square(f), has_opposition);
			}
		}

		return score;
	}
}

namespace eval {
	float evaluate(const position& p, const Searchthread& t, const float& lazy_margin) { return do_eval(p, t, lazy_margin); }
}
