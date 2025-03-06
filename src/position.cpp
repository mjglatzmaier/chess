
#include "position.h"
#include "move.h"

position::position(std::istringstream& fen) {
	setup(fen);
}


position::position(const position& p) { 
	*this = p;
}


position& position::operator=(const position& p) {
	std::copy(std::begin(p.history), std::end(p.history), std::begin(history));
	root_moves = p.root_moves;
	ifo = p.ifo;
	pcs = p.pcs;
	stats = p.stats;
	hidx = p.hidx;
	thread_id = p.thread_id;
	nodes_searched = p.nodes_searched;
	qnodes_searched = p.qnodes_searched;
	params = p.params;
	debug_search = p.debug_search;
	return *(this);
}


piece_data& piece_data::operator=(const piece_data& pd) {
	std::copy(std::begin(pd.bycolor), std::end(pd.bycolor), std::begin(bycolor));
	std::copy(std::begin(pd.king_sq), std::end(pd.king_sq), std::begin(king_sq));
	std::copy(std::begin(pd.color_on), std::end(pd.color_on), std::begin(color_on));
	std::copy(std::begin(pd.piece_on), std::end(pd.piece_on), std::begin(piece_on));
	std::copy(std::begin(pd.number_of), std::end(pd.number_of), std::begin(number_of));
	std::copy(std::begin(pd.bitmap), std::end(pd.bitmap), std::begin(bitmap));
	std::copy(std::begin(pd.piece_idx), std::end(pd.piece_idx), std::begin(piece_idx));
	std::copy(std::begin(pd.square_of), std::end(pd.square_of), std::begin(square_of));
	return (*this);
}


void position::setup(std::istringstream& fen) {
	clear();

	std::string token;
	fen >> token;
	Square s = Square::A8;

	for (auto& c : token) {
		if (isdigit(c)) s += int(c - '0');
		else if (c == '/') s -= 16;
		else { set_piece(c, s); ++s; }
	}

	// side to move
	fen >> token;
	ifo.stm = (token == "w" ? Color::white : Color::black);
	ifo.key ^= zobrist::stm(ifo.stm);
	ifo.repkey ^= zobrist::stm(ifo.stm);

	// the castle rights
	fen >> token;
	ifo.cmask = U16(0);
	for (auto& c : token) {
		U16 cr = CastleRights.at(c);
		ifo.cmask |= cr;
		ifo.key ^= zobrist::castle(ifo.stm, cr);
		ifo.repkey ^= zobrist::castle(ifo.stm, cr);
	}


	// ep square
	fen >> token;
	ifo.eps = Square::no_square;
	Row row = Row::no_row; Col col = Col::no_col;
	for (auto& c : token) {
		if (c >= 'a' && c <= 'h') col = Col(c - 'a');
		if (c == '3' || c == '6') row = Row(c - '1');
	}
	ifo.eps = Square(8 * row + col);

	if (!util::on_board(ifo.eps)) ifo.eps = Square::no_square;

	if (ifo.eps != Square::no_square) {
		ifo.key ^= zobrist::ep(util::col(ifo.eps));
		ifo.repkey ^= zobrist::ep(util::col(ifo.eps));
	}

	// half-moves since last pawn move/capture
	fen >> token;

	ifo.move50 = (token != "-" ? U8(std::stoi(token)) : 0);

	ifo.key ^= zobrist::mv50(ifo.move50);

	// move counter
	fen >> token;
	ifo.hmvs = (token != "-" ? U16(std::stoi(token)) : 0);
	ifo.key ^= zobrist::hmvs(ifo.hmvs);

	// check info
	Color stm = to_move();
	ifo.ks[stm] = pcs.king_sq[stm];
	ifo.incheck = is_attacked(ifo.ks[stm], stm, Color(stm ^ 1));

	ifo.checkers = (in_check() ? attackers_of2(ifo.ks[stm], Color(stm ^ 1)) : 0ULL);
	ifo.pinned[stm] = pinned(stm);
	ifo.pinned[stm ^ 1] = pinned(Color(stm ^ 1));
}


std::string position::to_fen() const {
	std::string fen = "";
	for (int r = 7; r >= 0; --r) {
		int empties = 0;

		for (int c = 0; c < 8; ++c) {
			int s = r * 8 + c;
			if (piece_on(Square(s)) == no_piece) { ++empties; continue; }
			if (empties > 0) fen += std::to_string(empties);
			empties = 0;

			fen += SanPiece[(color_on(Square(s)) == black ?
				piece_on(Square(s)) + 6 : piece_on(Square(s)))];
		}

		if (empties > 0) fen += std::to_string(empties);
		if (r > 0) fen += "/";
	}

	fen += (to_move() == white ? " w" : " b");

	// castle rights
	std::string c_str = "";
	if ((ifo.cmask & wks) == wks) c_str += "K";
	if ((ifo.cmask & wqs) == wqs) c_str += "Q";
	if ((ifo.cmask & bks) == bks) c_str += "k";
	if ((ifo.cmask & bqs) == bqs) c_str += "q";
	fen += (c_str == "" ? " -" : " " + c_str);

	// ep-square
	std::string ep_sq = "";
	if (ifo.eps != no_square) {
		ep_sq += SanCols[util::col(ifo.eps)] + std::to_string(util::row(ifo.eps) + 1);
	}
	fen += (ep_sq == "" ? " -" : " " + ep_sq);

	// move50
	std::string mv50 = std::to_string(ifo.move50);
	fen += " " + mv50;

	// half-mvs
	std::string half_mvs = std::to_string(ifo.hmvs);
	fen += " " + half_mvs;

	return fen;
}


bool position::is_draw() {

	if (ifo.move50 > 99) 
		return true;

	U64 kcurrent = ifo.repkey;
	unsigned same_count = 0;
	int idx = hidx - 2;
	while (same_count == 0 && idx >= 0) {
		same_count += (kcurrent == history[idx].repkey);
		idx -= 2;
	}

	return same_count > 0;
}


void position::do_move(const Move& m) {
	history[hidx++] = ifo;
	const Square from = Square(m.f);
	const Square to = Square(m.t);
	const Movetype t = Movetype(m.type);
	const Piece p = piece_on(from);
	const Color us = to_move();

	// king square update and castle rights update
	if (p == king) {
		pcs.king_sq[us] = to;
		ifo.ks[us] = to;
		if (can_castle_ks() || can_castle_qs()) {
			ifo.cmask &= (us == white ? clearw : clearb);
			ifo.key ^= (us == white ? zobrist::castle(white, clearw) : zobrist::castle(black, clearb));
			ifo.repkey ^= (us == white ? zobrist::castle(white, clearw) : zobrist::castle(black, clearb));
		}
	}
	else if (p == rook) {
		if (us == white && can_castle<white>()) {
			if (from == A1) {
				ifo.cmask &= clearwqs;
				ifo.key ^= zobrist::castle(white, clearwqs);
				ifo.repkey ^= zobrist::castle(white, clearwqs);
			}
			else if (from == H1) {
				ifo.cmask &= clearwks;
				ifo.key ^= zobrist::castle(white, clearwks);
				ifo.repkey ^= zobrist::castle(white, clearwks);
			}
		}
		else if (us == black && can_castle<black>()) {
			if (from == A8) {
				ifo.cmask &= clearbqs;
				ifo.key ^= zobrist::castle(black, clearbqs);
				ifo.repkey ^= zobrist::castle(black, clearbqs);
			}
			else if (from == H8) {
				ifo.cmask &= clearbks;
				ifo.key ^= zobrist::castle(black, clearbks);
				ifo.repkey ^= zobrist::castle(black, clearbks);
			}
		}
	}

	ifo.captured = no_piece;

	if (t == quiet) {
		pcs.do_quiet(us, p, from, to, ifo);
	}

	else if (t == capture) {
		ifo.captured = piece_on(to);
		pcs.do_cap(us, p, from, to, ifo);
	}

	else if (t == ep) {
		ifo.captured = pawn;
		pcs.do_ep(us, from, to, ifo);
	}

	else if (t < capture_promotion_q) pcs.do_promotion(us, (t == promotion_q ? queen :
		t == promotion_r ? rook :
		t == promotion_b ? bishop :
		knight), from, to, ifo);

	else if (t < castle_ks) {
		ifo.captured = piece_on(to);
		pcs.do_promotion_cap(us, (t == capture_promotion_q ? queen :
			t == capture_promotion_r ? rook :
			t == capture_promotion_b ? bishop :
			knight), from, to, ifo);
	}

	else if (t == castle_ks) {
		pcs.do_castle_ks(us, from, to, ifo);
		ifo.cmask &= (us == white ? clearw : clearb);
		ifo.has_castled[us] = true;
	}

	else if (t == castle_qs) {
		pcs.do_castle_qs(us, from, to, ifo);
		ifo.cmask &= (us == white ? clearw : clearb);
		ifo.has_castled[us] = true;
	}

	// eps
	ifo.eps = no_square;
	if (p == pawn && abs(from - to) == 16) {
		ifo.eps = Square(from + (us == white ? 8 : -8));
		ifo.key ^= zobrist::ep(util::col(to));
		ifo.repkey ^= zobrist::ep(util::col(to));
	}

	// move50
	if (p == pawn || t == capture) ifo.move50 = 0;
	else ifo.move50++;
	ifo.key ^= zobrist::mv50(ifo.move50);

	// half-moves
	ifo.hmvs++;
	ifo.key ^= zobrist::hmvs(ifo.hmvs);

	// side to move
	ifo.stm = Color(ifo.stm ^ 1);
	ifo.key ^= zobrist::stm(ifo.stm);
	ifo.repkey ^= zobrist::stm(ifo.stm);

	ifo.incheck = is_attacked(king_square(), ifo.stm, us);
	ifo.checkers = (ifo.incheck ? attackers_of2(king_square(), Color(ifo.stm ^ 1)) : 0ULL);
	ifo.pinned[ifo.stm] = pinned(ifo.stm);
	ifo.pinned[ifo.stm ^ 1] = pinned(Color(ifo.stm ^ 1));
	++nodes_searched;
}

void position::undo_move(const Move& m) {
	const Square from = Square(m.t);
	const Square to = Square(m.f);
	const Movetype t = Movetype(m.type);
	const Piece p = piece_on(from);
	const Color us = Color(to_move() ^ 1);
	Piece cp = ifo.captured;

	if (t == quiet) pcs.do_quiet(us, p, from, to, ifo);

	else if (t == capture) {
		pcs.do_quiet(us, p, from, to, ifo);
		pcs.add_piece(to_move(), cp, from, ifo);
	}

	else if (t == ep) {
		pcs.do_quiet(us, p, from, to, ifo);
		pcs.add_piece(to_move(), cp, Square(from + (us == white ? -8 : 8)), ifo);
	}

	else if (t < capture_promotion_q) {
		pcs.remove_piece(us, piece_on(from), from, ifo);
		pcs.add_piece(us, pawn, to, ifo);
	}

	else if (t < castle_ks) {
		pcs.remove_piece(us, piece_on(from), from, ifo);
		pcs.add_piece(to_move(), cp, from, ifo);
		pcs.add_piece(us, pawn, to, ifo);
	}

	else if (t == castle_ks) {
		Square rt = (us == white ? H1 : H8);
		Square rf = (us == white ? F1 : F8);
		pcs.do_quiet(us, king, from, to, ifo);
		pcs.do_quiet(us, rook, rf, rt, ifo);
	}

	else if (t == castle_qs) {
		Square rf = (us == white ? D1 : D8);
		Square rt = (us == white ? A1 : A8);
		pcs.do_quiet(us, king, from, to, ifo);
		pcs.do_quiet(us, rook, rf, rt, ifo);
	}
	ifo = history[--hidx];
}


void position::do_null_move() {
	const Color us = to_move();
	const Color them = Color(us ^ 1);

	history[hidx++] = ifo;

	// eps square
	if (ifo.eps != Square::no_square) {
		ifo.key ^= zobrist::ep(util::col(ifo.eps));
		ifo.repkey ^= zobrist::ep(util::col(ifo.eps));
		ifo.eps = Square::no_square;
	}

	// side to move
	ifo.stm = them;
	ifo.key ^= zobrist::stm(ifo.stm);
	ifo.repkey ^= zobrist::stm(ifo.stm);

	// move50
	ifo.move50++;
	ifo.key ^= zobrist::mv50(ifo.move50);

	// half-moves
	ifo.hmvs++;
	ifo.key ^= zobrist::hmvs(ifo.hmvs);
}


void position::undo_null_move() {
	ifo = history[--hidx];
}


std::vector<int> mvals{ 100, 300, 315, 480, 910, 2000 };

int position::see(const Move& m) const {

	if (m.type == Movetype::ep) return 0;

	else if (m.type == Movetype::capture &&
		(mvals[piece_on(Square(m.f))] <= mvals[piece_on(Square(m.t))])) {
		return (mvals[piece_on(Square(m.t))] - mvals[piece_on(Square(m.f))]);
	}

	else if (m.type == Movetype::capture_promotion_q ||
		m.type == Movetype::capture_promotion_r ||
		m.type == Movetype::capture_promotion_b ||
		m.type == Movetype::capture_promotion_n) {
		int fval = (m.type == Movetype::capture_promotion_q ? mvals[queen] :
			m.type == Movetype::capture_promotion_r ? mvals[rook] :
			m.type == Movetype::capture_promotion_b ? mvals[bishop] :
			mvals[knight]) - mvals[0];
		int tval = mvals[piece_on(Square(m.t))];

		if (fval <= tval) return fval - tval;
	}


	return see_move(m);
}

struct SeePiece {
	SeePiece(const Piece& pc, const int16& v) : p(pc), score(v) { }
	Piece p;
	int16 score;
	inline bool operator<(const SeePiece& o) const { return score < o.score; }
	inline bool operator>(const SeePiece& o) const { return score > o.score; }
};

int position::see_move(const Move& m) const {

	Square bks = king_square(black);
	Square wks = king_square(white);
	Square to = Square(m.t);
	Square from = Square(m.f);
	U64 pieces = all_pieces();
	U64 attackers = 0ULL;

	U64 white_bb = get_pieces<white>() ^ pinned<white>();
	U64 black_bb = get_pieces<black>() ^ pinned<black>();

	std::vector<SeePiece> black_list;
	std::vector<SeePiece> white_list;
	Piece atkr = Piece::no_piece;

	while (true) { // while loop for pieces behind currently attacking pieces
		U64 a = attackers_of(to, pieces) & pieces;
		if (a) {
			pieces ^= a;

			if (is_attacked(wks, white, black, pieces) || is_attacked(bks, black, white, pieces))
			{
				return 0; // let search handle discovered checking sequences
			}

			attackers |= a;
		}
		else break;

		U64 white_attackers = a & white_bb;
		if (white_attackers) {
			while (white_attackers) {
				Square s = Square(bits::pop_lsb(white_attackers));
				if (s == from) {
					atkr = piece_on(Square(s));
					continue; // first attacker is handled below
				}
				white_list.emplace_back(SeePiece(piece_on(s), mvals[piece_on(s)]));
			}
		}

		U64 black_attackers = a & black_bb;
		if (black_attackers) {
			while (black_attackers) {
				Square s = Square(bits::pop_lsb(black_attackers));
				if (s == from) {
					atkr = piece_on(s);
					continue;
				}
				black_list.emplace_back(SeePiece(piece_on(s), mvals[piece_on(s)]));
			}
		}

		if (white_list.size() == 0 && black_list.size() == 0 && atkr == Piece::no_piece) return 0;
	}

	std::sort(white_list.begin(), white_list.end());
	std::sort(black_list.begin(), black_list.end());


	int i = 0;
	unsigned w = 0;
	unsigned b = 0;
	Color color = to_move();


	if (color == black) { black_list.insert(black_list.begin(), SeePiece(atkr, 0)); }
	else { white_list.insert(white_list.begin(), SeePiece(atkr, 0)); }

	int score = 0;
	int prev = score;

	while (true) {

		Piece victim = Piece::no_piece;
		if (i == 0) {
			Piece v = piece_on(to);
			if (v == Piece::king) return 0; // illegal
			score += (v == Piece::no_piece ? 0 : mvals[v]);
			color = Color(color ^ 1);
			prev = score;
			++i;
			continue;
		}

		if ((w >= white_list.size() || b >= black_list.size())) break;

		victim = (color == white ? black_list[b++].p : white_list[w++].p);

		int av = -1;
		Piece attacker = Piece::no_piece;
		if (color == white && w < white_list.size()) {
			attacker = white_list[w].p;
			av = mvals[attacker];
		}
		else if (color == black && b < black_list.size()) {
			attacker = black_list[b].p;
			av = mvals[attacker];
		}

		if (attacker == Piece::no_piece) break;


		color = Color(color ^ 1);
		int vv = mvals[victim];



		if (vv < av || victim == king) {
			if (attacker == king) {
				if ((color == black && b < black_list.size()) ||
					(color == white && w < white_list.size())) {
					return score; // illegal
				}
			}


			if ((victim == king && ((color == black && w < white_list.size()) || (color == white && b < black_list.size()))) ||
				(victim != king && ((color == black && (black_list.size() > white_list.size())) ||
					(color == white && (white_list.size() > black_list.size()))))) {
				score = prev;
				return score;
			}
		}
		score += ((i & 1) == 1 ? -vv : vv);
		++i;
		prev = score;
	}

	return score;
}

inline bool _is_promotion(const Movetype& mt) {
	return (mt == promotion ||
		mt == promotion_q ||
		mt == promotion_r ||
		mt == promotion_b ||
		mt == promotion_n);
}

bool position::is_cap_promotion(const Movetype& mt) {
	return (mt == capture_promotion_q ||
		mt == capture_promotion_r ||
		mt == capture_promotion_b ||
		mt == capture_promotion_n);
}

bool position::is_promotion(const U8& mt)
{
	return _is_promotion(Movetype(mt)) || is_cap_promotion(Movetype(mt));
}


bool position::gives_check(const Move& m) {
	// Note: assumes the input move is legal (!)
	auto mask = 0ULL;
	auto c = to_move();
	auto them = Color(c ^ 1);
	auto isCapture = m.type == Movetype::capture ||
		m.type == Movetype::capture_promotion ||
		m.type == Movetype::capture_promotion_b ||
		m.type == Movetype::capture_promotion_n ||
		m.type == Movetype::capture_promotion_q ||
		m.type == Movetype::capture_promotion_r;
	auto isPromotion = is_promotion(m.type);
	auto isEp = m.type == Movetype::ep;
	auto isQuiet = m.type == Movetype::quiet;
	auto isCastles = m.type == Movetype::castles ||
		m.type == Movetype::castle_ks ||
		m.type == Movetype::castle_qs;
	auto from = m.f;
	auto to = m.t;
	auto capSq = Square::no_square;
	auto promotePiece = Piece::no_piece;
	auto piece = piece_on(Square(from));
	auto capPiece = Piece::no_piece;
	if (piece == king && !isCastles)
		return false;

	// Move our piece
	if (isCastles) {
		if (c == white) {
			to = (m.type == Movetype::castle_ks ? F1 : D1);
			from = (m.type == Movetype::castle_ks ? H1 : A1);
		}
		else if (c == black) {
			to = (m.type == Movetype::castle_ks ? F8 : D8);
			from = (m.type == Movetype::castle_ks ? H8 : A8);
		}
		pcs.bitmap[c][rook] ^= (bitboards::squares[from] | bitboards::squares[to]);
	}
	else {
		pcs.bitmap[c][piece] ^= (bitboards::squares[from] | bitboards::squares[to]);
		if (isPromotion)
			pcs.bitmap[c][piece] ^= bitboards::squares[to];
	}

	// Remove the captured piece
	if (isCapture || isEp) {
		if (isEp)
			capSq = Square(to + (c == white ? -8 : 8));
		else
			capSq = Square(to);

		capPiece = piece_on(Square(capSq));
		pcs.bitmap[them][capPiece] ^= bitboards::squares[capSq];
		mask ^= bitboards::squares[capSq];
	}

	// Add promotion piece
	if (isPromotion) {
		if (m.type == Movetype::capture_promotion_n || m.type == Movetype::promotion_n)
			promotePiece = knight;
		if (m.type == Movetype::capture_promotion_b || m.type == Movetype::promotion_b)
			promotePiece = bishop;
		if (m.type == Movetype::capture_promotion_r || m.type == Movetype::promotion_r)
			promotePiece = rook;
		if (m.type == Movetype::capture_promotion_q || m.type == Movetype::promotion_q)
			promotePiece = queen;
		pcs.bitmap[c][promotePiece] |= bitboards::squares[to];
	}

	// Make the check mask
	mask = (all_pieces() ^ bitboards::squares[from]) | bitboards::squares[to];

	auto target = king_square(them);
	auto checks = is_attacked(target, them, c, mask);

	// Reset the pcs bitmap
	if (isCastles) {
		pcs.bitmap[c][rook] ^= (bitboards::squares[to] | bitboards::squares[from]);
	}
	else {
		// Move our piece back
		pcs.bitmap[c][piece] ^= (bitboards::squares[to] | bitboards::squares[from]);

		// Place back the captured piece and remove promotion piece
		if (isCapture || isEp)
			pcs.bitmap[them][capPiece] |= bitboards::squares[capSq];
		if (isPromotion) {
			pcs.bitmap[c][promotePiece] ^= bitboards::squares[to];
		}
	}

	return checks;
}

bool position::quiet_gives_dangerous_check(const Move& m)
{
	if (m.type != Movetype::quiet)
		return false;

	U64 msk = (all_pieces() ^ bitboards::squares[m.f]) |
		bitboards::squares[m.t];
	auto us = to_move();
	auto them = Color(us ^ 1);
	auto target = king_square(them);
	auto friends = (us == white ? get_pieces<white>() : get_pieces<black>());
	auto attackers = attackers_of(target, msk) & friends;
	
	if (attackers == 0ULL)
		return false;

	// define double-checks as dangerous
	if (bits::more_than_one(attackers))
		return true;
	
	// contact checks from rooks/pawn/queen is dangerous
	auto s = bits::pop_lsb(attackers);
	auto piece = piece_on(Square(s));
	auto row_dist = util::row_dist(s, target);
	auto col_dist = util::col_dist(s, target);
	auto dangerous_checker = (piece == pawn || piece == rook || piece == queen);
	return dangerous_checker && row_dist <= 1 && col_dist <= 1;
}

bool position::is_legal(const Move& m) {
	Square f = Square(m.f);
	Square t = Square(m.t);
	Piece p = piece_on(f);
	Movetype mt = Movetype(m.type);
	if (mt == Movetype::no_type) 
		return false;
	Square ks = king_square();
	Color us = to_move();
	Color them = Color(us ^ 1);
	Square eks = king_square(them);
	auto pc = pcs.bitmap[them];
	bool ispromotion = _is_promotion(mt);
	bool iscappromotion = is_cap_promotion(mt);
	bool slider = (p == rook || p == bishop || p == queen);
	bool pawncapture = iscappromotion || mt == capture || mt == ep;

	// basic checks on hash moves
	if (p == Piece::no_piece) return false;
	if (f == t) return false;
	if (t == eks) return false;
	if (color_on(t) == us) return false;
	if (color_on(f) != us) return false;
	if ((mt == ep || mt == quiet || ispromotion) && piece_on(t) != Piece::no_piece) 
		return false;

	if ((ispromotion || iscappromotion) && p != pawn) 
		return false;

	if ((mt == capture || iscappromotion) && (color_on(t) != them))
		return false;

	if ((mt == capture || iscappromotion) && piece_on(t) == Piece::king)
		return false;

	if (mt == ep && t != ifo.eps)
		return false;

	if (mt == ep && p != pawn)
		return false;
	
	if (p == pawn) {
		if (util::row_dist(f, t) != 1 && util::row_dist(f, t) != 2)
			return false;

		if (us == black && util::row(f) < util::row(t)) 
			return false;

		if (us == white && util::row(f) > util::row(t))
			return false;

		if ((mt == quiet || ispromotion) && util::col_dist(f, t) != 0)
			return false;

		if ((!ispromotion && !iscappromotion) && (util::row(t) == Row::r1 || util::row(t) == Row::r8))
			return false;

		if (pawncapture && (util::row_dist(f, t) != 1 || util::col_dist(f, t) != 1)) 
			return false;

		if (mt == quiet && util::row_dist(f, t) == 2) {

			if (us == white && util::row(f) != 1) return false;
			if (us == black && util::row(f) != 6) return false;

			Square s = Square(f + (us == white ? 8 : -8));
			if (piece_on(s) != no_piece)
				return false;
		}
	}

	if (p == knight) {
		int rd = util::row_dist(f, t);
		int cd = util::col_dist(f, t);
		if (std::min(rd, cd) != 1 || std::max(rd, cd) != 2) 
			return false;
	}

	if (p == rook) {
		if (!util::same_row(f, t) && !util::same_col(f, t)) return false;
	}

	if (p == bishop) {
		if (!util::on_diagonal(f, t)) return false;
	}

	if (p == queen) {
		if (!util::same_row(f, t) && !util::same_col(f, t) && !util::on_diagonal(f, t)) return false;
	}

	if (p == king)
	{
		if (!util::same_row(f, t) && !util::same_col(f, t) && !util::on_diagonal(f, t))
			return false;

		if ((mt == quiet || mt == capture) && util::same_row(f, t) && util::col_dist(f, t) != 1)
			return false;

		if ((mt == quiet || mt == capture) && util::same_col(f, t) && util::row_dist(f, t) != 1)
			return false;

		if (util::on_diagonal(f, t) && (util::row_dist(f, t) != 1 || util::col_dist(f, t) != 1))
			return false;
	}

	// pinned
	if ((bitboards::squares[f] & ifo.pinned[ifo.stm]) && !util::aligned(ks, f, t)) 
		return false;

	// ep can uncover a discovered check
	if (mt == ep) {
		Square csq = Square(t + (them == white ? 8 : -8));
		U64 msk = (all_pieces() ^ bitboards::squares[f] ^ bitboards::squares[csq]) |
			bitboards::squares[t];

		return 
			(
			((magics::attacks<bishop>(msk, ks) & (pc[queen] | pc[bishop])) == 0ULL) &&
			((magics::attacks<rook>(msk, ks) & (pc[queen] | pc[rook])) == 0ULL)
			);
	}

	// can castle flag has already been checked in movegen
	if (mt == castle_ks || mt == castle_qs) {
		if (in_check())
			return false;
		if (p != king)
			return false;
		if (piece_on(us == white ? E1 : E8) != king) 
			return false;
		if (us == white && color_on(E1) != white) 
			return false;
		if (us == black && color_on(E8) != black) 
			return false;
		
		Square s1 = no_square;
		Square s2 = no_square;
		if (mt == castle_ks)
		{
			s1 = (us == white ? F1 : F8);
			s2 = (us == white ? G1 : G8);
		}
		else
		{
			s1 = (us == white ? D1 : D8);
			s2 = (us == white ? C1 : C8);
		}

		if (mt == castle_ks && !can_castle_ks())
			return false;
		if (mt == castle_qs && !can_castle_qs())
			return false;

		if (mt == castle_ks) {
			if (piece_on(us == white ? F1 : F8) != no_piece) return false;
			if (piece_on(us == white ? G1 : G8) != no_piece) return false;
			if (piece_on(us == white ? H1 : H8) != rook) return false;
			if (us == white && color_on(H1) != white) return false;
			if (us == black && color_on(H8) != black) return false;
		}
		if (mt == castle_qs) {
			if (piece_on(us == white ? B1 : B8) != no_piece) return false;
			if (piece_on(us == white ? C1 : C8) != no_piece) return false;
			if (piece_on(us == white ? D1 : D8) != no_piece) return false;
			if (piece_on(us == white ? A1 : A8) != rook) return false;
			if (us == white && color_on(A1) != white) return false;
			if (us == black && color_on(A8) != black) return false;
		}

		if (is_attacked(s1, us, them) || is_attacked(s2, us, them))
			return false;

		return true;
	}

	// is the king move legal
	else if (p == king) {

		U64 msk = (all_pieces() ^ bitboards::squares[ks]);
		if (is_attacked(t, us, them, msk))
			return false;
	}

	// if we are in check, hash move should either capture the checker
	// or block the check (king moves checked previously)
	if (in_check() && p != king) {
		U64 checks = checkers();
		if (bits::more_than_one(checks)) 
			return false;

		Square check_f = Square(bits::pop_lsb(checks));

		if ((mt == capture || iscappromotion) && t != check_f)
			return false;

		Piece checker = piece_on(check_f);

		if ((mt == quiet || ispromotion) && (checker == pawn || checker == knight))
			return false;

		if ((mt == quiet || ispromotion) && (checker == bishop || checker == rook || checker == queen)) {
			U64 empty = ~all_pieces();
			U64 evasion_target = bitboards::between[check_f][king_square()] & empty;
			U64 block_bb = evasion_target & bitboards::squares[t];
			if (block_bb == 0ULL) 
				return false;
		}
	}
	
	if (slider) {
		U64 bb = bitboards::between[f][t];
		bb ^= bitboards::squares[f];
		bb ^= bitboards::squares[t];
		bb &= all_pieces();
		if (bb != 0ULL) 
			return false;
	}

	return true;
}

U64 position::pinned(const Color us) {
	const Color them = Color(us ^ 1);
	const Square ks = king_square(us);
	U64 pinned = 0ULL;
	U64 bs = pcs.bitmap[them][bishop] | pcs.bitmap[them][queen];
	U64 rs = pcs.bitmap[them][rook] | pcs.bitmap[them][queen];

	U64 sliders = (bs & bitboards::battks[ks]) | (rs & bitboards::rattks[ks]);

	if (sliders == 0ULL) {
		return pinned;
	}

	do {
		int sq = bits::pop_lsb(sliders);

		if (!util::aligned(sq, ks)) continue;

		U64 tmp = (bitboards::between[sq][ks] & all_pieces()) ^
			(bitboards::squares[ks] | bitboards::squares[sq]);

		if (!bits::more_than_one(tmp)) pinned |= tmp;

	} while (sliders);

	return pinned & pcs.bycolor[us];
}

bool position::in_check() const {
	return ifo.incheck;
}

bool position::in_dangerous_check()
{
	if (!in_check())
		return false;

	if (ifo.checkers == 0ULL)
		return false;

	if (bits::more_than_one(ifo.checkers))
		return true;

	auto c = bits::pop_lsb(ifo.checkers);
	auto piece = piece_on(Square(c));
	auto row_dist = util::row_dist(c, (int)king_square());
	auto col_dist = util::col_dist(c, (int)king_square());
	auto us = to_move();
	auto them = to_move() ^ 1;
	auto dangerous_checker = (piece == pawn || piece == rook || piece == queen);

	return dangerous_checker && row_dist <= 1 && col_dist <= 1 && is_attacked(Square(c), Color(us), Color(them));
}


bool position::is_attacked(const Square& s, const Color& us, const Color& them, U64 m) const {
	// is square owned by "us" attacked by "them"
	auto p = pcs.bitmap[them];
	U64 stepper_attacks = ((bitboards::pattks[us][s] & p[pawn]) |
		(bitboards::nmask[s] & p[knight]) |
		(bitboards::kmask[s] & p[king]));

	if (stepper_attacks != 0ULL) return true;

	if (m == 0ULL) m = all_pieces();

	return ((magics::attacks<bishop>(m, s) & (p[queen] | p[bishop]) ||
		(magics::attacks<rook>(m, s) & (p[queen] | p[rook]))));
}

U64 position::attackers_of(const Square& s, const U64& m) const {
	auto p = [this](const Color& c, const Piece& p) { return pcs.bitmap[c][p]; };
	U64 battck = magics::attacks<bishop>(m, s);
	U64 rattck = magics::attacks<rook>(m, s);
	U64 qattck = battck | rattck;

	return ((bitboards::pattks[black][s] & p(white, pawn)) |
		(bitboards::pattks[white][s] & p(black, pawn)) |
		(bitboards::nmask[s] & (p(black, knight) | p(white, knight))) |
		(bitboards::kmask[s] & (p(black, king) | p(white, king))) |
		(battck & (p(white, bishop) | p(black, bishop))) |
		(rattck & (p(white, rook) | p(black, rook))) |
		(qattck & (p(white, queen) | p(black, queen))));
}

U64 position::attackers_of2(const Square& s, const Color& c) const {
	// attackers of square "s" by color "c"
	U64 m = all_pieces();
	auto p = pcs.bitmap[c];
	U64 battck = magics::attacks<bishop>(m, s);
	U64 rattck = magics::attacks<rook>(m, s);
	U64 qattck = battck | rattck;

	return ((bitboards::pattks[c ^ 1][s] & p[pawn]) |
		(bitboards::nmask[s] & p[knight]) |
		(bitboards::kmask[s] & p[king]) |
		(battck & p[bishop]) |
		(rattck & p[rook]) |
		(qattck & p[queen]));
}

void position::set_piece(const char& p, const Square& s) {
	auto idx = std::distance(SanPiece.begin(), std::find(SanPiece.begin(), SanPiece.end(), p));
	if (idx < 0) return; // error

	Color color = (idx < 6 ? white : black);
	Piece piece = Piece(idx < 6 ? idx : idx - 6);
	pcs.set(color, piece, s, ifo);
	if (piece == king) ifo.ks[color] = s;
}

void position::clear() {
	pcs.clear();
	stats.clear();
	hidx = 0;
	thread_id = 0;
	nodes_searched = 0;
	qnodes_searched = 0;
	std::memset(&ifo, 0, sizeof(info));
	ifo = {};
	ifo.repkey = 0ULL;
	ifo.key = 0ULL;
	ifo.pawnkey = 0ULL;
}


void position::print() const {
	std::cout << "   +---+---+---+---+---+---+---+---+" << std::endl;
	for (Row r = r8; r >= r1; --r) {
		std::cout << " " << r + 1 << " ";

		for (Col c = A; c <= H; ++c) {
			Square s = Square(8 * r + c);
			if (pcs.piece_on[s] != no_piece) {
				Piece p = pcs.piece_on[s];
				std::cout << "| "
					<< (pcs.color_on[s] == Color::white ? SanPiece[p] : SanPiece[p + 6])
					<< " ";
			}
			else std::cout << "|   ";
		}
		std::cout << "|" << std::endl;
		std::cout << "   +---+---+---+---+---+---+---+---+" << std::endl;
	}
	std::cout << "     a   b   c   d   e   f   g   h  " << std::endl;
}
