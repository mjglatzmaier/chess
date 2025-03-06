#include "order.h"
#include "move.h"
#include "position.h"
#include "utils.h"
#include "uci.h"
#include "squares.h"
#include "material.h"
#include "evaluate.h"

namespace haVoc {



	std::vector<int> mvals{ 10, 30, 33, 48, 91, 200 };
	bool endgame = false;


	//---------------- Sorting lambdas ---------------//
	Score score_captures(const position& p, const Move& m, const Move& prev, const Move& followup, const Move& threat, node* stack) {
		Score s = Score(p.see(m));
		s = Score(s + stack->bestMoveHistory->bm[p.to_move()][m.f][m.t]);
		return s;
	}

	Score score_qcaptures(const position& p, const Move& m, const Move& prev, const Move& followup, const Move& threat, node* stack) {
		Score s = Score(p.see(m));
		return s;
	}

	Score score_quiets(const position& p, const Move& m, const Move& prev, const Move& followup, const Move& threat, node* stack) {
		auto tomove = p.to_move();
		auto stats = p.history_stats();
		auto s = Score(stats->score(m, (Color)(tomove), prev, followup, threat));
		s = Score(s + stack->bestMoveHistory->bm[tomove][m.f][m.t]);

		// Use a LUT to order quiet moves without any history data
		if (s == Score::draw) {
			auto piece = p.piece_on(Square(m.f));
			auto ss = (p.to_move() == white ?
				square_score<white>(piece, Square(m.t)) - square_score<white>(piece, Square(m.f)) :
				square_score<black>(piece, Square(m.t)) - square_score<black>(piece, Square(m.f)));
			ss *= 10;
			s = Score((int)ss + s);
		}
		s = std::max(Score::ninf, s);
		return s;
	}



	//-------------- Move history impl ----------------//
	Movehistory& Movehistory::operator=(const Movehistory& mh) {
		std::copy(std::begin(mh.history), std::end(mh.history), std::begin(history));
		std::copy(std::begin(mh.counters), std::end(mh.counters), std::begin(counters));
		return (*this);
	}

	void Movehistory::update(const position& p,
		const Move& m,
		const Move& previous,
		const int16& depth,
		const Score& eval,
		const std::vector<Move>& quiets,
		Move* killers) {

		const Color c = p.to_move();
		int score = pow(depth, 2);
		if (m.type == Movetype::quiet) {
			history[c][m.f][m.t] += score;
			counters[previous.f][previous.t] = m;
			if (eval < Score::mate_max_ply &&
				m != killers[2] &&
				m != killers[3] &&
				m != killers[0]) {
				killers[1] = killers[0];
				killers[0] = m;
			}
			for (auto& q : quiets) {
				if (m.f == q.f) continue;
				history[c][q.f][q.t] -= score;
			}
		}

		// mate killers
		if (eval >= Score::mate_max_ply &&
			m != killers[0] &&
			m != killers[1] &&
			m != killers[2]) {
			killers[3] = killers[2];
			killers[2] = m;
		}
	}

	void Movehistory::clear() {
		for (auto& v : history) { for (auto& w : v) { std::fill(w.begin(), w.end(), 0); } }

		Move empty; empty.set(0, 0, Movetype::no_type);
		for (auto& v : counters) { std::fill(v.begin(), v.end(), empty); }
	}

	int Movehistory::score(const Move& m, const Color& c) const {
		return history[c][m.f][m.t];
	}

	int Movehistory::score(const Move& m,
		const Color& c,
		const Move& previous,
		const Move& followup,
		const Move& threat) const {
		int score = history[c][m.f][m.t];
		if (counters[previous.f][previous.t] == m)
			score += counter_move_bonus;
		if (followup.type != Movetype::no_type && followup.f == m.t && followup.t == m.f)
			score -= counter_move_bonus;

		return score;
	}

	//---------------- Scored move impl ---------------//
	ScoredMove& ScoredMove::operator=(const ScoredMove& o) {
		this->m = o.m;
		this->s = o.s;
		return *this;
	}



	//---------------- Scored moves array ---------------//
	void ScoredMoves::load_and_score(const position& p, Movegen* moves, const std::vector<Move>& filters, const Move& previous, const Move& followup, const Move& threat, node* stack, ScoreFunc score_lambda)
	{
		m_start = m_end = 0;
		for (int i = 0; i < moves->size(); ++i) {
			
			auto m = (*moves)[i];

			// skip hash moves and killers
			if (m == filters[0] || m == filters[1] || m == filters[2] ||
				m == filters[3] || m == filters[4])
				continue;

			Score sc = score_lambda(p, m, previous, followup, threat, stack);
			m_moves.emplace_back(ScoredMove(m, sc));
		}
	}

	void ScoredMoves::sort(const Score& cutoff)
	{
		unsigned N = m_moves.size();
		ScoredMove key;
		int j;
		for (unsigned i = m_start + 1; i < N; ++i) {
			key = m_moves[i];
			j = i - 1;

			while (j >= 0 && m_moves[j] < key ) {
				m_moves[j + 1] = m_moves[j];
				--j;
			}
			m_moves[j + 1] = key;
		}
		create_chunk(cutoff);
	}

	void ScoredMoves::create_chunk(const Score& cutoff) {
		m_start = m_end;
		for (int i = m_start; i < m_moves.size(); ++i) {
			if (m_moves[i].s >= cutoff)
				m_end++;
		}
	}


	
	//---------------- Moveorder impl ---------------//
	Moveorder::Moveorder() { }

	Moveorder::Moveorder(position& p, Move& hashmove, node* stack) {
		m_incheck = p.in_check();
		
		// TODO: FIXME
		//einfo ei = {};
		//ei.me = mtable.fetch(p);
		m_isendgame = false; // ei.me->is_endgame();
		endgame = m_isendgame;
		m_stack = stack;
		
		m_movegen = std::make_shared<Movegen>(p);
		_killerMoves = { hashmove, stack->killers[0], stack->killers[1], stack->killers[2], stack->killers[3] };
		
	}

	Moveorder::Moveorder(position& p, Move& hashmove, node* stack, bool debug) {
		//einfo ei = {};
		//ei.me = mtable.fetch(p);
		m_isendgame = false;
		endgame = m_isendgame;
		m_stack = stack;
		m_movegen = std::make_shared<Movegen>(p);
		_killerMoves = { hashmove, stack->killers[0], stack->killers[1], stack->killers[2], stack->killers[3] };
		m_debug = true;
	}

	bool Moveorder::next_move(position& pos, Move& m, const Move& previous, const Move& followup, const Move& threat, bool skipQuiets, bool rootMvs) {
		m = {};

		if (rootMvs)
		{
			if (_rootCounter < pos.root_moves.size())
			{
				m = pos.root_moves[_rootCounter++].pv[0];
				return true;
			}
			return false;
		}

		switch (m_phase) {
		case HashMove:
				m = _killerMoves[0];
			break;
		case MateKiller1:
			if (_killerMoves[3] != _killerMoves[0])
				m = _killerMoves[3];
			break;
		case MateKiller2:
			if (_killerMoves[4] != _killerMoves[0]) // && _killerMoves[4] != _killerMoves[3])
				m = _killerMoves[4];
			break;
		case Killer1:
			if (_killerMoves[1] != _killerMoves[0])
				m = _killerMoves[1];
			break;
		case Killer2:
			if ((_killerMoves[2] != _killerMoves[0]))// && _killerMoves[2] != _killerMoves[1])
				m = _killerMoves[2];
			break;
		case InitCaptures:
			m_movegen->reset();
			m_movegen->generate<capture, pieces>();
			m_captures = std::make_shared<ScoredMoves>(pos, m_movegen.get(), _killerMoves, previous, followup, threat, m_stack, score_captures,  Score::draw);
			break;
		case GoodCaptures:
		case BadCaptures:
			if (!m_captures->end()) {
				m = m_captures->front().m;
				m_captures->operator++();
			}
			break;
		case InitQuiets:
			if (!skipQuiets) {
				m_movegen->reset();
				m_movegen->generate<quiet, pieces>();
				m_quiets = std::make_shared<ScoredMoves>(pos, m_movegen.get(), _killerMoves, previous, followup, threat, m_stack, score_quiets, Score::draw);
			}
			else
				m_quiets = std::make_shared<ScoredMoves>();
			break;
		case GoodQuiets:
		case BadQuiets:
			if (skipQuiets) {
				m_quiets->skip_rest();
			}
			else if (!m_quiets->end()) {
				m = m_quiets->front().m;
				m_quiets->operator++();
			}
			break;
		case End:
			return false;
		}
		next_phase();
		return true;  
	}


	void Moveorder::next_phase() {
		if ((m_phase == HashMove) ||
				(m_phase == MateKiller1) ||
				(m_phase == MateKiller2) ||
				(m_phase == Killer1) ||
				(m_phase == Killer2) ||
				(m_phase == InitCaptures) ||
				(m_phase == InitQuiets))
			m_phase = Phase(m_phase + 1);
		else if ((m_phase == GoodCaptures || m_phase == BadCaptures) && m_captures->end())
		{
			m_captures->create_chunk(Score::ninf); // mark bad captures
			m_phase = Phase(m_phase + 1);
		}
		else if ((m_phase == GoodQuiets || m_phase == BadQuiets) && m_quiets->end())
		{
			m_quiets->create_chunk(Score::ninf); // mark bad quiet moves
			m_phase = Phase(m_phase + 1);
		}
	}




	//---------------- QMoveorder impl ---------------//
	QMoveorder::QMoveorder() { }

	QMoveorder::QMoveorder(position& p, Move& hashmove, node* stack) : Moveorder(p, hashmove, stack) { }


	bool QMoveorder::next_move(position& pos, Move& m, const Move& previous, const Move& followup, const Move& threat, bool skipQuiets, bool rootMoves) {
		m = {};
		switch (m_phase) {
		case HashMove:
			if (valid_qmove(_killerMoves[0]) && _killerMoves[0].type != Movetype::no_type)
				m = _killerMoves[0];
			break;
		case MateKiller1:
			if (valid_qmove(_killerMoves[3]) && _killerMoves[3] != _killerMoves[0])
				m = _killerMoves[3];
			break;
		case MateKiller2:
			if (valid_qmove(_killerMoves[4]) && _killerMoves[4] != _killerMoves[0] && _killerMoves[4] != _killerMoves[3])
				m = _killerMoves[4];
			break;
		case Killer1:
			if (valid_qmove(_killerMoves[1]) && _killerMoves[1] != _killerMoves[0])
				m = _killerMoves[1];
			break;
		case Killer2:
			if (valid_qmove(_killerMoves[2]) && (_killerMoves[2] != _killerMoves[0]) && _killerMoves[2] != _killerMoves[1])
				m = _killerMoves[2];
			break;
		case InitCaptures:
			m_movegen->reset();
			m_movegen->generate<capture, pieces>();
			m_captures = std::make_shared<ScoredMoves>(pos, m_movegen.get(), _killerMoves, previous, followup, threat, m_stack, score_qcaptures, Score::draw);
			break;
		case GoodCaptures:
		case BadCaptures:
			if (m_captures.get() && !m_captures->end()) {
				m = m_captures->front().m;
				m_captures->operator++();
			}
			break;
		case InitQuiets:
			if (m_incheck && !skipQuiets) {
				m_movegen->reset();
				m_movegen->generate<quiet, pieces>();
				m_quiets = std::make_shared<ScoredMoves>(pos, m_movegen.get(), _killerMoves, previous, followup, threat, m_stack, score_quiets, Score::draw);
			}
			else 
				m_quiets = std::make_shared<ScoredMoves>();
			break;
		case GoodQuiets:
		case BadQuiets:
			if (skipQuiets)
				break;
			if (!m_quiets->end()) {
				m = m_quiets->front().m;
				m_quiets->operator++();
			}
			break;
		case End:
			return false;
		}
		next_phase();
		return true;
	}

	void QMoveorder::next_phase() {
		if ((m_phase == HashMove) ||
			(m_phase == MateKiller1) ||
			(m_phase == MateKiller2) ||
			(m_phase == Killer1) ||
			(m_phase == Killer2) ||
			(m_phase == InitCaptures) ||
			(m_phase == InitQuiets))
			m_phase = Phase(m_phase + 1);
		else if ((m_phase == GoodCaptures || m_phase == BadCaptures) && m_captures->end()) {
			m_captures->create_chunk(Score::ninf); // mark bad captures
			m_phase = Phase(m_phase + 1);
		}
		else if ((m_phase == GoodQuiets || m_phase == BadQuiets) && m_quiets->end()) {
			m_quiets->create_chunk(Score::ninf); // mark bad quiet moves
			m_phase = Phase(m_phase + 1);
		}
	}

	bool QMoveorder::valid_qmove(const Move& m) {
		return m_incheck ||
			m.type == Movetype::capture ||
			m.type == Movetype::ep ||
			m.type == Movetype::capture_promotion_q ||
			m.type == Movetype::capture_promotion_r ||
			m.type == Movetype::capture_promotion_b ||
			m.type == Movetype::capture_promotion_n;
	}
}