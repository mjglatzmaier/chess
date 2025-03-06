#pragma once
#ifndef HAVOC_MOVE_ORDER_H
#define HAVOC_MOVE_ORDER_H

#include <array>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <functional>

#include "types.h"

class position;
class Movegen;


namespace haVoc {


	struct Movehistory {
	private:
		std::array<std::array<std::array<int, squares>, squares>, colors> history;
		std::array<std::array<Move, squares>, squares> counters;
		float counter_move_bonus = 1.0f;
		float threat_evasion_bonus = 1.0f;

	public:
		Movehistory() { 
			clear(); 
		}

		Movehistory& operator=(const Movehistory& mh);
		
		void update(const position& p,
			const Move& m,
			const Move& previous,
			const int16& depth,
			const Score& eval,
			const std::vector<Move>& quiets,
			Move* killers);

		void clear();
		
		int score(const Move& m, 
			const Color& color,
			const Move& previous,
			const Move& followup,
			const Move& threat) const;

		int score(const Move& m, const Color& c) const;
	};

	struct ScoredMove {
		ScoredMove() : m(Move{}), s(Score::ninf) { }
		ScoredMove(Move& mv, Score& sc) : m(mv), s(sc) { }
		ScoredMove(Move&& mv, Score&& sc) : m(mv), s(sc) { }
		Move m;
		Score s;
		bool operator>(const ScoredMove& o) { return s > o.s; }
		bool operator<(const ScoredMove& o) { return s < o.s; }
		ScoredMove& operator=(const ScoredMove& o);
	};


	typedef std::function<Score(const position& p, const Move& m, const Move& prev, const Move& followup, const Move& threat, node* stack )> ScoreFunc;
	
	class ScoredMoves {
	private:
		std::vector<ScoredMove> m_moves;
		unsigned m_start = 0;
		unsigned m_end = 0;

		void load_and_score(const position& p, Movegen* moves, const std::vector<Move>& filters, const Move& previous, const Move& followup, const Move& threat, node* stack, ScoreFunc score_lambda);
		void sort(const Score& cutoff);

	public:
		ScoredMoves() { m_moves.clear(); }
		ScoredMoves(const position& p, Movegen* m, const std::vector<Move>& filters, const Move& previous, const Move& followup, const Move& threat, node* stack, ScoreFunc score_lambda, Score cutoff) {
			m_moves.clear();
			load_and_score(p, m, filters, previous, followup, threat, stack, score_lambda);
			sort(cutoff);
		}
		~ScoredMoves() {}

		int operator++() { return m_start++; }
		int operator--() { return m_start--; }
		int start() { return m_start; }
		ScoredMove front() { return m_moves[m_start]; }
		ScoredMove operator[](int idx) { return m_moves[idx]; }
		bool end() { return m_start >= m_end; }
		unsigned size() { return m_end - m_start; }
		void skip_rest() { m_start = m_end; }
		void create_chunk(const Score& cutoff);
	};


	class Moveorder {
	protected:
		std::shared_ptr<ScoredMoves> m_captures;
		std::shared_ptr<ScoredMoves> m_quiets;
		std::shared_ptr<Movegen> m_movegen;
		std::vector<Move> _killerMoves;
		node* m_stack;

		bool m_incheck = false;
		bool m_isendgame = false;
		bool m_debug = false;
		int _rootCounter = 0;

		enum Phase { HashMove, MateKiller1, MateKiller2, InitCaptures, GoodCaptures, Killer1, Killer2, InitQuiets, GoodQuiets, BadCaptures, BadQuiets, End };
		Phase m_phase = HashMove;


		virtual void next_phase();

	public:
		Moveorder();
		Moveorder(position& p, Move& hashmove, node* stack, bool debug);
		Moveorder(position& p, Move& hashmove, node* stack);
		Moveorder(const Moveorder& mo) = delete;
		Moveorder(const Moveorder&& mo) = delete;
		Moveorder& operator=(const Moveorder& o) = delete;
		Moveorder& operator=(const Moveorder&& o) = delete;
		virtual ~Moveorder() { }

		virtual bool next_move(position& pos, Move& m, const Move& previous, const Move& followup, const Move& threat, bool skipQuiets, bool rootMvs);
	};



	class QMoveorder : public Moveorder {
	protected:
		enum Phase { HashMove, MateKiller1, MateKiller2,  InitCaptures, GoodCaptures, Killer1, Killer2, BadCaptures, InitQuiets, GoodQuiets, BadQuiets, End };
		Phase m_phase = HashMove;

		bool valid_qmove(const Move& m);
		void next_phase() override;

	public:
		QMoveorder();
		QMoveorder(position& p, Move& hashmove, node* stack);
		QMoveorder(const QMoveorder& mo) = delete;
		QMoveorder(const QMoveorder&& mo) = delete;
		QMoveorder& operator=(const QMoveorder& o) = delete;
		QMoveorder& operator=(const QMoveorder&& o) = delete;
		~QMoveorder() { }

		bool next_move(position& pos, Move& m, const Move& previous, const Move& followup, const Move& threat, bool skipQuiets, bool rootMoves = false) override;
	};
}

#endif