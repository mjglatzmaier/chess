
#include <memory>
#include <condition_variable>
#include <iostream>
#include <fstream>

#include "position.h"
#include "types.h"
#include "hashtable.h"
#include "utils.h"
#include "evaluate.h"
#include "order.h"
#include "material.h"
#include "options.h"


std::ofstream debug_file;
std::condition_variable cv;
volatile double elapsed = 0;

struct thread_entry {
	bool searching;
	int id = -1;
};
const size_t mv_hash_sz = 1024;
thread_entry searching_positions[mv_hash_sz];


inline size_t get_idx(position& p, const Move& m) {
	return (p.key() & (mv_hash_sz - 1));
}

inline bool is_searching(position& p, const Move& m, const int& id) {
	auto e = searching_positions[get_idx(p, m)];
	return e.id != id && e.searching;
}

inline bool main_thread(position& p) {
	return p.id() == 0;
}

inline void set_searching(position& p, const Move& m, const int& id) {
	size_t idx = get_idx(p, m);
	if (!searching_positions[idx].searching) {
		searching_positions[idx].searching = true;
		searching_positions[idx].id = id;
	}
}

inline void unset_searching(position& p, const Move& m) {
	size_t idx = get_idx(p, m);
	searching_positions[idx].searching = false;
	searching_positions[idx].id = -1;
}

inline unsigned reduction(bool pv_node, bool improving, int d, int mc) {
	return bitboards::reductions[(int)pv_node][(int)improving]
		[std::max(0, std::min(d, 64 - 1))][std::max(0, std::min(mc, 64 - 1))];
}

inline float razor_margin(int depth) {
	return 950 * (1 - exp((depth - 64.0) / 20.0));
}

inline float lazy_eval_margin(int depth, bool advanced_pawn) { //, bool pv_node, bool improving) {
	return (advanced_pawn ? -1 : 225 * (1 - exp((depth - 64.0) / 20.0)));
}

inline float lazy_eval_margin_search(int depth, bool advanced_pawn) {
	return (advanced_pawn ? -1 : 225 * (1 - exp((depth - 64.0) / 20.0)));
}

inline int futility_move_count(const bool& improving, const U16& depth) {
	return (6 + depth * depth) / (2 - improving);
}



// ------- Main searching methods ------ //
std::vector<std::unique_ptr<position>> mPositions;
int selDepth = 0;
size_t hashHits = 0;
std::mutex search_mtx;
const std::vector<float> material_vals{ 100.0f, 300.0f, 315.0f, 480.0f, 910.0f };

void Search::start(position& p, limits& lims, bool silent) {

	mPositions.clear();

	Threadpool<Workerthread> timer_thread(1);

	elapsed = 0;
	UCI_SIGNALS.stop = false;

	p.set_nodes_searched(0);
	p.set_qnodes_searched(0);

	// load the root moves
	Movegen mvs(p);
	mvs.generate<pseudo_legal, pieces>();
	p.root_moves.clear();
	for (int i = 0; i < mvs.size(); ++i) {
		if (!p.is_legal(mvs[i]))
			continue;
		p.root_moves.push_back(Rootmove(mvs[i]));
	}

	for (unsigned i = 0; i < SearchThreads.size(); ++i) {
		mPositions.emplace_back(std::make_unique<position>(p));
		mPositions[i]->set_id(i);
	}

	U16 depth = (lims.depth > 0 ? lims.depth : 64); // maxdepth
	searching = true;

	timer_thread.enqueue(search_timer, p, lims);


	// Launch worker threads
	if (SearchThreads.size() > 1) {
		for (unsigned i = 1; i < SearchThreads.size(); ++i)
			SearchThreads.enqueue(iterative_deepening, *mPositions[i], depth, silent);
	}

	// Launch main thread
	SearchThreads.enqueue(iterative_deepening, *mPositions[0], depth, silent);


	SearchThreads.wait_finished();
	UCI_SIGNALS.stop = true;


	U64 nodes = 0ULL;
	U64 qnodes = 0ULL;
	Rootmoves bestRoots;
	Score max = Score::ninf;
	for (auto& t : mPositions) {
		if (!silent) {
			std::cout << "id: " << t->id() << " " << t->nodes() << " " << t->qnodes() << std::endl;
		}
		nodes += t->nodes();
		qnodes += t->qnodes();

		if (t->root_moves.size() > 0 && t->root_moves[0].score > max) {
			max = t->root_moves[0].score;
			bestRoots = t->root_moves;
		}
	}

	if (bestRoots.size() <= 0)
		bestRoots = mPositions[0]->root_moves;

	if (!silent) {
		std::cout << "bestmove " << uci::move_to_string(bestRoots[0].pv[0]);
		if (bestRoots[0].pv.size() > 1)
			std::cout << " ponder " << uci::move_to_string(bestRoots[0].pv[1]);
		std::cout << std::endl;
	}

	searching = false;

	if (p.debug_search) {
		debug_file.close();
	}
}

void Search::search_timer(position& p, limits& lims) {
	util::clock c;
	c.start();
	bool fixed_time = lims.movetime > 0;
	int delay = 1; // ms
	double time_limit = estimate_max_time(p, lims);
	auto sleep = [delay]() { std::this_thread::sleep_for(std::chrono::milliseconds(delay)); };

	if (fixed_time) {
		do {
			elapsed += c.elapsed_ms();
			sleep();
		} while (!UCI_SIGNALS.stop && searching && elapsed <= lims.movetime);
	}
	else if (time_limit > -1) {
		// dynamic time estimate in a real game
		do {
			elapsed += c.elapsed_ms();
			sleep();
		} while (!UCI_SIGNALS.stop && searching && elapsed <= time_limit);
	}
	else {
		do {
			// analysis mode (infinite time)
			elapsed += c.elapsed_ms();
			sleep();
		} while (!UCI_SIGNALS.stop && searching);
	}
	UCI_SIGNALS.stop = true;
	return;
}

double Search::estimate_max_time(position& p, limits& lims) {
	double time_per_move_ms = 0;
	if (lims.infinite || lims.ponder || lims.depth > 0) return -1;
	else {
		bool sudden_death = lims.movestogo == 0; // no moves until next time control
		bool exact_time = lims.movetime != 0; // searching for an exact number of ms?
		double remainder_ms = (p.to_move() == white ? lims.wtime + lims.winc : lims.btime + lims.binc);
		//material_entry * me = mtable.fetch(p);
		//bool endgame = me->endgame;
		double moves_to_go = 45.0 - 22.5; // (!endgame ? 22.5 : 30.0);

		if (sudden_death && !exact_time) {
			time_per_move_ms = 1.5 * remainder_ms / moves_to_go;
		}
		else if (exact_time) return (double)lims.movetime;
		else if (!sudden_death) {
			time_per_move_ms = remainder_ms / lims.movestogo;
		}
	}
	return time_per_move_ms;
}

void Search::iterative_deepening(position& p, U16 depth, bool silent) {
	int16 alpha = ninf;
	int16 beta = inf;
	int16 delta = 65;
	int16 smallDelta = 33;
	Score eval = ninf;


	if (p.params.fixed_depth > 0) {
		depth = p.params.fixed_depth;
	}

	const unsigned stack_size = 64 + 4;
	node stack[stack_size];
	Move pv[Depth::MAX_PLY + 4];

	(stack + 2)->pv = pv;


	// Main iterative deepening loop
	for (unsigned id = 1 + p.id(); id <= depth; ++id) {

		if (UCI_SIGNALS.stop)
			break;

		(stack+0)->ply = (stack + 1)->ply = (stack + 2)->ply = 0;

		auto failLow = false;
		auto failHigh = false;
		hashHits = 0;

		// 1. aspiration window search
		while (true) {
			if (id >= 2) {
				alpha = std::max(int16(eval - smallDelta), int16(ninf));
				beta = std::min(int16(eval + smallDelta), int16(inf));
				if (failLow) {
					beta = std::min(int16(beta + delta), int16(inf));
					failLow = false;
				}
				if (failHigh) {
					alpha = std::max(int16(alpha - delta), int16(ninf));
					failHigh = false;
				}
			}

			selDepth = 0;
			eval = search<root>(p, alpha, beta, id, stack + 2);

			// bring the best move to the front of the root move array
			std::stable_sort(p.root_moves.begin(), p.root_moves.end());


			if (UCI_SIGNALS.stop)
				break;

			if ((eval <= alpha || eval >= beta))
				readout_pv(stack, p.root_moves, eval, Score(alpha), Score(beta), id);

			if (eval <= alpha) {
				delta += delta / 4;
				failHigh = true;
			}
			else if (eval >= beta) {
				delta += delta / 4;
				failLow = true;
			}
			else break;
		}


		// 2. Print PV to UI
		if (main_thread(p) && !UCI_SIGNALS.stop) {

			if (!silent)
				readout_pv(stack, p.root_moves, eval, Score(alpha), Score(beta), id);

			if (id == depth) {
				UCI_SIGNALS.stop = true;
				break;
			}
		}
	}
}


template<Nodetype type>
Score Search::search(position& pos, int16 alpha, int16 beta, U16 depth, node* stack) {

	if (UCI_SIGNALS.stop)
		return Score::draw;

	assert(alpha < beta);

	Score bestScore = Score::ninf;
	Move best_move = {};
	best_move.type = Movetype::no_type;
	size_t deferred = 0;

	Move ttm = {}; ttm.type = Movetype::no_type; // refactor me
	Move pv[Depth::MAX_PLY + 4];
	Score ttvalue = Score::ninf;

	bool in_check = pos.in_check();
	std::vector<Move> quiets;
	stack->in_check = in_check;
	stack->ply = (stack - 1)->ply + 1;


	U16 root_dist = stack->ply;
	const bool root_node = (type == Nodetype::root && stack->ply == 1);
	const bool pvNode = (root_node || type == Nodetype::pv);
	if (pvNode && selDepth < stack->ply + 1 && main_thread(pos))
		selDepth++;

	if (!root_node && !in_check && pos.is_draw())
			return Score::draw;

	{ // mate distance pruning
		Score mating_score = Score(Score::mate - root_dist);
		beta = std::min(mating_score, Score(beta));
		if (alpha >= mating_score)
			return mating_score;

		Score mated_score = Score(Score::mated + root_dist);
		alpha = std::max(mated_score, Score(alpha));
		if (beta <= mated_score)
			return mated_score;
	}

	// hashtable lookup
	auto hashHit = false;
	{
		hash_data e;
		hashHit = ttable.fetch(pos.key(), e);
		if (hashHit) {
			ttm = e.move;
			hashHits++;
			ttvalue = Score(e.score);
			if (!pvNode &&
				e.depth >= depth &&
				(ttvalue >= beta ? e.bound == bound_low : e.bound == bound_high)) {
				pos.stats_update(ttm, (stack - 1)->curr_move, depth, ttvalue, quiets, stack->killers);
				return ttvalue;
			}
		}
	}

	// static evaluation
	const bool anyPawnsOn7th = pos.pawns_near_promotion(); // either side has pawns on 7th
	const bool weHavePawnsOn7th = pos.pawns_on_7th(); // only side to move has pawns on 7th

	Score static_eval = (ttvalue != Score::ninf ? ttvalue :
		(stack-2)->static_eval != ninf && !in_check && (stack-2)->static_eval >= (stack-1)->static_eval ? Score((stack-2)->static_eval + 15) :
		!in_check ? 
		Score(std::lround(eval::evaluate(pos, *SearchThreads[pos.id()], lazy_eval_margin_search(depth, anyPawnsOn7th)))) :
		Score::ninf);
	 
	//Score static_eval = Score::ninf;
	//if (ttvalue == Score::ninf &&
	//	!pvNode && 
	//	!in_check && 
	//	(stack - 2)->static_eval != Score::ninf &&
	//	(stack - 1)->best_move.type == Movetype::quiet)
	//	static_eval = Score((stack - 2)->static_eval + 33);
	//else if (!in_check)
	//	static_eval = Score(std::lround(eval::evaluate(pos, *SearchThreads[pos.id()], lazy_eval_margin_search(depth, anyPawnsOn7th))));
	stack->static_eval = static_eval;
	bool hasStaticValue = static_eval != Score::ninf;


	// 0. Define the forward pruning conditions
	const bool forward_prune =
		(!in_check &&
		!pvNode &&
		(stack - 1)->curr_move.type == Movetype::quiet &&
		!stack->null_search &&
		abs(alpha - beta) == 1 && // only prune in null windows (same condition as !pv_node)
		hasStaticValue);

	// 1. Futility pruning
	//if (forward_prune &&
	//	!weHavePawnsOn7th &&
	//	depth <= 1 &&
	//	static_eval > mated_max_ply &&
	//	static_eval + 1150 < alpha)
	//	return Score(static_eval);

	// 2. Null move pruning
	bool null_move_allowed = (pos.to_move() == white ?
		pos.non_pawn_material<white>() :
		pos.non_pawn_material<black>());
	if (forward_prune &&
		null_move_allowed &&
		depth >= 6 &&
		static_eval - 8 * (64 - depth) >= beta) {

		int16 R = (depth >= 6 ? depth / 2 : 2);
		int16 ndepth = depth - R;

		(stack + 1)->null_search = true;
		pos.do_null_move();
		Score null_eval = Score(ndepth <= 1 ?
			-qsearch<non_pv>(pos, -beta, -beta + 1, 0, stack + 1) :
			-search<non_pv>(pos, -beta, -beta + 1, ndepth, stack + 1));
		pos.undo_null_move();
		(stack + 1)->null_search = false;


		// verification
		//null_eval = Score(ndepth <= 1 ?
		//	qsearch<non_pv>(pos, beta-1, beta, 0, stack) :
		//	search<non_pv>(pos, beta-1, beta, ndepth, stack));

		if (null_eval >= beta)
			return Score(null_eval);
		else {
			Move tm = (stack + 1)->best_move;
			if (tm.type == Movetype::capture && beta - null_eval >= 500)
				stack->threat_move = tm;
		}
	}

	// Main search
	U16 moves_searched = 0;
	haVoc::Moveorder mvs(pos, ttm, stack);
	Move move;
	Move pre_move = (stack - 1)->curr_move;
	Move pre_pre_move = (stack - 2)->curr_move;
	bool improving = stack->static_eval - (stack - 2)->static_eval >= 0;
	auto to_mv = pos.to_move();
	int SEE = 0;
	auto skipQuiets = false;
	auto rootMoves = root_node && pos.root_moves[0].pv.size() > 4;

	set_searching(pos, {}, pos.id());

	while (mvs.next_move(pos, move, pre_move, pre_pre_move, stack->threat_move, skipQuiets, rootMoves)) {

		if (UCI_SIGNALS.stop)
			return Score::draw;

		if (move.type == Movetype::no_type || !pos.is_legal(move))
			continue;


		if (main_thread(pos) && root_node && elapsed > 3000)
			std::cout << "info depth " << depth
			<< " currmove " << uci::move_to_string(move)
			<< " currmovenumber " << moves_searched << std::endl;

		// Set move data
		auto hashOrKiller = (move == ttm) ||
			(move == stack->killers[0]) ||
			(move == stack->killers[1]) ||
			(move == stack->killers[2]) ||
			(move == stack->killers[3]);
		auto isPromotion = pos.is_promotion(move.type);
		auto isCapture =
			(move.type == Movetype::capture) ||
			(move.type == Movetype::ep) ||
			pos.is_cap_promotion(Movetype(move.type));
		auto isQuiet = move.type == Movetype::quiet;
		auto isEvasion = in_check;
		auto advancedPawnPush = (pos.piece_on(Square(move.t)) == Piece::pawn) &&
			(to_mv == white ? util::row(move.t) >= Row::r6 : util::row(move.t) <= Row::r3);
		auto quietFollowingCapture = (stack - 1)->curr_move.type == Movetype::capture && isQuiet;
		auto quietFollowup = (stack - 1)->curr_move.type == Movetype::quiet && isQuiet;
		auto captureFollowup = (stack - 1)->curr_move.type == Movetype::capture && isCapture;
		auto threatResponse = (stack->threat_move.type != Movetype::no_type && stack->threat_move.f == move.t) && isCapture;
		auto dangerousQuietCheck = isQuiet && pos.quiet_gives_dangerous_check(move);

		// 4. Skip moves with negative see scores
		if (isCapture &&
			!hashOrKiller &&
			!pvNode &&
			!isEvasion &&
			!isPromotion &&
			bestScore < alpha &&
			depth <= 1 &&
			moves_searched > 1 &&
			(SEE = pos.see(move)) < 0)
			continue;

		// Debug viz readout
		//{
		//	std::unique_lock<std::mutex> lock(search_mtx);
		//	std::cout << "moveReadout " <<
		//		"depth=" << depth <<
		//		" m=" << (pos.to_move() == white ? "W" : "B") << "|" <<
		//		SanPiece[pos.piece_on(Square(move.f))] << "|" <<
		//		util::col(move.f) << "|" << util::row(move.f) << "|" <<
		//		util::col(move.t) << "|" << util::row(move.t) << std::endl;
		//}
		
		pos.do_move(move);
		stack->curr_move = move;

		bool givesCheck = pos.in_check();
		int16 extensions = givesCheck;
		int16 reductions = 1;

		// 5. Reduce uninteresting quiet moves
		if (!pvNode &&
			!improving &&
			!hashOrKiller &&
			!isCapture &&
			!isEvasion &&
			!givesCheck &&
			!isPromotion &&
			!advancedPawnPush &&
			depth <= 2 &&
			bestScore <= alpha)
			reductions += 1;

		// 6. Extend likely interesting quiet moves 
		if (!pvNode &&
			!improving &&
			!hashOrKiller &&
			!isCapture &&
			depth <= 2 &&
			bestScore < alpha &&
			bestScore > mated_max_ply &&
			(dangerousQuietCheck || advancedPawnPush || threatResponse))
			extensions += 1;

		// 7. Reduce losing captures
		//if (isCapture &&
		//	!hashOrKiller &&
		//	!givesCheck &&
		//	!isPromotion &&
		//	!isEvasion &&
		//	bestScore < alpha &&
		//	!advancedPawnPush &&
		//	depth <= 1 &&
		//	SEE < 0)
		//	reductions += 1;

		// 8. Movecount pruning from Stockfish
		skipQuiets = moves_searched >= futility_move_count(improving, depth);

		// 9. Reduction if this position is being searched by another thread
		//if (is_searching(pos, move, pos.id()))
		//	reductions += 1;

		int16 newdepth = depth + extensions - reductions;
		(stack + 1)->pv = nullptr;

		Score score = Score::ninf;
		if (moves_searched < 3) {
			(stack + 1)->pv = pv;
			(stack + 1)->pv[0].set(A1, A1, Movetype::no_type);
			score = Score(newdepth <= 1 ? -qsearch<Nodetype::pv>(pos, -beta, -alpha, 0, stack + 1) :
				-search<Nodetype::pv>(pos, -beta, -alpha, newdepth - 1, stack + 1));
		}
		else {
			int16 LMR = newdepth;
			// Late move reduction for moves unlikely to raise alpha
			if (!threatResponse &&
				!hashOrKiller &&
				!dangerousQuietCheck &&
				!captureFollowup &&
				!advancedPawnPush &&
				!isPromotion &&
				!isEvasion &&
				!givesCheck &&
				!anyPawnsOn7th &&
				depth >= 3 &&
				bestScore <= alpha) {
				unsigned R = reduction(pvNode, improving, depth, moves_searched);
				LMR -= R;
			}

			score = Score(LMR <= 1 ? -qsearch<non_pv>(pos, -alpha - 1, -alpha, 0, stack + 1) :
				-search<non_pv>(pos, -alpha - 1, -alpha, LMR - 1, stack + 1));

			if (score > alpha) {
				(stack + 1)->pv = pv;
				(stack + 1)->pv[0].set(A1, A1, Movetype::no_type);

				score = Score(newdepth <= 1 ? -qsearch<Nodetype::pv>(pos, -beta, -alpha, 0, stack + 1) :
					-search<Nodetype::pv>(pos, -beta, -alpha, newdepth - 1, stack + 1));
			}
		}
		++moves_searched;

		if (move.type == Movetype::quiet)
			quiets.emplace_back(move);


		pos.undo_move(move);

		// return if stop requested - do not updated best move or pv
		if (UCI_SIGNALS.stop)
			return Score::draw;


		// root move update
		if (root_node) {
			auto& rm = *std::find(pos.root_moves.begin(), pos.root_moves.end(), move);

			if (moves_searched == 1 || score > alpha) {
				rm.score = score;
				rm.selDepth = selDepth;
				rm.pv.resize(1);
				for (Move* m = (stack + 1)->pv;; ++m) {
					if (m->f == m->t || m->type == Movetype::no_type)
						break;
					rm.pv.push_back(*m);
				}
			}
			else rm.score = Score::ninf;
		}

		if (score > bestScore) {
			bestScore = score;
			best_move = move;
			stack->best_move = move;

			if (score >= beta) {
				// Update mate killers and quiet move stats
				pos.stats_update(best_move,
					(stack - 1)->curr_move,
					depth, bestScore, quiets, stack->killers);
				break;
			}

			if (score > alpha) {

				if (pvNode)
					alpha = score;

				if (pvNode && !root_node)
					update_pv(stack->pv, move, (stack + 1)->pv);
			}
		}


		// Penalize captures failing to raise alpha
		//if (SEE >= 0 && !pv && score < alpha && move.f != move.t && move.type == Movetype::capture) {
		//	auto capturePenalty = 2 * depth;
		//	stack->capHistory[to_mv][move.f][move.t] -= capturePenalty;
		//}

	} // end moves loop




	unset_searching(pos, {});

	// Update best move stats
	auto bestMoveBonus = 2 * depth;
	if (bestScore >= alpha && bestScore < beta && best_move.f != best_move.t) {
		stack->bestMoveHistory->bm[to_mv][best_move.f][best_move.t] += bestMoveBonus;
	}



	if (moves_searched == 0) {
		return (in_check ? Score(Score::mated + root_dist) : Score::draw);
	}

	Bound bound = (bestScore >= beta ? bound_low :
		pvNode && (best_move.type != Movetype::no_type) ? bound_exact : bound_high);
	ttable.save(pos.key(), depth, U8(bound), stack->ply, best_move, bestScore, pvNode);

	return bestScore;
}


template<Nodetype type>
Score Search::qsearch(position& p, int16 alpha, int16 beta, U16 depth, node* stack) {

	if (UCI_SIGNALS.stop) 
		return Score::draw; 

	Score best_score = Score::ninf;
	Move best_move = {};
	best_move.type = Movetype::no_type;

	Move ttm = {};
	ttm.type = Movetype::no_type;
	Score ttvalue = Score::ninf;
	bool pv_type = type == Nodetype::pv;

	stack->ply = (stack - 1)->ply + 1;
	if (pv_type && selDepth < stack->ply + 1)
		selDepth++;
	U16 root_dist = stack->ply;

	bool in_check = p.in_check();
	stack->in_check = in_check;

	//if (/*!root_node &&*/ !in_check && p.is_draw()) {
	//	//alpha = draw;
	//	//if (alpha >= beta)
	//	return Score::draw;
	//}

	hash_data e;
	e.depth = 0;
	{  // hashtable lookup
		if (ttable.fetch(p.key(), e)) {
			ttm = e.move;
			ttvalue = Score(e.score);
			hashHits++;

			if (!pv_type &&
				e.depth >= depth &&
				(ttvalue >= beta ? e.bound == bound_low : e.bound == bound_high)) {
				return ttvalue;
			}
		}
	}

	U16 qsdepth = in_check ? 1 : 0;
	const bool anyPawnsOn7th = p.pawns_near_promotion(); // either side has pawns on 7th
	

	if (!in_check) {
		// Compute the best score
		if (!pv_type && ttvalue != Score::ninf && e.depth >= depth)
			best_score = ttvalue;
		else
			best_score = (Score)std::lround(eval::evaluate(p, *SearchThreads[p.id()], lazy_eval_margin(qsdepth, anyPawnsOn7th)));
		
		// Stand pat
		if (best_score >= beta)
			return best_score;

		// Delta pruning
		int deltaCut = 910; // queen value
		if (anyPawnsOn7th)
			deltaCut += 775;
		if (best_score < alpha - deltaCut)
			return Score(alpha);

		// Adjust alpha
		if (pv_type && alpha < best_score)
			alpha = best_score;
	}


	U16 moves_searched = 0;

	haVoc::QMoveorder mvs(p, ttm, stack);
	Move move;
	Move pre_move = (stack - 1)->curr_move;
	Move pre_pre_move = (stack - 2)->curr_move;
	Color to_mv = p.to_move();

	while (mvs.next_move(p, move, pre_move, pre_pre_move, stack->threat_move, false)) {


		if (UCI_SIGNALS.stop)
			return Score::draw;


		if (move.type == Movetype::no_type || !p.is_legal(move))
			continue;


		auto hashOrKiller = (move == ttm) ||
			(move == stack->killers[0]) ||
			(move == stack->killers[1]) ||
			(move == stack->killers[2]) ||
			(move == stack->killers[3]);
		auto isPromotion = p.is_promotion(move.type);
		auto isCapture =
			(move.type == Movetype::capture) ||
			(move.type == Movetype::ep) ||
			p.is_cap_promotion(Movetype(move.type));
		auto isQuiet = move.type == Movetype::quiet;
		auto isEvasion = in_check;
		auto advancedPawnPush = (p.piece_on(Square(move.t)) == Piece::pawn) &&
			(to_mv == white ? util::row(move.t) >= Row::r6 : util::row(move.t) <= Row::r3);
		//auto quietFollowingCapture = (stack - 1)->curr_move.type == Movetype::capture && isQuiet;
		//auto quietFollowup = (stack - 1)->curr_move.type == Movetype::quiet && isQuiet;
		//auto captureFollowup = (stack - 1)->curr_move.type == Movetype::capture && isCapture;
		//auto threatResponse = (stack->threat_move.type != Movetype::no_type && stack->threat_move.f == move.t) && isCapture;
		//auto dangerousQuietCheck = isQuiet && p.quiet_gives_dangerous_check(move);

		// Qsearch delta pruning for captures
		if (!isQuiet && !in_check && !hashOrKiller)
		{
			int idx = int(p.piece_on(Square(move.t)));
			float capture_score = (move.type == capture ? material_vals[idx] :
				move.type == ep ? material_vals[0] :
				move.type == capture_promotion_q ? material_vals[idx] + material_vals[queen] :
				move.type == capture_promotion_r ? material_vals[idx] + material_vals[rook] :
				move.type == capture_promotion_b ? material_vals[idx] + material_vals[bishop] :
				move.type == capture_promotion_n ? material_vals[idx] + material_vals[knight] : 0);
			int margin = 200;

			if (//!advancedPawnPush &&
				//!isPromotion &&
				capture_score > 0 &&
				(best_score + capture_score + margin < alpha))
				continue;

			if (//!isPromotion &&
				//!advancedPawnPush &&
				capture_score > 0 &&
				(best_score - capture_score - margin > beta))
				continue;
		}

		if (p.see(move) < 0)
			continue;

		p.do_move(move);
		p.adjust_qnodes(1);

		Score score = Score(-qsearch<type>(p, -beta, -alpha, 0, stack + 1));

		++moves_searched;

		p.undo_move(move);

		if (score > best_score) {
			best_score = score;
			best_move = move;
			stack->best_move = move;

			if (score >= beta) {
				break;
			}

			if (pv_type && score > alpha) {
				alpha = score;
			}
		}
	}


	if (moves_searched == 0 && in_check) {
		return Score(Score::mated + root_dist);
	}

	// update all stats
	
	Bound bound = (best_score >= beta ? bound_low :
	  pv_type && (best_move.type != Movetype::no_type) ? bound_exact : bound_high);
	ttable.save(p.key(), qsdepth, U8(bound), stack->ply, best_move, best_score, pv_type);

	return best_score;
}


void Search::update_pv(Move* root, const Move& move, Move* child)
{
	for (*root++ = move; child && root && child->f != child->t;)
		*root++ = *child++;

	root->set(A1, A1, Movetype::no_type);
}


void Search::readout_pv(node* stack, const Rootmoves& mRoots, const Score& eval, const Score& alpha, const Score& beta, const U16& depth) {

	std::unique_lock<std::mutex> lock(search_mtx);

	U64 nodes = 0;
	for (auto& t : mPositions) {
		nodes += t->nodes();
		nodes += t->qnodes();
	}

	auto numLines = opts->value<int>("multipv");

	for (int i = 0; i < numLines; ++i)
	{
		std::string res = "";


		for (auto& m : mRoots[i].pv) {
			if (m.f == m.t || m.type == Movetype::no_type)
				break;
			res += uci::move_to_string(m) + " ";
		}

		std::cout << "info"
			<< " depth " << depth
			<< (eval >= beta ? " lowerbound" : eval <= alpha ? " upperbound" : "")
			<< " seldepth " << mRoots[i].selDepth
			<< " multipv " << i
			<< " score cp " << eval // TODO: support multipv
			<< " nodes " << nodes
			<< " tbhits " << hashHits
			<< " time " << (int)elapsed
			//<< " nps " << (nodes * 1000 / elapsed)
			<< " pv " << res << std::endl;
	}

}
