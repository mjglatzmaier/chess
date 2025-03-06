
#ifndef SEARCH_H
#define SEARCH_H

#include <algorithm>

#include "types.h"
#include "threads.h"
#include "move.h"



namespace Search {

	std::atomic_bool searching;
	std::mutex mtx;
	void search_timer(position& p, limits& lims);
	void start(position& p, limits& lims, bool silent);
	void iterative_deepening(position& p, U16 depth, bool silent);
	void readout_pv(node* stack, const Rootmoves& mRoots, const Score& eval, const Score& alpha, const Score& beta, const U16& depth);
	double estimate_max_time(position& p, limits& lims);
	void update_pv(Move* root, const Move& move, Move* child);

	template<Nodetype type>
	Score search(position& p, int16 alpha, int16 beta, U16 depth, node* stack);

	template<Nodetype type>
	Score qsearch(position& p, int16 alpha, int16 beta, U16 depth, node* stack);
}


#include "search.hpp"

#endif
