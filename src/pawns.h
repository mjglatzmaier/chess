
#ifndef PAWNS_H
#define PAWNS_H

#include <memory>

#include "types.h"

class position;

struct pawn_entry {
	pawn_entry() : key(0ULL), score(0) { }

	U64 key;
	int16 score;

	U64 doubled[2];
	U64 isolated[2];
	U64 backward[2];
	U64 passed[2];
	U64 dark[2];
	U64 light[2];
	U64 king[2];
	U64 attacks[2];
	U64 undefended[2];
	U64 weak_squares[2];
	U64 chaintips[2];
	U64 chainbases[2];
	U64 queenside[2];
	U64 kingside[2];
	U64 semiopen[2];
	U64 qsidepawns[2];
	U64 ksidepawns[2];
	int16 center_pawn_count;
	bool locked_center;
};



class pawn_table {
private:
	size_t sz_mb = 0;
	size_t count = 0;
	std::shared_ptr<pawn_entry[]> entries;

	void init();

public:
	pawn_table();
	pawn_table(const pawn_table& o);
	pawn_table& operator=(const pawn_table& o);
	~pawn_table() {}

	void clear();
	pawn_entry* fetch(const position& p) const;
};


//extern pawn_table ptable; // global pawn hash table

#endif
