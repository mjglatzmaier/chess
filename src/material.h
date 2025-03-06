
#pragma once

#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>

#include "types.h"

class position;

struct material_entry {
	U64 key;
	int16 score;
	double endgame_coeff; // interpolation between middle and endgame
	EndgameType endgame = EndgameType::none;
	U8 number[5]; // knight, bishop, rook, queen
	inline bool is_endgame() { return endgame != EndgameType::none; }
};


class material_table {
private:
	size_t sz_mb;
	size_t count;
	std::shared_ptr<material_entry[]> entries;

	void init();

public:
	material_table();
	material_table(const material_table& o);
	material_table& operator=(const material_table& o);

	~material_table() {}

	void clear();
	material_entry* fetch(const position& p) const;

};


//extern material_table mtable;

#endif