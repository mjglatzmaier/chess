#pragma once

#ifndef MAGICS_H
#define MAGICS_H

#include <memory>
#include <algorithm>
#include <random>

#include "types.h"
#include "bits.h"
#include "utils.h"

namespace magics {

	template<Piece p>
	U64 attacks(const Square& s, const U64& block);

	template<Piece p>
	U64 attacks(const U64& occ, const Square& s);

	U64 next_magic(const unsigned int& bits, util::rand<unsigned int>& r);
	bool load();
}

#endif
