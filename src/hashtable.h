
#pragma once
#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <memory>

#include "types.h"
#include "move.h"

const U64 search_bit = (1ULL << 63);

struct entry {
	entry() : pkey(0ULL), dkey(0ULL) { }

	U64 pkey;  // zobrist hashing
	U64 dkey;  // 17 bit val, 8 bit depth, 8 bit bound, 8 bit f, 8 bit t, 8 bit type

	inline bool empty() { return pkey == 0ULL && dkey == 0ULL; }

	inline void encode(const U8& depth,
		const U8& bound,
		const U8& age,
		const Move& m,
		const int16& score) {
		dkey = 0ULL;
		dkey |= U64(m.f); // 8 bits;
		dkey |= (U64(m.t) << 8); // 8 bits
		dkey |= (U64(m.type) << 16); // 8 bits
		dkey |= (U64(bound) << 26); // 4 bits;
		dkey |= (U64(depth + 1) << 30); // 8 bits
		dkey |= (U64(score < 0 ? -score : score) << 38); // 16 bits
		dkey |= (U64(score < 0 ? 1ULL : 0ULL) << 54);   // 1 bit
		dkey |= (U64(age) << 55); // 9 bits .. 
	}

	inline U8 depth() { return U8((dkey & 0xFF0000000) >> 30); }
	inline U8 bound() { return U8((dkey & 0xF000000) >> 26); }
	inline U8 age() { return U8((dkey & 0x7F80000000000000 >> 55)); }
};


enum Bound { bound_low, bound_high, bound_exact, no_bound };

struct hash_data {
	char depth;
	U8 bound;
	U8 age;
	int16 score;
	U16 pkey;
	U16 dkey;
	Move move; // 3 bytes

	inline void decode(const U64& dkey) {

		U8 f = U8(dkey & 0xFF);
		U8 t = U8((dkey & 0xFF00) >> 8);
		Movetype type = Movetype((dkey & 0xFF0000) >> 16);
		bound = U8((dkey & 0xF000000) >> 26);
		depth = U8((dkey & 0xFF0000000) >> 30);
		score = int16((dkey & 0xFFFF000000000) >> 38);
		int sign = int(dkey & (1ULL << 54));
		score = (sign == 1 ? -score : score);
		age = U8(dkey & 0x7F80000000000000 >> 55);

		move.set(f, t, type);
	}
};

const unsigned cluster_size = 4;

struct hash_cluster {
	// based on entry size = 64 bits / 8 = 8 + 8 bytes
	// 16 * 4 = 64 bytes, leaving 0 bytes for cache padding
	entry cluster_entries[cluster_size];
};


class hash_table {
private:
	size_t sz_mb;
	size_t cluster_count;
	std::unique_ptr<hash_cluster[]> entries;
	void alloc(size_t sizeMb);

public:
	hash_table();
	hash_table(const hash_table& o) = delete;
	hash_table(const hash_table&& o) = delete;
	~hash_table() {}

	hash_table& operator=(const hash_table& o) = delete;
	hash_table& operator=(const hash_table&& o) = delete;

	void save(const U64& key,
		const U8& depth,
		const U8& bound,
		const U8& age,
		const Move& m,
		const int16& score, const bool& pv_node);
	bool fetch(const U64& key, hash_data& e);
	inline entry* first_entry(const U64& key);
	void clear();
	void resize(size_t sizeMb);
};

inline entry* hash_table::first_entry(const U64& key) {
	return &entries[key & (cluster_count - 1)].cluster_entries[0];
}

extern hash_table ttable; // global transposition table

#endif
