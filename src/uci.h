
#pragma once

#ifndef UCI_H
#define UCI_H

#include <iostream>
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <string>

#include "bits.h"
#include "types.h"

struct Move;

const std::string START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct limits {
	unsigned wtime, btime, winc, binc;
	unsigned movestogo, nodes, movetime, mate, depth;
	bool infinite, ponder;
};

struct signals {
	volatile bool stop, ponder_hit, times_up;
};

namespace uci {
	void loop();
	bool parse_command(const std::string& input);
	void load_position(const std::string& pos);
	std::string move_to_string(const Move& m);
}

extern signals UCI_SIGNALS;

#endif
