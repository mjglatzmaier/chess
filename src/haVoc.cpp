
#include <cstring>
#include <iostream>
#include <map>
#include <memory>

#include "options.h"
#include "info.h"
#include "bitboards.h"
#include "uci.h"
#include "magics.h"
#include "zobrist.h"


std::unique_ptr<options> opts;

int main(int argc, char* argv[]) {

	greeting();

	opts = std::unique_ptr<options>(new options(argc, argv));
	zobrist::load();
	bitboards::load();
	magics::load();
	uci::loop();

	return 0;
}
