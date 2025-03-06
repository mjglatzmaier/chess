
#include "uci.h"
#include "move.h"
#include "search.h"
#include "threads.h"
#include "hashtable.h"
#include "threads.h"

position uci_pos;
Move dbgmove;
Threadpool<Workerthread> worker(1);
signals UCI_SIGNALS;

void uci::loop() {
	uci_pos.params = eval::Parameters;

	int numThreads = std::max(opts->value<int>("threads"), 1);
	SearchThreads.init(numThreads);

	std::string input = "";
	while (std::getline(std::cin, input)) {
		if (!parse_command(input)) break;
	}
}


bool uci::parse_command(const std::string& input) {
	std::istringstream instream(input);
	std::string cmd;
	bool running = true;

	while (instream >> std::skipws >> cmd) {
		std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

		if (cmd == "position" && instream >> cmd) {

			std::string tmp;
			if (cmd == "startpos") {
				getline(instream, tmp);
				std::istringstream fen(START_FEN);
				uci_pos.setup(fen);
				load_position(tmp);
			}
			else {
				std::string sfen = "";
				while ((instream >> cmd) && cmd != "moves") sfen += cmd + " ";
				getline(instream, tmp);
				tmp = "moves " + tmp;
				std::istringstream fen(sfen);
				uci_pos.setup(fen);
				load_position(tmp);
			}
		}
		else if (cmd == "setoption" && instream >> cmd && instream >> cmd)
		{
			if (cmd == "hash" && instream >> cmd && instream >> cmd)
			{
				auto sz = atoi(cmd.c_str());
				opts->set("hashsize", sz);
				ttable.resize(sz);
				break;
			}
			if (cmd == "clear" && instream >> cmd)
			{
				if (cmd == "hash")
					ttable.clear();
			}
			if (cmd == "threads" && instream >> cmd && instream >> cmd)
			{
				opts->set("threads", atoi(cmd.c_str()));
				break;
			}
			if (cmd == "multipv" && instream >> cmd && instream >> cmd)
			{
				opts->set("multipv", atoi(cmd.c_str()));
				break;
			}
		}
		else if (cmd == "d") {
			uci_pos.print();
			std::cout << "position hash key: " << uci_pos.key() << std::endl;
			std::cout << "fen: " << uci_pos.to_fen() << std::endl;
		}
		else if (cmd == "eval") {
			uci_pos.print();
			std::cout << "position hash key: " << uci_pos.key() << std::endl;
			std::cout << "evaluation: " << eval::evaluate(uci_pos, *SearchThreads[0], -1) << std::endl;
		}
		else if (cmd == "undo") {
			uci_pos.undo_move(dbgmove);
		}
		else if (cmd == "fdepth" && instream >> cmd) {
			uci_pos.params.fixed_depth = atoi(cmd.c_str());
			std::cout << "fixed depth search: " << uci_pos.params.fixed_depth << std::endl;
		}

		else if (cmd == "see" && instream >> cmd) {
			Movegen mvs(uci_pos);
			mvs.generate<pseudo_legal, pieces>();
			Move move;


			for (int i = 0; i < mvs.size(); ++i) {

				if (!uci_pos.is_legal(mvs[i])) continue;

				if (move_to_string(mvs[i]) == cmd) {
					move = mvs[i];
					break;
				}

			}

			if (move.type != Movetype::no_type) {
				int score = uci_pos.see_move(move);
				std::cout << "See score:  " << score << std::endl;
			}
			else std::cout << " (dbg) See : error, illegal move." << std::endl;
		}
		//else if (cmd == "evaltune") {
		//	auto& tm = haVoc::Tuningmanager::instance();
		//	tm.tune_evaluation();
		//}
		else if (cmd == "domove" && instream >> cmd) {
			Movegen mvs(uci_pos);
			bool isok = false;
			mvs.generate<pseudo_legal, pieces>();
			for (int i = 0; i < mvs.size(); ++i) {
				if (!uci_pos.is_legal(mvs[i])) continue;
				std::string tmp = SanSquares[mvs[i].f] + SanSquares[mvs[i].t];
				std::string ps = "";
				Movetype t = Movetype(mvs[i].type);

				if (t >= 0 && t < capture_promotion_q) {
					ps = (t == 0 ? "q" :
						t == 1 ? "r" :
						t == 2 ? "b" : "n");
				}
				else if (t >= capture_promotion_q && t < castle_ks) {
					ps = (t == 4 ? "q" :
						t == 5 ? "r" :
						t == 6 ? "b" : "n");
				}
				tmp += ps;

				if (tmp == cmd) {
					dbgmove.set(mvs[i].f, mvs[i].t, Movetype(mvs[i].type));
					isok = true;
					break;
				}
			}
			if (isok) {
				std::cout << "doing mv " << std::endl;
				uci_pos.do_move(dbgmove);
			}
			else std::cout << cmd << " is not a legal move" << std::endl;
		}
		else if (cmd == "debug") {
			uci_pos.debug_search = !uci_pos.debug_search;
			std::cout << "debugging set to: " << uci_pos.debug_search << std::endl;
		}

		// game specific uci commands (refactor?)
		else if (cmd == "isready") {
			ttable.clear();
			std::cout << "readyok" << std::endl;
		}
		else if (!Search::searching && cmd == "go") {
			limits lims;
			memset(&lims, 0, sizeof(limits));

			while (instream >> cmd)
			{
				if (cmd == "wtime" && instream >> cmd) lims.wtime = atoi(cmd.c_str());
				else if (cmd == "btime" && instream >> cmd) lims.btime = atoi(cmd.c_str());
				else if (cmd == "winc" && instream >> cmd) lims.winc = atoi(cmd.c_str());
				else if (cmd == "binc" && instream >> cmd) lims.binc = atoi(cmd.c_str());
				else if (cmd == "movestogo" && instream >> cmd) lims.movestogo = atoi(cmd.c_str());
				else if (cmd == "nodes" && instream >> cmd) lims.nodes = atoi(cmd.c_str());
				else if (cmd == "movetime" && instream >> cmd) lims.movetime = atoi(cmd.c_str());
				else if (cmd == "mate" && instream >> cmd) lims.mate = atoi(cmd.c_str());
				else if (cmd == "depth" && instream >> cmd) lims.depth = atoi(cmd.c_str());
				else if (cmd == "infinite") lims.infinite = (cmd == "infinite" ? true : false);
				else if (cmd == "ponder") lims.ponder = atoi(cmd.c_str());
			}

			// Set search threads
			int numThreads = std::max(opts->value<int>("threads"), 1);
			if (numThreads != SearchThreads.num_workers())
				SearchThreads.init(numThreads);

			bool silent = false;
			worker.enqueue(Search::start, uci_pos, lims, silent);
		}
		else if (cmd == "stop") {
			UCI_SIGNALS.stop = true;
		}
		else if (cmd == "moves") {
			Movegen mvs(uci_pos);
			mvs.generate<pseudo_legal, pieces>();
			for (int i = 0; i < mvs.size(); ++i) {
				if (!uci_pos.is_legal(mvs[i])) 
					continue;
				std::cout << move_to_string(mvs[i]) << " ";
			}
			std::cout << std::endl;
		}
		else if (cmd == "ucinewgame") {
			ttable.clear();
			uci_pos.clear();
		}
		else if (cmd == "uci") {
			ttable.clear();
			uci_pos.clear();
			std::cout << "id name haVoc" << std::endl;
			std::cout << "id author M.Glatzmaier" << std::endl;
			std::cout << "option name Threads type spin default 1 min 1 max 1024" << std::endl;
			std::cout << "option name Hash type spin default 1024 min 1 max 33554432" << std::endl;
			std::cout << "option name MultiPV type spin default 1 min 1 max 4" << std::endl;
			std::cout << "uciok" << std::endl;
		}

		else if (cmd == "exit" || cmd == "quit") {
			running = false;
			break;
		}
		else std::cout << "unknown command: " << cmd << std::endl;

	}
	return running;
}


void uci::load_position(const std::string& pos) {
	std::string token;
	std::istringstream ss(pos);

	ss >> token; // eat the moves token
	while (ss >> token) {
		Movegen mvs(uci_pos);
		mvs.generate<pseudo_legal, pieces>();
		for (int j = 0; j < mvs.size(); ++j) {
			if (!uci_pos.is_legal(mvs[j])) continue;

			if (move_to_string(mvs[j]) == token) {
				uci_pos.do_move(mvs[j]);
				break;
			}

		}
	}

}

std::string uci::move_to_string(const Move& m) {
	auto fromto = SanSquares[m.f] + SanSquares[m.t];
	Movetype t = Movetype(m.type);

	auto ps = (t == capture_promotion_q ? "q" :
		t == capture_promotion_r ? "r" :
		t == capture_promotion_b ? "b" :
		t == capture_promotion_n ? "n" :
		t == promotion_q ? "q" :
		t == promotion_r ? "r" :
		t == promotion_b ? "b" :
		t == promotion_n ? "n" : "");

	return fromto + ps;
}

