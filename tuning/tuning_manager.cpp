
#include <sstream>

#include "position.h"
#include "evaluate.h"
#include "tuning_manager.h"
#include "xml_parser.h"
#include "threads.h"



namespace haVoc {

	Tuningmanager::Tuningmanager(token) {
		init();
	}

	Tuningmanager::~Tuningmanager() {
	}


	void Tuningmanager::tune_evaluation() {
		auto entries = m_evals->entries();
		U64 correct = 0;
		U64 total = entries.size();
		haVoc::EvalEntries incorrect;
		for (const auto& e : entries) {
			auto score = e.second;

			std::istringstream fen(e.first);
			position board(fen);
			auto eval = eval::evaluate(board, *SearchThreads[0], -1);
			auto diff = std::abs((float)score - eval);
			if (diff <= 50)
				correct++;
			else
				incorrect[e.first] = e.second;
		}
		std::cout << "..correct: " << (float)((float)correct / (float)total) << std::endl;

		size_t count = 0;
		for (const auto& e : incorrect) {
			std::cout << e.first << " " << e.second << std::endl;
			if (count++ >= 20)
				break;
		}

	}

	void Tuningmanager::init() {
		auto parser = std::make_shared<XML>("engine_tuning_config.xml");
		std::string filename = "";
		parser->try_get("evalTuneFile", "file", filename);
		m_evals = std::make_shared<XmlEval>(filename);
	}
}