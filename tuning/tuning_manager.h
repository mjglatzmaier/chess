#pragma once
#ifndef HAVOC_TUNING_MANAGER_H
#define HAVOC_TUNING_MANAGER_H

#include <memory>
#include <map>
#include <iostream>
#include <thread>

#include "singleton.h"
#include "xml_eval.h" // evaluation tuning


namespace haVoc {

	class Tuningmanager final : public Singleton<Tuningmanager> {
	private:

		std::shared_ptr<XmlEval> m_evals;

		/// <summary>
		/// Reads the tuning configuration file and loads all tuning data
		/// </summary>
		void init();

	public:

	public:
		Tuningmanager(token);
		~Tuningmanager();

		void tune_evaluation();
	};

}

#endif

