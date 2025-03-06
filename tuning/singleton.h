#pragma once
#ifndef HAVOC_SINGLETON_H
#define HAVOC_SINGLETON_H

namespace haVoc {

	template<typename T>
	class Singleton {
	protected:
		struct token {}; // allows calling subclass c'tor without needing friendship
		Singleton() {}

	public:

		Singleton(const Singleton&) = delete;
		Singleton& operator= (const Singleton) = delete;


		static T& instance()
		{
			static T instance{ token{} };
			return instance;
		}
	};

}

#endif
