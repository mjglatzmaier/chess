#pragma once

#include <string>

namespace haVoc {

	/// <summary>
	/// Wrapper class around preprocessor definitions for build types
	/// </summary>
	static class Platforminfo final {

	public:

		static bool is_windows()
		{
#ifdef _WIN32 
			return true;
#endif
			return false;
		}

		static bool is_macOS()
		{
#ifdef __APPLE__
			return true;
#endif
			return false;
		}

		static std::string platform_desc()
		{
			if (is_windows())
#ifdef _WIN64
				return "Windows 64-bit OS";
#else
				return "Windows 32-bit OS";
#endif

			if (is_macOS())
				return "MacOS";

#ifdef __linux__
			return "Linux OS";
#elif __unix__
			return "Unix OS";
#endif

			return "Unknown OS";
		}
	};
}