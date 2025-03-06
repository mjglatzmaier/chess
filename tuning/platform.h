#pragma once
#ifndef PLATFORM_H
#define PLATFORM_H

#ifdef _MSC_VER
#include <windows.h>
#endif

#include <string>
#include <iostream>
#include <fstream>


namespace utils {


  inline void execute_windows(const std::string& exename,
    const std::string& options) {
    /*
    CreateProcessA(
      exename.c_str(), // exe name
      options.c_str(), // the command to be executed

      )
      */
  }

  inline void execute_linux(const std::string& exe) {

  }


}

#endif