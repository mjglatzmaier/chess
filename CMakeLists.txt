cmake_minimum_required(VERSION 3.16.3)

set(PROGRAM haVoc)

project(haVoc
  VERSION 1.0.0
  DESCRIPTION "Hobby chess project"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################################################
# Options
###################################################################


###################################################################
# Source Files
###################################################################
set(SRC_FILES
  src/bitboards.cpp
  src/evaluate.cpp
  src/hashtable.cpp
  src/haVoc.cpp
  src/magics.cpp
  src/material.cpp
  src/order.cpp
  src/pawns.cpp
  src/pgn.cpp
  src/position.cpp
  src/threads.cpp
  src/uci.cpp
  src/zobrist.cpp
)

###################################################################
# Build Configuration
###################################################################
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build configuration not set, setting configuration to Release.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build configuration type: Debug Release.")
endif()

###################################################################
# Compilation Options
###################################################################
if(WIN32)
  if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /W3 /GR /EHsc /D_64BIT /D_CONSOLE /D_UNICODE /D_WIN32 /D_WIN64 /D_MSC_VER=1939 /std:c++20 /GS /GL /W3 /Gy /Zi /Gm- /O2 /Ob2 /Zc:inline /fp:precise /GT /WX- /Ot /FC /Oi /MD")
  elseif (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /JMC /EHsc /GS /W3 /ZI /Gm- /Od /Zc:inline /fp:precise /WX- /RTC1 /Gd /MDd /FC /D_DEBUG /D_64BIT /D_CONSOLE /D_UNICODE /D_WIN32 /D_WIN64 /D_MSC_VER=1939 /std:c++20")
  endif()
else()
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -fomit-frame-pointer -fstrict-aliasing -ffast-math -O3 -std=c++20 -D_64BIT -D_CONSOLE -D_UNICODE -mavx")
  elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa -Wall -g -ggdb -O0 -std=c++20 -D_DEBUG -D_64BIT -D_CONSOLE -D_UNICODE")
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)
add_executable(${PROGRAM} ${SRC_FILES})


