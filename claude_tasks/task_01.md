# Task 01 — Project Scaffold & Build System

## Goal
Set up the `remake/` directory with modern CMake, C++20, cross-platform support,
Google Test via FetchContent, code formatting/linting, and CI pipeline skeleton.
No engine code yet — just the build infrastructure that everything else sits on.

---

## Steps

### 1.1 — Directory Structure
Create the following layout inside `remake/`:
```
remake/
├── CMakeLists.txt                 # Root CMake
├── cmake/
│   ├── CompilerWarnings.cmake     # Warning flags per-compiler
│   ├── Sanitizers.cmake           # ASan/UBSan/TSan options
│   └── StaticAnalysis.cmake       # clang-tidy integration
├── include/
│   └── havoc/                     # Public headers (empty for now)
├── src/
│   └── main.cpp                   # Minimal stub (prints version, exits)
├── tests/
│   ├── CMakeLists.txt             # Test target config
│   └── test_smoke.cpp             # Smoke test: EXPECT_TRUE(true)
├── bench/
│   └── CMakeLists.txt             # Benchmark target config (empty)
├── .clang-format                  # Code formatting rules
├── .clang-tidy                    # Linting rules
└── .github/
    └── workflows/
        └── ci.yml                 # GitHub Actions: build + test (Linux/macOS/Windows)
```

### 1.2 — Root CMakeLists.txt
- `cmake_minimum_required(VERSION 3.20)`
- `project(haVoc VERSION 2.0.0 LANGUAGES CXX)`
- `set(CMAKE_CXX_STANDARD 20)` + `CMAKE_CXX_STANDARD_REQUIRED ON` + `CMAKE_CXX_EXTENSIONS OFF`
- Create a `havoc_core` STATIC library target (empty source list for now, add placeholder .cpp)
- Create `havoc` executable target linking `havoc_core`
- Use `target_compile_features` instead of global `CMAKE_CXX_FLAGS`
- Use `target_include_directories` with `PUBLIC` for include/ and `PRIVATE` for src/
- Include `cmake/CompilerWarnings.cmake`, `cmake/Sanitizers.cmake`
- Use generator expressions for platform-specific flags, NOT `if(WIN32)` blocks
- Default build type to Release if not set
- Add `option(HAVOC_ENABLE_TESTS "Build tests" ON)`
- Add `option(HAVOC_ENABLE_BENCH "Build benchmarks" OFF)`
- Add `option(HAVOC_ENABLE_SANITIZERS "Enable sanitizers in Debug" OFF)`
- Add `option(HAVOC_NATIVE "Enable -march=native" ON)` for SIMD/popcnt

### 1.3 — cmake/CompilerWarnings.cmake
Create a function `havoc_set_warnings(target)` that applies:
- **GCC/Clang**: `-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wcast-align -Woverloaded-virtual -Wconversion -Wsign-conversion -Wnull-dereference -Wformat=2`
- **MSVC**: `/W4 /permissive-`
- Use generator expressions: `$<$<CXX_COMPILER_ID:GNU,Clang>:...>`
- Do NOT use `-Werror` by default (too strict during development)

### 1.4 — cmake/Sanitizers.cmake
Create a function `havoc_enable_sanitizers(target)` guarded by `HAVOC_ENABLE_SANITIZERS`:
- **GCC/Clang Debug**: `-fsanitize=address,undefined -fno-omit-frame-pointer`
- **MSVC Debug**: `/fsanitize=address`
- Only applies in Debug builds

### 1.5 — cmake/StaticAnalysis.cmake
- Optionally set `CMAKE_CXX_CLANG_TIDY` if clang-tidy is found
- Don't fail build on tidy warnings (informational only during dev)

### 1.6 — Google Test via FetchContent
In `tests/CMakeLists.txt`:
```cmake
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # Windows fix
FetchContent_MakeAvailable(googletest)

add_executable(havoc_tests test_smoke.cpp)
target_link_libraries(havoc_tests PRIVATE havoc_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(havoc_tests)
```

### 1.7 — .clang-format
Use a consistent style. Recommended baseline:
```yaml
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
AllowShortFunctionsOnASingleLine: Inline
BreakBeforeBraces: Attach
PointerAlignment: Left
SortIncludes: CaseInsensitive
IncludeBlocks: Regroup
```

### 1.8 — .clang-tidy
Enable useful checks without being too noisy:
```yaml
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-*,
  cppcoreguidelines-avoid-goto,
  cppcoreguidelines-init-variables,
  cppcoreguidelines-slicing,
  misc-redundant-expression,
  modernize-use-auto,
  modernize-use-nullptr,
  modernize-use-override,
  modernize-use-using,
  performance-*,
  readability-braces-around-statements,
  readability-const-return-type,
  readability-redundant-string-cstr
```

### 1.9 — CI Pipeline (.github/workflows/ci.yml)
GitHub Actions workflow:
- **Matrix**: Ubuntu (GCC 12+, Clang 15+), macOS (AppleClang), Windows (MSVC 2022)
- **Steps**: checkout → configure → build → test
- Run `ctest --output-on-failure`
- Build both Debug and Release
- Cache CMake build directory for speed

### 1.10 — Stub Files
- `src/main.cpp`: Print "haVoc v2.0.0" and exit
- `tests/test_smoke.cpp`: Single `TEST(Smoke, CanRun) { EXPECT_TRUE(true); }`
- `include/havoc/version.hpp`: `constexpr` version string

### 1.11 — Verify
- Build on local machine (macOS): `cmake -B build -S remake && cmake --build build && ctest --test-dir build`
- Confirm smoke test passes
- Confirm clang-format runs clean on stub files
- Do NOT commit — leave for manual review

---

## Acceptance Criteria
- [ ] `cmake --build build` succeeds with zero warnings on macOS (Clang)
- [ ] `ctest` runs and smoke test passes
- [ ] clang-format produces no changes on all source files
- [ ] Directory structure matches layout above
- [ ] No hardcoded platform-specific compiler flags in CMakeLists.txt
- [ ] Google Test fetched and linked automatically
