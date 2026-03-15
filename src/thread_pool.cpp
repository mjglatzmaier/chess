#include "havoc/thread_pool.hpp"

// Threadpool is entirely header-only (template class).
// This translation unit ensures the header compiles cleanly in isolation
// and gives CMake a source file to compile.

namespace havoc {

// Explicit instantiations for the thread pool types we use.
template class Threadpool<Workerthread>;
template class Threadpool<Searchthread>;

} // namespace havoc
