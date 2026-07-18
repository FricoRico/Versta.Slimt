#pragma once
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#define LEANMT_BREAK std::raise(SIGTRAP)

#define LEANMT_TRACE(x)                                         \
  do {                                                         \
    std::cerr << __FILE__ << ":" << __LINE__;                  \
    std::cerr << " " << __FUNCTION__ << " ";                   \
    std::cerr << #x << ": " << std::scientific << (x) << '\n'; \
  } while (0)

#define LEANMT_TRACE_BLOCK(x)                                   \
  do {                                                         \
    std::cerr << __FILE__ << ":" << __LINE__;                  \
    std::cerr << " " << __FUNCTION__ << " \n\n";               \
    std::cerr << #x << ": " << std::scientific << (x) << '\n'; \
    std::cerr << "\n\n";                                       \
  } while (0);

#define LEANMT_TRACE2(x, y) \
  LEANMT_TRACE(x);          \
  LEANMT_TRACE(y)

#define LEANMT_TRACE3(x, y, z) \
  LEANMT_TRACE2(x, y);         \
  LEANMT_TRACE(z);

// LEANMT_ABORT* signals a corrupt input file or a leanmt invariant failure
// (programmer error). These are unrecoverable — abort gives a real stack
// trace and surfaces the bug rather than letting it propagate as a silent
// translation failure. For paths whose failures are user-input-driven and
// genuinely recoverable (e.g. malformed model input), throw
// `std::runtime_error` directly rather than reusing this macro.
#define LEANMT_ABORT_IF(condition, error) \
  do {                                   \
    if (condition) {                     \
      std::cerr << (error) << '\n';      \
      std::abort();                      \
    }                                    \
  } while (0)

#define LEANMT_ABORT(message) \
  do {                       \
    std::cerr << (message);  \
    std::abort();            \
  } while (0)

#ifdef LEANMT_ENABLE_LOGS
#define LOG(level, ...)              \
  do {                               \
    fprintf(stderr, "[%s]", #level); \
    fprintf(stderr, __VA_ARGS__);    \
    fprintf(stderr, "\n");           \
  } while (0)
#else  // LEANMT_ENABLE_LOGS
#define LOG(...) (void)0
#endif  // LEANMT_ENABLE_LOGS
