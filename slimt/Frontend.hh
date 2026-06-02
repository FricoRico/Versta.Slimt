#pragma once
#include <atomic>
#include <cstddef>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "slimt/Batcher.hh"
#include "slimt/Cache.hh"
#include "slimt/Export.hh"
#include "slimt/Response.hh"
#include "slimt/Types.hh"

namespace slimt {

class Model;
struct Options;
struct Response;

struct SLIMT_EXPORT Config {
  // NOLINTBEGIN
  size_t max_words = 1024;
  size_t cache_size = 1024;
  size_t workers = 1;
  float tgt_length_limit_factor = 1.5;
  size_t wrap_length = 128;
  // NOLINTEND

  template <class App>
  void setup_onto(App &app) {
    // clang-format off
    app.add_option("--limit-tgt", tgt_length_limit_factor, "Max length proportional to source target can have.");
    app.add_option("--max-words", max_words, "Maximum words in a batch.");
    app.add_option("--wrap-length", wrap_length, "Maximum length allowed for a sample, beyond which hard-wrap.");
    app.add_option("--workers", workers, "Number of workers threads to launch for translating.");
    // clang-format on
  }
};

class SLIMT_EXPORT Blocking {
 public:
  explicit Blocking(const Config &config);
  std::vector<Response> translate(const Ptr<Model> &model,
                                  std::vector<std::string> sources,
                                  const Options &options);
  std::vector<Response> pivot(const Ptr<Model> &first, const Ptr<Model> &second,
                              std::vector<std::string> sources,
                              const Options &options);

 private:
  size_t id() { return id_++; }

  Config config_;
  std::optional<TranslationCache> cache_;
  size_t id_ = 0;
};

class SLIMT_EXPORT Async {
 public:
  explicit Async(const Config &config);
  ~Async();

  Handle translate(const Ptr<Model> &model, std::string source,
                   const Options &options);
  Handle pivot(const Ptr<Model> &first, const Ptr<Model> &second,
               std::string source, const Options &options);

  /// Point at a caller-owned atomic byte (e.g. a Rust `AtomicBool`), or
  /// nullptr to disarm. While it is non-null and reads true, worker threads
  /// abort pending batches instead of running `forward()`, so an in-flight
  /// translate stops within ~one batch per worker. The pointee must outlive
  /// the translate call. Reads are relaxed atomic loads.
  void set_cancel_flag(const void *flag) { cancel_flag_.store(flag); }

 private:
  size_t id() { return id_++; }

  bool cancelled() const {
    const void *flag = cancel_flag_.load(std::memory_order_relaxed);
    return flag != nullptr &&
           __atomic_load_n(static_cast<const unsigned char *>(flag),
                           __ATOMIC_RELAXED) != 0;
  }

  Config config_;
  std::optional<TranslationCache> cache_;
  Threadsafe<AggregateBatcher> batcher_;
  std::vector<std::thread> workers_;
  std::atomic<const void *> cancel_flag_{nullptr};

  size_t id_ = 0;
};

}  // namespace slimt
