#pragma once
#include <atomic>
#include <cstddef>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "Batcher.hh"
#include "Cache.hh"
#include "Export.hh"
#include "Response.hh"
#include "Types.hh"

namespace leanmt {

class Model;
struct Options;
struct Response;

struct LEANMT_EXPORT Config {
  // NOLINTBEGIN
  size_t max_words = 1024;
  size_t cache_size = 1024;
  size_t workers = 1;
  float tgt_length_limit_factor = 1.5;
  size_t wrap_length = 128;
  // NOLINTEND
};

class LEANMT_EXPORT Blocking {
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

class LEANMT_EXPORT Async {
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

}  // namespace leanmt
