#include "slimt/Frontend.hh"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Batcher.hh"
#include "slimt/Input.hh"
#include "slimt/Model.hh"
#include "slimt/Request.hh"
#include "slimt/Response.hh"
#include "slimt/TextProcessor.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

namespace {

Input convert(const Batch &batch, uint32_t pad_id, float limit_factor) {
  const auto &segment_refs = batch.segment_refs();
  Input input(batch.size(), batch.max_length(), pad_id, limit_factor);
  for (const auto &segment_ref : segment_refs) {
    const Segment &segment = segment_ref.get();
    input.add(segment);
  }

  input.set_shortlist(batch.shortlist());

  // Carry each row's request-level alternatives choice onto the Input. All
  // requesting rows share the same config (set once by the caller), so the
  // first one seen fixes it; the per-row flags let forward() keep those rows
  // greedy while still beam-re-decoding the rest of the batch.
  std::optional<AlternativesConfig> alt_cfg;
  std::vector<bool> alt_rows(segment_refs.size(), false);
  for (size_t i = 0; i < segment_refs.size(); ++i) {
    const std::optional<AlternativesConfig> &req = segment_refs[i].request()->alternatives();
    if (req.has_value()) {
      alt_rows[i] = true;
      if (!alt_cfg.has_value()) {
        alt_cfg = req;
      }
    }
  }
  if (alt_cfg.has_value()) {
    input.set_alternatives(*alt_cfg, std::move(alt_rows));
  }

  // Carry each row's forced target prefix (for steered re-translation).
  std::vector<Words> forced(segment_refs.size());
  bool any_forced = false;
  for (size_t i = 0; i < segment_refs.size(); ++i) {
    forced[i] = segment_refs[i].request()->forced_prefix();
    any_forced = any_forced || !forced[i].empty();
  }
  if (any_forced) {
    input.set_forced(std::move(forced));
  }

  input.finalize();
  return input;
}

void exhaust(const Config &config, const Ptr<Model> &model, Batcher &batcher) {
  AverageMeter<float> wps;
  AverageMeter<float> occupancy;
  Batch batch = batcher.generate();
  while (!batch.empty()) {
    // Mirror the Async worker: a throw in convert/forward/complete must NOT
    // strand the in-flight promises forever. Forward the exception into
    // each parent Request via `batch.abort`; the per-Request `on_error_`
    // callback (set up by the caller) translates that into
    // `promise.set_exception` so the synchronous future.get() loop in
    // Blocking::translate / pivot rethrows on the calling thread instead
    // of hanging.
    try {
      Timer timer;
      Input input = convert(batch, model->vocabulary().pad_id(),
                            config.tgt_length_limit_factor);
      Histories histories = model->forward(input);
      batch.complete(histories);

      auto elapsed = static_cast<float>(timer.elapsed());
      float sample_wps = input.words().size() / elapsed;
      wps.record(sample_wps);
      occupancy.record(input.occupancy());
    } catch (...) {
      batch.abort(std::current_exception());
    }
    batch = batcher.generate();
  }
}

template <class Continuation, class OnError>
Ptr<Request> make_request(size_t id, const Ptr<Model> &model,
                          std::optional<TranslationCache> &cache,
                          AnnotatedText &&annotated_text, Segments &&segments,
                          Continuation &&continuation, OnError &&on_error,
                          bool with_alignment,
                          std::optional<AlternativesConfig> alternatives =
                              std::nullopt,
                          Words forced_prefix = {}) {
  std::shared_ptr<const Words> shortlist_words;
  if (model->shortlist_generator()) {
    Words context_words;
    for (const Segment &segment : segments) {
      context_words.insert(context_words.end(), segment.begin(), segment.end());
    }

    Shortlist shortlist = model->shortlist_generator()->generate(
        context_words, ShortlistGenerator::kMinCandidates);
    shortlist_words = std::make_shared<const Words>(shortlist.words());
  }

  auto request = std::make_shared<Request>(      //
      id, model->id(),                           //
      std::move(annotated_text),                 //
      std::move(segments),                       //
      // Decoding generated word IDs back to text needs the *target* vocab,
      // which is distinct from the source vocab on two-vocab models like
      // bergamot's en-zh / en-ja / en-ko / en-zh_hant / zh_hant-en.
      model->target_vocabulary(),                //
      std::move(shortlist_words),                //
      cache,                                     //
      std::forward<Continuation>(continuation),  //
      std::forward<OnError>(on_error),           //
      with_alignment,                            //
      std::move(alternatives),                   //
      std::move(forced_prefix)                   //
  );
  return request;
}

std::optional<TranslationCache> make_cache(size_t cache_size) {
  constexpr size_t kCacheBucketSize = 16;
  if (cache_size > 0) {
    return std::make_optional<TranslationCache>(cache_size, kCacheBucketSize);
  }
  return std::nullopt;
}

}  // namespace

Blocking::Blocking(const Config &config)
    : config_(config), cache_(make_cache(config.cache_size)) {}  // NOLINT

std::vector<Response> Blocking::translate(const Ptr<Model> &model,
                                          std::vector<std::string> sources,
                                          const Options &options) {
  Batcher batcher(config_.max_words, config_.wrap_length,
                  config_.tgt_length_limit_factor);

  std::vector<Promise> promises(sources.size());
  std::vector<Future> futures;
  futures.reserve(sources.size());

  for (size_t i = 0; i < sources.size(); i++) {
    std::string &source = sources[i];

    Promise &promise = promises[i];
    Future future = promise.get_future();
    futures.push_back(std::move(future));

    auto continuation = [&promise](Response &&response) {
      promise.set_value(std::move(response));
      return nullptr;
    };
    auto on_error = [&promise](std::exception_ptr eptr) {
      promise.set_exception(std::move(eptr));
    };

    const auto &processor = model->processor();
    auto [annotated, segments] =
        processor.process(std::move(source), config_.wrap_length);
    auto request = make_request(id(), model, cache_, std::move(annotated),
                                std::move(segments), continuation, on_error,
                                /*with_alignment=*/options.alignment,
                                options.alternatives, options.forced_prefix);

    batcher.enqueue(request);
  }

  exhaust(config_, model, batcher);

  std::vector<Response> responses;
  responses.reserve(futures.size());
  for (auto &future : futures) {
    future.wait();
    Response response = future.get();
    responses.push_back(std::move(response));
  }
  return responses;
}

std::vector<Response> Blocking::pivot(const Ptr<Model> &first,
                                      const Ptr<Model> &second,
                                      std::vector<std::string> sources,
                                      const Options &options) {
  // Translate source to pivots.
  std::vector<Response> source_to_pivots =
      translate(first, std::move(sources), options);

  // Translate pivots to targets, after we have outputs at pivot from first
  // round.
  std::vector<Response> responses(source_to_pivots.size());

  Batcher batcher(config_.max_words, config_.wrap_length,
                  config_.tgt_length_limit_factor);

  // Holds the first exception fired by any second-leg request, to be
  // rethrown from this thread after exhaust returns.
  std::exception_ptr pivot_error;

  for (size_t i = 0; i < source_to_pivots.size(); i++) {
    Response &source_to_pivot = source_to_pivots[i];
    Response &response = responses[i];

    auto continuation = [&source_to_pivot,
                         &response](Response &&pivot_to_target) {
      Response combined =
          combine(std::move(source_to_pivot), std::move(pivot_to_target));
      response = std::move(combined);
      return nullptr;
    };
    // Blocking::pivot is synchronous and uses out-parameters instead of a
    // promise; capture the exception in the closure so we can rethrow it
    // from this stack frame after exhaust returns.
    auto on_error = [&pivot_error](std::exception_ptr eptr) {
      pivot_error = std::move(eptr);
    };

    const TextProcessor &processor = second->processor();
    auto [annotated, segments] = processor.process(source_to_pivot.target);
    auto request = make_request(id(), second, cache_, std::move(annotated),
                                std::move(segments), continuation, on_error,
                                /*with_alignment=*/options.alignment);

    batcher.enqueue(request);
  }

  exhaust(config_, second, batcher);

  if (pivot_error) {
    std::rethrow_exception(pivot_error);
  }

  return responses;
}

Async::Async(const Config &config)
    : config_(config),
      cache_(make_cache(config.cache_size)),
      batcher_(config.max_words, config.wrap_length,
               config.tgt_length_limit_factor) {
  // Also creates consumers, starts listening.
  for (size_t i = 0; i < config.workers; i++) {
    workers_.emplace_back([this]() {
      Batch batch;
      Ptr<Model> model;
      std::tie(batch, model) = batcher_.generate();
      while (!batch.empty()) {
        // Cancellation: skip the expensive forward() and abort the batch so
        // each parent Request's promise still completes (otherwise the
        // wrapper's drain would hang). Workers keep pulling and aborting the
        // rest of the queue, so an in-flight translate unwinds within ~one
        // batch per worker — whatever was already mid-forward when the flag
        // flipped.
        if (cancelled()) {
          batch.abort(std::make_exception_ptr(
              std::runtime_error("slimt: translation cancelled")));
          batch = Batch();
          model.reset();
          std::tie(batch, model) = batcher_.generate();
          continue;
        }
        // Catch exceptions so the worker thread doesn't die, but propagate
        // them through the in-flight Requests' failure callbacks instead of
        // silently completing the batch with empty Histories. Each parent
        // Request's `abort()` calls `on_error_`, which sets the exception on
        // the caller's promise — `future.get()` then rethrows on the calling
        // thread, the wrapper catches it, and the app sees a real error
        // rather than an empty translation.
        try {
          Input input = convert(batch, model->vocabulary().pad_id(),
                                config_.tgt_length_limit_factor);
          Histories histories = model->forward(input);
          batch.complete(histories);
        } catch (...) {
          batch.abort(std::current_exception());
        }
        // Release the batch (which holds Ptr<Request>) and the model before
        // re-entering the blocking generate() call. Otherwise an idle worker
        // pins the most recently used model alive — preventing eviction from
        // releasing the underlying mmap until either the service shuts down or
        // another translate request happens to swap this worker's model.
        batch = Batch();
        model.reset();
        std::tie(batch, model) = batcher_.generate();
      }
    });
  }
}

Handle Async::translate(const Ptr<Model> &model, std::string source,
                        const Options &options) {
  auto promise = std::make_shared<Promise>();
  auto future = promise->get_future();
  auto continuation = [promise, on_progress = options.on_progress](
                          Response &&response) {
    promise->set_value(std::move(response));
    if (on_progress) {
      on_progress();
    }
    return nullptr;
  };
  auto on_error = [promise](std::exception_ptr eptr) {
    promise->set_exception(std::move(eptr));
  };

  const TextProcessor &processor = model->processor();
  auto [annotated, segments] =
      processor.process(std::move(source), config_.wrap_length);
  auto request = make_request(id(), model, cache_, std::move(annotated),
                              std::move(segments), continuation, on_error,
                              /*with_alignment=*/options.alignment,
                              options.alternatives, options.forced_prefix);

  batcher_.enqueue(model, request);

  constexpr size_t parts = 1;  // NOLINT
  Handle handle(request, parts, std::move(future));
  return handle;
}

Handle Async::pivot(const Ptr<Model> &first, const Ptr<Model> &second,
                    std::string source, const Options &options) {
  // This is callback chaining or CPS due to async.
  auto promise = std::make_shared<Promise>();
  auto future = promise->get_future();

  bool with_alignment = options.alignment;
  // Both legs forward exceptions to the same outer promise. Once one leg's
  // on_error fires, Request::abort's idempotence prevents a duplicate
  // set_exception on the same promise.
  auto on_error = [promise](std::exception_ptr eptr) {
    promise->set_exception(std::move(eptr));
  };
  auto continuation = [this, promise, second, with_alignment, on_error,
                       on_progress = options.on_progress](
                          Response &&partial) -> Ptr<Request> {
    AnnotatedText intermediate = partial.target;
    auto joining_continuation =
        [source_to_pivot = std::move(partial), promise,
         on_progress](Response &&pivot_to_target) mutable -> Ptr<Request> {
      // We have both Responses at this callback, source_to_pivot is moved in,
      // second half will be available when complete.
      Response response =
          combine(std::move(source_to_pivot), std::move(pivot_to_target));
      promise->set_value(std::move(response));
      // Fire once per input, when both legs are done — same per-input
      // granularity as the direct translate path.
      if (on_progress) {
        on_progress();
      }
      return nullptr;
    };

    const TextProcessor &processor = second->processor();
    auto [annotated, segments] = processor.process(intermediate);

    auto request =
        make_request(id(), second, cache_, std::move(annotated),
                     std::move(segments), std::move(joining_continuation),
                     on_error, with_alignment);

    batcher_.enqueue(second, request);
    return request;
  };

  const TextProcessor &processor = first->processor();
  auto [annotated, segments] =
      processor.process(std::move(source), config_.wrap_length);
  auto request = make_request(id(), first, cache_, std::move(annotated),
                              std::move(segments), continuation, on_error,
                              /*with_alignment=*/options.alignment);

  batcher_.enqueue(first, request);

  constexpr size_t parts = 2;  // NOLINT
  Handle handle(request, parts, std::move(future));
  return handle;
}

Async::~Async() {
  batcher_.shutdown();
  for (std::thread &worker : workers_) {
    assert(worker.joinable());
    worker.join();
  }
  workers_.clear();
}

}  // namespace slimt
