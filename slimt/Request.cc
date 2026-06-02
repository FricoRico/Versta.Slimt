#include "slimt/Request.hh"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Cache.hh"
#include "slimt/Macros.hh"
#include "slimt/Response.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

size_t cache_key(size_t model_id, const Words &words, bool with_alignment,
                 bool with_alternatives, const Words &forced_prefix) {
  // Plain-text and alignment-bearing translations of the same sentence must
  // not share a cache entry: a History stored without alignment data could
  // otherwise be returned to a caller that asked for alignments. Fold the
  // bool into the key so both variants are independently cacheable. The same
  // reasoning applies to alternatives, and additionally the greedy-only decode
  // it forces can yield a different target than the robust path would.
  auto seed = model_id;
  hash_combine<size_t>(seed, with_alignment ? 1U : 0U);
  hash_combine<size_t>(seed, with_alternatives ? 1U : 0U);
  for (size_t word : words) {
    hash_combine<size_t>(seed, word);
  }
  hash_combine<size_t>(seed, forced_prefix.size());
  for (size_t word : forced_prefix) {
    hash_combine<size_t>(seed, word);
  }
  return seed;
}

// -----------------------------------------------------------------
Request::Request(size_t id, size_t model_id, AnnotatedText &&source,
                 Segments &&segments, const Vocabulary &vocabulary,
                 std::shared_ptr<const Words> shortlist_words,
                 std::optional<TranslationCache> &cache,
                 Continuation &&continuation, OnError &&on_error,
                 bool with_alignment,
                 std::optional<AlternativesConfig> alternatives,
                 Words forced_prefix)
    : id_(id),
      model_id_(model_id),
      source_(std::move(source)),
      segments_(std::move(segments)),
      shortlist_words_(std::move(shortlist_words)),
      vocabulary_(vocabulary),
      cache_(cache),
      continuation_(std::move(continuation)),
      on_error_(std::move(on_error)),
      with_alignment_(with_alignment),
      alternatives_(std::move(alternatives)),
      forced_prefix_(std::move(forced_prefix)) {
  counter_ = segments_.size();
  histories_.resize(segments_.size(), nullptr);

  // 1. If there are no segments_, we are never able to trigger the
  // response_builder calls from a different thread. This happens when the use
  // provides empty input, or the unit and subword preprocessing deems no
  // translatable units present. However, in this case we want an empty valid
  // response. There's no need to do any additional processing here.
  if (segments_.empty()) {
    complete(std::move(histories_));
  } else {
    counter_ = segments_.size();
    histories_.resize(segments_.size());

    // Word count book-keeping.
    words_total_ = 0;
    words_complete_ = 0;

    if (cache_) {
      // Iterate through segments, see if any can be prefilled from cache. If
      // prefilled, mark the particular segments as complete (non-empty
      // ProcessedSegmentRef). Also update accounting used elsewhere
      // (counter_) to reflect one less segment to translate.
      for (size_t idx = 0; idx < segments_.size(); idx++) {
        words_total_ += segments_[idx].size();
        size_t key = cache_key(model_id_, segment(idx), with_alignment_,
                               alternatives_.has_value(), forced_prefix_);
        auto [found, history] = cache_->find(key);
        if (found) {
          histories_[idx] = history;
          --counter_;
          words_complete_ += segments_[idx].size();
        }
      }
      // 2. Also, if cache somehow manages to decrease all counter prefilling
      // histories, then we'd have to trigger ResponseBuilder as well. No
      // segments go into batching and therefore no complete triggers.
      if (counter_.load() == 0) {
        complete(std::move(histories_));
      }
    } else {
      for (const Segment &segment : segments_) {
        words_total_ += segment.size();
      }
    }
  }
}

size_t Request::size() const { return segments_.size(); }

bool Request::cached(size_t index) const {
  return histories_[index] != nullptr;
}

std::pair<Fraction, Fraction> Request::progress() const {
  Fraction words{
      .p = static_cast<size_t>(words_complete_.load()),  //
      .q = words_total_                                  //
  };

  auto completed = static_cast<size_t>(counter_.load());
  Fraction segments{
      .p = segments_.size() - completed,  //
      .q = segments_.size()               //
  };

  return std::make_pair(std::move(words), std::move(segments));
}

size_t Request::word_count(size_t index) const {
  return segments_[index].size();
}

const Segment &Request::segment(size_t index) const { return segments_[index]; }

void Request::abort(std::exception_ptr eptr) {
  // First caller wins. Subsequent calls (e.g. from sibling SegmentRefs in
  // the same failed batch) are dropped — `on_error_` is allowed to assume
  // it fires at most once and downstream `promise->set_exception` would
  // throw `future_already_satisfied` on the second call.
  bool expected = false;
  if (!aborted_.compare_exchange_strong(expected, true)) {
    return;
  }
  if (on_error_) {
    on_error_(std::move(eptr));
  }
}

void Request::process(size_t index, History history) {
  // Concurrently called by multiple workers as a history from translation is
  // ready. The container storing histories is set with the value obtained.

  // Fill in placeholder from History obtained by freshly translating. Since
  // this was a cache-miss to have got through, update cache if available to
  // store the result.
  histories_[index] = std::move(history);
  if (cache_) {
    size_t key = cache_key(model_id_, segment(index), with_alignment_,
                           alternatives_.has_value(), forced_prefix_);
    cache_->store(key, histories_[index]);
  }

  words_complete_ += segments_[index].size();

  // In case this is last request in, completeRequest is called, which sets
  // the value of the promise.
  if (--counter_ == 0) {
    complete(std::move(histories_));
  }
}

void Request::complete(Histories &&histories) {
  SLIMT_ABORT_IF(source_.sentence_count() != histories.size(),
                 "Mismatch in source and translated sentences");
  Response response;

  // Move source_ into response.
  response.source = std::move(source_);
  // Reserving length at least as much as source_ seems like a reasonable
  // thing to do to avoid reallocations.
  response.target.text.reserve(response.source.text.size());

  for (size_t sentence_id = 0; sentence_id < histories.size(); sentence_id++) {
    Words words = histories[sentence_id]->target;
    std::string decoded;
    auto views = vocabulary_.decode(words, decoded, /*ignore_eos=*/false);

    // For each sentence, prepend the filler text between the corresponding
    // source-sentence and the source-sentence before.
    std::string_view pre = response.source.gap(sentence_id);
    response.target.append_sentence(pre, views.begin(), views.end());

    // If this is the last history to be decoded and translated-text
    // constructed, append the text till the end, which could be spaces or
    // empty.
    if (sentence_id + 1 == histories.size()) {
      response.target.append_ending_whitespace(
          response.source.gap(sentence_id + 1));
    }

    // Copy, don't move: the Hypothesis is shared via shared_ptr through the
    // translation cache, so multiple Requests may reference the same object.
    // Moving would empty the alignment in the cached entry — the next request
    // hitting the same cache entry would see `has_alignments == false` and
    // miss the data the alignment-bearing caller actually asked for.
    response.alignments.push_back(histories[sentence_id]->alignment);
    if (alternatives_.has_value()) {
      response.alternatives.push_back(histories[sentence_id]->alternatives);
    }
  }

  next_ = continuation_(std::move(response));
}

}  // namespace slimt
