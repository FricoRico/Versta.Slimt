#pragma once
#include <cstddef>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

#include "slimt/Cache.hh"

namespace slimt {

/// Range stores indices for half-interval [begin, end) in a string. Can be
/// used to represent a sentence, word.
struct Range {
  size_t begin;
  size_t end;
  size_t size() const { return end - begin; }
};

// Convenience shorthand to represent a fraction. Don't want to use
// std::pair<size_t, size_t> and the more verbose .first, .second, hence .p, .q,
// where p/q represents the fraction.
struct Fraction {
  size_t p;
  size_t q;
};

inline bool operator==(Range &a, Range b) {
  return a.begin == b.begin && a.end == b.end;
}

using Word = uint32_t;
using Words = std::vector<Word>;

// Per batch row, the output-vocabulary candidates of the row's own request
// (rows of one request share the same list); a null entry means that row's
// request has no shortlist and decodes over the full vocabulary. Keeping
// the candidates per row makes a sentence's translation independent of
// whichever other requests happened to be co-batched — batch composition
// varies with worker timing, and sampling from a batch-level union let junk
// candidates from unrelated requests win the argmax.
using RowShortlists = std::vector<std::shared_ptr<const Words>>;

struct View {
  void *data = nullptr;
  size_t size = 0;
};

using Views = std::vector<std::string_view>;

using Segment = Words;
using Segments = std::vector<Segment>;
using Sentences = std::vector<Words>;

template <class T>
using Ptr = std::shared_ptr<T>;

using Distribution = std::vector<float>;
using Alignment = std::vector<Distribution>;
using Alignments = std::vector<Alignment>;

// One candidate from a decode step's distribution: the token sequence forming a
// complete target word (the first subword plus its greedy continuation to the
// next word boundary) and the softmax probability of its first subword. The
// chosen (argmax) token is the first entry of a step's candidate list, so its
// probability doubles as that token's confidence.
struct TokenAlternative {
  Words word;
  float prob;
};
using StepAlternatives = std::vector<TokenAlternative>;
// Per generated target token, the top-k candidates for that step. Parallel to
// `Hypothesis::target`. Empty unless alternatives were requested.
using TokenAlternatives = std::vector<StepAlternatives>;

// Knobs for harvesting per-token alternatives during greedy decode. `top_k`
// candidates are kept per step, dropping any below `min_prob`.
struct AlternativesConfig {
  size_t top_k;
  float min_prob;
};

struct Hypothesis {
  Segment target;
  Alignment alignment;
  TokenAlternatives alternatives;
};

using History = Ptr<Hypothesis>;
using Histories = std::vector<History>;
using TranslationCache = AtomicCache<size_t, History>;

struct Response;

using Promise = std::promise<Response>;
using Future = std::future<Response>;

enum class Encoding {
  Byte,  //
  UTF8   //
};

}  // namespace slimt
