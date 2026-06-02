#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "slimt/Types.hh"

namespace slimt {
class AnnotatedText;
class Vocabulary;

/// TextProcessor wraps the SentencePiece vocabulary used to tokenise an
/// incoming string into model-ready Words plus their byte ranges.
///
/// Sentence splitting USED to live here too (regex/ssplit-cpp + PCRE2).
/// It now lives in the Rust caller (`translator::sentence_split`), which
/// pre-splits inputs into one-sentence-per-call before the model runs.
/// Each `process()` invocation therefore produces a single segment per
/// input — `wrap()` still chops sentences that exceed `wrap_length` so
/// pathologically long single sentences stay decodable.
class TextProcessor {
 public:
  TextProcessor(const Vocabulary &vocabulary);

  std::tuple<AnnotatedText, Segments> process(std::string &&input,
                                              size_t wrap_length) const;
  std::tuple<AnnotatedText, Segments> process(AnnotatedText &source) const;

 private:
  Segment tokenize(const std::string_view &segment,
                   std::vector<std::string_view> &word_ranges) const;

  void wrap(Segment &segment, std::vector<std::string_view> &word_ranges,
            Segments &segments, AnnotatedText &source,
            size_t wrap_length) const;

  const Vocabulary &vocabulary_;
};

}  // namespace slimt
