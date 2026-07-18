#include "TextProcessor.hh"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "Annotation.hh"
#include "Types.hh"
#include "Vocabulary.hh"

namespace leanmt {

Segment TextProcessor::tokenize(
    const std::string_view &segment,
    std::vector<std::string_view> &word_ranges) const {
  auto [words, views] = vocabulary_.encode(segment, /*add_eos=*/false);
  word_ranges.reserve(word_ranges.size() +
                      distance(views.begin(), views.end()));
  word_ranges.insert(word_ranges.end(), views.begin(), views.end());
  return words;
}

TextProcessor::TextProcessor(const Vocabulary &vocabulary)
    : vocabulary_(vocabulary) {}

std::tuple<AnnotatedText, Segments> TextProcessor::process(
    std::string &&input, size_t wrap_length) const {
  // The caller has pre-split this input into a single sentence (see
  // translator::sentence_split on the Rust side). Tokenise the whole
  // input as one segment, then let `wrap()` chop pathologically long
  // sentences so they stay under the model's positional-encoding limit.
  AnnotatedText source(std::move(input));
  Segments segments;
  std::string_view sentence(source.text.data(), source.text.size());

  std::vector<std::string_view> word_ranges;
  Segment segment = tokenize(sentence, word_ranges);

  if (!segment.empty()) {
    wrap(segment, word_ranges, segments, source, wrap_length);
  }
  return {std::move(source), std::move(segments)};
}

void TextProcessor::wrap(Segment &segment,
                         std::vector<std::string_view> &word_ranges,
                         Segments &segments, AnnotatedText &source,
                         size_t wrap_length) const {
  // There's an EOS token added to the words, manually.
  // SentencePiece/marian-vocab is set to not append EOS. Marian requires EOS to
  // be at the end as a marker to start translating. So while we're supplied
  // wrap_length_ from outside, we need to ensure there's space for EOS in
  // each wrapped segment.
  Word eos_id = vocabulary_.eos_id();
  size_t wrap_step_size = wrap_length - 1;

  for (size_t offset = 0; offset < segment.size(); offset += wrap_step_size) {
    auto start = segment.begin() + offset;

    // Restrict the range within bounds.
    size_t left = segment.size() - offset;
    size_t diff = std::min(wrap_step_size, left);

    segments.emplace_back(start, start + diff);
    segments.back().push_back(eos_id);

    auto astart = word_ranges.begin() + offset;

    // Construct a part vector of std::string_view representing wrapped segment,
    // use the last std::string_view to create an EOS std::string_view manually.
    std::vector<std::string_view> partial_word_ranges(astart, astart + diff);
    std::string_view &last = partial_word_ranges.back();
    const char *end = last.data() + last.size();
    partial_word_ranges.emplace_back(end, 0);
    // diff > 0
    source.record_existing_sentence(partial_word_ranges.begin(),
                                    partial_word_ranges.end(), astart->data());
  }
}

std::tuple<AnnotatedText, Segments> TextProcessor::process(
    AnnotatedText &source) const {
  // The difference here is that there is no wrap involved.
  Segments segments;
  std::string text = source.text;
  AnnotatedText replacement(std::move(text));

  for (size_t s = 0; s < source.sentence_count(); s++) {
    // This is our sentence_stream
    Range sentence_range = source.sentence_as_range(s);

    // Fool tokenization using Ranges into looking at replacement. They're
    // same, so okay.
    std::string_view sentence{&replacement.text[sentence_range.begin],
                              sentence_range.size()};

    std::vector<std::string_view> word_ranges;
    Segment segment = tokenize(sentence, word_ranges);

    // Manually add EoS
    Word eos_id = vocabulary_.eos_id();
    segment.push_back(eos_id);

    if (!word_ranges.empty()) {
      std::string_view &last =
          word_ranges.back();  // this is a possible segfault if
                               // word_ranges is empty. So guard.
      const char *end = last.data() + last.size();
      word_ranges.emplace_back(end, 0);
    } else {
      const char *end = sentence.data() + sentence.size();
      word_ranges.emplace_back(end, 0);
    }

    segments.push_back(std::move(segment));
    replacement.record_existing_sentence(word_ranges.begin(), word_ranges.end(),
                                         word_ranges.begin()->data());
  }

  return {std::move(replacement), std::move(segments)};
}

}  // namespace leanmt
