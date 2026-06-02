#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>

#include "sentencepiece_processor.h"
#include "slimt/Types.hh"

namespace slimt {

class Vocabulary {
 public:
  explicit Vocabulary(const std::string &fpath);
  explicit Vocabulary(View view);
  std::tuple<Words, Views> encode(const std::string_view &line,
                                  bool add_eos = false) const;
  Views decode(const Words &words, std::string &decoded,
               bool ignore_eos = true) const;

  Word pad_id() const { return std::max(0, processor_.pad_id()); }
  Word eos_id() const { return processor_.eos_id(); }
  size_t size() const { return processor_.GetPieceSize(); }

  // Whether `word`'s SentencePiece piece begins a new word — its piece starts
  // with the metaspace marker U+2581 ("▁", bytes E2 96 81). Continuation
  // subwords and control symbols return false.
  bool is_word_start(Word word) const {
    if (word >= size()) return false;
    const std::string &piece = processor_.IdToPiece(static_cast<int>(word));
    return piece.size() >= 3 &&
           static_cast<unsigned char>(piece[0]) == 0xE2 &&
           static_cast<unsigned char>(piece[1]) == 0x96 &&
           static_cast<unsigned char>(piece[2]) == 0x81;
  }

  // Whether `word` ends the current word: it starts a new word (▁), or its
  // piece begins with ASCII punctuation (a standalone token like "." or ","
  // that attaches without a space). Mid-word continuation subwords — including
  // multi-byte accented letters — return false, so expansion keeps consuming
  // them.
  bool ends_word(Word word) const {
    if (word >= size()) return true;
    if (is_word_start(word)) return true;
    const std::string &piece = processor_.IdToPiece(static_cast<int>(word));
    if (piece.empty()) return true;
    unsigned char c0 = static_cast<unsigned char>(piece[0]);
    bool word_char = (c0 & 0x80) || std::isalnum(c0);
    return !word_char;
  }

 private:
  sentencepiece::SentencePieceProcessor processor_;
};

}  // namespace slimt
