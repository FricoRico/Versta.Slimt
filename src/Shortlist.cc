#include "Shortlist.hh"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <stdexcept>

#include "Macros.hh"
#include "Types.hh"
#include "Utils.hh"
#include "Vocabulary.hh"

namespace leanmt {

namespace {

// Shortlist binary-integrity errors are user-fixable (re-download / replace
// the lex.*.s2t.bin file), so they throw `std::runtime_error` rather than
// going through `LEANMT_ABORT*`. Leanmt's frontend catches these and the
// caller sees a recoverable Result::Err.
[[noreturn]] void shortlist_throw(const std::string& message) {
  throw std::runtime_error(std::string("[leanmt] ") + message);
}

#define SHORTLIST_THROW_IF(cond, message) \
  do {                                    \
    if (cond) {                           \
      shortlist_throw(message);           \
    }                                     \
  } while (0)

}  // namespace

bool ShortlistGenerator::content_check() {
  bool fail_flag = false;
  // The offset table has to be within the size of shortlists.
  for (size_t i = 0; i < word_to_offset_size_ - 1; i++) {
    fail_flag |= word_to_offset_[i] >= shortlist_size_;
  }

  SHORTLIST_THROW_IF(fail_flag, "shortlist file: offset table not within shortlist size (corrupt lex.*.s2t.bin)");

  // The last element of word_to_offset_ must equal shortlist_size_
  fail_flag |= word_to_offset_[word_to_offset_size_ - 1] != shortlist_size_;

  SHORTLIST_THROW_IF(fail_flag, "shortlist file: word_to_offset != shortlist_size (corrupt lex.*.s2t.bin)");

  // The vocabulary indices have to be within the vocabulary size.
  size_t v_size = target_.size();
  for (size_t j = 0; j < shortlist_size_; j++) {
    fail_flag |= shortlist_[j] >= v_size;
  }

  SHORTLIST_THROW_IF(fail_flag, "shortlist file: indices out of bounds (corrupt lex.*.s2t.bin or wrong vocab)");
  return fail_flag;
}

// load shortlist from buffer
void ShortlistGenerator::load(const void* data, size_t blob_size,
                              bool check /*= true*/) {
  /* File layout:
   * header
   * word_to_offset array
   * short_lists array
   */
  (void)blob_size;
  SHORTLIST_THROW_IF(blob_size < sizeof(Header),
                     "shortlist file too short to contain a header (size " +
                         std::to_string(blob_size) +
                         "); re-download lex.*.s2t.bin");

  const char* ptr = static_cast<const char*>(data);
  const Header& header = *reinterpret_cast<const Header*>(ptr);
  ptr += sizeof(Header);
  SHORTLIST_THROW_IF(header.magic != kMagic,
                     "shortlist file: wrong magic number (not a leanmt "
                     "lex.*.s2t.bin, or corrupt)");

  uint64_t expected_size = sizeof(Header) +
                           header.word_to_offset_size * sizeof(uint64_t) +
                           header.shortlist_size * sizeof(Word);

  SHORTLIST_THROW_IF(expected_size != blob_size,
                     "shortlist file size mismatch: header claims " +
                         std::to_string(expected_size) + " bytes but file is " +
                         std::to_string(blob_size) + " (truncated or corrupt)");

  if (check) {
    size_t length = (       //
        blob_size           //
        - sizeof(uint64_t)  // header.magic
        - sizeof(uint64_t)  // header.checksum
    );
    auto checksum_actual = hash_bytes<uint64_t, uint64_t>(
        &header.frequent, (length) / sizeof(uint64_t));

    SHORTLIST_THROW_IF(checksum_actual != header.checksum,
                       "shortlist file: checksum mismatch (corrupt "
                       "lex.*.s2t.bin); re-download the model");
  }

  frequent_ = header.frequent;
  best_ = header.best;
  LOG(info, "[data] Lexical short list frequent %lu and best %lu", frequent_,
      best_);

  word_to_offset_size_ = header.word_to_offset_size;
  shortlist_size_ = header.shortlist_size;

  // Offsets right after header.
  word_to_offset_ = reinterpret_cast<const uint64_t*>(ptr);
  ptr += word_to_offset_size_ * sizeof(uint64_t);

  shortlist_ = reinterpret_cast<const Word*>(ptr);

  // Verify offsets and vocab ids are within bounds if requested by user.
  if (check) {
    content_check();
  }
}

ShortlistGenerator::ShortlistGenerator(                        //
    View view,                                                 //
    const Vocabulary& source, const Vocabulary& target,        //
    size_t source_index /*= 0*/, size_t /*target_index = 1*/,  //
    bool shared /*= false*/, bool check /*= true*/)
    : source_(source),
      target_(target),
      source_index_(source_index),
      shared_(shared) {
  LOG(info, "[data] Loading binary shortlist from buffer with check=%d", check);
  load(view.data, view.size, check);

  (void)source_index_;
}

Shortlist ShortlistGenerator::generate(const Words& words,
                                       size_t min_candidates) const {
  size_t source_size = source_.size();
  size_t target_size = target_.size();

  // Since V=target_.size() is not large, anchor the time and space
  // complexity to O(V). Attempt to squeeze the truth tables into CPU cache
  std::vector<bool> source_table(source_size, false);
  std::vector<bool> target_table(target_size, false);

  // Add frequent most frequent words
  for (Word i = 0; i < frequent_ && i < target_size; ++i) {
    target_table[i] = true;
  }

  // Collect unique words from source.
  // Add aligned target words: mark target_table[word] to 1
  for (auto word : words) {
    if (shared_) {
      target_table[word] = true;
    }
    // If word has not been encountered, add the corresponding target
    // words
    if (!source_table[word]) {
      size_t begin = word_to_offset_[word];
      size_t end = word_to_offset_[word + 1];
      for (uint64_t j = begin; j < end; j++) {
        target_table[shortlist_[j]] = true;
      }
      source_table[word] = true;
    }
  }

  // Due to the 'multiple-of-eight' issue, the following O(N) patch is inserted
  size_t target_table_ones = 0;  // counter for no. of selected target words
  for (size_t i = 0; i < target_size; i++) {
    if (target_table[i]) {
      target_table_ones++;
    }
  }

  // Top up to min_candidates with the next-most-frequent ids, and ensure
  // the generated vocabulary items are a multiple of the vector extension
  // alignment.
  for (size_t i = frequent_;
       i < target_size && (target_table_ones < min_candidates ||
                           target_table_ones % kVExtAlignment != 0);
       i++) {
    if (!target_table[i]) {
      target_table[i] = true;
      target_table_ones++;
    }
  }

  // turn selected indices into vector and sort (Bucket sort: O(V))
  std::vector<Word> indices;
  for (Word i = 0; i < target_size; i++) {
    if (target_table[i]) {
      indices.push_back(i);
    }
  }

  return Shortlist(std::move(indices));
}



}  // namespace leanmt
