#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "Tensor.hh"
#include "Types.hh"

namespace leanmt {

class Input {
 public:
  Input(size_t batch_size, size_t sequence_length, uint32_t pad_id,
        float limit_factor);

  void add(const std::vector<uint32_t> &words);
  void finalize();

  const Tensor &indices() const { return batch_; }
  const Tensor &mask() const { return mask_; }
  const std::vector<uint32_t> &words() const { return words_; }
  void set_shortlist(RowShortlists rows) { shortlist_rows_ = std::move(rows); }
  const RowShortlists &shortlist_rows() const { return shortlist_rows_; }
  void set_alternatives(AlternativesConfig cfg, std::vector<bool> rows) {
    alternatives_ = cfg;
    alternatives_rows_ = std::move(rows);
  }
  const std::optional<AlternativesConfig> &alternatives() const {
    return alternatives_;
  }
  // Per batch row (original sentence index): whether that row requested
  // alternatives. Empty when no row did. Used to exclude those rows from the
  // robust beam re-decode so their greedy alternatives survive.
  const std::vector<bool> &alternatives_rows() const {
    return alternatives_rows_;
  }
  void set_forced(std::vector<Words> rows) { forced_ = std::move(rows); }
  // Per batch row: the target tokens to force before free-running (empty for
  // rows without a forced prefix). Empty overall when no row forces anything.
  const std::vector<Words> &forced() const { return forced_; }
  // Ceiling on the robust re-decode beam width for this batch. Carried from
  // Options::max_beam_width (default 3). 1 (or less) disables the beam pass.
  void set_max_beam_width(size_t width) { max_beam_width_ = width; }
  size_t max_beam_width() const { return max_beam_width_; }
  const std::vector<size_t> &lengths() const { return lengths_; }
  size_t index() const { return index_; }
  float occupancy();
  float limit_factor() const;

 private:
  std::vector<uint32_t> words_;
  RowShortlists shortlist_rows_;
  std::optional<AlternativesConfig> alternatives_;
  std::vector<bool> alternatives_rows_;
  std::vector<Words> forced_;
  std::vector<size_t> lengths_;
  Tensor batch_;
  Tensor mask_;
  size_t index_ = 0;
  uint32_t pad_id_ = 0;
  size_t used_ = 0;
  float limit_factor_;
  size_t max_beam_width_ = 3;
  bool finalized_ = false;
};
}  // namespace leanmt
