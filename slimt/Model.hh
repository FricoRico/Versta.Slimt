#pragma once
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "slimt/Annotation.hh"
#include "slimt/Export.hh"
#include "slimt/Io.hh"
#include "slimt/Shortlist.hh"
#include "slimt/TextProcessor.hh"
#include "slimt/Transformer.hh"
#include "slimt/Types.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

class Input;
class Tensor;
class Arena;

template <class Field>
struct Package {
  Field model;
  // Source-side vocabulary, used to tokenize input. For models whose source
  // and target vocabularies are the same (the bergamot tiny11 baseline), this
  // is the only vocabulary present and `target_vocabulary` should be left
  // empty / default-constructed; the Model will alias the target side to the
  // same Vocabulary instance.
  Field vocabulary;
  // Target-side vocabulary, used to decode generated token IDs back to text
  // and to look up the EOS id during greedy sampling. Must be set for
  // two-vocab models such as Mozilla's en-zh, en-ja, en-ko, en-zh_hant and
  // zh_hant-en, which ship `srcvocab.*.spm` + `trgvocab.*.spm` and have
  // separate `encoder_Wemb` / `decoder_Wemb` tensors in the model file.
  Field target_vocabulary;
  Field shortlist;
};

class SLIMT_EXPORT Model {
 public:
  struct SLIMT_EXPORT Config {
    // NOLINTBEGIN
    size_t encoder_layers = 6;
    size_t decoder_layers = 2;
    size_t feed_forward_depth = 2;
    size_t num_heads = 8;
    // NOLINTEND
  };

  explicit Model(const Config &config, const Package<std::string> &package);
  explicit Model(const Config &config, const Package<View> &package);

  Histories forward(const Input &input) const;

  const Config &config() const { return config_; }
  // Source-side vocabulary. Use this for tokenizing input and for the input
  // batch's pad_id.
  const Vocabulary &vocabulary() const { return src_vocabulary_; }
  // Target-side vocabulary, used for greedy_sample's eos check and to decode
  // generated word IDs back to text. Aliases the source vocabulary for
  // shared-vocab models.
  const Vocabulary &target_vocabulary() const {
    return tgt_vocabulary_ ? *tgt_vocabulary_ : src_vocabulary_;
  }
  const TextProcessor &processor() const { return processor_; }
  const Transformer &transformer() const { return transformer_; }
  size_t id() const { return id_; }  // NOLINT
  const std::optional<ShortlistGenerator> &shortlist_generator() const {
    return shortlist_generator_;
  }

 private:
  // Greedy decode over a batch. `mask`, `shortlist_rows`, `lengths` and
  // `limit_factor` are the per-row inputs (passed explicitly, not as an Input,
  // so a flagged subset can be re-decoded). When `out_deficits` is non-null it
  // receives each sentence's worst contiguous log-prob deficit — the two-pass
  // router's flag signal.
  Histories decode(const Tensor &encoder_out, const Tensor &mask,
                   const RowShortlists &shortlist_rows,
                   const std::vector<size_t> &lengths, float limit_factor,
                   Arena &arena, std::vector<double> *out_deficits,
                   const std::optional<AlternativesConfig> &alt_cfg,
                   const std::vector<Words> &forced_prefix) const;

  // Batched beam decode over `S` sentences with a per-sentence width
  // (`beam_widths[s]`). All Σ widths hypotheses step in one fused batched
  // decode (each sentence's beams share its encoder row via replication; the
  // SSRU states are gathered by surviving parent each step). Used by the
  // two-pass router to re-decode flagged sentences. Returns one Hypothesis per
  // sentence (best by length-normalized score).
  Histories decode_beam(const Tensor &encoder_out, const Tensor &mask,
                        const RowShortlists &shortlist_rows,
                        const std::vector<size_t> &lengths,
                        const std::vector<size_t> &beam_widths,
                        float limit_factor, Arena &arena) const;

  static std::optional<ShortlistGenerator> make_shortlist_generator(
      View view, const Vocabulary &source, const Vocabulary &target);

  size_t id_;
  Config config_;
  using Mmap = Package<io::MmapFile>;
  std::optional<Mmap> mmap_;
  Package<View> view_;

  // Source vocabulary is always present. Target vocabulary is null for
  // single-vocab models (the source vocab is reused), or owned-non-null for
  // two-vocab models such as the bergamot CJK pairs. unique_ptr is used
  // because Vocabulary embeds a non-copyable / non-movable
  // SentencePieceProcessor, so it can't sit inside std::optional.
  Vocabulary src_vocabulary_;
  std::unique_ptr<Vocabulary> tgt_vocabulary_;
  TextProcessor processor_;
  Transformer transformer_;
  std::optional<ShortlistGenerator> shortlist_generator_;
};

}  // namespace slimt
