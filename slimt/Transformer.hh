#pragma once
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Tensor.hh"
#include "slimt/Types.hh"

namespace slimt {

class Encoder {
 public:
  explicit Encoder(size_t layers, size_t num_heads, size_t feed_forward_depth);
  Tensor forward(const Tensor &embedding, const Tensor &mask) const;
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void prepare_biases();
  const std::vector<EncoderLayer> &encoder() const { return encoder_; }

 private:
  std::vector<EncoderLayer> encoder_;
};

class Decoder {
 public:
  Decoder(size_t layers, size_t num_heads, size_t feed_forward_depth,
          const Tensor &embedding);

  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void prepare_biases();

  std::vector<AttentionContext> prepare_contexts(
      const Tensor &encoder_out) const;
  std::vector<Tensor> start_states(size_t batch_size) const;

  // Pre-compute a shortlisted output projection (selected columns of `W`
  // and the matching slice of `prepared_bias`) once for the whole decode.
  // Model::decode builds one per request in the batch and applies it to that
  // request's rows every step, so the per-step path skips column-selection /
  // bias-gather entirely.
  SelectedAffine prepare_shortlisted_output(const Words &shortlist) const;

  // Returns the final hidden state per active row (pre output-projection)
  // and the guided-alignment attention. The output projection happens in
  // Model::decode, grouped by each row's request shortlist.
  std::tuple<Tensor, Tensor> step(const Tensor &encoder_out, const Tensor &mask,
                                  std::vector<Tensor> &states,
                                  const std::vector<AttentionContext> &contexts,
                                  const Words &previous_step,
                                  size_t step_index = 0) const;

  // Full-vocabulary output projection, for rows whose request carries no
  // shortlist.
  Tensor project(const Tensor &x) const;

 private:
  // Number of positions for which the sinusoidal table is precomputed at
  // model load. Bergamot ships with `wrap_length=128` and
  // `tgt_length_limit_factor=1.5`, giving practical decode lengths under
  // ~200; 1024 covers every realistic configuration. Decodes that somehow
  // exceed this fall back to the per-step `sinusoidal_signal` path.
  // Memory cost: kMaxCachedPositions × embed_dim × sizeof(float) per model.
  static constexpr size_t kMaxCachedPositions = 1024;

  const Tensor &embedding_;
  std::vector<DecoderLayer> decoder_;
  Affine output_;
  Tensor positions_;
};

class Vocabulary;

Words greedy_sample(const Tensor &logits, const Vocabulary &vocabulary,
                    size_t batch_size);
Words greedy_sample_from_words(const Tensor &logits,
                               const Vocabulary &vocabulary, const Words &words,
                               size_t batch_size);

void transform_embedding(Tensor &word_embedding, size_t start = 0);
void transform_embedding(Tensor &word_embedding, const Tensor &positions,
                         size_t start = 0);

class Transformer {
 public:
  explicit Transformer(size_t encoder_layers, size_t decoder_layers,
                       size_t num_heads, size_t feed_forward_depth, View model);

  // Source-side embedding (encoder input lookup). For shared-vocab models the
  // encoder and decoder both use this tensor; for two-vocab models it's
  // loaded from `encoder_Wemb` and is distinct from `decoder_embedding()`.
  const Tensor &embedding() const { return encoder_embedding_; }
  const Encoder &encoder() const { return encoder_; }
  const Decoder &decoder() const { return decoder_; }

 private:
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void load_parameters();
  void prepare_biases();

  std::vector<io::Item> items_;
  // Encoder input embedding. Loaded from `Wemb` (single-vocab) or
  // `encoder_Wemb` (two-vocab).
  Tensor encoder_embedding_;
  // Decoder input/output embedding (tied with output projection). Loaded
  // from `Wemb` (single-vocab — same tensor as encoder_embedding_'s source)
  // or `decoder_Wemb` (two-vocab).
  Tensor decoder_embedding_;
  Encoder encoder_;
  Decoder decoder_;
};

}  // namespace slimt
