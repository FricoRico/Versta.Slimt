#include "slimt/Transformer.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "slimt/Io.hh"
#include "slimt/Modules.hh"
#include "slimt/Tensor.hh"
#include "slimt/TensorOps.hh"
#include "slimt/Types.hh"
#include "slimt/Utils.hh"
#include "slimt/Vocabulary.hh"

namespace slimt {

void transform_embedding(Tensor &word_embedding, size_t start /* = 0*/) {
  // This is a pain, why does marian-transpose here, I do not get yet.

  uint64_t embed_dim = word_embedding.dim(-1);
  uint64_t sequence_length = word_embedding.dim(-2);
  uint64_t batch_size = word_embedding.dim(-3);

  auto *word_embedding_ptr = word_embedding.data<float>();

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L88
  mul_scalar(word_embedding_ptr, std::sqrt(static_cast<float>(embed_dim)),
             word_embedding.size(), word_embedding_ptr);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L105
  Tensor positional_embedding(word_embedding.type(),
                              Shape({sequence_length, embed_dim}),
                              "positional_embedding");
  auto *positional_embedding_ptr = positional_embedding.data<float>();
  sinusoidal_signal(start, sequence_length, embed_dim,
                    positional_embedding_ptr);

  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L109
  add_positional_embedding(word_embedding_ptr, positional_embedding_ptr,
                           batch_size, sequence_length, embed_dim,
                           word_embedding_ptr);
}

// Variant that reads from a precomputed sinusoidal table instead of running
// `sinusoidal_signal` every step. The table is owned by the Decoder and
// populated once at model load.
void transform_embedding(Tensor &word_embedding, const Tensor &positions,
                         size_t start /* = 0*/) {
  uint64_t embed_dim = word_embedding.dim(-1);
  uint64_t sequence_length = word_embedding.dim(-2);
  uint64_t batch_size = word_embedding.dim(-3);
  assert(positions.dim(-1) == embed_dim);
  assert(start + sequence_length <= positions.dim(-2));

  auto *word_embedding_ptr = word_embedding.data<float>();
  mul_scalar(word_embedding_ptr, std::sqrt(static_cast<float>(embed_dim)),
             word_embedding.size(), word_embedding_ptr);

  const float *positional_embedding_ptr =
      positions.data<float>() + start * embed_dim;
  add_positional_embedding(word_embedding_ptr, positional_embedding_ptr,
                           batch_size, sequence_length, embed_dim,
                           word_embedding_ptr);
}

Encoder::Encoder(size_t layers, size_t num_heads, size_t feed_forward_depth) {
  for (size_t i = 0; i < layers; i++) {
    encoder_.emplace_back(i + 1, feed_forward_depth, num_heads);
  }
}

Tensor Encoder::forward(const Tensor &word_embedding,
                        const Tensor &mask) const {
  auto [x, attn] = encoder()[0].forward(word_embedding, mask);

  for (size_t i = 1; i < encoder_.size(); i++) {
    const EncoderLayer &layer = encoder()[i];
    auto [y, attn_y] = layer.forward(x, mask);

    // Overwriting x so that x is destroyed and we need lesser working memory.
    x = std::move(y);
  }
  return std::move(x);
}

void Encoder::register_parameters(const std::string &prefix,
                                  ParameterMap &parameters) {
  for (EncoderLayer &layer : encoder_) {
    layer.register_parameters(prefix, parameters);
  }
}

void Encoder::prepare_biases() {
  for (EncoderLayer &layer : encoder_) {
    layer.prepare_biases();
  }
}

std::vector<Tensor> Decoder::start_states(size_t batch_size) const {
  std::vector<Tensor> states;
  for (const auto &layer : decoder_) {
    Tensor state = layer.start_state(batch_size);
    states.push_back(std::move(state));
  }
  return states;
}

std::vector<AttentionContext> Decoder::prepare_contexts(
    const Tensor &encoder_out) const {
  std::vector<AttentionContext> contexts;
  contexts.reserve(decoder_.size());
  for (const auto &layer : decoder_) {
    contexts.push_back(layer.prepare_context(encoder_out));
  }
  return contexts;
}

Transformer::Transformer(size_t encoder_layers, size_t decoder_layers,
                         size_t num_heads, size_t feed_forward_depth,
                         View model)
    : items_(io::load_items(model.data)),
      encoder_(encoder_layers, num_heads, feed_forward_depth),  //
      decoder_(decoder_layers, num_heads, feed_forward_depth,
               decoder_embedding_) {
  load_parameters();
  prepare_biases();
}

Decoder::Decoder(size_t layers, size_t num_heads, size_t feed_forward_depth,
                 const Tensor &embedding)
    : embedding_(embedding) {
  for (size_t i = 0; i < layers; i++) {
    decoder_.emplace_back(i + 1, feed_forward_depth, num_heads);
  }
}

void Decoder::register_parameters(const std::string &prefix,
                                  ParameterMap &parameters) {
  // Somehow we have historically ended up with `none_QuantMultA` being used for
  // Wemb_QuantMultA.
  // https://github.com/browsermt/marian-dev/blob/2be8344fcf2776fb43a7376284067164674cbfaf/scripts/alphas/extract_stats.py#L55
  // - none_QuantMultA is generated when used with shortlist
  // - Wemb_QuantMultA is generated when used without shortlist.
  //
  // For shared-vocab models the output projection's W tensor is the
  // PrepareB-form alias of `Wemb`, named `Wemb_intgemm8`. For two-vocab
  // models it's the alias of `decoder_Wemb`, named `decoder_Wemb_intgemm8`.
  // Register both — Io.cc only emits whichever matches the model file.
  parameters.emplace("Wemb_intgemm8", &output_.W);
  parameters.emplace("decoder_Wemb_intgemm8", &output_.W);
  parameters.emplace("none_QuantMultA", &output_.quant);
  parameters.emplace("decoder_Wemb_QuantMultA", &output_.quant);
  parameters.emplace("decoder_ff_logit_out_b", &output_.b);

  for (DecoderLayer &layer : decoder_) {
    layer.register_parameters(prefix, parameters);
  }
}

void Decoder::prepare_biases() {
  prepare_bias(output_);
  for (DecoderLayer &layer : decoder_) {
    layer.prepare_biases();
  }

  // Precompute the sinusoidal positional table once. `embedding_` has been
  // populated by load_parameters() (called before prepare_biases() in
  // Transformer::Transformer), so embed_dim is known here. The Decoder is
  // const after this — every Model::decode reads the same buffer with no
  // synchronization, and step() avoids the per-step alloc + transcendentals.
  size_t embed_dim = embedding_.dim(-1);
  positions_ = Tensor(Type::f32, Shape({kMaxCachedPositions, embed_dim}),
                      "decoder_positions");
  sinusoidal_signal(0, kMaxCachedPositions, embed_dim,
                    positions_.data<float>());
}

SelectedAffine Decoder::prepare_shortlisted_output(
    const Words &shortlist) const {
  return prepare_selected(output_, shortlist);
}

std::tuple<Tensor, Tensor> Decoder::step(
    const Tensor &encoder_out, const Tensor &mask, std::vector<Tensor> &states,
    const std::vector<AttentionContext> &contexts,
    const Words &previous_step, size_t step_index) const {
  // Infer batch-size from encoder_out.
  size_t encoder_feature_dim = encoder_out.dim(-1);
  size_t source_sequence_length = encoder_out.dim(-2);
  size_t batch_size = encoder_out.dim(-3);

  (void)encoder_feature_dim;
  (void)source_sequence_length;

  // Trying to re-imagine:
  // https://github.com/browsermt/marian-dev/blob/f436b2b7528927333da1629a74fde3779c0a96dd/src/models/decoder.h#L67
  auto from_sentences = [this](const Words &previous_step, size_t batch_size) {
    const std::string name = "target_embed";
    size_t embed_dim = embedding_.dim(-1);

    // If no words, generate one embedding with all 0s.
    if (previous_step.empty()) {
      size_t sequence_length = 1;
      Shape shape({batch_size, sequence_length, embed_dim});
      Tensor empty_embed(Type::f32, std::move(shape), name);
      empty_embed.fill_in_place(0.0F);
      return empty_embed;
    }

    size_t sequence_length = 1;
    Shape shape({batch_size, sequence_length, embed_dim});
    Tensor embedding(Type::f32, std::move(shape), name);
    const float *source = embedding_.data<float>();
    float *target = embedding.data<float>();
    for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
      size_t token = previous_step[batch_id];
      const float *row = source + token * embed_dim;
      std::copy(row, row + embed_dim, target + batch_id * embed_dim);
    }

    return embedding;
  };

  Tensor decoder_embed = from_sentences(previous_step, batch_size);
  // Marian's transformer.h adds positional encoding for the absolute decoder
  // position (`startPos`) at every step. Using start=0 here makes greedy
  // decoding repeat tokens (e.g. "Hello, Hello, Hello") because every step
  // sees the same position signal.
  assert(step_index < kMaxCachedPositions);
  transform_embedding(decoder_embed, positions_, step_index);

  auto [x, attn] =
      decoder_[0].forward(contexts[0], mask, states[0], decoder_embed);

  Tensor guided_alignment = std::move(attn);
  for (size_t i = 1; i < decoder_.size(); i++) {
    auto [y, _attn] = decoder_[i].forward(contexts[i], mask, states[i], x);
    x = std::move(y);
    if (i + 1 == decoder_.size()) {
      // Last decoder layer
      // https://github.com/marian-nmt/marian-dev/blob/53b0b0d7c83e71265fee0dd832ab3bcb389c6ec3/src/models/transformer.h#L826C31-L826C41
      guided_alignment = std::move(_attn);
    }
  }

  return {std::move(x), std::move(guided_alignment)};
}

Tensor Decoder::project(const Tensor &x) const {
  return affine(output_, x, "logits");
}

void Transformer::load_parameters() {
  // Get the parameterss from strings to target tensors to load.
  ParameterMap parameters;
  std::string prefix;
  register_parameters(prefix, parameters);

  auto debug = [&parameters]() {
    for (const auto &p : parameters) {
      std::cout << p.first << "\n";
    }
  };

  (void)debug;

  auto lookup = [&parameters](const std::string &name) -> Tensor * {
    auto query = parameters.find(name);
    if (query == parameters.end()) {
      return nullptr;
    }
    return query->second;
  };

  std::vector<std::string> missed;
  bool loaded_shared_wemb = false;
  for (io::Item &item : items_) {
    if (item.name.empty()) continue;  // placeholder slots from Io.cc
    Tensor *target = lookup(item.name);
    if (target) {
      target->load(item.view, item.type, item.shape, item.name);
      // For shared-vocab models the single `Wemb` (and its `Wemb_intgemm8`
      // alias) covers both encoder input embedding and the decoder's
      // input/output projection. The Decoder holds a reference to
      // `decoder_embedding_`; mirror the encoder copy into it below.
      if (item.name == "Wemb") loaded_shared_wemb = true;
      parameters.erase(item.name);
    } else {
      missed.push_back(item.name);
    }
  }

  // Shared-vocab fallback: copy the loaded `Wemb` into `decoder_embedding_`
  // so Decoder::step reads the same data through its reference. Two-vocab
  // models populate `decoder_embedding_` directly from `decoder_Wemb` and
  // never need this copy.
  if (loaded_shared_wemb) {
    decoder_embedding_.load(encoder_embedding_.view(), encoder_embedding_.type(),
                            encoder_embedding_.shape(),
                            encoder_embedding_.name());
    // The decoder side's parameter slots for `decoder_Wemb*` legitimately
    // didn't load from this model file — drop them from `missed`/`parameters`
    // accounting.
    parameters.erase("decoder_Wemb");
  }

  for (std::string &entry : missed) {
    // `Wemb_QuantMultA` and `encoder_Wemb_QuantMultA` ship in the model
    // file as ig8 alpha markers. Io.cc un-quantizes them into real f32
    // values, but they have no parameter slot — the decoder output
    // projection's alpha comes from `none_QuantMultA` (shared-vocab) or
    // `decoder_Wemb_QuantMultA` (two-vocab), and the encoder doesn't
    // GEMM through `encoder_Wemb` at all. So these are harmlessly
    // unconsumed; don't warn.
    if (entry == "Wemb_QuantMultA" || entry == "encoder_Wemb_QuantMultA") {
      continue;
    }
    std::cerr << "[warn] Failed to ingest expected load of " << entry << "\n";
  }
  for (auto &parameter : parameters) {
    // Embedding-related slots are intentionally over-registered (Wemb,
    // encoder_Wemb, decoder_Wemb) — at most one set of names matches any
    // given model file. Suppress the warning for the unmatched ones.
    //
    // `none_QuantMultA` is the activation alpha for the decoder output
    // projection on shared-vocab models. Two-vocab models (en-ja, en-ko,
    // ...) ship neither it nor `decoder_Wemb_QuantMultA` with a real
    // value; Modules::affine falls back to dynamic activation
    // quantization in that case (see `a_quant_for`).
    const std::string &name = parameter.first;
    if (name == "Wemb" || name == "encoder_Wemb" || name == "decoder_Wemb" ||
        name == "Wemb_intgemm8" || name == "decoder_Wemb_intgemm8" ||
        name == "decoder_Wemb_QuantMultA" || name == "none_QuantMultA") {
      continue;
    }
    std::cerr << "[warn] Failed to complete expected load of ";
    std::cerr << parameter.first << "\n";
  }
}

void Transformer::register_parameters(const std::string &prefix,
                                      ParameterMap &parameters) {
  // Shared-vocab models (the bergamot tiny11 baseline) ship a single `Wemb`
  // that the encoder reads as input embedding AND the decoder reuses as
  // input lookup + output projection. Two-vocab models (en-zh, en-ja, ...)
  // ship separate `encoder_Wemb` and `decoder_Wemb` tensors of different
  // sizes. Register all three names; Io.cc + Transformer::load_parameters
  // populate whichever the model file actually contains. For shared-vocab
  // models the `Wemb` ingest path here populates `encoder_embedding_`, and
  // a post-load step copies it into `decoder_embedding_` so Decoder::step
  // (which holds a reference to decoder_embedding_) sees the same data.
  parameters.emplace("Wemb", &encoder_embedding_);
  parameters.emplace("encoder_Wemb", &encoder_embedding_);
  parameters.emplace("decoder_Wemb", &decoder_embedding_);
  encoder_.register_parameters(prefix, parameters);
  decoder_.register_parameters(prefix, parameters);
}

void Transformer::prepare_biases() {
  encoder_.prepare_biases();
  decoder_.prepare_biases();
}

Words greedy_sample(const Tensor &logits, const Vocabulary &vocabulary,
                    size_t batch_size) {
  Words sampled_words;
  sampled_words.reserve(batch_size);
  size_t stride = vocabulary.size();
  const auto *data = logits.data<float>();
  for (size_t i = 0; i < batch_size; i++) {
    size_t max_index = argmax(data + i * stride, stride);
    sampled_words.push_back(max_index);
  }
  return sampled_words;
}

Words greedy_sample_from_words(const Tensor &logits,
                               const Vocabulary & /*vocabulary*/,
                               const Words &words, size_t batch_size) {
  size_t stride = words.size();
  Words sampled_words;
  sampled_words.reserve(batch_size);
  const auto *data = logits.data<float>();
  for (size_t i = 0; i < batch_size; i++) {
    const float *row = data + i * stride;
    size_t max_index = argmax(row, stride);
    sampled_words.push_back(words[max_index]);
  }
  return sampled_words;
}

}  // namespace slimt
