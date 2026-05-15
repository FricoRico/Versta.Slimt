#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt {
using ParameterMap = std::unordered_map<std::string, Tensor *>;

struct Affine {
  Tensor W, b;  // NOLINT
  Tensor quant;
  mutable Tensor prepared_bias;
  mutable bool prepared_bias_ready = false;
  mutable float prepared_bias_a_quant = 0.0F;
  mutable float prepared_bias_b_quant = 0.0F;
};

// Snapshot of an Affine restricted to a fixed set of output columns
// (a "shortlist"). Built once per decode in Model::decode and reused across
// every decoder step, so the per-step output projection skips both the
// column-selection (via `qmm::select_columns`) and the bias gather that
// would otherwise run inside every per-step call.
struct SelectedAffine {
  Tensor W;
  Tensor prepared_bias;
  float a_quant = 0.0F;
  float b_quant = 0.0F;
};

struct Linear {
  Tensor W;  // NOLINT
  Tensor quant;
};

// Column-concatenation of weight matrices that consume the same input
// tensor. Marian's calibration derives a GEMM's activation alpha from its
// input activations, so Affines sharing an input share an alpha (this holds
// exactly across the shipped models, verified for enes/enja/jaen), and one
// quantize + one multiply serves all of them. `W` views the module-owned
// `storage`, so its data pointer is stable for the model's lifetime and qmm
// caches its pack persistently. Built in prepare_biases(); `valid` stays
// false when alphas are missing or disagree and callers fall back to
// per-Affine multiplies.
struct ConcatAffine {
  Aligned storage;
  Tensor W;
  Tensor prepared_bias;
  float a_quant = 0.0F;
  std::vector<float> b_quants;
  std::vector<size_t> segment_cols;
  bool valid = false;
};

struct AttentionContext {
  Tensor keys;
  Tensor values;
};

class LayerNorm {
 public:
  explicit LayerNorm() = default;
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  Tensor forward(const Tensor &x) const;
  Tensor forward_add(const Tensor &a, const Tensor &b) const;  // layer_norm(a+b)

 private:
  Tensor bias_;
  Tensor scale_;
};

class Attention {
 public:
  explicit Attention(std::string name, size_t num_heads);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void prepare_biases();
  AttentionContext prepare_context(const Tensor &k, const Tensor &v) const;
  std::tuple<Tensor, Tensor> forward(const Tensor &q, const Tensor &k,
                                     const Tensor &v, const Tensor &mask) const;
  std::tuple<Tensor, Tensor> forward(const Tensor &q,
                                     const AttentionContext &context,
                                     const Tensor &mask) const;

 private:
  std::tuple<Tensor, Tensor> forward_split(const Tensor &q, Tensor split_yq,
                                           const AttentionContext &context,
                                           const Tensor &mask) const;

  std::string name_;
  Affine Q_, K_, V_, O_;
  // Self-attention consumes one tensor for q/k/v, so all three fuse. For
  // cross-attention Q's input (decoder state) differs from K/V's
  // (encoder_out), so only the K|V pair shares an alpha; kv_ is built when
  // the three-way fusion is rejected.
  ConcatAffine qkv_;
  ConcatAffine kv_;
  LayerNorm ln_;
  size_t num_heads_;
};

class SSRU {
 public:
  explicit SSRU() = default;
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void prepare_biases();
  Tensor forward(Tensor &state, const Tensor &x) const;
  Tensor start_state(size_t batch_size) const;

 private:
  Affine F_;
  Linear O_;
  ConcatAffine fo_;
  LayerNorm ln_;
};

class FFN {
 public:
  explicit FFN(size_t depth);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void prepare_biases();
  Tensor forward(const Tensor &x) const;

  // second(relu(first(x))) with the intermediate kept quantized: first's
  // epilogue applies bias, relu and second's input quantization in one pass
  // over the int32 accumulators, so the widest tensor of the forward pass
  // ([T × ffn_dim]) never round-trips through f32. Falls back to the
  // three-pass form when either GEMM lacks a calibrated activation alpha.
  static Tensor forward_chain(const FFN &first, const FFN &second,
                              const Tensor &x);

 private:
  Affine O_;
  size_t depth_;
};

class EncoderLayer {
 public:
  EncoderLayer(size_t depth, size_t ffn_count, size_t num_heads);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void prepare_biases();
  std::tuple<Tensor, Tensor> forward(const Tensor &x, const Tensor &mask) const;

 private:
  size_t depth_;
  Attention attention_;
  std::vector<FFN> ffn_;
  LayerNorm ffn_ffn_;
};

class DecoderLayer {
 public:
  explicit DecoderLayer(size_t depth, size_t ffn_count, size_t num_heads);
  void register_parameters(const std::string &prefix, ParameterMap &parameters);
  void prepare_biases();
  AttentionContext prepare_context(const Tensor &encoder_out) const;
  std::tuple<Tensor, Tensor> forward(const Tensor &encoder_out,
                                     const Tensor &mask, Tensor &state,
                                     const Tensor &x) const;
  std::tuple<Tensor, Tensor> forward(const AttentionContext &context,
                                     const Tensor &mask, Tensor &state,
                                     const Tensor &x) const;
  Tensor start_state(size_t batch_size) const {
    return rnn_.start_state(batch_size);
  }

 private:
  size_t depth_;
  Attention attention_;
  SSRU rnn_;
  std::vector<FFN> ffn_;
  LayerNorm ffn_ffn_;
};

SelectedAffine prepare_selected(const Affine &parameters,
                                const std::vector<uint32_t> &indices);

Tensor affine_with_selected(const SelectedAffine &parameters, const Tensor &x,
                            const std::string &name = "");

Tensor affine(const Affine &parameters, const Tensor &x,
              const std::string &name = "");

void prepare_bias(Affine &parameters);

}  // namespace slimt
