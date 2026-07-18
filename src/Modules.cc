#include "Modules.hh"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "QMM.hh"
#include "Tensor.hh"
#include "TensorOps.hh"

namespace leanmt {

float retrieve_quantization_multiplier(const Tensor &W) {
  const auto *b_end = W.end<int8_t>();
  float b_quant = *(reinterpret_cast<const float *>(b_end));
  return b_quant;
}

const Tensor &retrieve_prepared_bias(const Affine &parameters) {
  assert(parameters.prepared_bias_ready);
  return parameters.prepared_bias;
}

namespace {
// Activation alpha for the int8 GEMM's quantize() step.
//
// Ruy's i8 quantize clips activations to ±127. The bergamot calibration
// pipeline normally bakes a per-tensor alpha into the model file
// (`*_QuantMultA`) so that `alpha * max|x| ≤ 127`. Two-vocab models
// (en-ja, en-ko, en-zh, ...) ship no such alpha for the decoder output
// projection — `decoder_Wemb_QuantMultA` is an empty placeholder and
// `none_QuantMultA` is absent. Io.cc drops the placeholder name on load
// so `Affine::quant` for those models comes back unloaded; we detect
// that here and compute alpha dynamically from `max|x|`. Matches
// marian-decoder's runtime behavior on the same files.
float a_quant_for(const Affine &parameters, const Tensor &x) {
  if (parameters.quant.loaded()) {
    return parameters.quant.item<float>();
  }
  const float *data = x.data<float>();
  size_t n = x.size();
  float xmax = 0.0F;
  for (size_t i = 0; i < n; ++i) {
    float v = std::fabs(data[i]);
    if (v > xmax) xmax = v;
  }
  if (xmax == 0.0F) return qmm::kInt8Maxf;  // arbitrary, tensor is all zeros
  return qmm::kInt8Maxf / xmax;
}

// Same as a_quant_for but for the SelectedAffine path, where the static
// value (if any) was captured into a float at decode start.
float dynamic_a_quant(const Tensor &x) {
  const float *data = x.data<float>();
  size_t n = x.size();
  float xmax = 0.0F;
  for (size_t i = 0; i < n; ++i) {
    float v = std::fabs(data[i]);
    if (v > xmax) xmax = v;
  }
  if (xmax == 0.0F) return qmm::kInt8Maxf;
  return qmm::kInt8Maxf / xmax;
}
}  // namespace

void prepare_bias(Affine &parameters) {
  // For Ruy `qmm::prepare_bias` is a `b.clone()` that ignores both quant
  // values, so it's safe to run even when `quant` is unloaded. The stored
  // `prepared_bias_a_quant` is informational only and never consulted on
  // the Ruy path; leave it 0 in that case.
  float a_quant =
      parameters.quant.loaded() ? parameters.quant.item<float>() : 0.0F;
  float b_quant = retrieve_quantization_multiplier(parameters.W);
  parameters.prepared_bias = qmm::prepare_bias(  //
      parameters.W, parameters.b,                //
      a_quant, b_quant,                          //
      "prepared_bias"                            //
  );
  parameters.prepared_bias_a_quant = a_quant;
  parameters.prepared_bias_b_quant = b_quant;
  parameters.prepared_bias_ready = true;
}

namespace {

struct ConcatSource {
  const Tensor *W;
  const Tensor *quant;
  const Tensor *prepared_bias;  // nullptr: bias-less (linear) segment
};

ConcatAffine concat_affines(const std::vector<ConcatSource> &sources) {
  ConcatAffine fused;
  for (const ConcatSource &source : sources) {
    if (!source.W->loaded() || !source.quant->loaded()) {
      return fused;
    }
    if (source.quant->item<float>() != sources.front().quant->item<float>()) {
      return fused;
    }
  }

  size_t width = sources.front().W->size() / sources.front().W->dim(-1);
  size_t total_cols = 0;
  for (const ConcatSource &source : sources) {
    if (source.W->size() / source.W->dim(-1) != width) {
      return fused;
    }
    total_cols += source.W->dim(-1);
  }

  Shape shape({width, total_cols});
  fused.storage = Tensor::allocate(Type::i8, shape);
  fused.W.load(View{fused.storage.data(), width * total_cols}, Type::i8,
               shape, "concat_W");
  fused.prepared_bias =
      Tensor(Type::f32, Shape({total_cols}), "concat_prepared_bias");

  auto *w_out = fused.W.data<int8_t>();
  auto *bias_out = fused.prepared_bias.data<float>();
  for (const ConcatSource &source : sources) {
    size_t cols = source.W->dim(-1);
    std::memcpy(w_out, source.W->data<int8_t>(), width * cols);
    w_out += width * cols;
    if (source.prepared_bias != nullptr) {
      std::memcpy(bias_out, source.prepared_bias->data<float>(),
                  cols * sizeof(float));
    } else {
      std::fill(bias_out, bias_out + cols, 0.0F);
    }
    bias_out += cols;
    fused.b_quants.push_back(retrieve_quantization_multiplier(*source.W));
    fused.segment_cols.push_back(cols);
  }
  fused.a_quant = sources.front().quant->item<float>();
  fused.valid = true;
  return fused;
}

// Multiply x against the segments [first, first+count): col-major storage
// makes any segment range a contiguous sub-block, viewed without copying.
std::vector<Tensor> concat_forward(const ConcatAffine &fused, const Tensor &x,
                                   size_t first, size_t count,
                                   const std::string &name) {
  size_t width = fused.W.dim(0);
  size_t offset = 0;
  for (size_t s = 0; s < first; ++s) {
    offset += fused.segment_cols[s];
  }
  size_t cols = 0;
  for (size_t s = first; s < first + count; ++s) {
    cols += fused.segment_cols[s];
  }

  Tensor W;
  W.load(View{const_cast<int8_t *>(fused.W.data<int8_t>()) + offset * width,
              width * cols},
         Type::i8, Shape({width, cols}), "concat_W");
  Tensor prepared_bias;
  prepared_bias.load(
      View{const_cast<float *>(fused.prepared_bias.data<float>() + offset),
           cols * sizeof(float)},
      Type::f32, Shape({cols}), "concat_prepared_bias");

  std::vector<float> b_quants(fused.b_quants.begin() + first,
                              fused.b_quants.begin() + first + count);
  std::vector<size_t> segment_cols(fused.segment_cols.begin() + first,
                                   fused.segment_cols.begin() + first + count);
  return qmm::affine_segmented(x, W, prepared_bias, fused.a_quant, b_quants,
                               segment_cols, name);
}

}  // namespace

std::tuple<Tensor, Tensor> scaled_dot_product_attention(const Tensor &q,
                                                        const Tensor &k,
                                                        const Tensor &v,
                                                        const Tensor &mask) {
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/models/transformer.h#L228

  // attn = softmax((q . k^T)/d_k) . v
  size_t batch_size = q.dim(-4);
  size_t num_heads = q.dim(-3);
  size_t query_length = q.dim(-2);
  size_t dim_head = q.dim(-1);

  size_t value_length = v.dim(-2);

  // Compute QKT
  Shape shape({batch_size, num_heads, query_length, value_length});
  Tensor qkt(q.type(), shape, "qkt");

  // scaling to avoid extreme values due to matrix multiplication
  float d_k = 1.0F / std::sqrt(dim_head);
  size_t reinterpreted_batch_size = batch_size * num_heads;
  batch_matrix_multiply(                               //
      q.data<float>(), k.data<float>(),                //
      reinterpreted_batch_size,                        //
      query_length, dim_head, value_length, dim_head,  //
      /*trans_a=*/false, /*trans_b=*/true,             //
      /*alpha=*/d_k,                                   //
      qkt.data<float>()                                //
  );

  // softmax (QKT/d_k), restricted to each batch row's valid prefix. The
  // mask is additive (0 for tokens, ~-1e8 for the padding suffix), so the
  // padded attention weights come out exactly 0; softmax over the prefix
  // plus a zero-filled tail computes the same values without materializing
  // a mask-add pass over the full [B, heads, Tq, Tk] score tensor.
  Tensor attn(v.type(), qkt.shape(), "sdpa_attn");
  float *qkt_data = qkt.data<float>();
  const float *mask_data = mask.data<float>();
  float *attn_data = attn.data<float>();
  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    const float *mask_row = mask_data + batch_id * value_length;
    size_t valid = value_length;
    while (valid > 0 && mask_row[valid - 1] != 0.0F) {
      valid--;
    }
    size_t rows = num_heads * query_length;
    if (valid == value_length) {
      // Length-bucketed batches are mostly unpadded; one batched call.
      size_t offset = batch_id * rows * value_length;
      softmax(qkt_data + offset, rows, value_length, attn_data + offset);
      continue;
    }
    for (size_t r = 0; r < rows; ++r) {
      size_t offset = (batch_id * rows + r) * value_length;
      softmax(qkt_data + offset, 1, valid, attn_data + offset);
      std::fill(attn_data + offset + valid, attn_data + offset + value_length,
                0.0F);
    }
  }

  // softmax (QKT/d_k) * V
  Tensor out(q.type(), q.shape(), "sdpa_out");
  batch_matrix_multiply(                                   //
      attn.data<float>(), v.data<float>(),                 //
      reinterpreted_batch_size,                            //
      query_length, value_length, value_length, dim_head,  //
      /*trans_a=*/false, /*trans_b=*/false,                //
      /*alpha =*/1.0,                                      //
      out.data<float>()                                    //
  );

  return std::make_tuple(std::move(out), std::move(attn));
}

Tensor split_heads(const Tensor &x, size_t num_heads) {
  size_t batch_size = x.dim(-3);
  size_t sequence_length = x.dim(-2);
  size_t feature_dim = x.dim(-1);
  size_t dim_head = feature_dim / num_heads;

  // Currently          [B x T x num_heads * dim_head]
  // Need to become     [num_heads, B, T, dim_head]
  //
  // So that T x T attention matrices are formed.
  //
  // This requires a reshaping. i.e, two axes has to be permuted.
  //
  // First, perceive input matrix as:
  //    [B x T  x num_heads x dim_head]
  //
  // Given the layout, this is easy, just a view will do, so add a 1 dimension.
  //
  // What remains is permute(1, 2), which transposes the T and num_heads
  // dimensions respectively. This will require changes to the underlying
  // storage.
  //
  // In marian, this is achieved by TransposeND
  // https://github.com/browsermt/marian-dev/blob/14c9d9b0e732f42674e41ee138571d5a7bf7ad94/src/tensors/cpu/tensor_operators.cpp#L370
  assert(feature_dim % num_heads == 0);

  Shape shape({batch_size, sequence_length, num_heads, dim_head});
  // LEANMT_TRACE(x.shape());
  // LEANMT_TRACE(shape);

  Tensor y(x.type(), shape.transpose(-3, -2), x.name());

  transpose_3120(x.data<float>(), batch_size, sequence_length, num_heads,
                 dim_head, y.data<float>());

  // LEANMT_TRACE(y.shape());

  return y;
}

Tensor join_heads(const Tensor &x) {
  size_t batch_size = x.dim(-4);
  size_t num_heads = x.dim(-3);
  size_t sequence_length = x.dim(-2);
  size_t dim_depth = x.dim(-1);

  size_t dim_model = num_heads * dim_depth;
  Shape shape({batch_size, sequence_length, dim_model});
  Tensor y(x.type(), shape, "concat");

  // B x N x T x H -> B x T x N x H
  transpose_3120(x.data<float>(), batch_size, num_heads, sequence_length,
                 dim_depth, y.data<float>());

  return y;
}

Tensor affine(const Affine &parameters, const Tensor &x,
              const std::string &name /* = ""*/) {
  Tensor local_prepared_bias;
  const Tensor *prepared_bias = nullptr;
  if (parameters.prepared_bias_ready) {
    prepared_bias = &retrieve_prepared_bias(parameters);
  } else {
    float a_quant_for_prep =
        parameters.quant.loaded() ? parameters.quant.item<float>() : 0.0F;
    local_prepared_bias = qmm::prepare_bias(             //
        parameters.W, parameters.b,                      //
        a_quant_for_prep,                                //
        retrieve_quantization_multiplier(parameters.W),  //
        "prepared_bias"                                  //
    );
    prepared_bias = &local_prepared_bias;
  }

  Tensor y = qmm::affine_with_prepared_bias(           //
      x,                                               //
      parameters.W,                                    //
      *prepared_bias,                                  //
      a_quant_for(parameters, x),                      //
      retrieve_quantization_multiplier(parameters.W),  //
      name                                             //
  );
  return y;
}

SelectedAffine prepare_selected(const Affine &parameters,
                                const std::vector<uint32_t> &indices) {
  // Caller is expected to have run prepare_bias on the Affine at load time —
  // shortlisting is only worthwhile when we're amortizing this across many
  // decoder steps, and that path always goes through prepare_biases() first.
  assert(parameters.prepared_bias_ready);

  Tensor selected_W = qmm::select_columns(parameters.W, indices, "selected_W");

  Tensor selected_prepared_bias(Type::f32, Shape({indices.size()}),
                                "selected_prepared_bias");
  const float *src = parameters.prepared_bias.data<float>();
  float *dst = selected_prepared_bias.data<float>();
  for (size_t i = 0; i < indices.size(); ++i) {
    dst[i] = src[indices[i]];
  }

  // Capture 0 as the "no calibrated alpha, use dynamic" sentinel; the file
  // ships an empty `decoder_Wemb_QuantMultA` placeholder for two-vocab
  // models, which Io.cc drops on load (see Tensor::loaded). We can't read
  // a calibrated value here, but we don't have an `x` to compute dynamic
  // either — defer that to `affine_with_selected`.
  float a_quant =
      parameters.quant.loaded() ? parameters.quant.item<float>() : 0.0F;

  return SelectedAffine{
      .W = std::move(selected_W),
      .prepared_bias = std::move(selected_prepared_bias),
      .a_quant = a_quant,
      .b_quant = retrieve_quantization_multiplier(parameters.W),
  };
}

Tensor affine_with_selected(const SelectedAffine &parameters, const Tensor &x,
                            const std::string &name /*= ""*/) {
  // a_quant == 0 is the "dynamic" sentinel set by `prepare_selected` when
  // the underlying Affine had no calibrated alpha.
  float a_quant =
      parameters.a_quant > 0.0F ? parameters.a_quant : dynamic_a_quant(x);
  return qmm::affine_with_prepared_bias(x, parameters.W, parameters.prepared_bias,
                                        a_quant, parameters.b_quant,
                                        name);
}

Tensor linear(const Linear &parameters, const Tensor &x,
              const std::string &name = "") {
  Tensor y = qmm::dot(                                 //
      x, parameters.W,                                 //
      parameters.quant.item<float>(),                  //
      retrieve_quantization_multiplier(parameters.W),  //
      name                                             //
  );
  return y;
}

Tensor SSRU::start_state(size_t batch_size) const {
  // auto start = graph->constant({1, 1, dimBatch, dim}, inits::zeros());
  size_t feature_dim = O_.W.dim(-1);
  Tensor start(Type::f32, Shape({batch_size, feature_dim}), "start");
  start.fill_in_place(0.0F);
  return start;
}

Tensor SSRU::forward(Tensor &state, const Tensor &x) const {
  // From Research to Production and Back: Ludicrously Fast Neural Machine
  // Translation (https://aclanthology.org/D19-5632.pdf) Section 3.1 describes
  // SSRU. SSRU is described by the following recurrent equations - which
  // is formulated using output (y), forget-gate (f), cell-states (c). for a
  // given input (x).
  //
  // f(t) = σ(Wt . x(t) + bf )
  // c(t) = f(t) ⊙  c(t−1) + (1 − ft) ⊙  Wx(t)
  // y(t) = ReLU(c(t));
  // h(t) = α LayerNorm( y(t) + x(t)) + β
  //
  // ⊙  indicates elementwise-multiplication.
  //
  // The notion of adding forget-gates dependent on the input to do alpha x +
  // beta y allowing the network to learn skip connections are described in
  // Highway Networks (https://arxiv.org/pdf/1505.00387.pdf)
  //
  // The term highway appears here because marian uses it in a similar capacity.
  // https://github.com/browsermt/marian-dev/blob/0f4196c767afd1070fbb20eb348a5777d0376283/src/tensors/cpu/tensor_operators.cpp#L1593
  //
  //       Wx(t)  is a linear operation (it's a linear transform).
  // Wfx(t) + bf  is an affine transform.

  Tensor &c = state;  // Load context from saved-state.

  // Forward parameter multiplications.
  Tensor f;    // Forget gate? NOLINT
  Tensor Wxt;  // NOLINT
  if (fo_.valid) {
    std::vector<Tensor> fo = concat_forward(fo_, x, /*first=*/0,
                                            /*count=*/2, "rnn");
    f = std::move(fo[0]);
    Wxt = std::move(fo[1]);
  } else {
    f = affine(F_, x, "rnn_f");
    Wxt = linear(O_, x, "rnn_o");
  }

  // https://github.com/browsermt/marian-dev/blob/77e886ae7ae6016981c6307c312650bf74b50487/src/rnn/cells.h#L1058
  // c(t) = f(t) ⊙  c(t−1) + (1 − ft) ⊙  Wx(t)
  // Tensor c_t = highway(c, f, Wxt);
  Tensor c_t = highway(c, Wxt, f);

  // https://github.com/browsermt/marian-dev/blob/77e886ae7ae6016981c6307c312650bf74b50487/src/rnn/cells.h#L1059
  // y(t) = ReLU(c(t));
  Tensor y = relu(c_t);

  // h(t) = α LayerNorm(y(t) + x(t)) + β
  Tensor h = ln_.forward_add(x, y);

  // The recurrent state outlives the current decoder step; copy c_t's data
  // into state's existing buffer instead of move-assigning a new one (which
  // would adopt c_t's possibly-arena-backed memory).
  std::copy(c_t.data<float>(), c_t.data<float>() + state.size(),
            state.data<float>());

  return h;
}

std::tuple<Tensor, Tensor> DecoderLayer::forward(const Tensor &encoder_out,
                                                 const Tensor &mask,
                                                 Tensor &state,
                                                 const Tensor &x) const {
  AttentionContext context = prepare_context(encoder_out);
  return forward(context, mask, state, x);
}

AttentionContext DecoderLayer::prepare_context(
    const Tensor &encoder_out) const {
  return attention_.prepare_context(encoder_out, encoder_out);
}

std::tuple<Tensor, Tensor> DecoderLayer::forward(
    const AttentionContext &context, const Tensor &mask, Tensor &state,
    const Tensor &x) const {
  Tensor decoder_out = rnn_.forward(state, x);

  auto [out, attn] = attention_.forward(decoder_out, context, mask);

  Tensor ffn2_out = FFN::forward_chain(ffn_[0], ffn_[1], out);
  // Post Norm (fused residual add)
  Tensor normalized_ffn_out = ffn_ffn_.forward_add(ffn2_out, out);
  return std::make_tuple(std::move(normalized_ffn_out), std::move(attn));
}

EncoderLayer::EncoderLayer(size_t depth, size_t ffn_count, size_t num_heads)
    : depth_(depth), attention_("self", num_heads) {
  for (size_t i = 0; i < ffn_count; i++) {
    ffn_.emplace_back(i + 1);
  }
}

DecoderLayer::DecoderLayer(size_t depth, size_t ffn_count, size_t num_heads)
    : depth_(depth), attention_("context", num_heads) {
  for (size_t i = 0; i < ffn_count; i++) {
    ffn_.emplace_back(i + 1);
  }
}

FFN::FFN(size_t depth) : depth_(depth) {}

void FFN::prepare_biases() { prepare_bias(O_); }

Tensor FFN::forward(const Tensor &x) const {
  Tensor y = affine(O_, x, "ffn" + std::to_string(depth_));
  return y;
}

Tensor FFN::forward_chain(const FFN &first, const FFN &second,
                          const Tensor &x) {
  const Affine &f1 = first.O_;
  const Affine &f2 = second.O_;
  if (!f1.quant.loaded() || !f2.quant.loaded()) {
    return second.forward(relu(first.forward(x)));
  }

  assert(f1.prepared_bias_ready && f2.prepared_bias_ready);
  float a2_quant = f2.quant.item<float>();
  Tensor intermediate = qmm::affine_relu_requantize(
      x, f1.W, f1.prepared_bias, f1.quant.item<float>(),
      retrieve_quantization_multiplier(f1.W), a2_quant,
      "ffn" + std::to_string(first.depth_));
  return qmm::affine_quantized(intermediate, f2.W, f2.prepared_bias, a2_quant,
                               retrieve_quantization_multiplier(f2.W),
                               "ffn" + std::to_string(second.depth_));
}

void Attention::prepare_biases() {
  prepare_bias(Q_);
  prepare_bias(K_);
  prepare_bias(V_);
  prepare_bias(O_);
  qkv_ = concat_affines({{&Q_.W, &Q_.quant, &Q_.prepared_bias},
                         {&K_.W, &K_.quant, &K_.prepared_bias},
                         {&V_.W, &V_.quant, &V_.prepared_bias}});
  if (!qkv_.valid) {
    kv_ = concat_affines({{&K_.W, &K_.quant, &K_.prepared_bias},
                          {&V_.W, &V_.quant, &V_.prepared_bias}});
  }
}

void SSRU::prepare_biases() {
  prepare_bias(F_);
  fo_ = concat_affines({{&F_.W, &F_.quant, &F_.prepared_bias},
                        {&O_.W, &O_.quant, nullptr}});
}

void EncoderLayer::prepare_biases() {
  attention_.prepare_biases();
  for (FFN &ffn : ffn_) {
    ffn.prepare_biases();
  }
}

void DecoderLayer::prepare_biases() {
  attention_.prepare_biases();
  rnn_.prepare_biases();
  for (FFN &ffn : ffn_) {
    ffn.prepare_biases();
  }
}

Tensor LayerNorm::forward(const Tensor &x) const {
  Tensor y = layer_norm(x, scale_, bias_);
  return y;
}

Tensor LayerNorm::forward_add(const Tensor &a, const Tensor &b) const {
  return layer_norm_add(a, b, scale_, bias_);
}

std::tuple<Tensor, Tensor> Attention::forward(const Tensor &q, const Tensor &k,
                                              const Tensor &v,
                                              const Tensor &mask) const {
  if (qkv_.valid && q.data<float>() == k.data<float>() &&
      k.data<float>() == v.data<float>()) {
    std::vector<Tensor> qkv = concat_forward(qkv_, q, /*first=*/0,
                                             /*count=*/3, "qkv");
    AttentionContext context{
        .keys = split_heads(qkv[1], num_heads_),
        .values = split_heads(qkv[2], num_heads_),
    };
    return forward_split(q, split_heads(qkv[0], num_heads_), context, mask);
  }

  AttentionContext context = prepare_context(k, v);
  return forward(q, context, mask);
}

AttentionContext Attention::prepare_context(const Tensor &k,
                                            const Tensor &v) const {
  // We have a B x T x H sequence coming in, for q, k and v.
  if (k.data<float>() == v.data<float>()) {
    if (qkv_.valid) {
      std::vector<Tensor> kv = concat_forward(qkv_, k, /*first=*/1,
                                              /*count=*/2, "kv");
      return {
          .keys = split_heads(kv[0], num_heads_),
          .values = split_heads(kv[1], num_heads_),
      };
    }
    if (kv_.valid) {
      std::vector<Tensor> kv = concat_forward(kv_, k, /*first=*/0,
                                              /*count=*/2, "kv");
      return {
          .keys = split_heads(kv[0], num_heads_),
          .values = split_heads(kv[1], num_heads_),
      };
    }
  }

  Tensor yk = affine(K_, k, "k");
  Tensor yv = affine(V_, v, "v");

  return {
      .keys = split_heads(yk, num_heads_),
      .values = split_heads(yv, num_heads_),
  };
}

std::tuple<Tensor, Tensor> Attention::forward(
    const Tensor &q, const AttentionContext &context,
    const Tensor &mask) const {
  Tensor yq = affine(Q_, q, "q");
  return forward_split(q, split_heads(yq, num_heads_), context, mask);
}

std::tuple<Tensor, Tensor> Attention::forward_split(
    const Tensor &q, Tensor split_yq, const AttentionContext &context,
    const Tensor &mask) const {
  // Apply individual scaled-dot-product-attention (SDPA)
  auto [attn_out, attn] = scaled_dot_product_attention(
      split_yq, context.keys, context.values, mask);

  // Join heads.
  Tensor out = join_heads(attn_out);

  // Project to output size.
  Tensor yo = affine(O_, out, "o");

  // Add and norm (fused residual add)
  const Tensor &x = q;
  Tensor y = ln_.forward_add(x, yo);

  return std::make_tuple(std::move(y), std::move(attn));
}

std::tuple<Tensor, Tensor> EncoderLayer::forward(const Tensor &x,
                                                 const Tensor &mask) const {
  // TODO(fill code):
  auto [out, attention] = attention_.forward(x, x, x, mask);

  Tensor ffn2_out = FFN::forward_chain(ffn_[0], ffn_[1], out);
  // Post Norm (fused residual add)
  Tensor normalized_ffn_out = ffn_ffn_.forward_add(ffn2_out, out);

  return std::make_tuple(std::move(normalized_ffn_out), std::move(attention));
}

void EncoderLayer::register_parameters(const std::string &prefix,
                                       ParameterMap &parameters) {
  std::string encoder_prefix = prefix + "encoder_l" + std::to_string(depth_);
  attention_.register_parameters(encoder_prefix, parameters);
  for (FFN &ffn : ffn_) {
    ffn.register_parameters(encoder_prefix, parameters);
  }

  ffn_ffn_.register_parameters(encoder_prefix + "_ffn_ffn", parameters);
}

void DecoderLayer::register_parameters(const std::string &prefix,
                                       ParameterMap &parameters) {
  std::string decoder_prefix = prefix + "decoder_l" + std::to_string(depth_);
  attention_.register_parameters(decoder_prefix, parameters);
  for (FFN &ffn : ffn_) {
    ffn.register_parameters(decoder_prefix, parameters);
  }
  rnn_.register_parameters(decoder_prefix, parameters);
  ffn_ffn_.register_parameters(decoder_prefix + "_ffn_ffn", parameters);
}

Attention::Attention(std::string name, size_t num_heads)
    : name_(std::move(name)), num_heads_(num_heads) {}

void Attention::register_parameters(const std::string &prefix,
                                    ParameterMap &parameters) {
  auto register_affine = [&](const std::string &suffix, Affine &affine) {
    std::string local_prefix = prefix + ("_" + name_ + "_");
    parameters.emplace(local_prefix + "W" + suffix, &affine.W);
    parameters.emplace(local_prefix + "b" + suffix, &affine.b);
    parameters.emplace(local_prefix + "W" + suffix + "_QuantMultA",
                       &affine.quant);
  };

  register_affine("q", Q_);
  register_affine("k", K_);
  register_affine("v", V_);
  register_affine("o", O_);

  std::string wo_prefix = prefix + ("_" + name_ + "_") + "Wo";
  ln_.register_parameters(wo_prefix, parameters);
}

void FFN::register_parameters(const std::string &prefix,
                              ParameterMap &parameters) {
  // std::string param_name = prefix + "_ffn_W" + std::to_string(depth_);
  parameters.emplace(prefix + "_ffn_W" + std::to_string(depth_), &O_.W);
  parameters.emplace(prefix + "_ffn_b" + std::to_string(depth_), &O_.b);
  parameters.emplace(prefix + "_ffn_W" + std::to_string(depth_) + "_QuantMultA",
                     &O_.quant);
}

void SSRU::register_parameters(const std::string &prefix,
                               ParameterMap &parameters) {
  const std::string local_prefix = prefix + "_rnn_";
  parameters.emplace(local_prefix + "W", &O_.W);
  parameters.emplace(local_prefix + "W_QuantMultA", &O_.quant);

  parameters.emplace(local_prefix + "Wf", &F_.W);
  parameters.emplace(local_prefix + "bf", &F_.b);
  parameters.emplace(local_prefix + "Wf_QuantMultA", &F_.quant);

  ln_.register_parameters(local_prefix + "ffn", parameters);
}

void LayerNorm::register_parameters(const std::string &prefix,
                                    ParameterMap &parameters) {
  parameters.emplace(prefix + "_ln_bias", &bias_);
  parameters.emplace(prefix + "_ln_scale", &scale_);
}

}  // namespace leanmt
