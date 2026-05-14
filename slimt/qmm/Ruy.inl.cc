#include "ruy/ruy.h"

#include <cmath>

#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#endif

namespace slimt::qmm::detail {

using Index = uint64_t;

namespace {
ruy::Context& thread_context() {
  thread_local ruy::Context context;
  return context;
}

ruy::Context& standalone_thread_context() {
  thread_local ruy::Context context;
  return context;
}

#if defined(__aarch64__)
// Ruy's int8 kernel computes an 8-row block regardless of m, so a 1-row
// multiply (the per-request output projection during decode: requests are
// usually single sentences, so each group is one row) wastes ~8× the
// arithmetic and packs both operands first. W is col-major, every column
// contiguous, so a direct sdot gemv needs no packing and accumulates the
// same exact int32 values. Detected at runtime; dotprod is armv8.2 and the
// generic arm64 build can't assume it at compile time.
bool dotprod_available() {
  static const bool available = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
  return available;
}

__attribute__((target("+dotprod"))) void gemv_i8_dotprod(
    const int8_t* x, const int8_t* W, Index width, Index cols, int32_t* out) {
  // Four columns per iteration: one shared x load feeds four independent
  // sdot accumulator chains, otherwise the loop is bound by sdot's latency
  // instead of its throughput.
  Index j = 0;
  for (; j + 4 <= cols; j += 4) {
    const int8_t* c0 = W + (j + 0) * width;
    const int8_t* c1 = W + (j + 1) * width;
    const int8_t* c2 = W + (j + 2) * width;
    const int8_t* c3 = W + (j + 3) * width;
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);
    Index k = 0;
    for (; k + 16 <= width; k += 16) {
      int8x16_t xv = vld1q_s8(x + k);
      acc0 = vdotq_s32(acc0, xv, vld1q_s8(c0 + k));
      acc1 = vdotq_s32(acc1, xv, vld1q_s8(c1 + k));
      acc2 = vdotq_s32(acc2, xv, vld1q_s8(c2 + k));
      acc3 = vdotq_s32(acc3, xv, vld1q_s8(c3 + k));
    }
    int32x4_t sums = vpaddq_s32(vpaddq_s32(acc0, acc1), vpaddq_s32(acc2, acc3));
    for (; k < width; ++k) {
      int32_t lane[4] = {c0[k], c1[k], c2[k], c3[k]};
      sums = vaddq_s32(sums, vmulq_n_s32(vld1q_s32(lane), x[k]));
    }
    vst1q_s32(out + j, sums);
  }
  for (; j < cols; ++j) {
    const int8_t* column = W + j * width;
    int32x4_t acc = vdupq_n_s32(0);
    Index k = 0;
    for (; k + 16 <= width; k += 16) {
      acc = vdotq_s32(acc, vld1q_s8(x + k), vld1q_s8(column + k));
    }
    int32_t sum = vaddvq_s32(acc);
    for (; k < width; ++k) {
      sum += x[k] * column[k];
    }
    out[j] = sum;
  }
}
#endif

// Threshold under which the row-at-a-time gemv beats ruy's padded 8-row
// kernel (8-row block cost is flat in m, gemv cost is linear; crossover
// is below 8 because ruy's kernel is better software-pipelined).
constexpr Index kGemvRowLimit = 4;

bool gemv(const int8_t* A, const int8_t* W, Index rows, Index width,
          Index cols, int32_t* out) {
#if defined(__aarch64__)
  if (rows > kGemvRowLimit || !dotprod_available()) {
    return false;
  }
  for (Index r = 0; r < rows; ++r) {
    gemv_i8_dotprod(A + r * width, W, width, cols, out + r * cols);
  }
  return true;
#else
  (void)A;
  (void)W;
  (void)rows;
  (void)width;
  (void)cols;
  (void)out;
  return false;
#endif
}

void multiply_i8_in(const int8_t* A, const Tensor& W, Index A_rows,
                    Index width, Index B_cols, int32_t* out,
                    ruy::Context& context) {
  if (gemv(A, W.data<int8_t>(), A_rows, width, B_cols, out)) {
    return;
  }

  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(A_rows, width, ruy::Order::kRowMajor,
                        lhs.mutable_layout());
  lhs.set_data(A);

  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, B_cols, ruy::Order::kColMajor,
                        rhs.mutable_layout());
  rhs.set_data(W.data<int8_t>());
  rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(A_rows, B_cols, ruy::Order::kRowMajor,
                        dst.mutable_layout());
  dst.set_data(out);

  // When Dst is int32, mul_params is unused.
  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);
}

void multiply_i8(const int8_t* A, const Tensor& W, Index A_rows, Index width,
                 Index B_cols, int32_t* out) {
  // Ruy's pack cache is keyed by data pointer, so it is only safe while the
  // pointer is stable. Mmap-backed model weights (`standalone() == false`)
  // are stable for the model's lifetime and live in the never-cleared
  // per-thread context. Heap-owned weights (SelectedAffine.W from
  // select_columns) are decode-scoped — without caching, the per-step
  // output projection repacks the same shortlisted W every step (~5% of
  // runtime). Their packs go into a separate context that Model::decode
  // clears before the tensors are freed, so a later allocation reusing the
  // address can't hit a stale pack and the cache can't grow across decodes.
  ruy::Context& context =
      W.standalone() ? standalone_thread_context() : thread_context();
  multiply_i8_in(A, W, A_rows, width, B_cols, out, context);
}
}  // namespace

void quantize(const float* input, float scale, Index rows, Index width,
              int8_t* output) {
  const Index size = rows * width;
  for (size_t i = 0; i < size; i++) {
    // Round half-to-even (nearbyintf, default FE_TONEAREST) to match intgemm's
    // `_mm*_cvtps_epi32`, which marian uses. roundf rounds half-away-from-zero;
    // the ±1 int8 disagreements on tie values shift the alignment argmax at
    // tail cross-attention knife-edges (the off-by-one cascade).
    float value = nearbyintf(scale * input[i]);

    // Since float can store bigger values, we threshold anything that's gone
    // higher and can't fit in int8.
    value = std::max<float>(-kInt8Maxf, value);
    value = std::min<float>(kInt8Maxf, value);

    // Finally a static cast.
    output[i] = static_cast<int8_t>(value);
  };
}

template <class Scalar>
void transpose(const Scalar* input, Index rows, Index cols, Scalar* output) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      output[j * rows + i] = input[i * cols + j];
    }
  }
}

void unquantize(const int32_t* input, float unquant_multiplier, Index rows_A,
                Index cols_B, float* output) {
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_B; j++) {
      Index idx = i * cols_B + j;
      output[idx] = (input[idx] * unquant_multiplier);
    }
  }
}

void unquantizeAddBias(const int32_t* input, const float* input_bias_prepared,
                       float unquant_multiplier, Index rows_A, Index cols_B,
                       float* output) {
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_B; j++) {
      Index idx = i * cols_B + j;
      output[idx] = (input[idx] * unquant_multiplier) + input_bias_prepared[j];
    }
  }
}

void unquantizeAddBiasReluQuantize(const int32_t* input,
                                   const float* input_bias_prepared,
                                   float unquant_multiplier,
                                   float quant_multiplier, Index rows_A,
                                   Index cols_B, int8_t* output) {
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_B; j++) {
      Index idx = i * cols_B + j;
      float value =
          (input[idx] * unquant_multiplier) + input_bias_prepared[j];
      value = value > 0.0F ? value : 0.0F;
      // No lower clamp: the relu already guarantees value >= 0.
      value = std::min(roundf(value * quant_multiplier), kInt8Maxf);
      output[idx] = static_cast<int8_t>(value);
    }
  }
}

// Ruy.
template <>
Tensor affine<Provider::Ruy>(const Tensor& x, const Tensor& W, const Tensor& b,
                             float a_quant, float b_quant,
                             const std::string& name) {
  const Tensor& A = x;  // NOLINT
  const Tensor& B = W;  // NOLINT
  const Tensor& bias = b;

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = B_rows;

  (void)name;
  // Prepare A: Quantize from f32 -> i8
  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT

  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, out_shape, name + "_out");  // NOLINT

  multiply_i8(prepared_A.data<int8_t>(), W, A_rows, width, B_cols,
              AB.data<int32_t>());

  // PrepareBias: ?
  // Actualyl there is no need.
  const Tensor& prepared_bias = bias;

  // Unquantizes, then adds bias in a single statement on the output.
  Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantizeAddBias(AB.data<int32_t>(), prepared_bias.data<float>(),
                            unquant_multiplier, A_rows, B_cols,
                            y.data<float>());
  return y;
}

template <>
Tensor affine_relu_requantize<Provider::Ruy>(const Tensor& x, const Tensor& W,
                                             const Tensor& prepared_bias,
                                             float a_quant, float b_quant,
                                             float out_quant,
                                             const std::string& name) {
  size_t A_cols = x.dim(-1);          // NOLINT
  size_t B_cols = W.dim(-1);          // NOLINT
  size_t A_rows = x.size() / A_cols;  // NOLINT
  size_t width = W.size() / B_cols;

  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT
  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, out_shape, name + "_acc");  // NOLINT
  multiply_i8(prepared_A.data<int8_t>(), W, A_rows, width, B_cols,
              AB.data<int32_t>());

  Tensor y(Type::i8, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantizeAddBiasReluQuantize(
      AB.data<int32_t>(), prepared_bias.data<float>(), unquant_multiplier,
      out_quant, A_rows, B_cols, y.data<int8_t>());
  return y;
}

template <>
std::vector<Tensor> affine_segmented<Provider::Ruy>(
    const Tensor& x, const Tensor& W, const Tensor& prepared_bias,
    float a_quant, const std::vector<float>& b_quants,
    const std::vector<size_t>& segment_cols, const std::string& name) {
  size_t A_cols = x.dim(-1);          // NOLINT
  size_t B_cols = W.dim(-1);          // NOLINT
  size_t A_rows = x.size() / A_cols;  // NOLINT
  size_t width = W.size() / B_cols;

  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT
  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  Shape acc_shape = x.shape();
  acc_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, acc_shape, name + "_acc");  // NOLINT
  // The concatenated weights live in module-owned storage that is stable for
  // the model's lifetime, so the pack belongs in the persistent context.
  multiply_i8_in(prepared_A.data<int8_t>(), W, A_rows, width, B_cols,
                 AB.data<int32_t>(), thread_context());

  std::vector<Tensor> outputs;
  outputs.reserve(segment_cols.size());
  const auto* acc = AB.data<int32_t>();
  const auto* bias = prepared_bias.data<float>();
  size_t offset = 0;
  for (size_t s = 0; s < segment_cols.size(); ++s) {
    size_t cols = segment_cols[s];
    Shape out_shape = x.shape();
    out_shape.set_dim(-1, cols);
    Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
    float unquant_multiplier = 1.0F / (a_quant * b_quants[s]);
    auto* out = y.data<float>();
    for (size_t i = 0; i < A_rows; ++i) {
      const int32_t* row = acc + i * B_cols + offset;
      const float* bias_row = bias + offset;
      for (size_t j = 0; j < cols; ++j) {
        out[i * cols + j] = (row[j] * unquant_multiplier) + bias_row[j];
      }
    }
    outputs.push_back(std::move(y));
    offset += cols;
  }
  return outputs;
}

template <>
Tensor affine_quantized<Provider::Ruy>(const Tensor& x, const Tensor& W,
                                       const Tensor& prepared_bias,
                                       float a_quant, float b_quant,
                                       const std::string& name) {
  size_t A_cols = x.dim(-1);          // NOLINT
  size_t B_cols = W.dim(-1);          // NOLINT
  size_t A_rows = x.size() / A_cols;  // NOLINT
  size_t width = W.size() / B_cols;

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, out_shape, name + "_acc");  // NOLINT
  multiply_i8(x.data<int8_t>(), W, A_rows, width, B_cols, AB.data<int32_t>());

  Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantizeAddBias(AB.data<int32_t>(), prepared_bias.data<float>(),
                            unquant_multiplier, A_rows, B_cols,
                            y.data<float>());
  return y;
}

template <>
Tensor select_columns<Provider::Ruy>(const Tensor& W,
                                     const std::vector<uint32_t>& indices,
                                     const std::string& name) {
  // B is stored col-major as `width × cols`, so each output column lives in a
  // contiguous `width`-byte slab. Memcpy the requested columns into a fresh
  // `width × |indices|` block; the result has the same layout, so any
  // downstream affine call can treat it as a normal weight tensor.
  const Tensor& B = W;                // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT
  size_t width = B_rows;

  Tensor selected_B(Type::i8, Shape({width, indices.size()}),  // NOLINT
                    name.empty() ? "selected_B" : name);

  const auto* B_data = B.data<int8_t>();      // NOLINT
  auto* sB_data = selected_B.data<int8_t>();  // NOLINT
  for (size_t c = 0; c < indices.size(); ++c) {
    std::memcpy(&sB_data[c * width], &B_data[indices[c] * width], width);
  }
  return selected_B;
}

template <>
Tensor dot<Provider::Ruy>(const Tensor& x, const Tensor& W, float a_quant,
                          float b_quant, const std::string& name) {
  const Tensor& A = x;  // NOLINT
  const Tensor& B = W;  // NOLINT

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = B_rows;

  (void)name;
  // Prepare A: Quantize from f32 -> i8
  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT

  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, out_shape, name + "_out");  // NOLINT

  multiply_i8(prepared_A.data<int8_t>(), W, A_rows, width, B_cols,
              AB.data<int32_t>());

  // Unquantizes, then adds bias in a single statement on the output.
  Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantize(AB.data<int32_t>(), unquant_multiplier, A_rows, B_cols,
                     y.data<float>());
  return y;
}

template <>
void prepare_weight_transposed<Provider::Ruy>(const float* weights,
                                              int8_t* prepared,
                                              float quantization_multiplier,
                                              size_t cols, size_t rows) {
  detail::quantize(weights, quantization_multiplier, cols, rows, prepared);
}

template <>
void prepare_weight_quantized_transposed<Provider::Ruy>(const int8_t* input,
                                                        int8_t* output,
                                                        size_t rows,
                                                        size_t cols) {
  std::memcpy(output, input,
              /*count=*/sizeof(int8_t) * (rows * cols));
}

// The legacy "prepared bias" optimization folded the
// shift-quantization compensation (column sums of W * a_quant_shift) into
// the bias once so subsequent affine calls can skip the per-call
// compensation pass. Ruy uses signed-signed int8 multiplication and applies
// the bias as a post-process, so no preparation is needed — pass the bias
// through unchanged and route the "with prepared bias" entry point to the
// regular one. Without these stubs, Modules.cc refers to undefined
// `prepare_bias<Ruy>` / `affine_with_prepared_bias<Ruy>`, producing a
// libslimt.a that silently fails to dlopen on ARM.
template <>
Tensor prepare_bias<Provider::Ruy>(const Tensor& W, const Tensor& b,
                                   float a_quant, float b_quant,
                                   const std::string& name) {
  (void)W;
  (void)a_quant;
  (void)b_quant;
  return b.clone(name.empty() ? "prepared_bias" : name);
}

template <>
Tensor affine_with_prepared_bias<Provider::Ruy>(const Tensor& x,
                                                const Tensor& W,
                                                const Tensor& prepared_bias,
                                                float a_quant, float b_quant,
                                                const std::string& name) {
  return affine<Provider::Ruy>(x, W, prepared_bias, a_quant, b_quant, name);
}

template <>
void clear_standalone_pack_cache<Provider::Ruy>() {
  standalone_thread_context().ClearPrepackedCache();
}
}  // namespace slimt::qmm::detail
