#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt::qmm {

constexpr float kInt8Maxf = 127.0F;

namespace detail {

enum class Provider {
  Ruy,  //
};

template <enum Provider>
Tensor affine(const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name = "");

template <enum Provider>
Tensor prepare_bias(const Tensor& W, const Tensor& b, float a_quant,
                    float b_quant, const std::string& name = "");

template <enum Provider>
Tensor affine_with_prepared_bias(const Tensor& x, const Tensor& W,
                                 const Tensor& prepared_bias, float a_quant,
                                 float b_quant,
                                 const std::string& name = "");

// affine() whose epilogue is fused: the int32 accumulators are unquantized,
// biased, relu'd and requantized with `out_quant` in one pass, returning the
// i8 tensor a following affine_quantized() consumes directly. The f32
// intermediate is never materialized.
template <enum Provider>
Tensor affine_relu_requantize(const Tensor& x, const Tensor& W,
                              const Tensor& prepared_bias, float a_quant,
                              float b_quant, float out_quant,
                              const std::string& name = "");

// affine() for an input already quantized with `a_quant`.
template <enum Provider>
Tensor affine_quantized(const Tensor& x, const Tensor& W,
                        const Tensor& prepared_bias, float a_quant,
                        float b_quant, const std::string& name = "");

// One multiply of x against a column-concatenation of weight matrices that
// share x's quantized form (their calibrated activation alphas are equal
// because they consume the same tensor). x is quantized once, multiplied
// once, and each segment is dequantized with its own weight multiplier into
// its own output tensor. `W` must have model-lifetime-stable storage: its
// pack is cached in the persistent per-thread context.
template <enum Provider>
std::vector<Tensor> affine_segmented(const Tensor& x, const Tensor& W,
                                     const Tensor& prepared_bias,
                                     float a_quant,
                                     const std::vector<float>& b_quants,
                                     const std::vector<size_t>& segment_cols,
                                     const std::string& name = "");

template <enum Provider>
Tensor select_columns(const Tensor& W, const std::vector<uint32_t>& indices,
                      const std::string& name = "");

template <enum Provider>
Tensor dot(const Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

template <enum Provider>
void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows);
template <enum Provider>
void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols);

// Drops the calling thread's cached packs of heap-owned (standalone) weight
// tensors. Must run before those tensors are freed — the cache is keyed by
// data pointer, and a later allocation reusing the address would hit a stale
// pack.
template <enum Provider>
void clear_standalone_pack_cache();

}  // namespace detail

Tensor affine(const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name = "");

Tensor prepare_bias(const Tensor& W, const Tensor& b, float a_quant,
                    float b_quant, const std::string& name = "");

Tensor affine_with_prepared_bias(const Tensor& x, const Tensor& W,
                                 const Tensor& prepared_bias, float a_quant,
                                 float b_quant,
                                 const std::string& name = "");

Tensor affine_relu_requantize(const Tensor& x, const Tensor& W,
                              const Tensor& prepared_bias, float a_quant,
                              float b_quant, float out_quant,
                              const std::string& name = "");

Tensor affine_quantized(const Tensor& x, const Tensor& W,
                        const Tensor& prepared_bias, float a_quant,
                        float b_quant, const std::string& name = "");

std::vector<Tensor> affine_segmented(const Tensor& x, const Tensor& W,
                                     const Tensor& prepared_bias,
                                     float a_quant,
                                     const std::vector<float>& b_quants,
                                     const std::vector<size_t>& segment_cols,
                                     const std::string& name = "");

Tensor select_columns(const Tensor& W, const std::vector<uint32_t>& indices,
                      const std::string& name = "");

Tensor dot(const Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows);
void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols);

void clear_standalone_pack_cache();

}  // namespace slimt::qmm
