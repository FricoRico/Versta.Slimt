#include "slimt/QMM.hh"

namespace slimt::qmm::detail {
constexpr Provider kProvider = Provider::Ruy;
}
// NOLINTNEXTLINE: The C++ file inclusion is intended.
#include "slimt/qmm/Ruy.inl.cc"

namespace slimt::qmm {
Tensor affine(const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name) {
  using detail::affine;
  using detail::kProvider;
  return affine<kProvider>(x, W, b, a_quant, b_quant, name);
}

Tensor prepare_bias(const Tensor& W, const Tensor& b, float a_quant,
                    float b_quant, const std::string& name) {
  using detail::kProvider;
  using detail::prepare_bias;
  return prepare_bias<kProvider>(W, b, a_quant, b_quant, name);
}

Tensor affine_with_prepared_bias(const Tensor& x, const Tensor& W,
                                 const Tensor& prepared_bias, float a_quant,
                                 float b_quant, const std::string& name) {
  using detail::affine_with_prepared_bias;
  using detail::kProvider;
  return affine_with_prepared_bias<kProvider>(x, W, prepared_bias, a_quant,
                                              b_quant, name);
}

Tensor affine_relu_requantize(const Tensor& x, const Tensor& W,
                              const Tensor& prepared_bias, float a_quant,
                              float b_quant, float out_quant,
                              const std::string& name) {
  using detail::affine_relu_requantize;
  using detail::kProvider;
  return affine_relu_requantize<kProvider>(x, W, prepared_bias, a_quant,
                                           b_quant, out_quant, name);
}

Tensor affine_quantized(const Tensor& x, const Tensor& W,
                        const Tensor& prepared_bias, float a_quant,
                        float b_quant, const std::string& name) {
  using detail::affine_quantized;
  using detail::kProvider;
  return affine_quantized<kProvider>(x, W, prepared_bias, a_quant, b_quant,
                                     name);
}

std::vector<Tensor> affine_segmented(const Tensor& x, const Tensor& W,
                                     const Tensor& prepared_bias,
                                     float a_quant,
                                     const std::vector<float>& b_quants,
                                     const std::vector<size_t>& segment_cols,
                                     const std::string& name) {
  using detail::affine_segmented;
  using detail::kProvider;
  return affine_segmented<kProvider>(x, W, prepared_bias, a_quant, b_quants,
                                     segment_cols, name);
}

Tensor select_columns(const Tensor& W, const std::vector<uint32_t>& indices,
                      const std::string& name) {
  using detail::kProvider;
  using detail::select_columns;
  return select_columns<kProvider>(W, indices, name);
}

Tensor dot(const Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name) {
  using detail::dot;
  using detail::kProvider;
  return dot<kProvider>(x, W, a_quant, b_quant, name);
}

void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows) {
  using detail::kProvider;
  using detail::prepare_weight_transposed;
  prepare_weight_transposed<kProvider>(weights, prepared,
                                       quantization_multiplier, cols, rows);
}

void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols) {
  using detail::kProvider;
  using detail::prepare_weight_quantized_transposed;
  prepare_weight_quantized_transposed<kProvider>(input, output, rows, cols);
}

void clear_standalone_pack_cache() {
  using detail::clear_standalone_pack_cache;
  using detail::kProvider;
  clear_standalone_pack_cache<kProvider>();
}

}  // namespace slimt::qmm
