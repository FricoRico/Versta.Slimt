#include "QMM.hh"

// NOLINTNEXTLINE: The C++ file inclusion is intended.
#include "qmm/Ruy.inl.cc"

namespace leanmt::qmm {

Tensor affine(const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name) {
  return detail::affine(x, W, b, a_quant, b_quant, name);
}

Tensor prepare_bias(const Tensor& W, const Tensor& b, float a_quant,
                    float b_quant, const std::string& name) {
  return detail::prepare_bias(W, b, a_quant, b_quant, name);
}

Tensor affine_with_prepared_bias(const Tensor& x, const Tensor& W,
                                 const Tensor& prepared_bias, float a_quant,
                                 float b_quant, const std::string& name) {
  return detail::affine_with_prepared_bias(x, W, prepared_bias, a_quant, b_quant,
                                           name);
}

Tensor affine_relu_requantize(const Tensor& x, const Tensor& W,
                              const Tensor& prepared_bias, float a_quant,
                              float b_quant, float out_quant,
                              const std::string& name) {
  return detail::affine_relu_requantize(x, W, prepared_bias, a_quant, b_quant,
                                        out_quant, name);
}

Tensor affine_quantized(const Tensor& x, const Tensor& W,
                        const Tensor& prepared_bias, float a_quant,
                        float b_quant, const std::string& name) {
  return detail::affine_quantized(x, W, prepared_bias, a_quant, b_quant, name);
}

std::vector<Tensor> affine_segmented(const Tensor& x, const Tensor& W,
                                     const Tensor& prepared_bias, float a_quant,
                                     const std::vector<float>& b_quants,
                                     const std::vector<size_t>& segment_cols,
                                     const std::string& name) {
  return detail::affine_segmented(x, W, prepared_bias, a_quant, b_quants,
                                  segment_cols, name);
}

Tensor select_columns(const Tensor& W, const std::vector<uint32_t>& indices,
                      const std::string& name) {
  return detail::select_columns(W, indices, name);
}

Tensor dot(const Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name) {
  return detail::dot(x, W, a_quant, b_quant, name);
}

void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows) {
  detail::prepare_weight_transposed(weights, prepared, quantization_multiplier,
                                    cols, rows);
}

void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols) {
  detail::prepare_weight_quantized_transposed(input, output, rows, cols);
}

void clear_standalone_pack_cache() { detail::clear_standalone_pack_cache(); }

}  // namespace leanmt::qmm
