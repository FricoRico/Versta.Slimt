#include <cmath>
#include <cstddef>
#include <cstdint>

namespace slimt {

// NOLINTBEGIN
enum class VExt {
  w0,  //
  w1,  //
  w4,  //
  w8,  //
};

// NOLINTEND
template <enum VExt>
struct VDatum;

template <enum VExt>
struct Ops;

}  // namespace slimt

// Naive implementation

// SSE2 is given for x86-64.
#if defined(USE_SSE2)
#include "slimt/simd/sse.h"
#define VEXT_W4_AVAILABLE

namespace slimt {
using F32x4 = VDatum<VExt::w4>;
}
#endif

// NEON is expected by default for aarch64.
#if defined(USE_NEON)
#include "slimt/simd/neon.h"
#define VEXT_W4_AVAILABLE
namespace slimt {
using F32x4 = VDatum<VExt::w4>;
}
#endif

namespace slimt::vext {

template <VExt Width>
void add(const float* a, const float* b, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::add(va[i], vb[i]);
  }
}

template <VExt Width>
void sub(const float* a, const float* b, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::sub(va[i], vb[i]);
  }
}

template <VExt Width>
void mul(const float* a, const float* b, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  const auto* vb = reinterpret_cast<const Element*>(b);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::mul(va[i], vb[i]);
  }
}

template <VExt Width>
void relu(const float* a, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::relu(va[i]);
  }
}

template <VExt Width>
void sigmoid(const float* a, size_t size, float* c) {
  using Element = VDatum<Width>;
  const auto* va = reinterpret_cast<const Element*>(a);
  size_t steps = size / Element::kWidth;

  auto* vc = reinterpret_cast<Element*>(c);
  for (size_t i = 0; i < steps; i++) {
    vc[i] = Ops<Width>::sigmoid(va[i]);
  }
}

template <VExt Width>
void highway(const float* x, const float* y, const float* g, size_t size,
             float* out) {
  // out[i] = sigmoid(g[i]) * x[i] + (1 - sigmoid(g[i])) * y[i]
  //        = y[i] + sigmoid(g[i]) * (x[i] - y[i])
  using Element = VDatum<Width>;
  size_t v_steps = size / Element::kWidth;

  const auto* vx = reinterpret_cast<const Element*>(x);
  const auto* vy = reinterpret_cast<const Element*>(y);
  const auto* vg = reinterpret_cast<const Element*>(g);
  auto* vout = reinterpret_cast<Element*>(out);

  for (size_t i = 0; i < v_steps; ++i) {
    Element sg = Ops<Width>::sigmoid(vg[i]);
    Element diff = Ops<Width>::sub(vx[i], vy[i]);
    vout[i] = Ops<Width>::add(vy[i], Ops<Width>::mul(sg, diff));
  }
}

template <VExt Width>
void layer_norm(const float* in, const float* scale, const float* bias,
                float eps, size_t rows, size_t cols, float* out) {
  using Element = VDatum<Width>;
  using Scalar = typename Ops<Width>::Scalar;
  constexpr size_t kWidth = Element::kWidth;
  size_t v_cols = cols / kWidth;

  const auto* vscale = reinterpret_cast<const Element*>(scale);
  const auto* vbias = reinterpret_cast<const Element*>(bias);
  Scalar inv_cols = static_cast<Scalar>(1) / static_cast<Scalar>(cols);

  for (size_t j = 0; j < rows; ++j) {
    const auto* vx = reinterpret_cast<const Element*>(in + j * cols);
    auto* vy = reinterpret_cast<Element*>(out + j * cols);

    // Pass 1: mean.
    Element vsum(0.0F);
    for (size_t i = 0; i < v_cols; ++i) {
      vsum = Ops<Width>::add(vsum, vx[i]);
    }
    Scalar mean = Ops<Width>::Reduce::sum(vsum) * inv_cols;
    Element vmean(mean);

    // Pass 2: variance, centered.
    Element vsumsq(0.0F);
    for (size_t i = 0; i < v_cols; ++i) {
      Element centered = Ops<Width>::sub(vx[i], vmean);
      vsumsq = Ops<Width>::add(vsumsq, Ops<Width>::mul(centered, centered));
    }
    Scalar var = Ops<Width>::Reduce::sum(vsumsq) * inv_cols;
    Element vinv_sigma(static_cast<Scalar>(1) / std::sqrt(var + eps));

    // Pass 3: y = scale * ((x - mean) / sigma) + bias.
    for (size_t i = 0; i < v_cols; ++i) {
      Element centered = Ops<Width>::sub(vx[i], vmean);
      Element normalized = Ops<Width>::mul(centered, vinv_sigma);
      Element scaled = Ops<Width>::mul(normalized, vscale[i]);
      vy[i] = Ops<Width>::add(scaled, vbias[i]);
    }
  }
}

// Fused residual + layer norm: out = layer_norm(a + b). The summed residual
// feeds only the norm, so it is never materialized; a[j]+b[j] is recomputed
// in each pass while the row stays hot in L1, saving the x_plus_y round-trip.
template <VExt Width>
void layer_norm_add(const float* a, const float* b, const float* scale,
                    const float* bias, float eps, size_t rows, size_t cols,
                    float* out) {
  using Element = VDatum<Width>;
  using Scalar = typename Ops<Width>::Scalar;
  constexpr size_t kWidth = Element::kWidth;
  size_t v_cols = cols / kWidth;

  const auto* vscale = reinterpret_cast<const Element*>(scale);
  const auto* vbias = reinterpret_cast<const Element*>(bias);
  Scalar inv_cols = static_cast<Scalar>(1) / static_cast<Scalar>(cols);

  for (size_t j = 0; j < rows; ++j) {
    const auto* va = reinterpret_cast<const Element*>(a + j * cols);
    const auto* vb = reinterpret_cast<const Element*>(b + j * cols);
    auto* vy = reinterpret_cast<Element*>(out + j * cols);

    Element vsum(0.0F);
    for (size_t i = 0; i < v_cols; ++i) {
      vsum = Ops<Width>::add(vsum, Ops<Width>::add(va[i], vb[i]));
    }
    Scalar mean = Ops<Width>::Reduce::sum(vsum) * inv_cols;
    Element vmean(mean);

    Element vsumsq(0.0F);
    for (size_t i = 0; i < v_cols; ++i) {
      Element centered = Ops<Width>::sub(Ops<Width>::add(va[i], vb[i]), vmean);
      vsumsq = Ops<Width>::add(vsumsq, Ops<Width>::mul(centered, centered));
    }
    Scalar var = Ops<Width>::Reduce::sum(vsumsq) * inv_cols;
    Element vinv_sigma(static_cast<Scalar>(1) / std::sqrt(var + eps));

    for (size_t i = 0; i < v_cols; ++i) {
      Element centered = Ops<Width>::sub(Ops<Width>::add(va[i], vb[i]), vmean);
      Element normalized = Ops<Width>::mul(centered, vinv_sigma);
      Element scaled = Ops<Width>::mul(normalized, vscale[i]);
      vy[i] = Ops<Width>::add(scaled, vbias[i]);
    }
  }
}

template <VExt Width>
void softmax(const float* logits_in, size_t batch_size, size_t num_classes,
             float* out_in) {
  // Attention softmax operates over variable-length sequences, so per-row
  // pointers are not 32-byte aligned in general — use unaligned load/store.
  //
  // The trailing `< kWidth` lanes used to fall through to scalar `std::exp`,
  // which routed to libc's `__expf_fma` and showed up as ~1 % of total wall
  // in the flamegraph — multiplied by every attention call. Avoid the
  // scalar tail by using a back-shifted vector load that overlaps the
  // already-processed tail of the row: re-loading and re-exping a few
  // elements with the same vector kernel writes back the same values, so
  // the only correction needed is to add only the *new* lanes to the row
  // sum.
  using Element = VDatum<Width>;
  using Scalar = typename Ops<Width>::Scalar;
  constexpr size_t kWidth = Element::kWidth;
  size_t aligned = (num_classes / kWidth) * kWidth;
  size_t tail = num_classes - aligned;

  for (size_t j = 0; j < batch_size; ++j) {
    const float* logit = logits_in + j * num_classes;
    float* p = out_in + j * num_classes;

    // Rows shorter than one vector lane fall through to scalar — there's no
    // safe back-shifted load.
    if (num_classes < kWidth) {
      Scalar max_scalar = logit[0];
      for (size_t i = 1; i < num_classes; ++i) {
        max_scalar = std::max(max_scalar, logit[i]);
      }
      Scalar sum_scalar = 0.0F;
      for (size_t i = 0; i < num_classes; ++i) {
        Scalar e = std::exp(logit[i] - max_scalar);
        p[i] = e;
        sum_scalar += e;
      }
      for (size_t i = 0; i < num_classes; ++i) p[i] /= sum_scalar;
      continue;
    }

    // Pass 1: max.
    Element vmax = Ops<Width>::loadu(logit);
    for (size_t i = kWidth; i < aligned; i += kWidth) {
      vmax = Ops<Width>::max(vmax, Ops<Width>::loadu(logit + i));
    }
    if (tail > 0) {
      vmax = Ops<Width>::max(
          vmax, Ops<Width>::loadu(logit + num_classes - kWidth));
    }
    Scalar max_scalar = Ops<Width>::Reduce::max(vmax);
    Element vmax_broadcast(max_scalar);

    // Pass 2: shift by max, exp, accumulate sum, store exp values.
    Element vsum(0.0F);
    for (size_t i = 0; i < aligned; i += kWidth) {
      Element x = Ops<Width>::loadu(logit + i);
      Element e = Ops<Width>::exp(Ops<Width>::sub(x, vmax_broadcast));
      vsum = Ops<Width>::add(vsum, e);
      Ops<Width>::storeu(p + i, e);
    }
    Scalar sum_scalar = Ops<Width>::Reduce::sum(vsum);
    if (tail > 0) {
      // Back-shifted tail vector: covers [num_classes - kWidth, num_classes).
      // The first (kWidth - tail) lanes overlap with the last aligned vector;
      // re-storing the same exp values is harmless. Only the last `tail`
      // lanes contribute new sum mass.
      size_t start = num_classes - kWidth;
      Element x = Ops<Width>::loadu(logit + start);
      Element e = Ops<Width>::exp(Ops<Width>::sub(x, vmax_broadcast));
      Ops<Width>::storeu(p + start, e);
      alignas(64) Scalar tmp[kWidth];
      Ops<Width>::storeu(tmp, e);
      for (size_t k = kWidth - tail; k < kWidth; ++k) sum_scalar += tmp[k];
    }

    // Pass 3: normalize. The aligned region divides by vector; the tail
    // divides by scalar over [aligned, num_classes) only. A back-shifted
    // vector here would re-divide the (kWidth - tail) overlap lanes that the
    // aligned loop already normalized, yielding exp/sum^2 for those elements.
    Element vsum_broadcast(sum_scalar);
    for (size_t i = 0; i < aligned; i += kWidth) {
      Element e = Ops<Width>::loadu(p + i);
      Ops<Width>::storeu(p + i, Ops<Width>::div(e, vsum_broadcast));
    }
    for (size_t i = aligned; i < num_classes; ++i) {
      p[i] /= sum_scalar;
    }
  }
}

}  // namespace slimt::vext
