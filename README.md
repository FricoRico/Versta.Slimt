# Versta.Leanmt

**Leanmt** is an Android-specific port of
[slimt](https://github.com/jerinphilip/slimt), the lightweight inference
frontend for tiny neural machine translation models. It lives on as a leaner, 
mobile-oriented fork. This version also contains patches made by David Ventura's
[fork](https://github.com/DavidVentura/slimt) of slimt.

Where upstream slimt aims at desktop, Python and command-line usage, leanmt
strips the project down to the parts that matter for shipping translation on
Android: a small, dependency-light C++ inference core plus the JNI surface that
lets it be called from Kotlin/Java. It is slimmed down, intended to use only
with Firefox translation models. The name is a playful nod to its parent -
*slimt* becomes *leanmt* ("lean machine translation").

## Background

[bergamot-translator](https://github.com/browsermt/bergamot-translator/) builds
on top of [marian-dev](https://github.com/marian-nmt/marian-dev) and reuses
marian's inference code-path. marian is a capable neural-network library focused
on machine translation, but most of what it carries autograd, a wide range of
sequence-to-sequence architectures, beam search is unnecessary for running
inference on client machines. leanmt is just the *tiny* part of marian, retuned
for on-device translation.

The same approach inspired contemporary efforts such as
[ggerganov/ggml](https://github.com/ggerganov/ggml) and
[karpathy/llama2.c](https://github.com/karpathy/llama2.c). The *tiny* models
roughly follow the [transformer architecture](https://arxiv.org/abs/1706.03762)
with [Simpler Simple Recurrent Units](https://aclanthology.org/D19-5632/)
(SSRU) in the decoder, and are the models used by Mozilla Firefox's
[offline translation addon](https://addons.mozilla.org/en-US/firefox/addon/firefox-translations/).

## How leanmt differs from slimt

leanmt is a deliberate slimdown of upstream slimt for the Android use case:

* **Android deployment focus.** Code and assets not relevant to shipping on
  Android have been removed, keeping the tree small and the build fast.
* **Single, embedded compute backend.** Matrix-multiply is hard-wired to
  [ruy](https://github.com/google/ruy) (the sole `int8_t` backend on every
  platform); the BLAS/intgemm/gemmology alternatives from upstream have been
  dropped.
* **NEON tuned.** SIMD paths target ARM NEON for on-device performance; x86
  SSE2/AVX2 paths are retained for development and testing on the host.
* **Self-contained vocabulary.** Vocabulary handling uses a bundled
  [sentencepiece](https://github.com/browsermt/sentencepiece); sentence
  splitting lives outside the library (in the calling Android/Kotlin layer).

The large dependency set of bergamot-translator is reduced to:

* [ruy](https://github.com/google/ruy) - the sole `int8_t` matrix-multiply
  backend.
* [sentencepiece](https://github.com/browsermt/sentencepiece) - vocabulary /
  subword decoding.

## Repository layout

```
src/              # the C++ inference library (namespace leanmt)
  leanmt.hh        # umbrella header: Frontend + Model + Version
  Frontend.hh/.cc  # translation API, batching, request/response
  Model.hh/.cc     # model loading and decoding
  Transformer.*    # encoder/decoder transformer blocks
  TensorOps.*      # quantized matmul, SIMD (NEON/SSE) kernels
  Simd.hh          # SIMD dispatch (vext namespace)
  qmm/             # ruy-backed quantized matrix-multiply
  simd/            # per-ISA SIMD implementations (neon, sse)
third_party/      # bundled ruy and sentencepiece submodules
```

Public headers install into `<prefix>/include/leanmt`, and the library target
is exported as `leanmt` (with `leanmt::` namespace when packaging is enabled).

## Building

leanmt uses CMake and is meant to be consumed both as a native library in an
Android build (via the NDK + JNI).

### Desktop / host (development)

Clone with submodules:

```bash
git clone --recursive https://github.com/FricoRico/Versta.Leanmt.git
cd Versta.Leanmt
```

Configure and build. On an x86_64 host enable SSE2; on aarch64/armv7 with NEON
use `-DUSE_NEON=ON`:

```bash
ARGS=(
    -DUSE_SSE2=ON            # or -DUSE_NEON=ON on ARM
    -DCMAKE_BUILD_TYPE=Release
)

cmake -B build -S $PWD "${ARGS[@]}"
cmake --build build --target all
```

This produces the `leanmt` shared/static library under `build/`.

### Android (NDK + JNI)

For on-device use, build `leanmt` with the Android NDK toolchain and expose it
through the JNI bindings. A typical invocation points CMake at the NDK and
selects an ARM ABI with NEON:

```bash
cmake -B build-android -S $PWD \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DUSE_NEON=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build-android --target all
```

The resulting `libleanmt` is linked into the Android application and called from
Kotlin/Java through the JNI layer shipped alongside this repository in the
`Versta` packaging. Sentence-splitting and higher-level text handling are
performed on the Kotlin side; leanmt handles model loading, encoding/decoding
and the quantized inference.

## Using the library

The public API centers on the `leanmt::Frontend`, which turns a loaded
`leanmt::Model` into a translator:

```cpp
#include "leanmt/leanmt.hh"

leanmt::Model model;
model.load(root, modelPath, vocabularyPath, shortlistPath);

leanmt::Frontend frontend;
leanmt::Request request{"Hello world"};
leanmt::Response response = frontend.translate(model, request);
std::cout << response.target.text << "\n";
```

See the headers under `leanmt/` for the full request/response and annotation
surface (alignments, scores, sentence boundaries).

## Status

leanmt is a work-in-progress port. Core text translation works for the
English–German tiny models and parity in features and speed with marian and
bergamot-translator (where relevant) is ongoing. Support for `base` models is
planned. Contributions are welcome.

## Credits & license

leanmt is derived from
[slimt](https://github.com/jerinphilip/slimt) by **Jerin Philip**
(`jerinphilip@live.in`) and **George Tom** (`georg3tom@gmail.com`), including patches
from **David Ventura**. Upstream slimt itself reuses code from
[browsermt/bergamot-translator](https://github.com/browsermt/bergamot-translator)
and [browsermt/marian-dev](https://github.com/browsermt/marian-dev).

leanmt is free software, distributed under the GNU General Public License
version 2 or (at your option) any later version. See `LICENSE` for the full text,
exceptions and third-party attributions 
(`browsermt/ssplit`, Apache-2.0; bergamot-translator / marian-dev, MPL-2.0).
