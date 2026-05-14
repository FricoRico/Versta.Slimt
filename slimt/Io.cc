#include "slimt/Io.hh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "slimt/Aligned.hh"
#include "slimt/QMM.hh"
#include "slimt/Tensor.hh"
#include "slimt/Types.hh"

namespace slimt::io {

namespace {

template <typename Element>
Element read(void*& read_head) {
  Element value;
  std::memcpy(&value, read_head, sizeof(Element));
  read_head = static_cast<char*>(read_head) + sizeof(Element);
  return value;
}

template <typename Element>
std::vector<Element> read_vector(void*& read_head, uint64_t size) {
  std::vector<Element> values(size);
  if (size > 0) {
    std::memcpy(values.data(), read_head, size * sizeof(Element));
  }
  read_head = static_cast<char*>(read_head) + size * sizeof(Element);
  return values;
}

char* read_bytes(void*& read_head, uint64_t size) {
  auto* begin = static_cast<char*>(read_head);
  read_head = begin + size;
  return begin;
}

// clang-format off
// NOLINTBEGIN
// Internal to types.h, don't use. Use test functions below.
enum class TypeClass : size_t {
  signed_type   = 0x0100,
  unsigned_type = 0x0200,
  float_type    = 0x0400,

  packed_type   = 0x0800, // special packed (CPU cache friendly) type class, used in FBGEMM, not meant to be used anywhere else
  avx2_type     = 0x1000, // processor-specific layout for avx2, currently used for FBGEMM only
  avx512_type   = 0x2000, // processor-specific layout for avx512, currently used for FBGEMM only

  intgemm_type = 0x4000, // legacy intgemm-format quantized models


  size_mask     = 0x00FF,
  class_mask    = 0xFF00
};

constexpr inline size_t operator+(TypeClass typeClass, size_t val) {
  return (size_t)typeClass + val;
}

constexpr inline size_t operator+(size_t val, TypeClass typeClass) {
  return val + (size_t)typeClass;
}



enum class OGType : size_t {
  int8     = TypeClass::signed_type + 1u,
  int16    = TypeClass::signed_type + 2u,
  int32    = TypeClass::signed_type + 4u,
  int64    = TypeClass::signed_type + 8u,

  uint8    = TypeClass::unsigned_type + 1u,
  uint16   = TypeClass::unsigned_type + 2u,
  uint32   = TypeClass::unsigned_type + 4u,
  uint64   = TypeClass::unsigned_type + 8u,

  float16  = TypeClass::float_type + 2u,
  float32  = TypeClass::float_type + 4u,
  float64  = TypeClass::float_type + 8u,

  packed16      = TypeClass::packed_type + 2u,                          // special type for FBGEMM, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint16) is meaningless.
  packed8avx2   = TypeClass::packed_type + 1u + TypeClass::avx2_type,   // special type for FBGEMM with AVX2, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint8) is meaningless.
  packed8avx512 = TypeClass::packed_type + 1u + TypeClass::avx512_type, // special type for FBGEMM with AVX512, not meant to be used anywhere else, not meant to be accessed invidually. Internal actual type (uint8) is meaningless.

  intgemm8      = TypeClass::signed_type + 1u + TypeClass::intgemm_type, // legacy int8 quantized matrices
  intgemm16     = TypeClass::signed_type + 2u + TypeClass::intgemm_type // legacy int16 quantized matrices
};

// clang-format on
// NOLINTEND

Type intercept(uint64_t value) {
  auto type = static_cast<OGType>(value);
  switch (type) {
    case OGType::intgemm8:
      return Type::ig8;
    case OGType::int8:
      return Type::i8;
    case OGType::float32:
      return Type::f32;
    default:
      throw std::runtime_error(
          "[slimt] model file contains an incompatible tensor type (" +
          std::to_string(value) + ")");
  }
}

}  // namespace

void set_item(Item& item, Aligned&& aligned) {
  item.aligned = std::move(aligned);
  item.view = View{
      .data = reinterpret_cast<char*>(item.aligned.data()),  //
      .size = item.aligned.size()                            //
  };
}

std::vector<io::Item> load_items(void* current) {
  uint64_t binary_file_version = read<uint64_t>(current);
  if (binary_file_version != kBinaryFileVersion) {
    throw std::runtime_error(
        "[slimt] model file binary version mismatch: file is v" +
        std::to_string(binary_file_version) + ", slimt expects v" +
        std::to_string(kBinaryFileVersion) +
        " (re-download or regenerate the model)");
  }

  // Read number of headers and based on the information, the headers.
  uint64_t num_headers = read<uint64_t>(current);
  std::vector<Header> headers = read_vector<Header>(current, num_headers);

  // prepopulate items with meta data from headers
  std::vector<io::Item> items;
  items.resize(num_headers);
  for (uint64_t i = 0; i < num_headers; ++i) {
    items[i].type = intercept(headers[i].type);

    // Can someone explain the -1? Remains a mystery to me.
    size_t length = headers[i].name_length;
    char* name = read_bytes(current, length);
    items[i].name = std::string(name, length - 1);
  }

  // read in actual shape and data
  for (uint64_t i = 0; i < num_headers; ++i) {
    Item& item = items[i];

    uint64_t size = headers[i].shape_length;
    std::vector<int> shape = read_vector<int>(current, size);

    // This copy has to be incurred, because metadata.
    item.shape.set(shape.data(), shape.data() + shape.size());
  }

  // move by offset bytes, aligned to 256-bytes boundary
  uint64_t offset = read<uint64_t>(current);
  read_bytes(current, offset);

  // Keep extra items for the int8-prepared (PrepareB-form) versions of any
  // embeddings that double as the decoder output projection. For shared-vocab
  // models that's a single `Wemb_intgemm8`. For two-vocab models with
  // separate `decoder_Wemb`, that's `decoder_Wemb_intgemm8`. Encoder-only
  // embeddings (`encoder_Wemb`) don't need a PrepareB form since they're
  // never used as a GEMM B-matrix.
  Item embedding_processed;
  Item decoder_embedding_processed;

  // Recognises the three embedding tensor names slimt knows about. Anything
  // else with name ending in "_QuantMultA" stays a quantization-multiplier
  // marker (no-op view); other ig8 tensors are GEMM B-matrices that need
  // PrepareBQuantizedTransposed.
  auto is_embedding_name = [](const std::string& name) {
    return name == "Wemb" || name == "encoder_Wemb" || name == "decoder_Wemb";
  };
  auto is_quant_mult_a_marker = [](const std::string& name) {
    return name == "Wemb_QuantMultA" ||
           name == "encoder_Wemb_QuantMultA" ||
           name == "decoder_Wemb_QuantMultA";
  };

  for (uint64_t i = 0; i < num_headers; ++i) {
    Item& item = items[i];
    uint64_t size = headers[i].data_length;
    char* ptr = read_bytes(current, size);
    // We're about to read-data.
    // We can either make it point to mmap, which is aligned,
    // or we can create a new aligned.
    if (item.type == Type::ig8) {
      // since Embedding layer quantized weights need to be dequantised, we
      // have a special case for the embedding tensors. Two-vocab models
      // (en-zh, en-ja, ...) have separate `encoder_Wemb` / `decoder_Wemb`;
      // single-vocab models have a single `Wemb`. The handling is the same:
      // unquantize the int8 storage to float for embedding lookup, and (for
      // the embedding that doubles as the decoder output projection) also
      // produce a PrepareB-prepared int8 alias for the GEMM.
      if (is_quant_mult_a_marker(item.name)) {
        // `*_Wemb_QuantMultA` tensors aren't empty placeholders despite
        // their shape-[1,1] size — marian stores them in the ig8 layout
        // (`shape.elements()` int8 values followed by the f32 `quantMult`)
        // and `marian-dev/.../integer_common.h::unquantizeWemb` reads them
        // back as `alpha[i] = int8[i] / quantMult`. For these alpha
        // tensors that's one int8 (=127, the saturated post-quantize value)
        // plus the multiplier, which un-quantizes to the original
        // calibrated activation alpha (≈4–8 for tiny11 bergamot models).
        //
        // We only need to materialize the value for `decoder_Wemb_QuantMultA`
        // — on shared-vocab models the same alpha is also shipped as a
        // real f32 `none_QuantMultA` that the parameter map already
        // consumes, and `encoder_Wemb` is an embedding-lookup-only tensor
        // with no output-projection GEMM. Drop those by clearing the name;
        // for `decoder_Wemb_QuantMultA` (the only source on two-vocab
        // models like en-ja / zh_hant-en) un-quantize and store as f32.
        size_t num_elements = item.shape.elements();
        char* mult_addr = ptr + num_elements;
        float quant_mult;
        std::memcpy(&quant_mult, mult_addr, sizeof(float));

        Aligned aligned(kAlignWidth, sizeof(float));
        auto* out = reinterpret_cast<float*>(aligned.data());
        out[0] =
            static_cast<float>(reinterpret_cast<int8_t*>(ptr)[0]) / quant_mult;
        set_item(item, std::move(aligned));
        item.type = Type::f32;
      } else if (is_embedding_name(item.name)) {  // NOLINT
        size_t num_elements = item.shape.elements();
        // At the end of items is the quantization multiplier.So we do some
        // pointer arithmetic to move ahead of the elements to extract the
        // quantization multiplier.
        char* end = ptr + num_elements;
        auto* quantization_multiplier_addr = reinterpret_cast<float*>(end);
        float quantization_multiplier = *(quantization_multiplier_addr);

        // Allocate aligned storage to write out unquantized embeddings.
        size_t size_as_float = num_elements * sizeof(float);
        Aligned aligned(kAlignWidth, size_as_float);

        auto* quantized_weights = reinterpret_cast<int8_t*>(ptr);
        auto* weights = reinterpret_cast<float*>(aligned.data());
        unquantize_embedding_weights(quantized_weights, quantization_multiplier,
                                     num_elements, weights);
        set_item(item, std::move(aligned));
        item.type = Type::f32;

        size_t rows = item.shape.dim(-2);
        size_t cols = item.shape.dim(-1);
        assert((rows * cols) % 8 == 0);

        // Decide which name the PrepareB-form alias should take. Skip the
        // alias entirely for `encoder_Wemb` — encoder embeddings are never
        // used as a GEMM B-matrix. For `decoder_Wemb` and the legacy single
        // `Wemb`, build the int8 alias for the decoder output projection.
        const bool needs_prepared_alias = (item.name != "encoder_Wemb");
        if (needs_prepared_alias) {
          Item& processed = (item.name == "decoder_Wemb")
                                ? decoder_embedding_processed
                                : embedding_processed;
          processed.name = (item.name == "decoder_Wemb")
                               ? "decoder_Wemb_intgemm8"
                               : "Wemb_intgemm8";
          processed.shape = Shape({cols, rows});
          processed.type = Type::i8;
          size_t prepared_size =
              processed.shape.elements() * sizeof(int8_t) + sizeof(float);
          Aligned embedding_aligned(kAlignWidth, prepared_size);
          auto* prepared = reinterpret_cast<int8_t*>(embedding_aligned.data());
          qmm::prepare_weight_transposed(weights, prepared,
                                         quantization_multiplier, cols, rows);

          auto* embedding_quantization_multiplier_addr =
              reinterpret_cast<float*>(prepared + (rows * cols));
          *embedding_quantization_multiplier_addr = quantization_multiplier;

          set_item(processed, std::move(embedding_aligned));
        }
      } else {
        // The matrix has to be processed to the format expected by the QMM backend.
        size_t rows = item.shape.dim(-2);
        size_t cols = item.shape.dim(-1);
        auto* input = reinterpret_cast<int8_t*>(ptr);

        Aligned aligned(kAlignWidth, rows * cols + sizeof(float));

        auto* output = reinterpret_cast<int8_t*>(aligned.data());
        qmm::prepare_weight_quantized_transposed(input, output, rows, cols);

        // Set b_quant at end.
        auto* output_end = reinterpret_cast<float*>(output + rows * cols);
        auto* input_end = reinterpret_cast<float*>(input + rows * cols);
        *output_end = *input_end;

        set_item(item, std::move(aligned));
        item.type = Type::i8;

        // This debug function exists here to inspect if need be.
        auto debug = [&]() {
          Tensor input_view;
          View original = {
              .data = ptr,                                         //
              .size = static_cast<size_t>(headers[i].data_length)  //
          };

          input_view.load(original, item.type, item.shape, item.name);
          std::cerr << "input" << input_view << "\n";

          Tensor output_view;
          output_view.load(item.view, item.type, item.shape, item.name);
          std::cerr << "output" << output_view << "\n";
          std::abort();
        };

        (void)debug;
      }
    } else {
      item.view = View{
          .data = ptr,                       //
          .size = static_cast<size_t>(size)  //
      };
    }
  }

  // Append both prepared aliases. Items left default-constructed (e.g. when
  // the model has no `decoder_Wemb`) appear as no-name entries that the
  // Transformer's parameter map never looks up — harmless.
  items.push_back(std::move(embedding_processed));
  items.push_back(std::move(decoder_embedding_processed));
  return items;
}

void unquantize_embedding_weights(const int8_t* quantized_weights,
                                  float quantization_multiplier, size_t size,
                                  float* weights) {
  // Now proceed to unquantize the int8_ts into floats.
  for (size_t i = 0; i < size; i++) {
    weights[i] = static_cast<float>(quantized_weights[i]) *
                 (1 / quantization_multiplier);
  }
}

std::ostream& operator<<(std::ostream& out, const Item& item) {
  out << "Item(" << item.name << ", ";
  out << to_string(item.type) << ", ";
  out << item.shape << ")";
  return out;
}

MmapFile::MmapFile(const std::string& filepath) {
  fd_ = open(filepath.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw std::runtime_error("Failed to open file: " + filepath);
  }

  struct stat st;
  if (fstat(fd_, &st) == -1) {
    close(fd_);
    throw std::runtime_error("Failed to get file size: " + filepath);
  }
  size_ = st.st_size;

  data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (data_ == MAP_FAILED) {  // NOLINT
    close(fd_);
    throw std::runtime_error("Failed to mmap file: " + filepath);
  }
}

MmapFile::~MmapFile() {
  if (data_ != nullptr) {
    munmap(data_, size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

MmapFile::MmapFile(MmapFile&& from) noexcept
    : fd_(from.fd_), data_(from.data_), size_(from.size_) {
  from.reset();
}

MmapFile& MmapFile::operator=(MmapFile&& from) noexcept {
  if (this == &from) {
    return *this;
  }
  consume(from);
  return *this;
}

void MmapFile::consume(MmapFile& from) {
  fd_ = (from.fd_);
  data_ = (from.data_);
  size_ = (from.size_);
  from.reset();
}

void MmapFile::reset() {
  fd_ = -1;
  data_ = nullptr;
  size_ = 0;
}

}  // namespace slimt::io
