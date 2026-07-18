#include "Aligned.hh"

#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "Arena.hh"

namespace leanmt {

Aligned::Aligned(size_t alignment, size_t size) : size_(size) {
  if (Arena* arena = active_arena()) {
    data_ = arena->allocate(size, alignment);
    from_arena_ = true;
  } else {
    data_ = allocate(alignment, size);
  }
}

Aligned::~Aligned() { release(); }

void* Aligned::data() const { return data_; }
size_t Aligned::size() const { return size_; }

char* Aligned::begin() const { return reinterpret_cast<char*>(data_); }
char* Aligned::end() const { return reinterpret_cast<char*>(data_) + size_; }

bool Aligned::empty() const { return size_ == 0; }

Aligned::Aligned(Aligned&& from) noexcept {
  if (this != &from) {
    release();
    consume(from);
  }
}

Aligned& Aligned::operator=(Aligned&& from) noexcept {
  if (this != &from) {
    release();
    consume(from);
  }
  return *this;
}

void Aligned::consume(Aligned& from) {
  data_ = from.data_;
  size_ = from.size_;
  from_arena_ = from.from_arena_;

  from.data_ = nullptr;
  from.size_ = 0;
  from.from_arena_ = false;
}

void* Aligned::allocate(size_t alignment, size_t size) {
  size_t aligned_size = (size / alignment) * alignment;
  if (size % alignment != 0) {
    aligned_size += alignment;
  }
  assert(aligned_size >= size);
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
    return nullptr;
  }
  return ptr;
}

void Aligned::release() {
  if (data_ != nullptr && !from_arena_) {
    free(data_);
  }
  data_ = nullptr;
  size_ = 0;
  from_arena_ = false;
}
}  // namespace leanmt
