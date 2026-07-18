#include "Arena.hh"

#include <cassert>
#include <cstdlib>

#include "Aligned.hh"

namespace leanmt {

namespace {
thread_local Arena* g_active_arena = nullptr;

void* aligned_malloc(size_t alignment, size_t size) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
}
}  // namespace

Arena* active_arena() { return g_active_arena; }

Arena::Chunk::Chunk(size_t cap)
    : data(static_cast<uint8_t*>(aligned_malloc(kAlignWidth, cap)),
           &std::free),
      capacity(cap) {}

Arena::Arena(size_t initial_chunk_bytes)
    : next_chunk_capacity_(initial_chunk_bytes) {
  size_t aligned = ((initial_chunk_bytes + kAlignWidth - 1) / kAlignWidth) *
                   kAlignWidth;
  chunks_.emplace_back(aligned);
}

void* Arena::allocate(size_t bytes, size_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }
  // Chunks are allocated at kAlignWidth which covers the alignments leanmt asks
  // for; fresh chunks start at offset 0 (aligned). Round size up to alignment
  // so vectorized stores past the logical size don't clobber the next
  // allocation (this matches what aligned_alloc does on the heap path).
  assert(alignment <= kAlignWidth);
  size_t padded_bytes = ((bytes + alignment - 1) / alignment) * alignment;
  size_t aligned_cursor = (cursor_ + alignment - 1) & ~(alignment - 1);
  if (aligned_cursor + padded_bytes > chunks_[current_index_].capacity) {
    grow(padded_bytes);
    aligned_cursor = 0;
  }
  void* p = chunks_[current_index_].data.get() + aligned_cursor;
  cursor_ = aligned_cursor + padded_bytes;
  return p;
}

void Arena::grow(size_t at_least) {
  // First try to advance into an already-allocated later chunk that's big
  // enough; this is the common case after the first decode iteration has
  // sized the arena.
  for (size_t i = current_index_ + 1; i < chunks_.size(); ++i) {
    if (chunks_[i].capacity >= at_least) {
      current_index_ = i;
      cursor_ = 0;
      return;
    }
  }
  // No existing chunk fits — allocate a new one, doubling from the largest
  // existing chunk so amortized growth stays bounded.
  size_t cap = next_chunk_capacity_ * 2;
  if (cap < at_least) {
    cap = at_least;
  }
  size_t aligned = ((cap + kAlignWidth - 1) / kAlignWidth) * kAlignWidth;
  chunks_.emplace_back(aligned);
  current_index_ = chunks_.size() - 1;
  cursor_ = 0;
  next_chunk_capacity_ = aligned;
}

void Arena::reset() {
  current_index_ = 0;
  cursor_ = 0;
}

ArenaScope::ArenaScope(Arena& arena) : previous_(g_active_arena) {
  g_active_arena = &arena;
}

ArenaScope::~ArenaScope() { g_active_arena = previous_; }

}  // namespace leanmt
