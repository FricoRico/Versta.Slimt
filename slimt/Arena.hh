#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace slimt {

// Bump-pointer arena for per-decode-step transient tensors.
//
// Allocations are taken from a linked list of chunks; reset() rewinds the
// cursor to the first chunk so the same physical memory is reused across
// iterations. Tensors allocated from an arena must not outlive its next
// reset(), which is enforced by convention (only used inside a scoped region
// in Model::decode).
class Arena {
 public:
  explicit Arena(size_t initial_chunk_bytes);

  void* allocate(size_t bytes, size_t alignment);
  void reset();

 private:
  struct Chunk {
    std::unique_ptr<uint8_t, void (*)(void*)> data;
    size_t capacity;
    Chunk(size_t cap);
  };

  void grow(size_t at_least);

  std::vector<Chunk> chunks_;
  size_t current_index_ = 0;
  size_t cursor_ = 0;
  size_t next_chunk_capacity_ = 0;
};

// Thread-local accessor for the currently active arena. nullptr means
// allocations go through the regular heap path. Set/cleared via ArenaScope.
Arena* active_arena();

class ArenaScope {
 public:
  explicit ArenaScope(Arena& arena);
  ~ArenaScope();
  ArenaScope(const ArenaScope&) = delete;
  ArenaScope& operator=(const ArenaScope&) = delete;

 private:
  Arena* previous_;
};

}  // namespace slimt
