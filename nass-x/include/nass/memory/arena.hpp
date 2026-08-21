#ifndef NASS_MEMORY_ARENA_HPP
#define NASS_MEMORY_ARENA_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include "nass/core/types.hpp"

namespace nass {

// Cache line size for alignment (typically 64 bytes)
constexpr size_t CACHE_LINE_SIZE = 64;

// Memory arena for zero-copy allocations
class Arena {
public:
    explicit Arena(size_t size_bytes, bool use_huge_pages = false);
    ~Arena();
    
    // Non-copyable, movable only
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;
    
    // Allocate memory from arena (cache-line aligned)
    void* allocate(size_t bytes, size_t alignment = CACHE_LINE_SIZE);
    
    // Reset arena (reuse memory without deallocation)
    void reset();
    
    // Get remaining capacity
    size_t available() const { return capacity_ - offset_; }
    
    // Get total capacity
    size_t capacity() const { return capacity_; }
    
    // Check if pointer belongs to this arena
    bool contains(void* ptr) const;
    
    // Create buffer view in arena
    template<typename T>
    BufferView<T> create_view(size_t count) {
        void* mem = allocate(count * sizeof(T), alignof(T));
        return BufferView<T>(static_cast<T*>(mem), count);
    }
    
private:
    uint8_t* buffer_;
    size_t capacity_;
    size_t offset_;
    bool owns_buffer_;
    
    void* align_pointer(void* ptr, size_t alignment);
};

// Thread-local arena wrapper for lock-free allocation
class ThreadLocalArena {
public:
    static ThreadLocalArena& instance(size_t arena_size_mb = 256);
    
    Arena& get() { return *arena_; }
    
    void* allocate(size_t bytes, size_t alignment = CACHE_LINE_SIZE) {
        return arena_->allocate(bytes, alignment);
    }
    
    template<typename T>
    BufferView<T> create_view(size_t count) {
        return arena_->create_view<T>(count);
    }
    
    void reset() { arena_->reset(); }
    
private:
    ThreadLocalArena(size_t arena_size_mb);
    ~ThreadLocalArena();
    
    std::unique_ptr<Arena> arena_;
};

// Aligned allocation helper
inline void* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(size, alignment);
#else
    posix_memalign(&ptr, alignment, size);
#endif
    return ptr;
}

inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace nass

#endif // NASS_MEMORY_ARENA_HPP
