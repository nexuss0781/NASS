#include "nass/memory/arena.hpp"
#include <cstring>
#include <thread>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace nass {

Arena::Arena(size_t size_bytes, bool use_huge_pages)
    : buffer_(nullptr)
    , capacity_(size_bytes)
    , offset_(0)
    , owns_buffer_(true) {
    
    // Try to allocate with huge pages if requested (Linux only)
    if (use_huge_pages) {
#if defined(__linux__)
        buffer_ = static_cast<uint8_t*>(mmap(
            nullptr,
            capacity_,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0
        ));
        if (buffer_ == MAP_FAILED) {
            buffer_ = nullptr;
        }
#endif
    }
    
    // Fallback to regular aligned allocation
    if (buffer_ == nullptr) {
        buffer_ = static_cast<uint8_t*>(aligned_alloc(CACHE_LINE_SIZE, capacity_));
    }
    
    if (buffer_ == nullptr) {
        throw std::bad_alloc();
    }
}

Arena::~Arena() {
    if (owns_buffer_ && buffer_ != nullptr) {
#if defined(__linux__)
        // Check if it was huge page allocation
        if (munmap(buffer_, capacity_) != 0) {
#endif
            aligned_free(buffer_);
#if defined(__linux__)
        }
#endif
    }
}

Arena::Arena(Arena&& other) noexcept
    : buffer_(other.buffer_)
    , capacity_(other.capacity_)
    , offset_(other.offset_)
    , owns_buffer_(other.owns_buffer_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
    other.owns_buffer_ = false;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        if (owns_buffer_ && buffer_ != nullptr) {
            aligned_free(buffer_);
        }
        
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        owns_buffer_ = other.owns_buffer_;
        
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
        other.owns_buffer_ = false;
    }
    return *this;
}

void* Arena::allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) {
        return nullptr;
    }
    
    // Align the current offset
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
    
    // Check if we have enough space
    if (aligned_offset + bytes > capacity_) {
        return nullptr;  // Out of memory
    }
    
    void* ptr = buffer_ + aligned_offset;
    offset_ = aligned_offset + bytes;
    
    return ptr;
}

void Arena::reset() {
    offset_ = 0;
}

bool Arena::contains(void* ptr) const {
    return (ptr >= buffer_ && ptr < buffer_ + capacity_);
}

void* Arena::align_pointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
}

// Thread-local arena implementation
ThreadLocalArena::ThreadLocalArena(size_t arena_size_mb)
    : arena_(std::make_unique<Arena>(arena_size_mb * 1024 * 1024)) {}

ThreadLocalArena::~ThreadLocalArena() = default;

ThreadLocalArena& ThreadLocalArena::instance(size_t arena_size_mb) {
    thread_local static ThreadLocalArena instance(arena_size_mb);
    return instance;
}

} // namespace nass
