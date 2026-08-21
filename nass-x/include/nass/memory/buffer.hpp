#ifndef NASS_MEMORY_BUFFER_HPP
#define NASS_MEMORY_BUFFER_HPP

#include <cstddef>
#include <memory>
#include "nass/core/types.hpp"
#include "nass/memory/arena.hpp"

namespace nass {

// Managed buffer with optional arena allocation
template<typename T>
class Buffer {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    
    // Empty buffer
    Buffer() : data_(nullptr), size_(0), capacity_(0), arena_(nullptr) {}
    
    // Allocate from heap
    explicit Buffer(size_t size) 
        : data_(new T[size]())
        , size_(size)
        , capacity_(size)
        , arena_(nullptr) {}
    
    // Allocate from arena (zero-copy)
    Buffer(Arena* arena, size_t size)
        : data_(static_cast<T*>(arena->allocate(size * sizeof(T), alignof(T))))
        , size_(size)
        , capacity_(size)
        , arena_(arena) {}
    
    // Move constructor
    Buffer(Buffer&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , capacity_(other.capacity_)
        , arena_(other.arena_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        other.arena_ = nullptr;
    }
    
    // Move assignment
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            clear();
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            arena_ = other.arena_;
            
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
            other.arena_ = nullptr;
        }
        return *this;
    }
    
    ~Buffer() {
        clear();
    }
    
    // Non-copyable
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    // Accessors
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }
    
    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }
    
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
    
    // Create view
    BufferView<T> view() {
        return BufferView<T>(data_, size_);
    }
    
    BufferView<const T> view() const {
        return BufferView<const T>(data_, size_);
    }
    
    // Reset buffer
    void reset() {
        clear();
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
        arena_ = nullptr;
    }
    
    // Resize (only for heap-allocated buffers)
    void resize(size_t new_size) {
        if (arena_ != nullptr) {
            // Cannot resize arena-allocated buffers
            return;
        }
        
        if (new_size <= capacity_) {
            size_ = new_size;
            return;
        }
        
        // Grow capacity
        size_t new_capacity = (new_size > capacity_ * 2) ? new_size : capacity_ * 2;
        T* new_data = new T[new_capacity]();
        
        // Copy existing data
        for (size_t i = 0; i < size_; ++i) {
            new_data[i] = data_[i];
        }
        
        delete[] data_;
        data_ = new_data;
        capacity_ = new_capacity;
        size_ = new_size;
    }
    
private:
    void clear() {
        if (data_ != nullptr && arena_ == nullptr) {
            delete[] data_;
        }
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }
    
    T* data_;
    size_t size_;
    size_t capacity_;
    Arena* arena_;
};

// Audio sample buffer type aliases
using FloatBuffer = Buffer<Float32>;
using DoubleBuffer = Buffer<Float64>;
using Int16Buffer = Buffer<Int16>;
using ComplexFloatBuffer = Buffer<ComplexFloat>;

} // namespace nass

#endif // NASS_MEMORY_BUFFER_HPP
