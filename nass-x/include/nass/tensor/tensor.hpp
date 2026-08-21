#ifndef NASS_TENSOR_TENSOR_HPP
#define NASS_TENSOR_TENSOR_HPP

#include <cstddef>
#include <vector>
#include <array>
#include <memory>
#include <numeric>
#include <stdexcept>
#include "nass/core/types.hpp"
#include "nass/memory/arena.hpp"

namespace nass {

// Multi-dimensional tensor with strided access
template<typename T, size_t MaxDims = 4>
class Tensor {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    
    // Empty tensor
    Tensor() : data_(nullptr), size_(0), dims_(0), arena_(nullptr), is_view_(false), shape_(), strides_() {}
    
    // Create tensor with shape
    template<typename... Dims>
    explicit Tensor(Dims... dims) 
        : data_(nullptr)
        , size_(0)
        , dims_(sizeof...(dims))
        , arena_(nullptr)
        , is_view_(false)
        , shape_{static_cast<size_t>(dims)...}
        , strides_() {
        compute_strides();
        allocate_heap();
    }
    
    // Create tensor from shape array
    explicit Tensor(const std::array<size_t, MaxDims>& shape, size_t ndim)
        : data_(nullptr)
        , size_(0)
        , dims_(ndim)
        , arena_(nullptr)
        , is_view_(false)
        , shape_(shape)
        , strides_() {
        compute_strides();
        allocate_heap();
    }
    
    // Create tensor in arena (zero-copy)
    template<typename... Dims>
    Tensor(Arena* arena, Dims... dims)
        : data_(nullptr)
        , size_(0)
        , dims_(sizeof...(dims))
        , arena_(arena)
        , is_view_(false)
        , shape_{static_cast<size_t>(dims)...}
        , strides_() {
        compute_strides();
        allocate_arena();
    }
    
    // Move constructor
    Tensor(Tensor&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , dims_(other.dims_)
        , arena_(other.arena_)
        , is_view_(other.is_view_)
        , shape_(other.shape_)
        , strides_(other.strides_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.dims_ = 0;
        other.arena_ = nullptr;
        other.is_view_ = false;
    }
    
    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            clear();
            data_ = other.data_;
            size_ = other.size_;
            dims_ = other.dims_;
            arena_ = other.arena_;
            is_view_ = other.is_view_;
            shape_ = other.shape_;
            strides_ = other.strides_;
            
            other.data_ = nullptr;
            other.size_ = 0;
            other.dims_ = 0;
            other.arena_ = nullptr;
            other.is_view_ = false;
        }
        return *this;
    }
    
    ~Tensor() {
        clear();
    }
    
    // Non-copyable
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Accessors
    size_t ndim() const { return dims_; }
    size_t size() const { return size_; }
    size_t dim(size_t i) const { return (i < dims_) ? shape_[i] : 0; }
    
    const size_t* shape() const { return shape_.data(); }
    const size_t* strides() const { return strides_.data(); }
    
    // Raw data access
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    // Element access (no bounds checking for performance)
    template<typename... Indices>
    T& at(Indices... indices) {
        return data_[offset(indices...)];
    }
    
    template<typename... Indices>
    const T& at(Indices... indices) const {
        return data_[offset(indices...)];
    }
    
    // Element access with bounds checking
    template<typename... Indices>
    T& operator()(Indices... indices) {
        size_t idx = offset(indices...);
        if (idx >= size_) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        return data_[idx];
    }
    
    // Create slice/view (no copy) - returns a non-owning view
    Tensor<T, MaxDims> slice(size_t dim, size_t start, size_t end) {
        if (dim >= dims_ || end <= start) {
            throw std::invalid_argument("Invalid slice parameters");
        }
        
        Tensor<T, MaxDims> view;
        view.data_ = data_ + start * strides_[dim];
        view.size_ = size_ / shape_[dim] * (end - start);
        view.dims_ = dims_;
        view.shape_ = shape_;
        view.strides_ = strides_;
        view.arena_ = arena_;
        view.shape_[dim] = end - start;
        view.is_view_ = true;  // Mark as non-owning view
        
        return view;
    }
    
    // Flatten to 1D view
    BufferView<T> flat_view() {
        return BufferView<T>(data_, size_);
    }
    
    // Reshape (only if total size matches) - returns a non-owning view
    template<typename... NewDims>
    Tensor<T, MaxDims> reshape(NewDims... new_dims) {
        size_t new_size = (new_dims * ...);
        if (new_size != size_) {
            throw std::invalid_argument("Reshape size mismatch");
        }
        
        Tensor<T, MaxDims> reshaped;
        reshaped.data_ = data_;
        reshaped.size_ = size_;
        reshaped.dims_ = sizeof...(new_dims);
        reshaped.shape_ = {static_cast<size_t>(new_dims)...};
        reshaped.strides_ = strides_;
        reshaped.compute_strides();
        reshaped.arena_ = arena_;
        reshaped.is_view_ = true;  // Mark as non-owning view
        
        return reshaped;
    }
    
    // Transpose dimensions - returns a non-owning view
    Tensor<T, MaxDims> transpose(const std::array<size_t, MaxDims>& perm) {
        Tensor<T, MaxDims> transposed;
        transposed.data_ = data_;
        transposed.size_ = size_;
        transposed.dims_ = dims_;
        transposed.arena_ = arena_;
        transposed.is_view_ = true;  // Mark as non-owning view
        
        for (size_t i = 0; i < dims_; ++i) {
            transposed.shape_[i] = shape_[perm[i]];
            transposed.strides_[i] = strides_[perm[i]];
        }
        
        return transposed;
    }
    
    // Zero-fill
    void zero() {
        if (data_) {
            std::fill(data_, data_ + size_, T{});
        }
    }
    
private:
    template<typename... Indices>
    size_t offset(Indices... indices) const {
        constexpr size_t NumIndices = sizeof...(indices);
        static_assert(NumIndices <= MaxDims, "Too many indices");
        
        std::array<size_t, NumIndices> idx_array{static_cast<size_t>(indices)...};
        size_t off = 0;
        for (size_t i = 0; i < NumIndices; ++i) {
            off += idx_array[i] * strides_[i];
        }
        return off;
    }
    
    void compute_strides() {
        if (dims_ == 0) {
            size_ = 0;
            return;
        }
        strides_[dims_ - 1] = 1;
        for (int i = static_cast<int>(dims_) - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
        
        size_ = 1;
        for (size_t i = 0; i < dims_; ++i) {
            size_ *= shape_[i];
        }
    }
    
    void allocate_heap() {
        if (size_ > 0) {
            data_ = new T[size_]();
        }
    }
    
    void allocate_arena() {
        if (arena_ && size_ > 0) {
            data_ = static_cast<T*>(arena_->allocate(size_ * sizeof(T), alignof(T)));
        }
    }
    
    void clear() {
        // Only delete if we own the data (not a view) and it was heap-allocated
        if (data_ && !arena_ && !is_view_) {
            delete[] data_;
            data_ = nullptr;
        }
        // Reset state but preserve is_view_ for debugging
        data_ = nullptr;
        size_ = 0;
        dims_ = 0;
        arena_ = nullptr;
        is_view_ = false;
    }
    
    T* data_;
    size_t size_;
    size_t dims_;
    Arena* arena_;
    bool is_view_;  // True if this tensor is a non-owning view
    std::array<size_t, MaxDims> shape_;
    std::array<size_t, MaxDims> strides_;
};

// Type aliases for common tensor types
using FloatTensor = Tensor<Float32>;
using DoubleTensor = Tensor<Float64>;
using ComplexFloatTensor = Tensor<ComplexFloat>;
using Int16Tensor = Tensor<Int16>;

// Spectrogram tensor (frames x bins)
using SpectrogramTensor = Tensor<ComplexFloat, 2>;

} // namespace nass

#endif // NASS_TENSOR_TENSOR_HPP
