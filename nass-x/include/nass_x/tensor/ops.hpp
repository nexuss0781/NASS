#ifndef NASS_TENSOR_OPS_HPP
#define NASS_TENSOR_OPS_HPP

#include <cmath>
#include <algorithm>
#include "nass_x/tensor/tensor.hpp"

namespace nass_x::tensor {

// Tensor operations namespace
namespace tensor_ops {

// Element-wise addition
template<typename T, size_t MaxDims>
void add(const Tensor<T, MaxDims>& a, const Tensor<T, MaxDims>& b, 
         Tensor<T, MaxDims>& result) {
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Tensor size mismatch for addition");
    }
    
    const T* data_a = a.data();
    const T* data_b = b.data();
    T* data_res = result.data();
    
    for (size_t i = 0; i < a.size(); ++i) {
        data_res[i] = data_a[i] + data_b[i];
    }
}

// Element-wise subtraction
template<typename T, size_t MaxDims>
void subtract(const Tensor<T, MaxDims>& a, const Tensor<T, MaxDims>& b,
              Tensor<T, MaxDims>& result) {
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Tensor size mismatch for subtraction");
    }
    
    const T* data_a = a.data();
    const T* data_b = b.data();
    T* data_res = result.data();
    
    for (size_t i = 0; i < a.size(); ++i) {
        data_res[i] = data_a[i] - data_b[i];
    }
}

// Element-wise multiplication
template<typename T, size_t MaxDims>
void multiply(const Tensor<T, MaxDims>& a, const Tensor<T, MaxDims>& b,
              Tensor<T, MaxDims>& result) {
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Tensor size mismatch for multiplication");
    }
    
    const T* data_a = a.data();
    const T* data_b = b.data();
    T* data_res = result.data();
    
    for (size_t i = 0; i < a.size(); ++i) {
        data_res[i] = data_a[i] * data_b[i];
    }
}

// Scalar multiplication
template<typename T, size_t MaxDims>
void scale(Tensor<T, MaxDims>& tensor, T scalar) {
    T* data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i) {
        data[i] *= scalar;
    }
}

// Absolute value
template<typename T, size_t MaxDims>
void abs(Tensor<T, MaxDims>& tensor) {
    T* data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i) {
        data[i] = std::abs(data[i]);
    }
}

// Complex magnitude (for ComplexFloat tensors)
template<size_t MaxDims>
void magnitude(const Tensor<ComplexFloat, MaxDims>& complex,
               Tensor<Float32, MaxDims>& magnitude_out) {
    if (complex.size() != magnitude_out.size()) {
        throw std::invalid_argument("Tensor size mismatch for magnitude");
    }
    
    const ComplexFloat* data_c = complex.data();
    Float32* data_m = magnitude_out.data();
    
    for (size_t i = 0; i < complex.size(); ++i) {
        data_m[i] = std::abs(data_c[i]);
    }
}

// Complex phase
template<size_t MaxDims>
void phase(const Tensor<ComplexFloat, MaxDims>& complex,
           Tensor<Float32, MaxDims>& phase_out) {
    if (complex.size() != phase_out.size()) {
        throw std::invalid_argument("Tensor size mismatch for phase");
    }
    
    const ComplexFloat* data_c = complex.data();
    Float32* data_p = phase_out.data();
    
    for (size_t i = 0; i < complex.size(); ++i) {
        data_p[i] = std::arg(data_c[i]);
    }
}

// Polar to complex conversion
template<size_t MaxDims>
void polar_to_complex(const Tensor<Float32, MaxDims>& magnitude,
                      const Tensor<Float32, MaxDims>& phase,
                      Tensor<ComplexFloat, MaxDims>& complex_out) {
    if (magnitude.size() != phase.size() || 
        magnitude.size() != complex_out.size()) {
        throw std::invalid_argument("Tensor size mismatch for polar conversion");
    }
    
    const Float32* data_m = magnitude.data();
    const Float32* data_p = phase.data();
    ComplexFloat* data_c = complex_out.data();
    
    for (size_t i = 0; i < magnitude.size(); ++i) {
        data_c[i] = std::polar(data_m[i], data_p[i]);
    }
}

// Mean reduction
template<typename T, size_t MaxDims>
T mean(const Tensor<T, MaxDims>& tensor) {
    if (tensor.size() == 0) return T{};
    
    const T* data = tensor.data();
    T sum = T{};
    for (size_t i = 0; i < tensor.size(); ++i) {
        sum += data[i];
    }
    return sum / static_cast<T>(tensor.size());
}

// Sum reduction
template<typename T, size_t MaxDims>
T sum(const Tensor<T, MaxDims>& tensor) {
    const T* data = tensor.data();
    T total = T{};
    for (size_t i = 0; i < tensor.size(); ++i) {
        total += data[i];
    }
    return total;
}

// L2 norm
template<typename T, size_t MaxDims>
T norm_l2(const Tensor<T, MaxDims>& tensor) {
    const T* data = tensor.data();
    T sum_sq = T{};
    for (size_t i = 0; i < tensor.size(); ++i) {
        sum_sq += data[i] * data[i];
    }
    return std::sqrt(sum_sq);
}

// Copy tensor data
template<typename T, size_t MaxDims>
void copy(const Tensor<T, MaxDims>& src, Tensor<T, MaxDims>& dst) {
    if (src.size() != dst.size()) {
        throw std::invalid_argument("Tensor size mismatch for copy");
    }
    std::copy(src.data(), src.data() + src.size(), dst.data());
}

// Zero-fill tensor
template<typename T, size_t MaxDims>
void zero(Tensor<T, MaxDims>& tensor) {
    std::fill(tensor.data(), tensor.data() + tensor.size(), T{});
}

// Fill with constant
template<typename T, size_t MaxDims>
void fill(Tensor<T, MaxDims>& tensor, T value) {
    std::fill(tensor.data(), tensor.data() + tensor.size(), value);
}

// Matrix multiplication (2D tensors only)
template<typename T>
void matmul(const Tensor<T, 2>& A, const Tensor<T, 2>& B,
            Tensor<T, 2>& C) {
    if (A.dim(1) != B.dim(0)) {
        throw std::invalid_argument("Matrix dimension mismatch for multiplication");
    }
    if (C.dim(0) != A.dim(0) || C.dim(1) != B.dim(1)) {
        throw std::invalid_argument("Result matrix dimension mismatch");
    }
    
    // Simple O(n^3) implementation - can be optimized with BLAS later
    for (size_t i = 0; i < A.dim(0); ++i) {
        for (size_t j = 0; j < B.dim(1); ++j) {
            T sum = T{};
            for (size_t k = 0; k < A.dim(1); ++k) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}

} // namespace tensor_ops

} // namespace nass_x::tensor

#endif // NASS_TENSOR_OPS_HPP
