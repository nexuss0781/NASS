#ifndef NASS_CORE_TYPES_HPP
#define NASS_CORE_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <complex>
#include <string>
#include <vector>

namespace nass_x::tensor {

// Precision types
using Float32 = float;
using Float64 = double;
using Int16 = int16_t;
using Int32 = int32_t;
using UInt8 = uint8_t;

// Complex type for FFT
template<typename T>
using Complex = std::complex<T>;

using ComplexFloat = Complex<Float32>;
using ComplexDouble = Complex<Float64>;

// Audio format enumeration
enum class AudioFormat {
    FLOAT32,
    FLOAT64,
    INT16,
    INT32,
    UINT8
};

// STFT parameters
struct STFTConfig {
    size_t fft_size = 2048;
    size_t hop_size = 512;      // 75% overlap (fft_size / 4)
    size_t win_size = 2048;
    bool normalized = true;
    AudioFormat format = AudioFormat::FLOAT32;
    
    size_t num_freq_bins() const {
        return fft_size / 2 + 1;
    }
    
    size_t frames_per_second(int sample_rate) const {
        return static_cast<size_t>(sample_rate) / hop_size;
    }
};

// Processing status codes
enum class Status : int {
    OK = 0,
    ERROR_INVALID_PARAM = -1,
    ERROR_OUT_OF_MEMORY = -2,
    ERROR_FILE_NOT_FOUND = -3,
    ERROR_UNSUPPORTED_FORMAT = -4,
    ERROR_FFT_FAILED = -5,
    ERROR_THREAD_FAILED = -6,
    WARNING_PRECISION_LOSS = 100
};

// Buffer view for zero-copy operations
template<typename T>
struct BufferView {
    T* data;
    size_t size;
    size_t stride;
    
    BufferView() : data(nullptr), size(0), stride(1) {}
    BufferView(T* d, size_t s, size_t str = 1) 
        : data(d), size(s), stride(str) {}
    
    T& operator[](size_t idx) { return data[idx * stride]; }
    const T& operator[](size_t idx) const { return data[idx * stride]; }
    
    T* begin() { return data; }
    T* end() { return data + size * stride; }
    const T* begin() const { return data; }
    const T* end() const { return data + size * stride; }
};

// Time-frequency representation
struct Spectrogram {
    BufferView<ComplexFloat> data;
    size_t num_frames;
    size_t num_bins;
    
    ComplexFloat& at(size_t frame, size_t bin) {
        return data[frame * num_bins + bin];
    }
    
    const ComplexFloat& at(size_t frame, size_t bin) const {
        return data[frame * num_bins + bin];
    }
};

} // namespace nass_x::tensor

#endif // NASS_CORE_TYPES_HPP
