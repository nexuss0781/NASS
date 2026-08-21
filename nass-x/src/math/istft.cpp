#include "nass/math/istft.hpp"
#include <cstring>
#include <cmath>

namespace nass {

ISTFT::ISTFT(const STFTConfig& config)
    : config_(config)
    , valid_(false)
    , reconstruction_error_(0.0f) {
    
    // Validate configuration
    if (config_.fft_size == 0 || config_.hop_size == 0 || config_.win_size == 0) {
        return;
    }
    
    // Initialize FFT engine
    fft_ = std::make_unique<FFTSimd>(config_.fft_size);
    if (!fft_->is_valid()) {
        return;
    }
    
    // Allocate buffers
    window_ = WindowFunction::generate(WindowFunction::Type::HANN, config_.win_size, true);
    ifft_buffer_ = Buffer<ComplexFloat>(config_.fft_size);
    overlap_buffer_ = Buffer<Float32>(config_.win_size);
    overlap_add_ = std::make_unique<OverlapAdd>(config_.hop_size, config_.win_size);
    
    valid_ = true;
}

ISTFT::~ISTFT() = default;

size_t ISTFT::calculate_output_size(size_t num_frames) const {
    return (num_frames - 1) * config_.hop_size + config_.win_size;
}

void ISTFT::apply_window(ComplexFloat* spectrum) {
    // Apply window in frequency domain (conjugate symmetry preserved)
    const Float32* win = window_.data();
    for (size_t i = 0; i < config_.fft_size; ++i) {
        spectrum[i] *= win[i % config_.win_size];
    }
}

Result<FloatBuffer> ISTFT::inverse(const ComplexFloat* spectrogram, size_t num_frames) {
    if (!valid_) {
        return Result<FloatBuffer>(Status::ERROR_INVALID_PARAM, "ISTFT not initialized");
    }
    
    size_t output_size = calculate_output_size(num_frames);
    FloatBuffer output(output_size);
    
    Status status = inverse(spectrogram, num_frames, output.data(), output_size);
    
    if (status != Status::OK) {
        return Result<FloatBuffer>(status, "ISTFT inversion failed");
    }
    
    return Result<FloatBuffer>(std::move(output));
}

Status ISTFT::inverse(const ComplexFloat* spectrogram, size_t num_frames,
                      Float32* output, size_t output_size) {
    if (!valid_) {
        return Status::ERROR_INVALID_PARAM;
    }
    
    size_t num_bins = config_.num_freq_bins();
    
    // Zero output buffer
    std::memset(output, 0, output_size * sizeof(Float32));
    
    // Overlap-add synthesis
    overlap_add_->reset();
    
    for (size_t frame = 0; frame < num_frames; ++frame) {
        // Copy spectrum to IFFT buffer (with Hermitian symmetry)
        for (size_t bin = 0; bin < num_bins; ++bin) {
            ifft_buffer_[bin] = spectrogram[frame * num_bins + bin];
        }
        
        // Fill negative frequencies (Hermitian symmetry for real output)
        for (size_t bin = 1; bin < config_.fft_size / 2; ++bin) {
            ifft_buffer_[config_.fft_size - bin] = std::conj(ifft_buffer_[bin]);
        }
        
        // Perform IFFT
        fft_->inverse(ifft_buffer_.data(), overlap_buffer_.data());
        
        // Scale by FFT size
        float scale = static_cast<float>(config_.fft_size);
        for (size_t i = 0; i < config_.win_size; ++i) {
            overlap_buffer_[i] *= scale;
        }
        
        // Overlap-add to output
        overlap_add_->add_frame(overlap_buffer_.data(), output, frame);
    }
    
    // Finalize with normalization
    overlap_add_->finalize(output, output_size);
    
    return Status::OK;
}

// ReconstructionValidator implementation
ReconstructionValidator::ReconstructionValidator(const STFTConfig& config)
    : config_(config)
    , stft_(std::make_unique<STFT>(config))
    , istft_(std::make_unique<ISTFT>(config)) {
    
    // Pre-allocate temporary buffer
    temp_buffer_ = Buffer<Float32>(config_.win_size * 4);
}

Float32 ReconstructionValidator::validate_round_trip(const Float32* audio, size_t num_samples) {
    if (!stft_->is_valid() || !istft_->is_valid()) {
        return -1.0f;  // Invalid state
    }
    
    // Forward transform
    auto stft_result = stft_->transform(audio, num_samples);
    if (!stft_result.ok()) {
        return -1.0f;
    }
    
    Spectrogram& spec = stft_result.value();
    
    // Inverse transform
    auto istft_result = istft_->inverse(spec.data.data, spec.num_frames);
    if (!istft_result.ok()) {
        return -1.0f;
    }
    
    FloatBuffer& reconstructed = istft_result.value();
    
    // Calculate reconstruction error (L2 norm)
    size_t compare_size = std::min(num_samples, reconstructed.size());
    Float32 error_sum = 0.0f;
    Float32 signal_power = 0.0f;
    
    for (size_t i = 0; i < compare_size; ++i) {
        Float32 diff = audio[i] - reconstructed[i];
        error_sum += diff * diff;
        signal_power += audio[i] * audio[i];
    }
    
    if (signal_power < 1e-10f) {
        return 0.0f;  // Silent signal
    }
    
    return std::sqrt(error_sum / signal_power);
}

bool ReconstructionValidator::is_perfect(Float32 error, Float32 tolerance) const {
    return error >= 0.0f && error <= tolerance;
}

} // namespace nass
