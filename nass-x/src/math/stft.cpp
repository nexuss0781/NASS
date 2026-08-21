#include "nass/math/stft.hpp"
#include <cstring>

namespace nass_x::tensor {

STFT::STFT(const STFTConfig& config)
    : config_(config)
    , valid_(false) {
    
    // Validate configuration
    if (config_.fft_size == 0 || config_.hop_size == 0 || config_.win_size == 0) {
        return;
    }
    
    if (config_.hop_size > config_.win_size) {
        return;
    }
    
    // Initialize FFT engine
    fft_ = std::make_unique<FFTSimd>(config_.fft_size);
    if (!fft_->is_valid()) {
        return;
    }
    
    // Allocate buffers
    window_ = WindowFunction::generate(WindowFunction::Type::HANN, config_.win_size, true);
    fft_buffer_ = Buffer<ComplexFloat>(config_.fft_size);
    frame_buffer_ = Buffer<Float32>(config_.win_size);
    
    valid_ = true;
}

STFT::~STFT() = default;

size_t STFT::calculate_num_frames(size_t num_samples) const {
    if (num_samples < config_.win_size) {
        return 1;
    }
    return 1 + (num_samples - config_.win_size) / config_.hop_size;
}

void STFT::apply_window(Float32* frame) {
    WindowFunction::apply(frame, window_.data(), config_.win_size);
}

Result<Spectrogram> STFT::transform(const Float32* audio, size_t num_samples) {
    if (!valid_) {
        return Result<Spectrogram>(Status::ERROR_INVALID_PARAM, "STFT not initialized");
    }
    
    size_t num_frames = calculate_num_frames(num_samples);
    size_t num_bins = config_.num_freq_bins();
    
    // Allocate output buffer
    auto spectrogram_data = new ComplexFloat[num_frames * num_bins];
    BufferView<ComplexFloat> view(spectrogram_data, num_frames * num_bins, 1);
    
    Status status = transform(audio, num_samples, spectrogram_data, num_frames);
    
    if (status != Status::OK) {
        delete[] spectrogram_data;
        return Result<Spectrogram>(status, "STFT transformation failed");
    }
    
    Spectrogram spec{view, num_frames, num_bins};
    return Result<Spectrogram>(spec);
}

Status STFT::transform(const Float32* audio, size_t num_samples,
                       ComplexFloat* spectrogram, size_t num_frames) {
    if (!valid_) {
        return Status::ERROR_INVALID_PARAM;
    }
    
    size_t num_bins = config_.num_freq_bins();
    
    for (size_t frame = 0; frame < num_frames; ++frame) {
        size_t start_sample = frame * config_.hop_size;
        
        // Extract frame and apply window
        size_t remaining = num_samples - start_sample;
        size_t frame_size = (remaining >= config_.win_size) ? config_.win_size : remaining;
        
        // Copy frame to buffer (zero-pad if needed)
        std::memcpy(frame_buffer_.data(), audio + start_sample, frame_size * sizeof(Float32));
        if (frame_size < config_.win_size) {
            std::memset(frame_buffer_.data() + frame_size, 0, 
                       (config_.win_size - frame_size) * sizeof(Float32));
        }
        
        // Apply window function
        apply_window(frame_buffer_.data());
        
        // Perform FFT
        fft_->forward(frame_buffer_.data(), fft_buffer_.data());
        
        // Copy result to spectrogram (only positive frequencies)
        for (size_t bin = 0; bin < num_bins; ++bin) {
            spectrogram[frame * num_bins + bin] = fft_buffer_[bin];
        }
    }
    
    return Status::OK;
}

// BatchSTFT implementation
BatchSTFT::BatchSTFT(const STFTConfig& config, size_t batch_size)
    : config_(config)
    , batch_size_(batch_size) {
    
    engines_.reserve(batch_size_);
    for (size_t i = 0; i < batch_size_; ++i) {
        engines_.push_back(std::make_unique<STFT>(config_));
    }
}

BatchSTFT::~BatchSTFT() = default;

Result<std::vector<Spectrogram>> BatchSTFT::transform_batch(
    const std::vector<const Float32*>& audio_chunks,
    const std::vector<size_t>& sample_counts) {
    
    if (audio_chunks.size() != sample_counts.size() || 
        audio_chunks.size() > batch_size_) {
        return Result<std::vector<Spectrogram>>(
            Status::ERROR_INVALID_PARAM, "Invalid batch parameters");
    }
    
    std::vector<Spectrogram> results;
    results.reserve(audio_chunks.size());
    
    // Process each chunk with corresponding engine
    for (size_t i = 0; i < audio_chunks.size(); ++i) {
        auto result = engines_[i]->transform(audio_chunks[i], sample_counts[i]);
        if (!result.ok()) {
            return Result<std::vector<Spectrogram>>(
                result.status(), result.error_message());
        }
        results.push_back(result.value());
    }
    
    return Result<std::vector<Spectrogram>>(results);
}

} // namespace nass_x::tensor
