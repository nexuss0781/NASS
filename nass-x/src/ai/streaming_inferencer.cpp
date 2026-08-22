#include "nass_x/ai/streaming_inferencer.hpp"
#include <cmath>
#include <cstring>
#include <chrono>

namespace nass_x::ai {

// AudioRingBuffer Implementation
AudioRingBuffer::AudioRingBuffer(size_t capacity) 
    : capacity_(capacity) {
#ifdef NASS_X_HAS_BOOST
    // Boost lockfree queue handles its own allocation
#else
    buffer_.resize(capacity * 2);  // Double buffer for wrap-around
#endif
}

bool AudioRingBuffer::push(const float* data, size_t count) {
#ifdef NASS_X_HAS_BOOST
    return buffer_.push_range(data, data + count) == count;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    size_t write = write_pos_.load(std::memory_order_relaxed);
    size_t read = read_pos_.load(std::memory_order_acquire);
    
    size_t available = (read > write) ? (read - write - 1) : (capacity_ - (write - read) - 1);
    if (count > available) return false;
    
    for (size_t i = 0; i < count; ++i) {
        buffer_[write % capacity_] = data[i];
        write++;
    }
    write_pos_.store(write, std::memory_order_release);
    return true;
#endif
}

bool AudioRingBuffer::pop(float* data, size_t count) {
#ifdef NASS_X_HAS_BOOST
    return buffer_.pop_range(data, data + count) == count;
#else
    std::lock_guard<std::mutex> lock(mutex_);
    size_t write = write_pos_.load(std::memory_order_relaxed);
    size_t read = read_pos_.load(std::memory_order_acquire);
    
    size_t available = (write >= read) ? (write - read) : (capacity_ - (read - write));
    if (count > available) return false;
    
    for (size_t i = 0; i < count; ++i) {
        data[i] = buffer_[read % capacity_];
        read++;
    }
    read_pos_.store(read, std::memory_order_release);
    return true;
#endif
}

size_t AudioRingBuffer::available() const {
#ifdef NASS_X_HAS_BOOST
    return buffer_.read_available();
#else
    size_t write = write_pos_.load(std::memory_order_acquire);
    size_t read = read_pos_.load(std::memory_order_acquire);
    return (write >= read) ? (write - read) : (capacity_ - (read - write));
#endif
}

void AudioRingBuffer::clear() {
#ifdef NASS_X_HAS_BOOST
    buffer_.reset();
#else
    read_pos_.store(0, std::memory_order_release);
    write_pos_.store(0, std::memory_order_release);
#endif
}

// StreamingInferencer Implementation
StreamingInferencer::StreamingInferencer(
    std::shared_ptr<NeuralEngine> engine,
    const StreamingConfig& config)
    : engine_(engine)
    , config_(config)
    , input_buffer_(config.max_queue_size * config.chunk_size)
    , output_buffer_(config.max_queue_size * config.chunk_size)
    , current_chunk_(config.chunk_size, 0.0f)
    , overlap_buffer_(config.chunk_size + config.hop_size, 0.0f)
{
    // Initialize Hann window for overlap-add
    if (config_.use_overlap_add) {
        window_.resize(config_.chunk_size);
        for (size_t i = 0; i < config_.chunk_size; ++i) {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (config_.chunk_size - 1)));
        }
    }
}

StreamingInferencer::~StreamingInferencer() {
    running_ = false;
}

void StreamingInferencer::push_audio(const float* samples, size_t count) {
    samples_written_ += count;
    
    if (!input_buffer_.push(samples, count)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.overruns++;
    }
    
    // Process chunks as they become available
    while (input_buffer_.available() >= config_.hop_size) {
        process_chunk();
    }
}

size_t StreamingInferencer::get_output(float* buffer, size_t max_samples) {
    size_t got = output_buffer_.pop(buffer, max_samples);
    samples_read_ += got;
    
    if (got < max_samples) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.underruns++;
    }
    
    return got;
}

bool StreamingInferencer::has_output() const {
    return output_buffer_.available() > 0;
}

float StreamingInferencer::get_latency_ms() const {
    size_t pending = input_buffer_.available() + output_buffer_.available();
    float sample_rate = 48000.0f;  // Assume 48kHz, could be configurable
    return (static_cast<float>(pending) / sample_rate) * 1000.0f;
}

void StreamingInferencer::reset() {
    input_buffer_.clear();
    output_buffer_.clear();
    std::fill(current_chunk_.begin(), current_chunk_.end(), 0.0f);
    std::fill(overlap_buffer_.begin(), overlap_buffer_.end(), 0.0f);
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
}

StreamingInferencer::Stats StreamingInferencer::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    Stats s = stats_;
    if (s.chunks_processed > 0) {
        s.avg_latency_ms = get_latency_ms();
    }
    return s;
}

void StreamingInferencer::process_chunk() {
    std::lock_guard<std::mutex> lock(process_mutex_);
    
    // Extract hop_size samples for processing
    std::vector<float> hop_samples(config_.hop_size);
    if (!input_buffer_.pop(hop_samples.data(), config_.hop_size)) {
        return;
    }
    
    // Shift chunk buffer and add new samples
    std::memmove(current_chunk_.data(), 
                 current_chunk_.data() + config_.hop_size,
                 (config_.chunk_size - config_.hop_size) * sizeof(float));
    std::memcpy(current_chunk_.data() + (config_.chunk_size - config_.hop_size),
                hop_samples.data(),
                config_.hop_size * sizeof(float));
    
    // Apply window if using overlap-add
    std::vector<float> windowed_chunk = current_chunk_;
    if (config_.use_overlap_add) {
        apply_window(windowed_chunk, window_);
    }
    
    // Convert to tensor and run inference
    Tensor input_tensor({1, config_.chunk_size}, windowed_chunk.data());
    Tensor output_tensor = engine_->infer(input_tensor);
    
    // Convert back and apply overlap-add
    std::vector<float> output_samples(output_tensor.shape[1]);
    std::memcpy(output_samples.data(), output_tensor.data, 
                output_samples.size() * sizeof(float));
    
    if (config_.use_overlap_add) {
        overlap_add(output_samples, 0);
    } else {
        output_buffer_.push(output_samples.data(), output_samples.size());
    }
    
    // Update stats
    {
        std::lock_guard<std::mutex> slock(stats_mutex_);
        stats_.chunks_processed++;
    }
}

void StreamingInferencer::apply_window(std::vector<float>& chunk, const std::vector<float>& window) {
    for (size_t i = 0; i < chunk.size() && i < window.size(); ++i) {
        chunk[i] *= window[i];
    }
}

void StreamingInferencer::overlap_add(const std::vector<float>& chunk, size_t output_offset) {
    // Simple overlap-add: add new chunk to overlap buffer
    for (size_t i = 0; i < chunk.size() && i < overlap_buffer_.size(); ++i) {
        overlap_buffer_[i] += chunk[i];
    }
    
    // Output the first hop_size samples from overlap buffer
    output_buffer_.push(overlap_buffer_.data(), config_.hop_size);
    
    // Shift overlap buffer
    std::memmove(overlap_buffer_.data(), 
                 overlap_buffer_.data() + config_.hop_size,
                 (overlap_buffer_.size() - config_.hop_size) * sizeof(float));
    std::memset(overlap_buffer_.data() + (overlap_buffer_.size() - config_.hop_size),
                0,
                config_.hop_size * sizeof(float));
}

} // namespace nass_x::ai
