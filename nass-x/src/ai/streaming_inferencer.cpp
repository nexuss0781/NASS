#include "streaming_inferencer.hpp"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace nass_x::ai {

StreamingInferencer::StreamingInferencer(
    std::shared_ptr<NeuralEngine> engine,
    const Config& config)
    : engine_(std::move(engine)), config_(config) {
    
    if (!engine_) {
        throw std::invalid_argument("NeuralEngine cannot be null");
    }
    
    // Initialize ring buffers
    input_buffer_.resize(config_.buffer_chunks);
    output_buffer_.resize(config_.buffer_chunks);
}

bool StreamingInferencer::push_chunk(const Tensor& chunk) {
    if (!running_.load()) {
        return false;
    }
    
    size_t current_write = write_pos_.load(std::memory_order_relaxed);
    size_t current_done = done_pos_.load(std::memory_order_acquire);
    
    // Check if buffer is full
    if (current_write >= current_done + config_.buffer_chunks) {
        return false; // Buffer full, drop chunk (or could block in real-time apps)
    }
    
    size_t idx = current_write % config_.buffer_chunks;
    
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        input_buffer_[idx] = chunk;
    }
    
    write_pos_.store(current_write + 1, std::memory_order_release);
    cv_.notify_one();
    
    return true;
}

bool StreamingInferencer::process_next(Tensor& output, int timeout_ms) {
    size_t current_read = read_pos_.load(std::memory_order_relaxed);
    size_t current_write = write_pos_.load(std::memory_order_acquire);
    
    // Wait for available data
    if (current_read >= current_write) {
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
            [this, current_read]() {
                return write_pos_.load(std::memory_order_acquire) > current_read || !running_.load();
            })) {
            return false; // Timeout
        }
    }
    
    if (!running_.load() && read_pos_.load() >= write_pos_.load()) {
        return false; // Stopped and no more data
    }
    
    size_t idx = current_read % config_.buffer_chunks;
    
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (output_buffer_[idx].shape().empty()) {
            // Output not ready yet, wait a bit more
            return false;
        }
        output = output_buffer_[idx];
    }
    
    read_pos_.store(current_read + 1, std::memory_order_release);
    done_pos_.store(current_read + 1, std::memory_order_release);
    
    return true;
}

void StreamingInferencer::start() {
    if (running_.load()) {
        return; // Already running
    }
    
    should_stop_.store(false);
    running_.store(true);
    write_pos_.store(0);
    read_pos_.store(0);
    done_pos_.store(0);
    
    worker_thread_ = std::thread(&StreamingInferencer::worker_loop, this);
}

void StreamingInferencer::stop() {
    should_stop_.store(true);
    cv_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    running_.store(false);
}

StreamingInferencer::LatencyStats StreamingInferencer::getLatencyStats() const {
    LatencyStats stats{0.0, 0.0, 0.0, chunks_processed_.load()};
    
    if (!config_.enable_monitoring || latency_samples_.empty()) {
        return stats;
    }
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    if (latency_samples_.empty()) {
        return stats;
    }
    
    double sum = std::accumulate(latency_samples_.begin(), latency_samples_.end(), 0.0);
    stats.avg_latency_ms = sum / latency_samples_.size();
    stats.max_latency_ms = *std::max_element(latency_samples_.begin(), latency_samples_.end());
    stats.min_latency_ms = *std::min_element(latency_samples_.begin(), latency_samples_.end());
    
    return stats;
}

void StreamingInferencer::worker_loop() {
    while (running_.load() && !should_stop_.load()) {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        size_t current_write = write_pos_.load(std::memory_order_acquire);
        
        if (current_read >= current_write) {
            // Wait for input
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(100), [this, current_read]() {
                return write_pos_.load(std::memory_order_acquire) > current_read || should_stop_.load();
            });
            continue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Get input chunk
        size_t idx = current_read % config_.buffer_chunks;
        Tensor input_chunk;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            input_chunk = input_buffer_[idx];
        }
        
        // Run inference
        auto outputs = engine_->infer({input_chunk});
        
        if (!outputs.empty()) {
            // Apply overlap-add if needed
            Tensor processed = outputs[0]; // Simplified: assume single output
            
            // Store output
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                output_buffer_[idx] = std::move(processed);
            }
        }
        
        // Record latency
        if (config_.enable_monitoring) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                latency_samples_.push_back(latency_ms);
                // Keep only last 100 samples
                if (latency_samples_.size() > 100) {
                    latency_samples_.erase(latency_samples_.begin());
                }
            }
        }
        
        chunks_processed_.fetch_add(1, std::memory_order_relaxed);
        read_pos_.store(current_read + 1, std::memory_order_release);
    }
}

Tensor StreamingInferencer::overlap_add(const Tensor& chunk, size_t offset) {
    // Simplified placeholder for overlap-add logic
    // In production, this would maintain a state buffer and perform
    // windowed addition of overlapping chunks
    
    // For now, just return the chunk as-is
    return chunk;
}

} // namespace nass_x::tensor_x::ai
