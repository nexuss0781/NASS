#pragma once

#include "neural_engine.hpp"
#include "nass_x/tensor/tensor.hpp"
#include "../io/audio_reader.hpp"
#include "../io/audio_writer.hpp"
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace nass_x::ai {

/**
 * @brief Real-time streaming inference engine with lock-free ring buffer.
 * 
 * Processes continuous audio streams in chunks using neural models,
 * handling overlap-add reconstruction and latency monitoring.
 * Designed for live performance, broadcasting, and real-time communication.
 */
class StreamingInferencer {
public:
    struct Config {
        size_t chunk_size = 1024;       // Samples per chunk
        size_t hop_size = 512;          // Hop size for overlap-add
        size_t buffer_chunks = 32;      // Ring buffer capacity
        int sample_rate = 48000;        // Audio sample rate
        bool enable_monitoring = true;  // Enable latency stats
    };

    explicit StreamingInferencer(
        std::shared_ptr<NeuralEngine> engine,
        const Config& config = Config());

    /**
     * @brief Push audio chunk into the input ring buffer.
     * Non-blocking. Returns false if buffer is full.
     */
    bool push_chunk(const Tensor& chunk);

    /**
     * @brief Process next available chunk and return result.
     * Blocking wait if no chunks available (with timeout).
     * Returns false if stream ended or error occurred.
     */
    bool process_next(Tensor& output, int timeout_ms = 100);

    /**
     * @brief Start background processing thread.
     */
    void start();

    /**
     * @brief Stop processing and flush buffers.
     */
    void stop();

    /**
     * @brief Get current latency statistics.
     */
    struct LatencyStats {
        double avg_latency_ms;
        double max_latency_ms;
        double min_latency_ms;
        size_t processed_chunks;
    };
    LatencyStats getLatencyStats() const;

    /**
     * @brief Check if inferencer is running.
     */
    bool is_running() const { return running_.load(); }

private:
    std::shared_ptr<NeuralEngine> engine_;
    Config config_;
    
    // Lock-free ring buffers (simplified with mutex for now)
    std::vector<Tensor> input_buffer_;
    std::vector<Tensor> output_buffer_;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    std::atomic<size_t> done_pos_{0};
    
    std::mutex buffer_mutex_;
    std::condition_variable cv_;
    
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // Latency monitoring
    mutable std::mutex stats_mutex_;
    std::vector<double> latency_samples_;
    std::atomic<size_t> chunks_processed_{0};
    
    // Worker loop
    void worker_loop();
    
    // Overlap-add reconstruction
    Tensor overlap_add(const Tensor& chunk, size_t offset);
};

} // namespace nass_x::tensor_x::ai
