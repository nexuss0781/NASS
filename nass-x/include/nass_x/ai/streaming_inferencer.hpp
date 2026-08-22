#pragma once

#include "nass_x/tensor/tensor.hpp"
#include "nass_x/ai/neural_engine.hpp"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#ifdef NASS_X_HAS_BOOST
#include <boost/lockfree/spsc_queue.hpp>
#endif

namespace nass_x::ai {

/**
 * @brief Configuration for streaming inference
 */
struct StreamingConfig {
    size_t chunk_size = 1024;        // Samples per chunk
    size_t hop_size = 256;           // Hop size for overlap-add
    size_t max_queue_size = 32;      // Max chunks in queue
    bool use_overlap_add = true;     // Enable OLA reconstruction
    float overlap_factor = 0.75f;    // Overlap ratio (0.0-1.0)
};

/**
 * @brief Thread-safe ring buffer for real-time audio streaming
 * 
 * Provides lock-free single-producer single-consumer queue for
 * low-latency audio processing pipelines.
 */
class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t capacity);
    
    // Single-producer, single-consumer
    bool push(const float* data, size_t count);
    bool pop(float* data, size_t count);
    
    size_t available() const;
    void clear();
    
private:
#ifdef NASS_X_HAS_BOOST
    boost::lockfree::spsc_queue<float, boost::lockfree::capacity<65536>> buffer_;
#else
    // Fallback to mutex-based queue if Boost not available
    std::vector<float> buffer_;
    std::atomic<size_t> read_pos_{0};
    std::atomic<size_t> write_pos_{0};
    mutable std::mutex mutex_;
#endif
    size_t capacity_;
};

/**
 * @brief Real-time streaming inference engine with overlap-add reconstruction
 * 
 * Manages chunked audio processing through neural models with seamless
 * reconstruction using windowed overlap-add technique.
 */
class StreamingInferencer {
public:
    explicit StreamingInferencer(
        std::shared_ptr<NeuralEngine> engine,
        const StreamingConfig& config = StreamingConfig());
    
    ~StreamingInferencer();
    
    /** Push audio samples for processing (thread-safe) */
    void push_audio(const float* samples, size_t count);
    
    /** Get processed output samples (thread-safe) */
    size_t get_output(float* buffer, size_t max_samples);
    
    /** Check if output is available */
    bool has_output() const;
    
    /** Get current latency in milliseconds */
    float get_latency_ms() const;
    
    /** Reset internal state */
    void reset();
    
    /** Get statistics */
    struct Stats {
        size_t chunks_processed = 0;
        size_t underruns = 0;
        size_t overruns = 0;
        float avg_latency_ms = 0.0f;
    };
    Stats get_stats() const;

private:
    void process_chunk();
    void apply_window(std::vector<float>& chunk, const std::vector<float>& window);
    void overlap_add(const std::vector<float>& chunk, size_t output_offset);
    
    std::shared_ptr<NeuralEngine> engine_;
    StreamingConfig config_;
    
    AudioRingBuffer input_buffer_;
    AudioRingBuffer output_buffer_;
    
    std::vector<float> current_chunk_;
    std::vector<float> overlap_buffer_;
    std::vector<float> window_;
    
    std::mutex process_mutex_;
    std::atomic<bool> running_{true};
    std::atomic<size_t> samples_written_{0};
    std::atomic<size_t> samples_read_{0};
    
    Stats stats_;
    mutable std::mutex stats_mutex_;
};

} // namespace nass_x::ai
