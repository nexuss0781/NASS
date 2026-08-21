#ifndef NASS_PIPELINE_ENGINE_HPP
#define NASS_PIPELINE_ENGINE_HPP

#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include "nass/core/types.hpp"
#include "nass/core/config.hpp"
#include "nass/pipeline/worker.hpp"

namespace nass {

// Processing task for pipeline
struct ProcessTask {
    const Float32* input_data;
    size_t num_samples;
    ComplexFloat* output_data;
    size_t output_frames;
    size_t chunk_id;
    
    Status status;
    std::string error_message;
};

// Pipeline processing engine with thread pool
class PipelineEngine {
public:
    explicit PipelineEngine(const EngineConfig& config);
    ~PipelineEngine();
    
    // Non-copyable
    PipelineEngine(const PipelineEngine&) = delete;
    PipelineEngine& operator=(const PipelineEngine&) = delete;
    
    // Initialize engine
    Status initialize();
    
    // Shutdown engine
    void shutdown();
    
    // Process audio chunk (blocking)
    Result<Spectrogram> process_chunk(const Float32* audio, size_t num_samples);
    
    // Process multiple chunks in parallel
    Result<std::vector<Spectrogram>> process_batch(
        const std::vector<const Float32*>& audio_chunks,
        const std::vector<size_t>& sample_counts);
    
    // Get number of worker threads
    size_t num_workers() const { return workers_.size(); }
    
    // Check if engine is running
    bool is_running() const { return running_; }
    
    // Get performance metrics
    struct Metrics {
        size_t chunks_processed;
        double avg_processing_time_ms;
        double throughput_samples_per_sec;
        size_t active_workers;
    };
    
    Metrics get_metrics() const;
    
private:
    EngineConfig config_;
    std::vector<std::unique_ptr<WorkerNode>> workers_;
    std::atomic<bool> running_;
    std::atomic<size_t> chunks_processed_;
    
    STFTConfig stft_config_;
    std::unique_ptr<STFT> stft_engine_;
    
    // Worker management
    void distribute_tasks(std::vector<ProcessTask>& tasks);
    void wait_for_completion();
};

// Chunked audio processor for streaming
class ChunkedProcessor {
public:
    explicit ChunkedProcessor(PipelineEngine& engine, size_t chunk_size);
    ~ChunkedProcessor();
    
    // Process audio stream in chunks
    Result<std::vector<Spectrogram>> process_stream(
        const Float32* audio, size_t total_samples);
    
    // Get current chunk index
    size_t current_chunk() const { return current_chunk_; }
    
    // Get total chunks
    size_t total_chunks(size_t total_samples) const;
    
private:
    PipelineEngine& engine_;
    size_t chunk_size_;
    size_t current_chunk_;
    Buffer<Float32> overlap_buffer_;
    size_t overlap_samples_;
};

} // namespace nass

#endif // NASS_PIPELINE_ENGINE_HPP
