#include "nass/pipeline/engine.hpp"
#include <chrono>

namespace nass {

PipelineEngine::PipelineEngine(const EngineConfig& config)
    : config_(config)
    , running_(false)
    , chunks_processed_(0)
    , stft_config_(config.default_stft) {
    
    stft_engine_ = std::make_unique<STFT>(stft_config_);
}

PipelineEngine::~PipelineEngine() {
    shutdown();
}

Status PipelineEngine::initialize() {
    if (running_) {
        return Status::OK;  // Already initialized
    }
    
    if (!stft_engine_->is_valid()) {
        return Status::ERROR_INVALID_PARAM;
    }
    
    // Create worker threads
    size_t num_workers = config_.num_threads;
    if (num_workers == 0) {
        num_workers = std::thread::hardware_concurrency();
        if (num_workers == 0) num_workers = 4;
    }
    
    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i) {
        workers_.push_back(std::make_unique<WorkerNode>(i));
        workers_[i]->start();
    }
    
    running_ = true;
    return Status::OK;
}

void PipelineEngine::shutdown() {
    if (!running_) return;
    
    running_ = false;
    
    for (auto& worker : workers_) {
        worker->stop();
    }
    workers_.clear();
}

Result<Spectrogram> PipelineEngine::process_chunk(const Float32* audio, size_t num_samples) {
    if (!running_ || !stft_engine_->is_valid()) {
        return Result<Spectrogram>(Status::ERROR_INVALID_PARAM, "Engine not initialized");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto result = stft_engine_->transform(audio, num_samples);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (result.ok()) {
        ++chunks_processed_;
    }
    
    return result;
}

Result<std::vector<Spectrogram>> PipelineEngine::process_batch(
    const std::vector<const Float32*>& audio_chunks,
    const std::vector<size_t>& sample_counts) {
    
    if (!running_ || audio_chunks.size() != sample_counts.size()) {
        return Result<std::vector<Spectrogram>>(
            Status::ERROR_INVALID_PARAM, "Invalid batch parameters");
    }
    
    std::vector<Spectrogram> results(audio_chunks.size());
    std::vector<Status> statuses(audio_chunks.size(), Status::OK);
    
    // Distribute tasks to workers
    for (size_t i = 0; i < audio_chunks.size(); ++i) {
        size_t chunk_idx = i;
        
        workers_[i % workers_.size()]->submit_task([this, &audio_chunks, &sample_counts, 
                                                     &results, &statuses, chunk_idx]() {
            auto result = stft_engine_->transform(audio_chunks[chunk_idx], sample_counts[chunk_idx]);
            if (result.ok()) {
                results[chunk_idx] = result.value();
                statuses[chunk_idx] = Status::OK;
            } else {
                statuses[chunk_idx] = result.status();
            }
            ++chunks_processed_;
        });
    }
    
    // Wait for completion (simple busy-wait for now)
    bool all_done = false;
    while (!all_done) {
        all_done = true;
        for (size_t i = 0; i < audio_chunks.size(); ++i) {
            if (statuses[i] == Status::OK) {
                // Completed successfully
            } else if (statuses[i] != Status::ERROR_INVALID_PARAM && 
                       statuses[i] != Status::ERROR_FFT_FAILED &&
                       statuses[i] != Status::ERROR_OUT_OF_MEMORY &&
                       statuses[i] != Status::ERROR_THREAD_FAILED) {
                // Still processing
                all_done = false;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    // Check for errors
    for (size_t i = 0; i < statuses.size(); ++i) {
        if (statuses[i] != Status::OK) {
            return Result<std::vector<Spectrogram>>(statuses[i], "Batch processing failed");
        }
    }
    
    return Result<std::vector<Spectrogram>>(results);
}

PipelineEngine::Metrics PipelineEngine::get_metrics() const {
    Metrics metrics{};
    metrics.chunks_processed = chunks_processed_;
    metrics.active_workers = 0;
    
    for (const auto& worker : workers_) {
        if (!worker->is_busy()) {
            ++metrics.active_workers;
        }
    }
    
    // Calculate average processing time and throughput
    // (would need timestamp tracking for accurate values)
    metrics.avg_processing_time_ms = 0.0;
    metrics.throughput_samples_per_sec = 0.0;
    
    return metrics;
}

// ChunkedProcessor implementation
ChunkedProcessor::ChunkedProcessor(PipelineEngine& engine, size_t chunk_size)
    : engine_(engine)
    , chunk_size_(chunk_size)
    , current_chunk_(0)
    , overlap_samples_(0) {
    
    // Overlap buffer for continuous processing
    overlap_samples_ = engine.get_metrics().active_workers > 0 ? 
                       chunk_size / 4 : 0;  // 25% overlap
    
    if (overlap_samples_ > 0) {
        overlap_buffer_ = Buffer<Float32>(overlap_samples_);
    }
}

ChunkedProcessor::~ChunkedProcessor() = default;

size_t ChunkedProcessor::total_chunks(size_t total_samples) const {
    if (chunk_size_ == 0) return 0;
    return (total_samples + chunk_size_ - 1) / chunk_size_;
}

Result<std::vector<Spectrogram>> ChunkedProcessor::process_stream(
    const Float32* audio, size_t total_samples) {
    
    std::vector<Spectrogram> results;
    size_t num_chunks = total_chunks(total_samples);
    
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start_sample = i * chunk_size_;
        size_t samples_remaining = total_samples - start_sample;
        size_t chunk_samples = (samples_remaining >= chunk_size_) ? 
                               chunk_size_ : samples_remaining;
        
        // Apply overlap from previous chunk if needed
        const Float32* chunk_data = audio + start_sample;
        
        auto result = engine_.process_chunk(chunk_data, chunk_samples);
        if (!result.ok()) {
            return Result<std::vector<Spectrogram>>(
                result.status(), result.error_message());
        }
        
        results.push_back(result.value());
        current_chunk_ = i + 1;
    }
    
    return Result<std::vector<Spectrogram>>(results);
}

} // namespace nass
