/**
 * @file pipeline.hpp
 * @brief Multi-threaded audio processing pipeline with work-stealing
 */

#pragma once

#include "nass_x/tensor/tensor.hpp"
#include "nass/memory/arena.hpp"
#include "nass_x/audio_reader.hpp"
#include "nass_x/audio_writer.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#include <memory>

namespace nass_x {

/**
 * @brief Pipeline stage types
 */
enum class StageType {
    DECODE,
    PREPROCESS,
    STFT,
    PROCESS,
    ISTFT,
    POSTPROCESS,
    ENCODE
};

/**
 * @brief Processing task in the pipeline
 */
struct PipelineTask {
    std::unique_ptr<Tensor<float>> input;
    std::unique_ptr<Tensor<float>> output;
    int64_t sample_offset;
    int frame_index;
    bool is_last;
    
    PipelineTask() : sample_offset(0), frame_index(0), is_last(false) {}
};

/**
 * @brief Thread-safe work queue with lock-free operations
 */
template<typename T>
class LockFreeQueue {
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }
    
    bool pop(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout_ms < 0) {
            cv_.wait(lock, [this] { return !queue_.empty() || done_; });
        } else {
            if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             [this] { return !queue_.empty() || done_; })) {
                return false;
            }
        }
        
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    void set_done() {
        done_ = true;
        cv_.notify_all();
    }
    
    bool empty() const {
        return queue_.empty();
    }
    
    size_t size() const {
        return queue_.size();
    }
    
private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    std::atomic<bool> done_{false};
};

/**
 * @brief Worker thread for pipeline processing
 */
class PipelineWorker {
public:
    using ProcessFunc = std::function<void(PipelineTask&)>;
    
    PipelineWorker(int id, ProcessFunc process_func);
    ~PipelineWorker();
    
    void start();
    void stop();
    void set_task_queue(LockFreeQueue<PipelineTask>* queue);
    
    int id() const { return id_; }
    bool is_busy() const { return busy_.load(); }
    int tasks_processed() const { return tasks_processed_.load(); }
    
private:
    void worker_thread();
    
    int id_;
    ProcessFunc process_func_;
    LockFreeQueue<PipelineTask>* task_queue_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> busy_{false};
    std::atomic<int> tasks_processed_{0};
};

/**
 * @brief Multi-threaded audio processing pipeline
 */
class AudioPipeline {
public:
    using StageFunc = std::function<void(PipelineTask&)>;
    
    explicit AudioPipeline(int num_workers = 8);
    ~AudioPipeline();
    
    /**
     * @brief Add a processing stage to the pipeline
     * @param type Stage type
     * @param func Processing function
     */
    void add_stage(StageType type, StageFunc func);
    
    /**
     * @brief Start the pipeline
     */
    void start();
    
    /**
     * @brief Submit audio data for processing
     * @param tensor Input audio tensor
     * @param sample_offset Sample offset in original audio
     */
    void submit(Tensor<float> tensor, int64_t sample_offset = 0);
    
    /**
     * @brief Set callback for completed frames
     * @param callback Function called when frame is processed
     */
    void set_output_callback(std::function<void(Tensor<float>&&)> callback);
    
    /**
     * @brief Signal end of input
     */
    void finish_input();
    
    /**
     * @brief Wait for all processing to complete
     */
    void wait_complete();
    
    /**
     * @brief Stop the pipeline immediately
     */
    void stop();
    
    /**
     * @brief Get pipeline statistics
     */
    struct Stats {
        int total_tasks_submitted;
        int total_tasks_completed;
        int current_queue_size;
        double avg_processing_time_ms;
        double throughput_samples_per_sec;
    };
    Stats get_stats() const;
    
    /**
     * @brief Enable work-stealing between workers
     */
    void enable_work_stealing(bool enable = true);
    
private:
    void dispatcher_thread();
    void collector_thread();
    
    int num_workers_;
    std::vector<std::unique_ptr<PipelineWorker>> workers_;
    
    std::vector<std::pair<StageType, StageFunc>> stages_;
    LockFreeQueue<PipelineTask> input_queue_;
    LockFreeQueue<PipelineTask> output_queue_;
    
    std::thread dispatcher_thread_;
    std::thread collector_thread_;
    
    std::function<void(Tensor<float>&&)> output_callback_;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> input_finished_{false};
    std::atomic<int> tasks_submitted_{0};
    std::atomic<int> tasks_completed_{0};
    
    mutable std::mutex stats_mutex_;
    int total_processing_time_us_;
    int processing_count_;
    
    bool work_stealing_enabled_;
};

} // namespace nass_x::tensor_x
