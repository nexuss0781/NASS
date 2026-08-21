#ifndef NASS_PIPELINE_WORKER_HPP
#define NASS_PIPELINE_WORKER_HPP

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include "nass/core/types.hpp"
#include "nass/math/stft.hpp"

namespace nass_x::tensor {

// Worker node for parallel processing
class WorkerNode {
public:
    explicit WorkerNode(size_t worker_id);
    ~WorkerNode();
    
    // Non-copyable
    WorkerNode(const WorkerNode&) = delete;
    WorkerNode& operator=(const WorkerNode&) = delete;
    
    // Start worker thread
    void start();
    
    // Stop worker thread
    void stop();
    
    // Submit task to worker
    void submit_task(std::function<void()> task);
    
    // Check if worker is busy
    bool is_busy() const { return busy_; }
    
    // Get worker ID
    size_t id() const { return worker_id_; }
    
    // Get tasks processed count
    size_t tasks_processed() const { return tasks_processed_; }
    
private:
    void worker_loop();
    
    size_t worker_id_;
    std::thread thread_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_;
    std::atomic<bool> busy_;
    std::atomic<size_t> tasks_processed_;
};

// Thread pool for work-stealing
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();
    
    // Submit task to pool
    void submit(std::function<void()> task);
    
    // Wait for all tasks to complete
    void wait_all();
    
    // Get number of threads
    size_t size() const { return workers_.size(); }
    
    // Shutdown pool
    void shutdown();
    
private:
    std::vector<std::unique_ptr<WorkerNode>> workers_;
    std::atomic<size_t> next_worker_;
    std::atomic<size_t> pending_tasks_;
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;
};

} // namespace nass_x::tensor

#endif // NASS_PIPELINE_WORKER_HPP
