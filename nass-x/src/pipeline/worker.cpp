#include "nass/pipeline/worker.hpp"

namespace nass {

WorkerNode::WorkerNode(size_t worker_id)
    : worker_id_(worker_id)
    , running_(false)
    , busy_(false)
    , tasks_processed_(0) {}

WorkerNode::~WorkerNode() {
    stop();
}

void WorkerNode::start() {
    if (running_) return;
    
    running_ = true;
    thread_ = std::thread(&WorkerNode::worker_loop, this);
}

void WorkerNode::stop() {
    if (!running_) return;
    
    running_ = false;
    cv_.notify_one();
    
    if (thread_.joinable()) {
        thread_.join();
    }
}

void WorkerNode::submit_task(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    cv_.notify_one();
}

void WorkerNode::worker_loop() {
    while (running_) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { 
                return !task_queue_.empty() || !running_; 
            });
            
            if (!running_ && task_queue_.empty()) {
                break;
            }
            
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }
        
        if (task) {
            busy_ = true;
            task();
            busy_ = false;
            ++tasks_processed_;
        }
    }
}

// ThreadPool implementation
ThreadPool::ThreadPool(size_t num_threads)
    : next_worker_(0)
    , pending_tasks_(0) {
    
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.push_back(std::make_unique<WorkerNode>(i));
        workers_[i]->start();
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::submit(std::function<void()> task) {
    size_t worker_idx = next_worker_++ % workers_.size();
    ++pending_tasks_;
    
    // Wrap task to track completion
    auto wrapped_task = [this, task = std::move(task)]() {
        task();
        --pending_tasks_;
        completion_cv_.notify_all();
    };
    
    workers_[worker_idx]->submit_task(std::move(wrapped_task));
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    completion_cv_.wait(lock, [this] { 
        return pending_tasks_ == 0; 
    });
}

void ThreadPool::shutdown() {
    for (auto& worker : workers_) {
        worker->stop();
    }
}

} // namespace nass
