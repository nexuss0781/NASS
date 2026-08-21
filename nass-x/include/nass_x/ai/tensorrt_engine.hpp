#pragma once

#include "neural_engine.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

namespace nass_x::ai {

/**
 * @brief NVIDIA TensorRT Backend Implementation
 * Provides high-performance FP16/INT8 inference on NVIDIA GPUs
 */
class TensorRTEngine : public NeuralEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine() override;

    bool load_model(const std::string& model_path, bool use_fp16 = true) override;
    bool infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
    std::vector<TensorInfo> get_input_info() const override;
    std::vector<TensorInfo> get_output_info() const override;
    std::string get_backend_name() const override { return "TensorRT"; }
    bool is_ready() const override { return engine_ != nullptr; }

private:
    // TensorRT components
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    nvinfer1::IRuntime* runtime_{nullptr};
    
    // Memory management
    std::vector<void*> gpu_buffers_;
    std::vector<size_t> input_sizes_;
    std::vector<size_t> output_sizes_;
    
    // Stream for async operations
    cudaStream_t stream_;
    
    // Helper methods
    bool build_engine_from_onnx(const std::string& onnx_path, bool use_fp16);
    bool allocate_buffers();
    void cleanup();
};

} // namespace nass_x::ai
