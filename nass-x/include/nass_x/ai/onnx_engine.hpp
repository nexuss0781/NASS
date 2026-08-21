#pragma once

#include "neural_engine.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

namespace nass_x::ai {

/**
 * @brief ONNX Runtime Backend Implementation
 * Cross-platform inference supporting CPU, CUDA, TensorRT, and DirectML
 */
class OnnxEngine : public NeuralEngine {
public:
    OnnxEngine();
    ~OnnxEngine() override;

    bool load_model(const std::string& model_path, bool use_fp16 = true) override;
    bool infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
    std::vector<TensorInfo> get_input_info() const override;
    std::vector<TensorInfo> get_output_info() const override;
    std::string get_backend_name() const override { return "ONNX"; }
    bool is_ready() const override { return session_ != nullptr; }

private:
    // ONNX Runtime components
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    // Memory info
    Ort::MemoryInfo memory_info_;
    
    // Input/Output names
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    
    // Helper methods
    void initialize_session_options(bool use_cuda);
    std::vector<TensorInfo> extract_tensor_info(const std::vector<std::string>& names, bool is_input);
};

} // namespace nass_x::ai
