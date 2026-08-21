#pragma once

#include "../../nass_x/tensor/tensor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace nass_x::ai {

// Bring Tensor into namespace (using float, 8 max dims as default)
using Tensor = nass_x::tensor::Tensor<float, 8>;

/**
 * @brief Metadata for model inputs/outputs
 */
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    enum class DataType { FLOAT32, FLOAT16, INT8, INT64 } dtype;
};

/**
 * @brief Abstract base class for Neural Inference Engines
 * Supports TensorRT, ONNX Runtime, and future backends
 */
class NeuralEngine {
public:
    virtual ~NeuralEngine() = default;

    /**
     * @brief Load model from file
     * @param model_path Path to .engine (TRT) or .onnx (ORT)
     * @param use_fp16 Enable half-precision inference
     */
    virtual bool load_model(const std::string& model_path, bool use_fp16 = true) = 0;

    /**
     * @brief Run inference
     * @param inputs Vector of input tensors
     * @param outputs Vector of output tensors (pre-allocated)
     * @return Success status
     */
    virtual bool infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) = 0;

    /**
     * @brief Get input/output metadata
     */
    virtual std::vector<TensorInfo> get_input_info() const = 0;
    virtual std::vector<TensorInfo> get_output_info() const = 0;

    /**
     * @brief Get backend name (e.g., "TensorRT", "ONNX")
     */
    virtual std::string get_backend_name() const = 0;

    /**
     * @brief Check if engine is ready
     */
    virtual bool is_ready() const = 0;
};

/**
 * @brief Factory function to create appropriate engine based on file extension or hint
 */
std::unique_ptr<NeuralEngine> create_engine(const std::string& backend_type);

} // namespace nass_x::tensor_x::ai
