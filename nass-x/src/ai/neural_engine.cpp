#include "nass_x/ai/neural_engine.hpp"
#include <stdexcept>
#include <algorithm>

// Forward declarations - implementations only compiled when backends enabled
#ifdef NASS_X_USE_TENSORRT
#include "nass_x/ai/tensorrt_engine.hpp"
#endif
#ifdef NASS_X_USE_ONNX
#include "nass_x/ai/onnx_engine.hpp"
#endif

namespace nass_x::ai {

std::unique_ptr<NeuralEngine> create_engine(const std::string& backend_type) {
    std::string type = backend_type;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    if (type == "tensorrt" || type == "trt") {
#ifdef NASS_X_USE_TENSORRT
        return std::make_unique<TensorRTEngine>();
#else
        throw std::runtime_error("TensorRT support not compiled. Enable NASS_X_USE_TENSORRT.");
#endif
    }
    
    if (type == "onnx" || type == "onnxruntime" || type == "ort") {
#ifdef NASS_X_USE_ONNX
        return std::make_unique<OnnxEngine>();
#else
        throw std::runtime_error("ONNX Runtime support not compiled. Enable NASS_X_USE_ONNX.");
#endif
    }

    // Auto-detect based on availability
#ifdef NASS_X_USE_TENSORRT
    return std::make_unique<TensorRTEngine>();
#elif defined(NASS_X_USE_ONNX)
    return std::make_unique<OnnxEngine>();
#else
    throw std::runtime_error("No neural backends available. Compile with TensorRT or ONNX Runtime support.");
#endif
}

} // namespace nass_x::ai
