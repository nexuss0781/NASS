#include "nass_x/ai/tensorrt_engine.hpp"
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>

namespace nass_x::ai {

// Logger for TensorRT
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static TRTLogger g_logger;

TensorRTEngine::TensorRTEngine() {
    cudaStreamCreate(&stream_);
}

TensorRTEngine::~TensorRTEngine() {
    cleanup();
    cudaStreamDestroy(stream_);
}

bool TensorRTEngine::load_model(const std::string& model_path, bool use_fp16) {
    // Check file extension
    if (model_path.size() > 7 && model_path.substr(model_path.size() - 7) == ".engine") {
        // Load serialized engine
        std::ifstream file(model_path, std::ios::binary);
        if (!file) {
            std::cerr << "[TensorRT] Failed to open engine file: " << model_path << std::endl;
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        
        runtime_ = nvinfer1::createInferRuntime(g_logger);
        if (!runtime_) return false;
        
        engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
        if (!engine_) {
            std::cerr << "[TensorRT] Failed to deserialize engine" << std::endl;
            return false;
        }
    } else if (model_path.size() > 5 && model_path.substr(model_path.size() - 5) == ".onnx") {
        // Build engine from ONNX (requires NvOnnxParser)
        return build_engine_from_onnx(model_path, use_fp16);
    } else {
        std::cerr << "[TensorRT] Unsupported model format: " << model_path << std::endl;
        return false;
    }
    
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "[TensorRT] Failed to create execution context" << std::endl;
        return false;
    }
    
    return allocate_buffers();
}

bool TensorRTEngine::build_engine_from_onnx(const std::string& onnx_path, bool use_fp16) {
    // Note: Full ONNX parsing requires NvOnnxParser library
    // This is a simplified placeholder for the build process
    std::cout << "[TensorRT] Building engine from ONNX: " << onnx_path 
              << " (FP16: " << (use_fp16 ? "yes" : "no") << ")" << std::endl;
    
    // In production: 
    // 1. Create builder
    // 2. Create network definition
    // 3. Parse ONNX file
    // 4. Configure builder (FP16/INT8)
    // 5. Build optimized engine
    // 6. Serialize to disk
    
    std::cerr << "[TensorRT] ONNX parsing requires NvOnnxParser. Using pre-built .engine files recommended." << std::endl;
    return false;
}

bool TensorRTEngine::allocate_buffers() {
    int nb_bindings = engine_->getNbBindings();
    gpu_buffers_.resize(nb_bindings);
    
    for (int i = 0; i < nb_bindings; ++i) {
        auto dims = engine_->getBindingDimensions(i);
        auto dtype = engine_->getBindingDataType(i);
        
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            vol *= dims.d[d];
        }
        
        size_t element_size = 4; // Default FP32
        if (dtype == nvinfer1::DataType::kHALF) element_size = 2;
        if (dtype == nvinfer1::DataType::kINT8) element_size = 1;
        if (dtype == nvinfer1::DataType::kINT32) element_size = 4;
        
        size_t tensor_size = vol * element_size;
        
        if (engine_->bindingIsInput(i)) {
            input_sizes_.push_back(tensor_size);
        } else {
            output_sizes_.push_back(tensor_size);
        }
        
        cudaMalloc(&gpu_buffers_[i], tensor_size);
    }
    
    return true;
}

bool TensorRTEngine::infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    if (!context_ || inputs.empty()) return false;
    
    // Copy inputs to GPU
    for (size_t i = 0; i < inputs.size() && i < input_sizes_.size(); ++i) {
        cudaMemcpyAsync(gpu_buffers_[i], inputs[i].data(), 
                       input_sizes_[i], cudaMemcpyHostToDevice, stream_);
    }
    
    // Run inference
    bool status = context_->executeV2(gpu_buffers_.data());
    
    // Copy outputs back
    for (size_t i = 0; i < outputs.size() && i < output_sizes_.size(); ++i) {
        size_t idx = inputs.size() + i;
        if (idx < gpu_buffers_.size()) {
            cudaMemcpyAsync(outputs[i].data(), gpu_buffers_[idx],
                           output_sizes_[i], cudaMemcpyDeviceToHost, stream_);
        }
    }
    
    cudaStreamSynchronize(stream_);
    return status;
}

std::vector<TensorInfo> TensorRTEngine::get_input_info() const {
    std::vector<TensorInfo> info;
    int nb_bindings = engine_ ? engine_->getNbBindings() : 0;
    
    for (int i = 0; i < nb_bindings; ++i) {
        if (!engine_->bindingIsInput(i)) continue;
        
        TensorInfo ti;
        ti.name = engine_->getBindingName(i);
        auto dims = engine_->getBindingDimensions(i);
        for (int d = 0; d < dims.nbDims; ++d) {
            ti.shape.push_back(dims.d[d]);
        }
        
        auto dtype = engine_->getBindingDataType(i);
        if (dtype == nvinfer1::DataType::kFLOAT) ti.dtype = TensorInfo::DataType::FLOAT32;
        else if (dtype == nvinfer1::DataType::kHALF) ti.dtype = TensorInfo::DataType::FLOAT16;
        else if (dtype == nvinfer1::DataType::kINT8) ti.dtype = TensorInfo::DataType::INT8;
        else ti.dtype = TensorInfo::DataType::FLOAT32;
        
        info.push_back(ti);
    }
    return info;
}

std::vector<TensorInfo> TensorRTEngine::get_output_info() const {
    std::vector<TensorInfo> info;
    int nb_bindings = engine_ ? engine_->getNbBindings() : 0;
    
    for (int i = 0; i < nb_bindings; ++i) {
        if (engine_->bindingIsInput(i)) continue;
        
        TensorInfo ti;
        ti.name = engine_->getBindingName(i);
        auto dims = engine_->getBindingDimensions(i);
        for (int d = 0; d < dims.nbDims; ++d) {
            ti.shape.push_back(dims.d[d]);
        }
        
        auto dtype = engine_->getBindingDataType(i);
        if (dtype == nvinfer1::DataType::kFLOAT) ti.dtype = TensorInfo::DataType::FLOAT32;
        else if (dtype == nvinfer1::DataType::kHALF) ti.dtype = TensorInfo::DataType::FLOAT16;
        else if (dtype == nvinfer1::DataType::kINT8) ti.dtype = TensorInfo::DataType::INT8;
        else ti.dtype = TensorInfo::DataType::FLOAT32;
        
        info.push_back(ti);
    }
    return info;
}

void TensorRTEngine::cleanup() {
    for (auto* buf : gpu_buffers_) {
        if (buf) cudaFree(buf);
    }
    gpu_buffers_.clear();
    
    if (context_) delete context_;
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
    
    context_ = nullptr;
    engine_ = nullptr;
    runtime_ = nullptr;
}

} // namespace nass_x::tensor_x::ai
