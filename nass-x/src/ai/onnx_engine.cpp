#include "nass_x/ai/onnx_engine.hpp"
#include <algorithm>
#include <iostream>

namespace nass_x::ai {

OnnxEngine::OnnxEngine() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "NASS_X_ONNX")
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    initialize_session_options(true); // Try CUDA first
}

OnnxEngine::~OnnxEngine() = default;

void OnnxEngine::initialize_session_options(bool use_cuda) {
#ifdef ORT_USE_CUDA
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[ONNX] Using CUDA execution provider" << std::endl;
    } else {
        std::cout << "[ONNX] Using CPU execution provider" << std::endl;
    }
#else
    std::cout << "[ONNX] CUDA not available, using CPU" << std::endl;
#endif

    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

bool OnnxEngine::load_model(const std::string& model_path, bool /*use_fp16*/) {
    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        size_t num_outputs = session_->GetOutputCount();
        
        input_names_.reserve(num_inputs);
        output_names_.reserve(num_outputs);
        input_names_cstr_.reserve(num_inputs);
        output_names_cstr_.reserve(num_outputs);
        
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.get());
            input_names_cstr_.push_back(input_names_.back().c_str());
        }
        
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
            output_names_cstr_.push_back(output_names_.back().c_str());
        }
        
        std::cout << "[ONNX] Model loaded: " << model_path 
                  << " (Inputs: " << num_inputs << ", Outputs: " << num_outputs << ")" << std::endl;
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[ONNX] Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

bool OnnxEngine::infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    if (!session_ || inputs.empty()) return false;
    
    try {
        // Prepare input tensors
        std::vector<Ort::Value> input_values;
        input_values.reserve(inputs.size());
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto shape = inputs[i].shape();
            std::vector<int64_t> int64_shape(shape.begin(), shape.end());
            
            input_values.emplace_back(
                Ort::Value::CreateTensor<float>(
                    memory_info_,
                    const_cast<float*>(static_cast<const float*>(inputs[i].data())),
                    inputs[i].size(),
                    int64_shape.data(),
                    int64_shape.size()
                )
            );
        }
        
        // Run inference
        auto output_values = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr_.data(),
            input_values.data(),
            inputs.size(),
            output_names_cstr_.data(),
            outputs.size()
        );
        
        // Copy outputs
        for (size_t i = 0; i < output_values.size() && i < outputs.size(); ++i) {
            if (output_values[i].IsTensor()) {
                auto* out_data = output_values[i].GetTensorMutableData<float>();
                size_t size = outputs[i].size();
                std::memcpy(outputs[i].data(), out_data, size * sizeof(float));
            }
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[ONNX] Inference failed: " << e.what() << std::endl;
        return false;
    }
}

std::vector<TensorInfo> OnnxEngine::get_input_info() const {
    return extract_tensor_info(input_names_, true);
}

std::vector<TensorInfo> OnnxEngine::get_output_info() const {
    return extract_tensor_info(output_names_, false);
}

std::vector<TensorInfo> OnnxEngine::extract_tensor_info(
    const std::vector<std::string>& names, bool is_input) const {
    
    std::vector<TensorInfo> info;
    if (!session_) return info;
    
    Ort::AllocatorWithDefaultOptions allocator;
    
    for (size_t i = 0; i < names.size(); ++i) {
        TensorInfo ti;
        ti.name = names[i];
        
        try {
            auto type_info = is_input ? 
                session_->GetInputTypeInfo(i) : 
                session_->GetOutputTypeInfo(i);
            
            auto tensor_info = type_info.GetTensorTypeAndShape();
            auto shape = tensor_info.GetShape();
            ti.shape.assign(shape.begin(), shape.end());
            
            auto dtype = tensor_info.GetElementType();
            switch (dtype) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    ti.dtype = TensorInfo::DataType::FLOAT32;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                    ti.dtype = TensorInfo::DataType::FLOAT16;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                    ti.dtype = TensorInfo::DataType::INT8;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    ti.dtype = TensorInfo::DataType::INT64;
                    break;
                default:
                    ti.dtype = TensorInfo::DataType::FLOAT32;
            }
        } catch (...) {
            // Skip on error
        }
        
        info.push_back(ti);
    }
    
    return info;
}

} // namespace nass_x::tensor_x::ai
