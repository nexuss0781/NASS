#include "nass_x/ai/audio_processor.hpp"
#include "nass_x/ai/neural_engine.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace nass_x::ai {

// ============================================================================
// AudioPreprocessor Implementation
// ============================================================================

AudioPreprocessor::AudioPreprocessor(const Config& config) : config_(config) {}

Tensor AudioPreprocessor::audio_to_tensor(const float* audio_data, size_t num_samples) {
    // Create tensor with shape [1, channels, samples] or [1, samples] for mono
    std::vector<int64_t> shape;
    if (config_.channels > 1) {
        shape = {1, config_.channels, static_cast<int64_t>(num_samples / config_.channels)};
    } else {
        shape = {1, static_cast<int64_t>(num_samples)};
    }
    
    Tensor output(shape, nass_x::DType::FLOAT32);
    float* out_data = static_cast<float*>(output.data());
    
    if (config_.normalize) {
        // Apply normalization
        for (size_t i = 0; i < num_samples; ++i) {
            out_data[i] = audio_data[i] / config_.norm_factor;
        }
    } else {
        std::memcpy(out_data, audio_data, num_samples * sizeof(float));
    }
    
    return output;
}

void AudioPreprocessor::tensor_to_audio(const Tensor& output_tensor, float* audio_buffer, size_t& num_samples) {
    const float* in_data = static_cast<const float*>(output_tensor.data());
    num_samples = output_tensor.size();
    
    // Denormalize if needed
    if (config_.normalize) {
        for (size_t i = 0; i < num_samples; ++i) {
            audio_buffer[i] = in_data[i] * config_.norm_factor;
        }
    } else {
        std::memcpy(audio_buffer, in_data, num_samples * sizeof(float));
    }
}

// ============================================================================
// ModelManager Implementation
// ============================================================================

ModelManager::ModelManager() = default;
ModelManager::~ModelManager() = default;

bool ModelManager::load_model(const std::string& name, const std::string& path, bool use_fp16) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        auto engine = create_engine(path);
        
        if (!engine->load_model(path, use_fp16)) {
            std::cerr << "[ModelManager] Failed to load model: " << path << std::endl;
            return false;
        }
        
        ModelEntry entry;
        entry.info.name = name;
        entry.info.path = path;
        entry.info.version = "1.0"; // Could extract from model metadata
        entry.info.is_loaded = true;
        entry.engine = std::move(engine);
        
        models_[name] = std::move(entry);
        std::cout << "[ModelManager] Loaded model '" << name << "' from " << path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ModelManager] Exception loading model: " << e.what() << std::endl;
        return false;
    }
}

bool ModelManager::unload_model(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end()) {
        return false;
    }
    
    models_.erase(it);
    std::cout << "[ModelManager] Unloaded model '" << name << "'" << std::endl;
    return true;
}

bool ModelManager::hot_swap_model(const std::string& name, const std::string& new_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end()) {
        std::cerr << "[ModelManager] Model not found for hot-swap: " << name << std::endl;
        return false;
    }
    
    // Load new model temporarily
    auto new_engine = create_engine(new_path);
    if (!new_engine->load_model(new_path, true)) {
        std::cerr << "[ModelManager] Failed to load new model for hot-swap: " << new_path << std::endl;
        return false;
    }
    
    // Atomic swap
    it->second.engine = std::move(new_engine);
    it->second.info.path = new_path;
    it->second.info.version = "1.1"; // Increment version
    
    std::cout << "[ModelManager] Hot-swapped model '" << name << "' to " << new_path << std::endl;
    return true;
}

NeuralEngine* ModelManager::get_engine(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end()) {
        return nullptr;
    }
    return it->second.engine.get();
}

ModelManager::ModelInfo ModelManager::get_model_info(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = models_.find(name);
    if (it == models_.end()) {
        return ModelInfo{};
    }
    return it->second.info;
}

std::vector<std::string> ModelManager::list_models() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> names;
    names.reserve(models_.size());
    
    for (const auto& [name, _] : models_) {
        names.push_back(name);
    }
    return names;
}

size_t ModelManager::get_total_vram_usage_mb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t total = 0;
    for (const auto& [_, entry] : models_) {
        total += entry.info.vram_usage_mb;
    }
    return total;
}

} // namespace nass_x::ai
