#pragma once

#include "neural_engine.hpp"
#include "../audio_reader.hpp"
#include "../../nass_x/tensor/tensor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace nass_x::ai {

// Bring Tensor into namespace for convenience (using default float type)
using Tensor = nass_x::tensor::Tensor<float, 8>;

/**
 * @brief Converts audio buffers to neural network input tensors
 * Handles normalization, chunking, and format conversion
 */
class AudioPreprocessor {
public:
    struct Config {
        float sample_rate = 48000.0f;
        int channels = 1;
        int hop_size = 512;
        int window_size = 2048;
        bool normalize = true;
        float norm_factor = 32768.0f; // For int16 -> float32
    };
    
    explicit AudioPreprocessor();
    explicit AudioPreprocessor(const Config& config);

    /**
     * @brief Convert audio frame to model input tensor
     * @param audio_data Raw audio samples (interleaved if multi-channel)
     * @param num_samples Number of samples in the buffer
     * @return Tensor ready for inference
     */
    Tensor audio_to_tensor(const float* audio_data, size_t num_samples);

    /**
     * @brief Convert model output tensor back to audio
     * @param output_tensor Model output
     * @param[out] audio_buffer Output buffer (must be pre-allocated)
     * @param[out] num_samples Number of samples written
     */
    void tensor_to_audio(const Tensor& output_tensor, float* audio_buffer, size_t& num_samples);

private:
    Config config_;
};

/**
 * @brief Manages neural models with hot-swap capability
 * Handles loading, unloading, versioning, and resource tracking
 */
class ModelManager {
public:
    struct ModelInfo {
        std::string name;
        std::string path;
        std::string version;
        size_t vram_usage_mb{0};
        bool is_loaded{false};
    };

    ModelManager();
    ~ModelManager();

    /**
     * @brief Load a model with automatic backend selection
     * @param name Unique identifier for the model
     * @param path Path to model file (.onnx or .engine)
     * @param use_fp16 Enable half-precision
     * @return Success status
     */
    bool load_model(const std::string& name, const std::string& path, bool use_fp16 = true);

    /**
     * @brief Unload a model to free resources
     */
    bool unload_model(const std::string& name);

    /**
     * @brief Hot-swap: replace running model without interruption
     * @param name Model identifier
     * @param new_path Path to new model version
     */
    bool hot_swap_model(const std::string& name, const std::string& new_path);

    /**
     * @brief Get engine for inference
     */
    NeuralEngine* get_engine(const std::string& name);

    /**
     * @brief Get model metadata
     */
    ModelInfo get_model_info(const std::string& name) const;

    /**
     * @brief List all loaded models
     */
    std::vector<std::string> list_models() const;

    /**
     * @brief Get total VRAM usage across all models
     */
    size_t get_total_vram_usage_mb() const;

private:
    struct ModelEntry {
        ModelInfo info;
        std::unique_ptr<NeuralEngine> engine;
    };

    std::unordered_map<std::string, ModelEntry> models_;
    mutable std::mutex mutex_;
};

} // namespace nass_x::tensor_x::ai
